import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - JinaEmbeddings v3 (jina-embeddings-v3, unified task model)

/// jina-embeddings-v3 是 XLM-RoBERTa + RoPE + task LoRA 的统一任务嵌入模型：
///   - word_emb + token_type_emb[0] -> emb_ln（LoRA 在 emb_ln 之前加回，同官方实现）
///   - 24 × postnorm 块：norm1(x + MHA(x, task)), norm2(x + MLP(x, task))
///   - 均值池化（非填充 token）+ L2 归一化（由 MLXModelRuntime 完成）
///
/// task LoRA（rank 4, alpha 1）在运行时按 taskId 选取对应分支叠加到 base 权重上，
/// 与 jina_v3_mlx.py 的 unified 前向一一对应。权重键名（word_embeddings /
/// layers.{i}.mixer.{Wqkv,out_proj,lora_*} / layers.{i}.mlp.{fc1,fc2,lora_*} /
/// norm1 / norm2 / lora_emb_* / lora_tt_*）与转换产物 weights.safetensors 逐键一致。
///
/// 本实现仅作为 MLX 后端的可加载模型，不经 MLXEmbedders 注册表（其无 jina-v3 架构）；
/// 由 MLXModelRuntime 走 MLXLMCommon.loadWeights（公开 API）加载与量化。

public struct JinaV3Configuration {
    public var modelType = "jina-v3-mlx"
    public var vocabSize = 250002
    public var hiddenSize = 1024
    public var numHiddenLayers = 24
    public var numAttentionHeads = 16
    public var intermediateSize = 4096
    public var layerNormEps: Float = 1e-5
    public var typeVocabSize = 1
    public var rotaryEmbBase: Float = 20000.0
    public var loraRank = 4
    public var loraAlpha = 1

    public init(data: Data) throws {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MLXRuntimeError.loadFailed("jina-v3: config.json 无法解析")
        }
        if let v = obj["model_type"] as? String { modelType = v }
        if let v = obj["vocab_size"] as? Int { vocabSize = v }
        if let v = obj["hidden_size"] as? Int { hiddenSize = v }
        if let v = obj["num_hidden_layers"] as? Int { numHiddenLayers = v }
        if let v = obj["num_attention_heads"] as? Int { numAttentionHeads = v }
        if let v = obj["intermediate_size"] as? Int { intermediateSize = v }
        if let v = obj["type_vocab_size"] as? Int { typeVocabSize = v }
        if let v = obj["layer_norm_eps"] as? Double { layerNormEps = Float(v) }
        if let v = obj["rotary_emb_base"] as? Double { rotaryEmbBase = Float(v) }
        if let v = obj["lora_rank"] as? Int { loraRank = v }
        if let v = obj["lora_alpha"] as? Int { loraAlpha = v }
    }
}

/// 精确 GELU（erf 版），对齐 transformers 的 F.gelu(approximate="none")
private func geluExact(_ x: MLXArray) -> MLXArray {
    x * 0.5 * (1 + MLX.erf(x / sqrt(2.0)))
}

/// GPT-NeoX 风格 RoPE 的逆频率表（head_dim 半维）
private func jinaV3RotaryInvFreq(base: Float, headDim: Int) -> MLXArray {
    let half = headDim / 2
    // 用 Float 构造（[Double] 会生成 float64 数组，GPU 不支持 float64）
    return MLXArray(
        (0..<half).map {
            Float(1.0 / pow(Double(base), Double($0 * 2) / Double(headDim)))
        })
}

/// GPT-NeoX 风格（非交错、半旋转）RoPE，作用于 (B,S,H,D) 的最后一维
private func jinaV3Rotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    // x: (B,S,H,D)，cos/sin: (1,S,1,D/2)
    let half = x.dim(-1) / 2
    let x1 = x[0..<half, axis: -1]
    let x2 = x[half..<x.dim(-1), axis: -1]
    return concatenated([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis: -1)
}

/// 非 Module 的普通持有类：Module 上直接存储 MLXArray 属性会被反射收为参数
/// （update 要求权重树里有同名键），而 invFreq / 任务 LoRA 不属于结构化参数树，
/// 因此放进普通类由 loadExtra 手动挂载。
final class JinaV3MHAExtra {
    var invFreq: MLXArray
    var loraWqkvA: MLXArray
    var loraWqkvB: MLXArray
    var loraOutA: MLXArray
    var loraOutB: MLXArray

    init(_ config: JinaV3Configuration, headDim: Int) {
        invFreq = jinaV3RotaryInvFreq(base: config.rotaryEmbBase, headDim: headDim)
        loraWqkvA = MLXArray.zeros([5, config.loraRank, config.hiddenSize])
        loraWqkvB = MLXArray.zeros([5, 3 * config.hiddenSize, config.loraRank])
        loraOutA = MLXArray.zeros([5, config.loraRank, config.hiddenSize])
        loraOutB = MLXArray.zeros([5, config.hiddenSize, config.loraRank])
    }
}

private class JinaV3MHA: Module {
    @ModuleInfo(key: "Wqkv") var Wqkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let extra: JinaV3MHAExtra

    let numHeads: Int
    let headDim: Int
    let scaling: Float

    init(_ config: JinaV3Configuration) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        scaling = Float(config.loraAlpha) / Float(config.loraRank)
        _Wqkv.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize)
        _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        extra = JinaV3MHAExtra(config, headDim: headDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, taskId: Int, addMask: MLXArray?) -> MLXArray {
        let b = x.dim(0), s = x.dim(1)
        let scale = 1.0 / sqrt(Double(headDim))
        var qkv = Wqkv(x)
        let a = extra.loraWqkvA[taskId]  // (r, h)
        let bb = extra.loraWqkvB[taskId]  // (3h, r)
        qkv = qkv + ((x.matmul(a.T)) .matmul(bb.T)) * scaling

        var qkv3 = qkv.reshaped(b, s, 3, numHeads, headDim)  // (B,S,3,H,D)
        let t = MLXArray((0..<s).map { Int32($0) }).asType(.float32)
        let freqs = t.expandedDimensions(axis: 1) * extra.invFreq.expandedDimensions(axis: 0)  // (S, D/2)
        // 官方实现 cos/sin 形状 (1,S,1,D/2)：仅作用于 Q、K，V 不旋转
        let cos = MLX.cos(freqs).expandedDimensions(axes: [0, 2])  // (1,S,1,D/2)
        let sin = MLX.sin(freqs).expandedDimensions(axes: [0, 2])

        let qRaw = qkv3[0, axis: 2]  // (B,S,H,D)
        let kRaw = qkv3[1, axis: 2]
        let v = qkv3[2, axis: 2].transposed(0, 2, 1, 3)  // (B,H,S,D)
        let q = jinaV3Rotary(qRaw, cos: cos, sin: sin).transposed(0, 2, 1, 3)  // (B,H,S,D)
        let k = jinaV3Rotary(kRaw, cos: cos, sin: sin).transposed(0, 2, 1, 3)

        var scores = (q * scale).matmul(k.transposed(0, 1, 3, 2))  // (B,H,S,S)
        if let addMask { scores = scores + addMask }
        let probs = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        // 官方实现：LoRA 加在 out_proj 输出上（out = out_proj(attn) + lora(attn)），
        // 不能把 delta 先加进 attn 再过 out_proj
        let attn = (probs.matmul(v)).transposed(0, 2, 1, 3).reshaped(b, s, numHeads * headDim)
        var out = outProj(attn)
        let oa = extra.loraOutA[taskId]
        let ob = extra.loraOutB[taskId]
        out = out + ((attn.matmul(oa.T)) .matmul(ob.T)) * scaling
        return out
    }
}

final class JinaV3MLPExtra {
    var loraFc1A: MLXArray
    var loraFc1B: MLXArray
    var loraFc2A: MLXArray
    var loraFc2B: MLXArray

    init(_ config: JinaV3Configuration) {
        loraFc1A = MLXArray.zeros([5, config.loraRank, config.hiddenSize])
        loraFc1B = MLXArray.zeros([5, config.intermediateSize, config.loraRank])
        loraFc2A = MLXArray.zeros([5, config.loraRank, config.intermediateSize])
        loraFc2B = MLXArray.zeros([5, config.hiddenSize, config.loraRank])
    }
}

private class JinaV3MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let extra: JinaV3MLPExtra

    let scaling: Float

    init(_ config: JinaV3Configuration) {
        scaling = Float(config.loraAlpha) / Float(config.loraRank)
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        extra = JinaV3MLPExtra(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, taskId: Int) -> MLXArray {
        var h = fc1(x)
        let a = extra.loraFc1A[taskId]
        let bb = extra.loraFc1B[taskId]
        h = h + ((x.matmul(a.T)) .matmul(bb.T)) * scaling
        h = geluExact(h)  // 精确 GELU（erf），对齐 torch F.gelu(approximate="none")
        var out = fc2(h)
        let oa = extra.loraFc2A[taskId]
        let ob = extra.loraFc2B[taskId]
        out = out + ((h.matmul(oa.T)) .matmul(ob.T)) * scaling
        return out
    }
}

private class JinaV3Block: Module {
    @ModuleInfo(key: "mixer") var mixer: JinaV3MHA
    @ModuleInfo(key: "mlp") var mlp: JinaV3MLP
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    init(_ config: JinaV3Configuration) {
        _mixer.wrappedValue = JinaV3MHA(config)
        _mlp.wrappedValue = JinaV3MLP(config)
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, taskId: Int, addMask: MLXArray?) -> MLXArray {
        let h = norm1(mixer(x, taskId: taskId, addMask: addMask) + x)
        return norm2(mlp(h, taskId: taskId) + h)
    }
}

final class JinaV3ModelExtra {
    var loraEmbA: MLXArray
    var loraEmbB: MLXArray
    var loraTtA: MLXArray
    var loraTtB: MLXArray

    init(_ config: JinaV3Configuration) {
        loraEmbA = MLXArray.zeros([5, config.vocabSize, config.loraRank])
        loraEmbB = MLXArray.zeros([5, config.loraRank, config.hiddenSize])
        loraTtA = MLXArray.zeros([5, config.typeVocabSize, config.loraRank])
        loraTtB = MLXArray.zeros([5, config.loraRank, config.hiddenSize])
    }
}

public final class JinaV3Model: Module, BaseLanguageModel {
    // 注意：v3 的 weights.safetensors 为扁平键（word_embeddings / token_type_embeddings /
    // emb_ln / layers.{i}.* 直接挂在顶层），因此这些权重直接挂在模型顶层（与 jina_v3_mlx.py 一致）。
    @ModuleInfo(key: "word_embeddings") private var wordEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") private var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "emb_ln") private var embLn: LayerNorm
    @ModuleInfo(key: "layers") private var layers: [JinaV3Block]
    let extra: JinaV3ModelExtra

    private let config: JinaV3Configuration
    private let scaling: Float

    public init(_ config: JinaV3Configuration) {
        self.config = config
        self.scaling = Float(config.loraAlpha) / Float(config.loraRank)
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _tokenTypeEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.typeVocabSize, dimensions: config.hiddenSize)
        _embLn.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in JinaV3Block(config) }
        extra = JinaV3ModelExtra(config)
        super.init()
    }

    /// 返回 last_hidden_state [B, S, H]（池化 + L2 由调用方完成）
    public func callAsFunction(
        _ inputIds: MLXArray,
        attentionMask: MLXArray?,
        tokenTypeIds: MLXArray? = nil,
        taskId: Int = 1
    ) -> MLXArray {
        let types = tokenTypeIds ?? MLXArray.zeros(inputIds.shape, dtype: .int32)
        var h = wordEmbeddings(inputIds) + tokenTypeEmbeddings(types)

        // embedding 级 task LoRA（emb_ln 之前加回，同官方实现）
        let dE = extra.loraEmbA[taskId].matmul(extra.loraEmbB[taskId])  // (vocab, h)
        h = h + dE[inputIds] * scaling
        let dT = extra.loraTtA[taskId].matmul(extra.loraTtB[taskId])  // (typeVocab, h)
        h = h + dT[types] * scaling

        h = embLn(h)

        var mask: MLXArray?
        if let attentionMask {
            let m = attentionMask.asType(.float32).expandedDimensions(axes: [1, 2])  // (B,1,1,S)
            // 官方实现用 -10000 的加性 bias（fp16 下 -max 会溢出成 -inf，行为不同）
            mask = (1.0 - m) * MLXArray(-10000.0)
        }
        for (li, layer) in layers.enumerated() {
            h = layer(h, taskId: taskId, addMask: mask)
        }
        return h
    }

    /// 手动挂载权重中的任务 LoRA 张量（这些不走 update 的参数树，
    /// 由加载方在结构化权重 update 完成后调用；weights 键与 safetensors 一致）。
    /// invFreq 不保存在权重中（Python 端由配置现算，本实现已在持有类初始化时算好），
    /// 若权重里带该键则覆盖。
    func loadExtra(weights: [String: MLXArray]) {
        for (i, layer) in layers.enumerated() {
            func get(_ key: String) -> MLXArray {
                guard let v = weights[key] else {
                    fatalError("JinaV3 loadExtra: 缺少权重键 \(key)")
                }
                return v
            }
            if let inv = weights["layers.\(i).mixer.invFreq"] {
                layer.mixer.extra.invFreq = inv
            }
            layer.mixer.extra.loraWqkvA = get("layers.\(i).mixer.lora_Wqkv_A")
            layer.mixer.extra.loraWqkvB = get("layers.\(i).mixer.lora_Wqkv_B")
            layer.mixer.extra.loraOutA = get("layers.\(i).mixer.lora_out_A")
            layer.mixer.extra.loraOutB = get("layers.\(i).mixer.lora_out_B")
            layer.mlp.extra.loraFc1A = get("layers.\(i).mlp.lora_fc1_A")
            layer.mlp.extra.loraFc1B = get("layers.\(i).mlp.lora_fc1_B")
            layer.mlp.extra.loraFc2A = get("layers.\(i).mlp.lora_fc2_A")
            layer.mlp.extra.loraFc2B = get("layers.\(i).mlp.lora_fc2_B")
        }
        extra.loraEmbA = weights["lora_emb_A"]!
        extra.loraEmbB = weights["lora_emb_B"]!
        extra.loraTtA = weights["lora_tt_A"]!
        extra.loraTtB = weights["lora_tt_B"]!
    }
}
