import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - JinaBERT v2 (jina-embeddings-v2-base-zh)

/// JinaBERT v2 与标准 BERT 的三点差异，导致它不能复用 MLXEmbedders 的 BertModel：
///  1. ALiBi 位置偏置（无 learned position embeddings，注意力分数内加相对距离偏置）
///  2. GEGLU 门控前馈（gated_layers 一次投影 [h, 2*inter]，chunk 后 GELU(gate)*up，
///     且 LayerNorm 在 FFN 块内部）
///  3. 句向量 = 非填充 token 均值池化 + L2 归一化（由 MLXModelRuntime 的 jina 分支完成）
///
/// 另外不注册进 EmbedderTypeRegistry：EmbeddingModelOutput 的构造器是库内 internal，
/// 应用侧无法构造 EmbeddingModel 的返回值；这里直接实现 BaseLanguageModel + 返回
/// hidden states，由 MLXModelRuntime 走 MLXLMCommon.loadWeights（公开 API）自行加载。
///
/// 模块树与权重键名（embeddings.word_embeddings / encoder.layer.N.attention.self.query /
/// encoder.layer.N.mlp.gated_layers / pooler.dense ...）与转换产物 safetensors 逐键一致；
/// 量化复用 loadWeights 的通用路径（config.json 的 quantization + 权重内 .scales/.biases
/// 兄弟键 → 自动转 QuantizedLinear）。

public struct JinaBertConfiguration {
    public var modelType = "jina_bert"
    public var vocabSize = 61056
    public var hiddenSize = 768
    public var numHiddenLayers = 12
    public var numAttentionHeads = 12
    public var intermediateSize = 3072
    public var typeVocabSize = 2
    public var layerNormEps: Float = 1e-12
    public var maxPositionEmbeddings = 8192

    public init(data: Data) throws {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MLXRuntimeError.loadFailed("jina_bert: config.json 无法解析")
        }
        if let v = obj["model_type"] as? String { modelType = v }
        if let v = obj["vocab_size"] as? Int { vocabSize = v }
        if let v = obj["hidden_size"] as? Int { hiddenSize = v }
        if let v = obj["num_hidden_layers"] as? Int { numHiddenLayers = v }
        if let v = obj["num_attention_heads"] as? Int { numAttentionHeads = v }
        if let v = obj["intermediate_size"] as? Int { intermediateSize = v }
        if let v = obj["type_vocab_size"] as? Int { typeVocabSize = v }
        if let v = obj["layer_norm_eps"] as? Double { layerNormEps = Float(v) }
        if let v = obj["max_position_embeddings"] as? Int { maxPositionEmbeddings = v }
    }
}

/// alibi[h, i, j] = -slope_h * |i - j|（与 jina-bert-implementation 一致）
private func jinaAlibiSlopes(_ numHeads: Int) -> [Float] {
    func slopesPowerOf2(_ n: Int) -> [Float] {
        // start = 2^(-(2^-(log2(n)-3)))；外层指数带负号（漏掉会得到 4 倍斜率）
        let start = Float(pow(2.0, -pow(2.0, -(log2(Double(n)) - 3.0))))
        return (0..<n).map { start * pow(start, Float($0)) }
    }
    let l = log2(Double(numHeads))
    if l == l.rounded(.down) { return slopesPowerOf2(numHeads) }
    let closest = Int(pow(2.0, l.rounded(.down)))
    let extra = slopesPowerOf2(2 * closest)
        .enumerated().filter { $0.offset % 2 == 0 }.map(\.element)
    return slopesPowerOf2(closest) + Array(extra.prefix(numHeads - closest))
}

private func buildAlibi(numHeads: Int, seqLen: Int) -> MLXArray {
    let slopes = MLXArray(jinaAlibiSlopes(numHeads).map { -$0 }).reshaped(numHeads, 1, 1)
    let ar = MLXArray(0..<Int32(seqLen))
    let rel = MLX.abs(ar.expandedDimensions(axes: [0]) - ar.expandedDimensions(axes: [1]))
        .asType(.float32)  // [S, S]
    return slopes * rel  // [H, S, S]
}

private class JinaBertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(_ config: JinaBertConfiguration) {
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _tokenTypeEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.typeVocabSize, dimensions: config.hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, tokenTypeIds: MLXArray?) -> MLXArray {
        let types = tokenTypeIds ?? MLXArray.zeros(inputIds.shape, dtype: .int32)
        return layerNorm(wordEmbeddings(inputIds) + tokenTypeEmbeddings(types))
    }
}

private class JinaBertSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    let numHeads: Int
    let headDim: Int

    init(_ config: JinaBertConfiguration) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        _query.wrappedValue = Linear(config.hiddenSize, numHeads * headDim)
        _key.wrappedValue = Linear(config.hiddenSize, numHeads * headDim)
        _value.wrappedValue = Linear(config.hiddenSize, numHeads * headDim)
        super.init()
    }

    private func splitHeads(_ x: MLXArray, _ lin: Linear) -> MLXArray {
        // [B, S, H*D] -> [B, H, S, D]
        lin(x).reshaped(x.dim(0), x.dim(1), numHeads, headDim).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, alibi: MLXArray?) -> MLXArray {
        let b = x.dim(0), s = x.dim(1)
        let q = splitHeads(x, query)
        let k = splitHeads(x, key)
        let v = splitHeads(x, value)

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) / Float(sqrt(Double(headDim)))
        if let alibi { scores = scores + alibi.expandedDimensions(axes: [0]) }  // [1,H,S,S]
        if let mask { scores = scores + mask }
        let probs = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        return matmul(probs, v).transposed(0, 2, 1, 3).reshaped(b, s, numHeads * headDim)
    }
}

private class JinaBertSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(_ config: JinaBertConfiguration) {
        _dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, _ residual: MLXArray) -> MLXArray {
        layerNorm(dense(hidden) + residual)
    }
}

private class JinaBertAttention: Module {
    @ModuleInfo(key: "self") var selfAttention: JinaBertSelfAttention
    @ModuleInfo(key: "output") var output: JinaBertSelfOutput

    init(_ config: JinaBertConfiguration) {
        _selfAttention.wrappedValue = JinaBertSelfAttention(config)
        _output.wrappedValue = JinaBertSelfOutput(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, alibi: MLXArray?) -> MLXArray {
        output(selfAttention(x, mask: mask, alibi: alibi), x)
    }
}

/// GEGLU 前馈：GELU(gate) * up，LayerNorm 在块内（post-LN）
private class JinaBertGLUMLP: Module {
    @ModuleInfo(key: "gated_layers") var gatedLayers: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm

    let intermediateSize: Int

    init(_ config: JinaBertConfiguration) {
        intermediateSize = config.intermediateSize
        _gatedLayers.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize * 2, bias: false)
        _wo.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = gatedLayers(x).split(parts: 2, axis: -1)
        return layerNorm(wo(gelu(h[0]) * h[1]) + x)
    }
}

private class JinaBertLayer: Module {
    @ModuleInfo(key: "attention") var attention: JinaBertAttention
    @ModuleInfo(key: "mlp") var mlp: JinaBertGLUMLP

    init(_ config: JinaBertConfiguration) {
        _attention.wrappedValue = JinaBertAttention(config)
        _mlp.wrappedValue = JinaBertGLUMLP(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, alibi: MLXArray?) -> MLXArray {
        mlp(attention(x, mask: mask, alibi: alibi))
    }
}

private class JinaBertEncoder: Module {
    @ModuleInfo(key: "layer") var layers: [JinaBertLayer]

    let numHeads: Int

    init(_ config: JinaBertConfiguration) {
        numHeads = config.numAttentionHeads
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in JinaBertLayer(config) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let a = buildAlibi(numHeads: numHeads, seqLen: x.dim(1)).asType(x.dtype)
        return layers.reduce(x) { $1($0, mask: mask, alibi: a) }
    }
}

/// 权重兼容占位（sentence-transformers 用均值池化，不用 pooler）
private class JinaBertPooler: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ config: JinaBertConfiguration) {
        _dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        tanh(dense(x))
    }
}

public final class JinaBertEmbeddingModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "embeddings") private var embeddings: JinaBertEmbeddings
    @ModuleInfo(key: "encoder") private var encoder: JinaBertEncoder
    @ModuleInfo(key: "pooler") private var pooler: JinaBertPooler

    private let config: JinaBertConfiguration

    public init(_ config: JinaBertConfiguration) {
        self.config = config
        _embeddings.wrappedValue = JinaBertEmbeddings(config)
        _encoder.wrappedValue = JinaBertEncoder(config)
        _pooler.wrappedValue = JinaBertPooler(config)
        super.init()
    }

    /// 返回 last_hidden_state [B, S, H]；池化 + L2 由调用方完成
    public func callAsFunction(
        _ inputIds: MLXArray,
        attentionMask: MLXArray?,
        tokenTypeIds: MLXArray? = nil
    ) -> MLXArray {
        let x = embeddings(inputIds, tokenTypeIds: tokenTypeIds)
        var mask: MLXArray?
        if let attentionMask {
            let m = attentionMask.asType(x.dtype).expandedDimensions(axes: [1, 2])  // [B,1,1,S]
            mask = (1.0 - m) * MLXArray(-Float.greatestFiniteMagnitude)
        }
        return encoder(x, mask: mask)
    }
}
