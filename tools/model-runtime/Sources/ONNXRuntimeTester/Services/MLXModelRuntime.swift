import Foundation
import MLX
import MLXEmbedders
import MLXRerankers
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

/// MLX 后端：使用 mlx-swift-lm 的 MLXEmbedders / MLXRerankers 加载并运行
/// MLX 格式的 embedding 与 reranker 模型（BERT / Qwen3 / NomicBert / BGE-M3 等，
/// 由库自动按 config.json 识别架构）。
///
/// MLX 自带分词器与池化/打分逻辑，无需 swift-transformers，也无需 ONNX Runtime。
final class MLXModelRuntime: @unchecked Sendable {
    let kind: ModelKind
    let name: String
    let loadDuration: TimeInterval
    let fileBytes: Int64
    let modelDirectory: URL
    /// config.json 的 model_type
    private let modelType: String

    private let embeddingContainer: EmbedderModelContainer?
    private let rerankerContainer: RerankerContainer?

    /// jina_bert 专用：库的 EmbeddingModelOutput 构造器 internal，无法注册进
    /// EmbedderTypeRegistry，改走自管加载（MLXLMCommon.loadWeights）+ 自管池化
    private struct JinaContext {
        let model: JinaBertEmbeddingModel
        let tokenizer: any MLXLMCommon.Tokenizer
    }
    private let jina: JinaContext?

    private init(
        kind: ModelKind,
        name: String,
        loadDuration: TimeInterval,
        fileBytes: Int64,
        modelDirectory: URL,
        modelType: String,
        embeddingContainer: EmbedderModelContainer?,
        rerankerContainer: RerankerContainer?,
        jina: JinaContext? = nil
    ) {
        self.kind = kind
        self.name = name
        self.loadDuration = loadDuration
        self.fileBytes = fileBytes
        self.modelDirectory = modelDirectory
        self.modelType = modelType
        self.embeddingContainer = embeddingContainer
        self.rerankerContainer = rerankerContainer
        self.jina = jina
    }

    /// 加载 MLX 模型：embedding 用 EmbedderModelFactory，reranker 用 RerankerModelFactory。
    /// 两者都从本地目录加载（目录需包含 config.json、权重与 tokenizer 文件）。
    /// jina_bert（JinaBERT v2，ALiBi + GEGLU）不在库的内置注册表中，走专用分支。
    static func load(directory: URL, kind: ModelKind) async throws -> MLXModelRuntime {
        let start = ContinuousClock.now
        let loader = #huggingFaceTokenizerLoader()

        let modelType = (try? Data(contentsOf: directory.appendingPathComponent("config.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["model_type"]
            as? String ?? ""

        let (embeddingContainer, rerankerContainer): (EmbedderModelContainer?, RerankerContainer?)
        do {
            if kind == .embedding, modelType == "jina_bert" {
                let configData = try Data(
                    contentsOf: directory.appendingPathComponent("config.json"))
                let baseConfig = try JSONDecoder.json5().decode(
                    BaseConfiguration.self, from: configData)
                let model = JinaBertEmbeddingModel(try JinaBertConfiguration(data: configData))
                try await loadWeights(
                    modelDirectory: directory, model: model,
                    perLayerQuantization: baseConfig.perLayerQuantization)
                let tokenizer = try await loader.load(from: directory)
                let elapsed = start.duration(to: .now)
                let loadDuration = Double(elapsed.components.seconds)
                    + Double(elapsed.components.attoseconds) / 1e18
                return MLXModelRuntime(
                    kind: kind, name: directory.lastPathComponent,
                    loadDuration: loadDuration, fileBytes: modelFileBytes(directory),
                    modelDirectory: directory, modelType: modelType,
                    embeddingContainer: nil, rerankerContainer: nil,
                    jina: JinaContext(model: model, tokenizer: tokenizer))
            }

            if kind == .embedding {
                embeddingContainer = try await EmbedderModelFactory.shared.loadContainer(
                    from: directory, using: loader)
                rerankerContainer = nil
            } else {
                embeddingContainer = nil
                rerankerContainer = try await RerankerModelFactory.shared.loadContainer(
                    from: directory, using: loader, allowUnverifiedModel: true)
            }
        } catch {
            throw MLXRuntimeError.loadFailed("\(error.localizedDescription)")
        }

        let elapsed = start.duration(to: .now)
        let loadDuration = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let fileBytes = modelFileBytes(directory)

        return MLXModelRuntime(
            kind: kind,
            name: directory.lastPathComponent,
            loadDuration: loadDuration,
            fileBytes: fileBytes,
            modelDirectory: directory,
            modelType: modelType,
            embeddingContainer: embeddingContainer,
            rerankerContainer: rerankerContainer
        )
    }

    var info: LoadedModel {
        let isEmbedding = embeddingContainer != nil || jina != nil
        return LoadedModel(
            kind: kind,
            backend: .mlx,
            name: name,
            path: modelDirectory.path,
            inputNames: isEmbedding ? ["input_ids", "attention_mask", "token_type_ids"] : [],
            outputNames: isEmbedding ? ["pooled_output"] : ["logits"],
            coreMLEnabled: false,
            loadDuration: loadDuration,
            fileBytes: fileBytes
        )
    }
}

enum MLXRuntimeError: LocalizedError {
    case loadFailed(String)
    case runFailed(String)
    case unsupported(String)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let m): return "MLX 模型加载失败：\(m)"
        case .runFailed(let m): return "MLX 推理失败：\(m)"
        case .unsupported(let m): return m
        }
    }
}

extension MLXModelRuntime: ModelRuntime {
    var backend: BackendKind { .mlx }
    var statusDescription: String { "MLX" }
    var inputNames: [String] {
        (embeddingContainer != nil || jina != nil)
            ? ["input_ids", "attention_mask", "token_type_ids"] : []
    }
    var outputNames: [String] {
        (embeddingContainer != nil || jina != nil) ? ["pooled_output"] : ["logits"]
    }

    func embed(texts: [String], maxLength: Int) async throws -> (vectors: [[Float]], tokens: Int) {
        if let jina {
            return try jinaEmbed(jina, texts: texts, maxLength: maxLength)
        }
        guard let container = embeddingContainer else {
            throw MLXRuntimeError.unsupported("该 MLX 模型不是 embedding 模型")
        }
        let (vectors, tokens) = await container.perform { ctx in
            let encoded = texts.map { ctx.tokenizer.encode(text: $0, addSpecialTokens: true) }
            let trimmed = encoded.map { Array($0.prefix(maxLength)) }
            let seq = trimmed.map { $0.count }.max() ?? 1
            let batch = trimmed.count
            var idFlat: [Int32] = []
            var maskFlat: [Float] = []
            for e in trimmed {
                for i in 0..<seq {
                    if i < e.count {
                        idFlat.append(Int32(e[i]))
                        maskFlat.append(1)
                    } else {
                        idFlat.append(0)
                        maskFlat.append(0)
                    }
                }
            }
            let ids = MLXArray(idFlat).reshaped([batch, seq])
            let mask = MLXArray(maskFlat).reshaped([batch, seq])
            let types = MLXArray([Int32](repeating: 0, count: batch * seq)).reshaped([batch, seq])

            let output = ctx.model(ids, positionIds: nil, tokenTypeIds: types, attentionMask: mask)
            let pooled = ctx.pooling(output, mask: mask, normalize: true, applyLayerNorm: true)
            MLX.eval(pooled)
            let flat = pooled.asArray(Float.self)
            let tokenCount = trimmed.reduce(0) { $0 + $1.count }
            return (flat, tokenCount)
        }

        let hidden = batchSize(for: vectors.count, batch: texts.count)
        var result: [[Float]] = []
        for b in 0..<texts.count {
            result.append(Array(vectors[b * hidden..<(b + 1) * hidden]))
        }
        return (result, tokens)
    }

    /// jina_bert：非填充 token 均值池化 + L2 归一化（无池化后 LayerNorm）
    private func jinaEmbed(
        _ jina: JinaContext, texts: [String], maxLength: Int
    ) throws -> (vectors: [[Float]], tokens: Int) {
        let encoded = texts.map { jina.tokenizer.encode(text: $0, addSpecialTokens: true) }
        let trimmed = encoded.map { Array($0.prefix(maxLength)) }
        let seq = max(1, trimmed.map { $0.count }.max() ?? 1)
        let batch = trimmed.count
        var idFlat: [Int32] = []
        var maskFlat: [Float] = []
        for e in trimmed {
            for i in 0..<seq {
                if i < e.count {
                    idFlat.append(Int32(e[i]))
                    maskFlat.append(1)
                } else {
                    idFlat.append(0)
                    maskFlat.append(0)
                }
            }
        }
        let ids = MLXArray(idFlat).reshaped([batch, seq])
        let mask = MLXArray(maskFlat).reshaped([batch, seq])

        let hidden = jina.model(ids, attentionMask: mask)
        let m = mask.asType(.float32)  // [B, S]
        let counts = MLX.maximum(m.sum(axis: -1, keepDims: true), MLXArray(Float(1e-9)))  // [B,1]
        var pooled = (hidden.asType(.float32) * m.expandedDimensions(axes: [-1])).sum(axis: 1) / counts
        let norm = MLX.sqrt((pooled * pooled).sum(axis: -1, keepDims: true))
        pooled = pooled / MLX.maximum(norm, MLXArray(Float(1e-12)))
        MLX.eval(pooled)

        let hiddenDim = pooled.dim(-1)
        var result: [[Float]] = []
        for b in 0..<batch {
            let row = pooled[b].asArray(Float.self)
            precondition(row.count == hiddenDim, "embedding 维度异常")
            result.append(row)
        }
        let tokenCount = trimmed.reduce(0) { $0 + $1.count }
        return (result, tokenCount)
    }

    func rerank(pairs: [(query: String, document: String)], maxLength: Int) async throws -> (scores: [Float], tokens: Int) {
        guard let container = rerankerContainer else {
            throw MLXRuntimeError.unsupported("该 MLX 模型不是 reranker 模型")
        }
        var scores: [Float] = []
        for pair in pairs {
            let response = try await container.scores(
                query: pair.query,
                documents: [pair.document],
                instruction: nil
            )
            let score = response.results.first?.score ?? 0
            scores.append(Float(score))
        }
        // MLX reranker 容器不对外暴露分词器，token 数无法精确统计，记为 0（itemsPerSecond 仍有效）
        return (scores, 0)
    }

    private func batchSize(for flatCount: Int, batch: Int) -> Int {
        batch > 0 ? max(1, flatCount / batch) : 1
    }
}

/// 计算目录（MLX 模型权重所在）的总字节数
func modelFileBytes(_ url: URL) -> Int64 {
    let fileManager = FileManager.default
    var isDir: ObjCBool = false
    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir) else { return 0 }
    if !isDir.boolValue {
        return (try? fileManager.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
    }
    var total: Int64 = 0
    if let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
    }
    return total
}
