import Foundation
import OnnxRuntimeBindings

/// 一次加载请求：模型路径 + 选择的后端 + 角色（Embedding / Reranker）
struct ModelLoadRequest {
    var path: String
    var backend: BackendKind
    var kind: ModelKind
    /// 仅对 jina-embeddings-v3 生效：指定 LoRA 任务分支
    /// （retrieval.query / retrieval.passage / separation / classification / text-matching），默认 retrieval.passage
    var task: String? = nil
}

/// 基准测试参数
struct BenchmarkConfig {
    var batchSize: Int = 16
    /// 测试条目数（句子数或 query-doc 对数）
    var testCount: Int = 128
    var maxLength: Int = 128
}

/// 基准测试引擎：管理 ORT 环境、并按后端加载若干模型槽（每个槽独立选择角色与后端），
/// 生成测试数据并测量推理速度。推理逻辑完全由 `ModelRuntime` 抽象屏蔽（ONNX / CoreML / MLX）。
actor BenchmarkEngine {

    private let env: ORTEnv

    /// 已加载的模型槽：每个槽有独立角色（Embedding/Reranker）与运行时
    private struct LoadedSlot {
        var index: Int
        var kind: ModelKind
        var runtime: any ModelRuntime
        var path: String
    }

    private var slots: [LoadedSlot] = []
    private var tokenizerDescription = ""

    init() throws {
        self.env = try ORTEngine.makeEnv()
    }

    // MARK: - 加载

    /// 加载若干模型槽（每个槽可独立选择角色与后端），并自动定位 swift-transformers 分词器
    func load(_ requests: [ModelLoadRequest?], coreML: CoreMLOptions) async throws -> LoadReport {
        slots = []
        tokenizerDescription = ""

        var report = LoadReport(models: [], tokenizerDescription: "")
        var tokenizerCandidates: [URL] = []

        for (index, req) in requests.enumerated() {
            guard let req else { continue }
            let (runtime, path) = try await loadRuntime(
                req, kind: req.kind, coreML: coreML, tokenizerCandidates: &tokenizerCandidates)
            var info = runtime.info
            info.path = path
            slots.append(LoadedSlot(index: index, kind: req.kind, runtime: runtime, path: path))
            report.models.append(info)
        }

        guard !slots.isEmpty else {
            throw ORTEngineError.sessionCreationFailed("请至少选择一个模型")
        }

        // swift-transformers 分词器：仅 ONNX / CoreML 后端需要（MLX 自带分词器）
        if !tokenizerCandidates.isEmpty {
            do {
                let (loaded, usedDir) = try await TokenizerService.load(fromDirectories: tokenizerCandidates)
                tokenizerDescription = "\(loaded.description)（来自 \(usedDir.path)）"
                assignTokenizer(loaded)
            } catch {
                tokenizerDescription = "未找到（\(error.localizedDescription)）"
            }
        } else {
            tokenizerDescription = "MLX 自带分词器（无需 swift-transformers）"
            assignTokenizer(nil)
        }
        report.tokenizerDescription = tokenizerDescription

        return report
    }

    /// 将共享分词器注入 ONNX / CoreML 运行时（MLX 运行时忽略）
    private func assignTokenizer(_ tokenizer: (any TextTokenizer)?) {
        for slot in slots {
            if let r = slot.runtime as? ORTModelRuntime { r.tokenizer = tokenizer }
            if let r = slot.runtime as? CoreMLModelRuntime { r.tokenizer = tokenizer }
        }
    }

    /// 按后端分发加载，返回运行时与用于展示的解析路径
    private func loadRuntime(
        _ req: ModelLoadRequest,
        kind: ModelKind,
        coreML: CoreMLOptions,
        tokenizerCandidates: inout [URL]
    ) async throws -> (runtime: any ModelRuntime, resolvedPath: String) {
        switch req.backend {
        case .onnx:
            let resolved = try ModelPathResolver.resolve(req.path)
            addUnique(resolved.modelURL.deletingLastPathComponent(), to: &tokenizerCandidates)
            addUnique(resolved.modelURL.deletingLastPathComponent().deletingLastPathComponent(), to: &tokenizerCandidates)
            let runtime = try ORTEngine.load(
                url: resolved.modelURL, kind: kind, env: env, coreML: coreML, task: req.task)
            return (runtime, resolved.modelURL.path)
        case .coreml:
            let url = URL(fileURLWithPath: req.path).standardizedFileURL
            addUnique(url.deletingLastPathComponent(), to: &tokenizerCandidates)
            addUnique(url.deletingLastPathComponent().deletingLastPathComponent(), to: &tokenizerCandidates)
            let runtime = try CoreMLModelRuntime.load(url: url, kind: kind, cpuOnly: coreML.cpuOnly)
            return (runtime, url.path)
        case .mlx:
            let url = URL(fileURLWithPath: req.path).standardizedFileURL
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
                throw MLXRuntimeError.loadFailed("路径不存在：\(req.path)")
            }
            let dir = isDir.boolValue ? url : url.deletingLastPathComponent()
            let runtime = try await MLXModelRuntime.load(directory: dir, kind: kind, task: req.task)
            return (runtime, dir.path)
        }
    }

    private func addUnique(_ url: URL, to list: inout [URL]) {
        if !list.contains(url) { list.append(url) }
    }

    func unload() {
        slots = []
        tokenizerDescription = ""
    }

    // MARK: - 基准测试

    nonisolated func runBenchmarks(config: BenchmarkConfig) -> AsyncStream<BenchmarkEvent> {
        AsyncStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.yield(.failed("引擎已释放"))
                    continuation.finish()
                    return
                }
                await self.performBenchmarks(config: config, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func performBenchmarks(
        config: BenchmarkConfig,
        continuation: AsyncStream<BenchmarkEvent>.Continuation
    ) async {
        do {
            for slot in slots {
                if Task.isCancelled { break }
                if slot.kind == .embedding {
                    try await runEmbeddingBenchmark(
                        slot: slot.index, runtime: slot.runtime, config: config, continuation: continuation)
                } else {
                    try await runRerankerBenchmark(
                        slot: slot.index, runtime: slot.runtime, config: config, continuation: continuation)
                }
                if Task.isCancelled { break }
            }
            continuation.finish()
        } catch {
            let message = error.localizedDescription
            continuation.yield(.failed(message + coreMLCompatibilityHint(message)))
            continuation.finish()
        }
    }

    // MARK: - Embedding

    private func runEmbeddingBenchmark(
        slot: Int,
        runtime: any ModelRuntime,
        config: BenchmarkConfig,
        continuation: AsyncStream<BenchmarkEvent>.Continuation
    ) async throws {

        let texts = TestCorpus.generateTexts(count: config.testCount)
        continuation.yield(.progress("Embedding（\(runtime.backend.rawValue)）：正在准备 \(texts.count) 条测试文本（最大长度 \(config.maxLength)）…"))

        let batches = texts.chunked(into: config.batchSize)
        guard !batches.isEmpty else { return }

        // 预热（不计入统计）
        _ = try await runtime.embed(texts: batches[0], maxLength: config.maxLength)

        var runLatencies: [Double] = []
        var batchLatencies: [Double] = []
        var totalTokens = 0

        for (index, batch) in batches.enumerated() {
            if Task.isCancelled { break }
            let start = ContinuousClock.now
            let (vectors, tokens) = try await runtime.embed(texts: batch, maxLength: config.maxLength)
            let elapsed = milliseconds(since: start)

            runLatencies.append(elapsed)
            batchLatencies.append(elapsed)
            totalTokens += tokens

            if index == 0, let first = vectors.first {
                continuation.yield(.progress("Embedding（\(runtime.backend.rawValue)）：向量维度 \(first.count)，开始计时…"))
            }
        }

        let metrics = makeMetrics(
            runLatencies: runLatencies,
            batchLatencies: batchLatencies,
            itemsPerBatch: config.batchSize,
            totalTokens: totalTokens
        )

        // 效果演示：相似度
        var samples: [SimilaritySample] = []
        for pair in TestCorpus.similarityPairs {
            let vectorA = try await runtime.embed(texts: [pair.a], maxLength: config.maxLength).vectors.first ?? []
            let vectorB = try await runtime.embed(texts: [pair.b], maxLength: config.maxLength).vectors.first ?? []
            samples.append(
                SimilaritySample(
                    label: pair.label,
                    textA: pair.a,
                    textB: pair.b,
                    similarity: cosineSimilarity(vectorA, vectorB)
                ))
        }

        let dimensionSample = try await runtime.embed(
            texts: [TestCorpus.embeddingTexts[0]], maxLength: config.maxLength).vectors.first ?? []
        let dimension = dimensionSample.count

        continuation.yield(
            .embeddingFinished(
                slot: slot,
                EmbeddingBenchmark(
                    metrics: metrics,
                    dimension: dimension,
                    similaritySamples: samples,
                    modelLabel: "模型 \(slot + 1) · \(runtime.backend.rawValue)"
                )))
    }

    // MARK: - Reranker

    private func runRerankerBenchmark(
        slot: Int,
        runtime: any ModelRuntime,
        config: BenchmarkConfig,
        continuation: AsyncStream<BenchmarkEvent>.Continuation
    ) async throws {

        // 生成测试数据：query 与候选文档的所有组合循环取样
        var pairs: [(query: String, document: String)] = []
        let queries = TestCorpus.rerankerQueries
        let documentGroups = TestCorpus.rerankerDocuments
        for i in 0..<config.testCount {
            let queryIndex = i % queries.count
            let documents = documentGroups[queryIndex % documentGroups.count]
            let documentIndex = (i / queries.count) % documents.count
            pairs.append((queries[queryIndex], documents[documentIndex]))
        }

        continuation.yield(.progress("Reranker（\(runtime.backend.rawValue)）：正在准备 \(pairs.count) 组 query-document 测试对…"))

        let batches = pairs.chunked(into: config.batchSize)
        guard !batches.isEmpty else { return }

        // 预热
        _ = try await runtime.rerank(pairs: batches[0], maxLength: config.maxLength)

        var runLatencies: [Double] = []
        var batchLatencies: [Double] = []
        var totalTokens = 0

        for batch in batches {
            if Task.isCancelled { break }
            let start = ContinuousClock.now
            let (scores, tokens) = try await runtime.rerank(pairs: batch, maxLength: config.maxLength)
            let elapsed = milliseconds(since: start)
            runLatencies.append(elapsed)
            batchLatencies.append(elapsed)
            totalTokens += tokens
            _ = scores
        }

        let metrics = makeMetrics(
            runLatencies: runLatencies,
            batchLatencies: batchLatencies,
            itemsPerBatch: config.batchSize,
            totalTokens: totalTokens
        )

        // 效果演示：第一个 query 对其候选文档的排序
        let demoQuery = TestCorpus.rerankerQueries[0]
        let demoDocuments = TestCorpus.rerankerDocuments[0]
        let demoPairs = demoDocuments.map { (demoQuery, $0) }
        let demoScores = try await runtime.rerank(pairs: demoPairs, maxLength: config.maxLength).scores
        let ranked = zip(demoDocuments, demoScores)
            .map { RankedSample(query: demoQuery, document: $0.0, score: $0.1) }
            .sorted { $0.score > $1.score }

        continuation.yield(
            .rerankerFinished(
                slot: slot,
                RerankerBenchmark(
                    metrics: metrics,
                    rankedSamples: ranked,
                    modelLabel: "模型 \(slot + 1) · \(runtime.backend.rawValue)"
                )))
    }

    // MARK: - 工具

    private func makeMetrics(
        runLatencies: [Double],
        batchLatencies: [Double],
        itemsPerBatch: Int,
        totalTokens: Int
    ) -> BenchmarkMetrics {
        guard !runLatencies.isEmpty else { return BenchmarkMetrics() }

        let sorted = runLatencies.sorted()
        let totalRunSeconds = runLatencies.reduce(0, +) / 1000.0
        let totalBatchSeconds = batchLatencies.reduce(0, +) / 1000.0
        let iterations = runLatencies.count

        func percentile(_ fraction: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            let index = min(sorted.count - 1, Int((Double(sorted.count - 1) * fraction).rounded()))
            return sorted[index]
        }

        return BenchmarkMetrics(
            batchSize: itemsPerBatch,
            iterations: iterations,
            avgLatencyMs: runLatencies.reduce(0, +) / Double(iterations),
            p50LatencyMs: percentile(0.5),
            p90LatencyMs: percentile(0.9),
            itemsPerSecond: totalBatchSeconds > 0 ? Double(iterations * itemsPerBatch) / totalBatchSeconds : 0,
            tokensPerSecond: totalRunSeconds > 0 ? Double(totalTokens) / totalRunSeconds : 0,
            totalItems: iterations * itemsPerBatch,
            totalTokens: totalTokens,
            totalSeconds: totalBatchSeconds
        )
    }

    private func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        let seconds = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
        return seconds * 1000.0
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (normA.squareRoot() * normB.squareRoot())
    }
}

extension Array {
    /// 将数组按指定大小分块
    func chunked(into size: Int) -> [[Element]] {
        guard size > 0 else { return [self] }
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
