import Foundation
import OnnxRuntimeBindings

/// 无界面基准测试模式，用于快速测量 Embedding / Reranker 模型速度：
///
///     swift run ONNXRuntimeTester --embedding <模型.onnx> --reranker <模型.onnx> \
///         [--batch 16] [--count 128] [--max-length 128] [--no-coreml] [--cpu-only] \
///         [--task retrieval.passage]（仅 jina-embeddings-v3：retrieval.query/passage/separation/classification/text-matching）
///
/// 流程与 GUI 完全一致：通过 onnxruntime-swift 加载模型（默认开启 CoreML
/// 执行提供程序），生成一批测试数据，测量推理速度并输出统计。
enum CLI {

    static func run() -> Never {
        var arguments = CommandLine.arguments
        arguments.removeFirst()

        func value(for flag: String) -> String? {
            guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else {
                return nil
            }
            return arguments[index + 1]
        }

        if arguments.contains("--help") || arguments.contains("-h") {
            print(usage)
            exit(0)
        }

        // embedding 与 reranker 至少提供一个
        let embeddingPath = value(for: "--embedding") ?? ""
        let rerankerPath = value(for: "--reranker") ?? ""
        guard !embeddingPath.isEmpty || !rerankerPath.isEmpty else {
            print(usage)
            exit(2)
        }
        let embeddingBackend = backend(from: value(for: "--embedding-backend") ?? "onnx")
        let rerankerBackend = backend(from: value(for: "--reranker-backend") ?? "onnx")
        let batchSize = Int(value(for: "--batch") ?? "16") ?? 16
        let testCount = Int(value(for: "--count") ?? "128") ?? 128
        let maxLength = Int(value(for: "--max-length") ?? "128") ?? 128
        let cpuOnly = arguments.contains("--cpu-only")
        // 仅对 jina-embeddings-v3 生效：LoRA 任务分支（retrieval.query / retrieval.passage /
        // separation / classification / text-matching），默认 retrieval.passage
        let taskFlag = value(for: "--task")

        // 通用「导出嵌入」模式：读取文本文件（JSON 数组），对指定 embedding 模型推理并输出向量
        if let dumpPath = value(for: "--embed-file"),
            let textsPath = value(for: "--texts-file"),
            let emb = value(for: "--embedding"), !emb.isEmpty
        {
            let done = DispatchSemaphore(value: 0)
            Task.detached {
                do {
                    let data = try Data(contentsOf: URL(fileURLWithPath: textsPath))
                    let texts = try JSONDecoder().decode([String].self, from: data)
                    let vectors: [[Float]]
                    let tokens: Int
                    if embeddingBackend == .mlx {
                        let runtime = try await MLXModelRuntime.load(
                            directory: URL(fileURLWithPath: emb), kind: .embedding, task: taskFlag)
                        (vectors, tokens) = try await runtime.embed(
                            texts: texts, maxLength: maxLength)
                    } else {
                        // ONNX / CoreML：加载后注入 swift-transformers 分词器
                        let runtime: any ModelRuntime
                        if embeddingBackend == .onnx {
                            let env = try ORTEnv(loggingLevel: .warning)
                            let resolved = try ModelPathResolver.resolve(emb)
                            runtime = try ORTEngine.load(
                                url: resolved.modelURL, kind: .embedding, env: env,
                                coreML: CoreMLOptions(
                                    enabled: !arguments.contains("--no-coreml") && !cpuOnly,
                                    cpuOnly: cpuOnly),
                                task: taskFlag)
                            let tok = try await TokenizerService.load(
                                fromDirectory: resolved.modelURL.deletingLastPathComponent())
                            (runtime as? ORTModelRuntime)?.tokenizer = tok
                        } else {
                            runtime = try CoreMLModelRuntime.load(
                                url: URL(fileURLWithPath: emb).standardizedFileURL,
                                kind: .embedding, cpuOnly: cpuOnly)
                            let tok = try await TokenizerService.load(
                                fromDirectory: URL(fileURLWithPath: emb).deletingLastPathComponent())
                            (runtime as? CoreMLModelRuntime)?.tokenizer = tok
                        }
                        (vectors, tokens) = try await runtime.embed(
                            texts: texts, maxLength: maxLength)
                    }
                    let out: [String: Any] = [
                        "vectors": vectors,
                        "tokens": tokens,
                        "dim": vectors.first?.count ?? 0,
                        "count": vectors.count,
                    ]
                    let json = try JSONSerialization.data(
                        withJSONObject: out, options: [.withoutEscapingSlashes])
                    try json.write(to: URL(fileURLWithPath: dumpPath))
                    print(
                        "✅ 已导出 \(vectors.count) 条向量（维度 \(vectors.first?.count ?? 0)，\(tokens) tokens）→ \(dumpPath)"
                    )
                } catch {
                    print("❌ 导出失败：\(error.localizedDescription)")
                    exit(1)
                }
                exit(0)
            }
            done.wait()
            exit(0)
        }

        let coreML = CoreMLOptions(
            // --cpu-only 与界面「仅使用 CPU」语义一致：彻底关闭 CoreML EP（而非仅 useCPUOnly），
            // 否则仍会挂上 CoreML EP，遇到不兼容的算子（如 nomic 的 rotary embedding）照样失败。
            enabled: !arguments.contains("--no-coreml") && !cpuOnly,
            cpuOnly: cpuOnly
        )

        let requests: [ModelLoadRequest?] = [
            embeddingPath.isEmpty ? nil : ModelLoadRequest(path: embeddingPath, backend: embeddingBackend, kind: .embedding, task: taskFlag),
            rerankerPath.isEmpty ? nil : ModelLoadRequest(path: rerankerPath, backend: rerankerBackend, kind: .reranker),
        ]

        let semaphore = DispatchSemaphore(value: 0)

        // 脱离主 actor 运行，避免阻塞主线程造成死锁
        Task.detached {
            do {
                try await runBenchmark(
                    requests: requests,
                    config: BenchmarkConfig(
                        batchSize: batchSize,
                        testCount: testCount,
                        maxLength: maxLength
                    ),
                    coreML: coreML
                )
            } catch {
                let message = error.localizedDescription
                print("\n❌ 失败：\(message)\(coreMLCompatibilityHint(message))")
                exit(1)
            }
            exit(0)
        }
        // 正常情况下 exit() 会先终止进程，这里仅作为兜底
        semaphore.wait()
        exit(0)
    }

    /// 解析后端参数：onnx / mlx / coreml（大小写不敏感，非法值回退 onnx）
    private static func backend(from raw: String) -> BackendKind {
        switch raw.lowercased() {
        case "onnx": return .onnx
        case "mlx": return .mlx
        case "coreml": return .coreml
        default: return .onnx
        }
    }

    private static let usage = """
        用法：
          ONNXRuntimeTester --embedding <模型路径> [--embedding-backend onnx|mlx|coreml] \\
              [--reranker <模型路径>] [--reranker-backend onnx|mlx|coreml] \\
              [--batch 16] [--count 128] [--max-length 128] [--no-coreml] [--cpu-only] \\
              [--task retrieval.passage]（仅 jina-embeddings-v3 生效）

        后端说明：
          onnx    ONNX Runtime 加载 .onnx（默认）
          mlx     mlx-swift-lm 加载 MLX 目录（需含 config.json 与权重）
          coreml  Core ML 加载 .mlpackage / .mlmodelc

        示例：
          swift run ONNXRuntimeTester --embedding ./tiny-embedding/model.onnx \\
              --reranker ./tiny-reranker/model.onnx --reranker-backend coreml

        不带 --embedding 参数运行时启动图形界面。
        """

    private static func runBenchmark(
        requests: [ModelLoadRequest?],
        config: BenchmarkConfig,
        coreML: CoreMLOptions
    ) async throws {
        print("ONNX Runtime \(ORTVersion() ?? "未知") · CoreML \(ORTIsCoreMLExecutionProviderAvailable() ? "可用" : "不可用")")
        print("🚀 正在加载模型（CoreML \(coreML.enabled ? (coreML.cpuOnly ? "仅 CPU" : "开启") : "关闭")）…")

        let engine = try BenchmarkEngine()
        let report = try await engine.load(requests, coreML: coreML)

        for (index, model) in report.models.enumerated() {
            print(
                "✅ 模型 \(index + 1)（\(model.kind.rawValue) · \(model.backend.rawValue)）：\(model.name)（\(String(format: "%.2f", model.loadDuration))s，\(model.statusDescription)，\(StatFormatter.formatBytes(UInt64(model.fileBytes)))）")
            print("   输入：\(model.inputNames.joined(separator: ", "))")
            print("   输出：\(model.outputNames.joined(separator: ", "))")
        }
        print("分词器：\(report.tokenizerDescription)")

        print("\n🧪 开始测速：批大小 \(config.batchSize) · 测试数量 \(config.testCount) · 最大长度 \(config.maxLength)")

        for await event in engine.runBenchmarks(config: config) {
            switch event {
            case .progress(let message):
                print("   … \(message)")
            case .embeddingFinished(let slot, let result):
                print("\n📊 模型 \(slot + 1) Embedding 结果")
                print(
                    "  吞吐：\(String(format: "%.1f", result.metrics.itemsPerSecond)) 句/s · \(String(format: "%.1f", result.metrics.tokensPerSecond)) tokens/s")
                print(
                    "  延迟：平均 \(String(format: "%.1f", result.metrics.avgLatencyMs))ms · P50 \(String(format: "%.1f", result.metrics.p50LatencyMs))ms · P90 \(String(format: "%.1f", result.metrics.p90LatencyMs))ms")
                print(
                    "  规模：\(result.metrics.iterations) 轮 × \(result.metrics.batchSize) 句 · 共 \(result.metrics.totalItems) 句 / \(result.metrics.totalTokens) tokens · 总耗时 \(String(format: "%.2f", result.metrics.totalSeconds))s")
                print("  向量维度：\(result.dimension)")
                if !result.similaritySamples.isEmpty {
                    print("  相似度效果：")
                    for sample in result.similaritySamples {
                        print(
                            "    [\(sample.label)] \(StatFormatter.formatSimilarity(sample.similarity))  \(sample.textA)  <->  \(sample.textB)")
                    }
                }
            case .rerankerFinished(let slot, let result):
                print("\n📊 模型 \(slot + 1) Reranker 结果")
                print(
                    "  吞吐：\(String(format: "%.1f", result.metrics.itemsPerSecond)) 对/s · \(String(format: "%.1f", result.metrics.tokensPerSecond)) tokens/s")
                print(
                    "  延迟：平均 \(String(format: "%.1f", result.metrics.avgLatencyMs))ms · P50 \(String(format: "%.1f", result.metrics.p50LatencyMs))ms · P90 \(String(format: "%.1f", result.metrics.p90LatencyMs))ms")
                print(
                    "  规模：\(result.metrics.iterations) 轮 × \(result.metrics.batchSize) 对 · 共 \(result.metrics.totalItems) 对 / \(result.metrics.totalTokens) tokens · 总耗时 \(String(format: "%.2f", result.metrics.totalSeconds))s")
                if !result.rankedSamples.isEmpty {
                    print("  排序效果（query：\(result.rankedSamples[0].query)）：")
                    for (index, sample) in result.rankedSamples.enumerated() {
                        print("    \(index + 1). [\(String(format: "%.4f", sample.score))] \(sample.document)")
                    }
                }
            case .failed(let message):
                print("\n❌ \(message)")
            }
        }
    }
}
