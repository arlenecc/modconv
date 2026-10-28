import Foundation
import OnnxRuntimeBindings

/// ONNX Runtime 引擎错误
/// CoreML 执行提供程序兼容性提示：当错误信息与 CoreML/MLModel 相关时，返回可操作建议，否则返回空串。
/// 覆盖加载期（编译 MIL 失败）与推理期（动态 shape 不支持）两类 CoreML EP 失败。
func coreMLCompatibilityHint(_ message: String) -> String {
    let lower = message.lowercased()
    let markers = ["coreml", "mlmodel", "model execution plan", "coreml_execution_provider"]
    guard markers.contains(where: { lower.contains($0) }) else { return "" }
    return "\n（该模型与 CoreML 执行提供程序不兼容：可在界面取消勾选「ONNX 启用 CoreML EP」或勾选「仅使用 CPU」后重试）"
}

/// jina-embeddings-v3 的 ONNX 是「统一任务」模型：需把任务前缀拼到文本前，并额外传入
/// 标量输入 task_id（retrieval.query=0 / retrieval.passage=1 / separation=2 /
/// classification=3 / text-matching=4）。输出 sentence_embedding 已是均值池化 + L2。
struct JinaV3TaskInfo {
    let taskId: Int
    let prompt: String
}

/// 从模型目录 config.json 的 task_id_map（无则用默认映射）与官方任务前缀构造 jina-v3 任务信息。
/// 官方 v3 任务前缀（config 未保存前缀，此处硬编码）：
private func jinaV3TaskInfo(directory: URL, task: String?) -> JinaV3TaskInfo {
    let prompts: [String: String] = [
        "retrieval.query": "Represent the query for retrieving evidence documents: ",
        "retrieval.passage": "Represent the document for retrieval: ",
        "separation": "Represent the sentence for semantic textual similarity: ",
        "classification": "Represent the document for classification: ",
        "text-matching": "Represent the sentence for semantic textual similarity: ",
    ]
    var idMap: [String: Int] = [
        "retrieval.query": 0, "retrieval.passage": 1, "separation": 2,
        "classification": 3, "text-matching": 4,
    ]
    if let cfg = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
        let obj = try? JSONSerialization.jsonObject(with: cfg) as? [String: Any],
        let map = obj["task_id_map"] as? [String: Int]
    {
        idMap = map
    }
    let taskKey = (task ?? "retrieval.passage").lowercased()
    let id = idMap[taskKey] ?? 1
    let prompt = prompts[taskKey] ?? prompts["retrieval.passage"]!
    return JinaV3TaskInfo(taskId: id, prompt: prompt)
}

enum ORTEngineError: LocalizedError {
    case envCreationFailed(String)
    case sessionCreationFailed(String)
    case runFailed(String)
    case tensorCreationFailed(String)
    case outputReadFailed(String)
    case missingInput(String)
    case invalidPath(String)

    var errorDescription: String? {
        switch self {
        case .envCreationFailed(let message):
            return "创建 ONNX Runtime 环境失败：\(message)"
        case .sessionCreationFailed(let message):
            return "创建推理会话失败：\(message)"
        case .runFailed(let message):
            return "模型推理失败：\(message)"
        case .tensorCreationFailed(let message):
            return "创建输入张量失败：\(message)"
        case .outputReadFailed(let message):
            return "读取输出张量失败：\(message)"
        case .missingInput(let name):
            return "模型缺少必需的输入：\(name)"
        case .invalidPath(let message):
            return message
        }
    }
}

/// 路径解析：用户可能选择 .onnx 文件，也可能选择模型目录。
/// 统一解析为「实际的 .onnx 文件 URL」与「模型所在目录（用于定位分词器）」。
///
/// - 目录：在其中查找 model.onnx / onnx/model.onnx 等常见位置，未找到则回退到目录内任意 .onnx
/// - 文件：必须以 .onnx 结尾
/// - 所有失败都给出可理解的中文提示，避免底层抛出晦涩的「不是目录」等错误
enum ModelPathResolver {
    static func resolve(_ rawPath: String) throws -> (modelURL: URL, directory: URL) {
        let path = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else {
            throw ORTEngineError.invalidPath("模型路径为空，请先选择模型")
        }

        let url = URL(fileURLWithPath: path).standardizedFileURL
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            throw ORTEngineError.invalidPath("指定的路径不存在：\(path)")
        }

        if isDir.boolValue {
            let candidates = [
                "model.onnx",
                "onnx/model.onnx",
                "encoder/model.onnx",
                "openvino/model.onnx",
                "onnx/model_qint8.onnx",
            ]
            var found: URL?
            for candidate in candidates {
                let candidateURL = url.appendingPathComponent(candidate)
                if FileManager.default.fileExists(atPath: candidateURL.path) {
                    found = candidateURL
                    break
                }
            }
            if found == nil {
                let contents = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
                found = contents.first { $0.pathExtension.lowercased() == "onnx" }
            }
            guard let modelURL = found else {
                throw ORTEngineError.invalidPath("目录中未找到 .onnx 模型文件：\(path)")
            }
            return (modelURL, url)
        }

        guard url.pathExtension.lowercased() == "onnx" else {
            throw ORTEngineError.invalidPath("指定的不是 .onnx 模型文件：\(path)")
        }
        return (url, url.deletingLastPathComponent())
    }
}

/// 一个已加载的模型运行时（仅在 BenchmarkEngine actor 内串行访问，标记为 unchecked Sendable）
final class ORTModelRuntime: @unchecked Sendable {
    let session: ORTSession
    let kind: ModelKind
    let name: String
    let inputNames: [String]
    let outputNames: [String]
    let coreMLEnabled: Bool
    let loadDuration: TimeInterval
    let fileBytes: Int64

    /// 分词器（由引擎在加载后注入；ONNX 与 CoreML 共用 swift-transformers 分词器）
    var tokenizer: (any TextTokenizer)?
    /// jina-embeddings-v3 专用：统一任务模型的任务前缀与 task_id（仅对带 task_id 输入的模型设置）
    var jinaV3: JinaV3TaskInfo?
    /// 首次 embedding 推理后缓存的向量维度
    private var cachedDimension: Int = 0

    /// 模型实际接受的输入 id 元素类型（部分导出使用 int32，默认 int64）
    private(set) var elementType: ORTTensorElementDataType = .int64

    init(
        session: ORTSession,
        kind: ModelKind,
        name: String,
        inputNames: [String],
        outputNames: [String],
        coreMLEnabled: Bool,
        loadDuration: TimeInterval,
        fileBytes: Int64
    ) {
        self.session = session
        self.kind = kind
        self.name = name
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.coreMLEnabled = coreMLEnabled
        self.loadDuration = loadDuration
        self.fileBytes = fileBytes
    }

    var info: LoadedModel {
        LoadedModel(
            kind: kind,
            backend: .onnx,
            name: name,
            path: "",
            inputNames: inputNames,
            outputNames: outputNames,
            coreMLEnabled: coreMLEnabled,
            loadDuration: loadDuration,
            fileBytes: fileBytes
        )
    }

    // MARK: - 输入张量构建

    /// 按模型声明的输入名匹配 id / mask / type_ids 并打包为 ORTValue。
    /// taskId 非 nil 时（jina-embeddings-v3）额外构造标量 task_id 输入。
    func buildInputs(
        ids: [[Int]],
        attentionMasks: [[Int]],
        tokenTypeIds: [[Int]],
        taskId: Int? = nil
    ) throws -> [String: ORTValue] {
        var inputs: [String: ORTValue] = [:]
        var matchedIds = false
        var matchedMask = false

        for name in inputNames {
            let lower = name.lowercased()
            if lower.contains("input_ids") || lower == "ids" || lower.contains("inputid") {
                inputs[name] = try makeTensor(ids)
                matchedIds = true
            } else if lower.contains("attention_mask") || lower.contains("mask") {
                inputs[name] = try makeTensor(attentionMasks)
                matchedMask = true
            } else if lower.contains("token_type_ids") || lower.contains("type_ids") || lower.contains("segment_ids") {
                inputs[name] = try makeTensor(tokenTypeIds)
            }
        }

        if let tid = taskId,
            let name = inputNames.first(where: { $0.lowercased() == "task_id" })
        {
            inputs[name] = try makeScalarTensor(Int64(tid))
        }

        guard matchedIds else {
            throw ORTEngineError.missingInput(inputNames.joined(separator: ", "))
        }
        guard matchedMask else {
            throw ORTEngineError.missingInput("attention_mask")
        }
        return inputs
    }

    /// 构造标量 int64 张量（如 jina-v3 的 task_id 输入）
    private func makeScalarTensor(_ value: Int64) throws -> ORTValue {
        let data = NSMutableData(bytes: [value], length: MemoryLayout<Int64>.size)
        return try ORTValue(tensorData: data, elementType: .int64, shape: [])
    }

    private func makeTensor(_ rows: [[Int]]) throws -> ORTValue {
        guard !rows.isEmpty else {
            throw ORTEngineError.tensorCreationFailed("空的批次")
        }
        let batch = rows.count
        let seqLen = rows[0].count
        let flat = rows.flatMap { $0 }
        let shape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: seqLen)]

        let data: NSMutableData
        switch elementType {
        case .int64:
            let values = flat.map { Int64($0) }
            data = values.withUnsafeBufferPointer { pointer in
                NSMutableData(
                    bytes: pointer.baseAddress!,
                    length: values.count * MemoryLayout<Int64>.size
                )
            }
        case .int32:
            let values = flat.map { Int32($0) }
            data = values.withUnsafeBufferPointer { pointer in
                NSMutableData(
                    bytes: pointer.baseAddress!,
                    length: values.count * MemoryLayout<Int32>.size
                )
            }
        default:
            throw ORTEngineError.tensorCreationFailed("不支持的元素类型")
        }

        do {
            return try ORTValue(tensorData: data, elementType: elementType, shape: shape)
        } catch {
            throw ORTEngineError.tensorCreationFailed(error.localizedDescription)
        }
    }

    // MARK: - 推理

    /// 执行一次批次推理；若模型实际需要 int32 输入会自动切换并重试一次
    func runBatch(
        ids: [[Int]],
        attentionMasks: [[Int]],
        tokenTypeIds: [[Int]],
        taskId: Int? = nil
    ) throws -> [String: ORTValue] {
        do {
            let inputs = try buildInputs(
                ids: ids, attentionMasks: attentionMasks, tokenTypeIds: tokenTypeIds, taskId: taskId)
            return try session.run(
                withInputs: inputs,
                outputNames: Set(outputNames),
                runOptions: nil
            )
        } catch {
            // 部分 ONNX 导出使用 int32 输入；若错误与数据类型相关，切换后重试
            if elementType == .int64, error.localizedDescription.lowercased().contains("type") {
                elementType = .int32
                do {
                    let inputs = try buildInputs(
                        ids: ids, attentionMasks: attentionMasks, tokenTypeIds: tokenTypeIds,
                        taskId: taskId)
                    return try session.run(
                        withInputs: inputs,
                        outputNames: Set(outputNames),
                        runOptions: nil
                    )
                } catch let retryError {
                    // 保留原始错误，避免误导
                    elementType = .int64
                    throw ORTEngineError.runFailed(error.localizedDescription + "（尝试 int32 输入亦失败：\(retryError.localizedDescription)）")
                }
            }
            throw ORTEngineError.runFailed(error.localizedDescription)
        }
    }
}

/// ONNX Runtime 引擎：创建共享环境并按需加载模型 session
enum ORTEngine {

    /// 创建 ORT 环境（进程内共享一个即可）
    static func makeEnv() throws -> ORTEnv {
        do {
            return try ORTEnv(loggingLevel: .warning)
        } catch {
            throw ORTEngineError.envCreationFailed(error.localizedDescription)
        }
    }

    /// 若模型使用外部数据（同目录下的 .onnx_data 文件），用 Python/onnx 将其内联为自包含 .onnx 后再加载。
    ///
    /// 背景：ONNX Runtime 对外部数据（.onnx_data）的路径解析在部分模型/布局下不稳定——
    /// 会误把 .onnx 模型文件当成目录去查找 .onnx_data（报错 "Not a directory"）。
    /// 将外部数据内联进 .onnx 后，ORT 按普通内嵌模型加载即可彻底规避该问题。
    /// 返回实际用于 ORT 加载的 .onnx 路径（无外部数据时返回原路径）。
    /// 解析 .onnx 内部 external_data 引用的外部数据文件名（location）。
    ///
    /// 直接在 protobuf 字节里查找 `StringStringEntryProto{key:"location", value:<文件名>}`，
    /// 与外部数据的具体命名规则（model.onnx_data / model_fp16.onnx_data / *.data 等）无关，
    /// 因此无论选取哪个版本的模型都能正确定位它自己的外部数据文件。
    private static func externalDataLocations(_ modelURL: URL) -> [String] {
        guard let data = try? Data(contentsOf: modelURL, options: .mappedIfSafe) else { return [] }
        var locations: [String] = []
        var cursor = data.startIndex
        while let keyRange = data.range(of: Self.locationKeyBytes, in: cursor..<data.endIndex) {
            var index = keyRange.upperBound
            // "location" 之后应紧跟 value 条目（tag 0x12 + varint 长度 + 字节）
            if index < data.endIndex, data[index] == 0x12,
                let (length, valueStart) = Self.readVarint(data, index + 1) {
                let valueEnd = valueStart + length
                if valueEnd <= data.endIndex,
                    let location = String(data: data[valueStart..<valueEnd], encoding: .utf8) {
                    locations.append(location)
                    cursor = valueEnd
                    continue
                }
            }
            cursor = keyRange.upperBound
        }
        return locations
    }

    /// 匹配 external_data 的 key 条目：field1(0x0A) + len(0x08) + "location"
    private static let locationKeyBytes: Data = {
        var d = Data([0x0A, 0x08])
        d.append(Data("location".utf8))
        return d
    }()

    /// 读取 protobuf varint（外部数据文件名长度的常见编码，兼容多字节）
    private static func readVarint(_ data: Data, _ start: Int) -> (value: Int, next: Int)? {
        var result = 0
        var shift = 0
        var index = start
        while index < data.endIndex {
            let byte = data[index]
            result |= Int(byte & 0x7F) << shift
            index += 1
            if byte & 0x80 == 0 {
                return (result, index)
            }
            shift += 7
            if shift > 28 { return nil }
        }
        return nil
    }

    /// 模型实际引用的外部数据文件（把 location 解析为模型同目录下存在的文件 URL）
    static func externalDataFiles(_ modelURL: URL) -> [URL] {
        let dir = modelURL.deletingLastPathComponent()
        var files: [URL] = []
        var seen = Set<String>()
        for location in externalDataLocations(modelURL) {
            // location 通常是相对模型目录的文件名（可能带 ./ 或子目录）
            let url = URL(fileURLWithPath: location, relativeTo: dir).standardizedFileURL
            guard seen.insert(url.path).inserted else { continue }
            if FileManager.default.fileExists(atPath: url.path) {
                files.append(url)
            }
        }
        return files
    }

    /// 判断模型是否引用了外部数据文件（解析 .onnx 内部的 location，兼容任意命名）
    private static func hasExternalData(_ modelURL: URL) -> Bool {
        !externalDataLocations(modelURL).isEmpty
    }

    private static func loadableModelURL(_ modelURL: URL) throws -> URL {
        guard Self.hasExternalData(modelURL) else { return modelURL }

        // 估算内联后体积（proto + 外部数据）。protobuf 单消息序列化上限约 2GB，
        // 超过则无法内联成单个自包含 .onnx（onnx.save_model 会报 EncodeError）。
        let externalBytes = Self.externalDataFiles(modelURL).reduce(0) {
            $0 + ((try? FileManager.default.attributesOfItem(atPath: $1.path)[.size] as? Int64) ?? 0)
        }
        let protoBytes = (try? FileManager.default.attributesOfItem(atPath: modelURL.path)[.size] as? Int64) ?? 0
        let inlinedBytes = protoBytes + externalBytes
        let inlineLimit: Int64 = 1_900_000_000  // ~1.9GB，留出序列化开销余量
        guard inlinedBytes <= inlineLimit else {
            throw ORTEngineError.sessionCreationFailed(
                "该 ONNX 模型使用外部数据，合并后约 \(inlinedBytes / 1_000_000) MB，超过单文件上限（约 1.9 GB）。当前 ONNX Runtime 1.24.2 既无法直接加载外部数据模型，也无法内联超大模型。请改用自包含的量化 / fp16 变体（如 model_fp16.onnx、model_int8.onnx、model_q4.onnx），它们不含外部数据、可直接加载。"
            )
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ort_inlined_\(UUID().uuidString).onnx")

        // 清理上次内联产生的临时文件，避免磁盘堆积
        for old in (try? FileManager.default.contentsOfDirectory(
            at: FileManager.default.temporaryDirectory, includingPropertiesForKeys: nil)) ?? [] {
            if old.lastPathComponent.hasPrefix("ort_inlined_") {
                try? FileManager.default.removeItem(at: old)
            }
        }

        // 内联为单个自包含 .onnx：ORT 1.24.2 对外部数据的相对路径解析有 bug
        // （会把 model.onnx 当成目录去找 model.onnx_data，报 "Not a directory"），
        // 内联成无外部数据的单文件可彻底规避。
        let script = "import onnx,sys\nm=onnx.load(sys.argv[1])\nonnx.save_model(m,sys.argv[2])"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-c", script, modelURL.path, tempURL.path]
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            throw ORTEngineError.sessionCreationFailed(
                "内联外部数据失败（无法启动 python3：请确认 python3 与 onnx 已安装）：\(error.localizedDescription)"
            )
        }
        if process.terminationStatus != 0 {
            let msg = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw ORTEngineError.sessionCreationFailed(
                "内联外部数据失败（请确认 python3 与 onnx 已安装）：\n\(msg)"
            )
        }
        return tempURL
    }

    /// 从本地路径加载模型，并按配置开启 CoreML 执行提供程序。
    /// CoreML 不可用或附加失败时自动降级为 CPU 执行提供程序（不影响加载）。
    @discardableResult
    static func load(
        url: URL,
        kind: ModelKind,
        env: ORTEnv,
        coreML: CoreMLOptions,
        task: String? = nil
    ) throws -> ORTModelRuntime {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: url.path) else {
            throw ORTEngineError.sessionCreationFailed("文件不存在：\(url.path)")
        }

        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(0)
        try options.setGraphOptimizationLevel(.all)

        var coreMLEnabled = false
        if coreML.enabled, ORTIsCoreMLExecutionProviderAvailable() {
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            coreMLOptions.enableOnSubgraphs = coreML.enableOnSubgraphs
            coreMLOptions.createMLProgram = coreML.createMLProgram
            coreMLOptions.useCPUOnly = coreML.cpuOnly
            do {
                try options.appendCoreMLExecutionProvider(with: coreMLOptions)
                coreMLEnabled = true
            } catch {
                // CoreML 附加失败时退回 CPU 执行提供程序，保证工具仍可用
                coreMLEnabled = false
            }
        }

        // 优先直接加载：ONNX Runtime 原生支持外部数据（.onnx_data 与 .onnx 同目录即可），
        // 无需内联，也避免内联大模型时触发 protobuf 单消息 2GB 序列化上限。
        // 仅当直接加载失败且模型带外部数据时，才回退到「内联」重试。
        let start = ContinuousClock.now
        let session: ORTSession
        do {
            session = try ORTSession(env: env, modelPath: url.path, sessionOptions: options)
        } catch let directError {
            guard Self.hasExternalData(url) else {
                throw ORTEngineError.sessionCreationFailed(directError.localizedDescription)
            }
            let modelURL: URL
            do {
                modelURL = try Self.loadableModelURL(url)
            } catch {
                throw ORTEngineError.sessionCreationFailed(
                    "直接加载失败：\(directError.localizedDescription)\n内联外部数据失败：\(error.localizedDescription)")
            }
            do {
                session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
            } catch {
                throw ORTEngineError.sessionCreationFailed(
                    "直接加载失败：\(directError.localizedDescription)\n内联后加载失败：\(error.localizedDescription)")
            }
        }
        let elapsed = start.duration(to: .now)
        let loadDuration = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18

        let inputNames = (try? session.inputNames()) ?? []
        let outputNames = (try? session.outputNames()) ?? []

        // 模型大小：含实际引用的外部数据文件，避免只显示外置指针文件大小
        var fileBytes: Int64 = (try? fileManager.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        for ext in Self.externalDataFiles(url) {
            fileBytes += (try? fileManager.attributesOfItem(atPath: ext.path)[.size] as? Int64) ?? 0
        }

        let runtime = ORTModelRuntime(
            session: session,
            kind: kind,
            name: url.lastPathComponent,
            inputNames: inputNames,
            outputNames: outputNames,
            coreMLEnabled: coreMLEnabled,
            loadDuration: loadDuration,
            fileBytes: fileBytes
        )
        // jina-embeddings-v3：模型含独立的 task_id 输入，需按任务前缀 + task_id 推理
        if inputNames.contains(where: { $0.lowercased() == "task_id" }) {
            runtime.jinaV3 = jinaV3TaskInfo(directory: url.deletingLastPathComponent(), task: task)
        }
        return runtime
    }
}

// MARK: - 输出张量读取工具

extension ORTModelRuntime {

    /// 读取输出张量为 Float 数组，并返回其形状
    func readFloatTensor(_ value: ORTValue) throws -> (floats: [Float], shape: [Int]) {
        let data: NSMutableData
        do {
            data = try value.tensorData()
        } catch {
            throw ORTEngineError.outputReadFailed(error.localizedDescription)
        }
        let count = data.length / MemoryLayout<Float>.size
        var floats = [Float](repeating: 0, count: count)
        memcpy(&floats, data.bytes, count * MemoryLayout<Float>.size)
        var shape: [Int] = []
        if let info = try? value.tensorTypeAndShapeInfo() {
            shape = info.shape.map { $0.intValue }
        }
        return (floats, shape)
    }
}

// MARK: - ModelRuntime 协议适配（ONNX 后端）

extension ORTModelRuntime: ModelRuntime {
    var backend: BackendKind { .onnx }

    var statusDescription: String {
        coreMLEnabled ? "ONNX · CoreML 已启用" : "ONNX · 纯 CPU"
    }

    func embed(texts: [String], maxLength: Int) async throws -> (vectors: [[Float]], tokens: Int) {
        guard let tokenizer else {
            throw ORTEngineError.sessionCreationFailed("分词器未加载，无法构造输入")
        }
        // jina-embeddings-v3：拼上任务前缀后再分词（retrieval.passage 等）
        let processed = jinaV3.map { info in texts.map { info.prompt + $0 } } ?? texts
        let encoded = processed.map { tokenizer.encode($0, maxLength: maxLength) }
        let (ids, masks, types) = padBatch(encoded)
        let outputs = try runBatch(ids: ids, attentionMasks: masks, tokenTypeIds: types, taskId: jinaV3?.taskId)
        let (floats, shape) = try readFirstOutput(outputs)
        let vectors = poolEmbeddingOutput(floats, shape: shape, masks: masks)
        if let first = vectors.first { cachedDimension = first.count }
        let tokens = masks.reduce(0) { $0 + $1.reduce(0, +) }
        return (vectors, tokens)
    }

    func rerank(pairs: [(query: String, document: String)], maxLength: Int) async throws -> (scores: [Float], tokens: Int) {
        guard let tokenizer else {
            throw ORTEngineError.sessionCreationFailed("分词器未加载，无法构造输入")
        }
        let encoded = pairs.map { tokenizer.encodePair(query: $0.query, document: $0.document, maxLength: maxLength) }
        let (ids, masks, types) = padBatch(encoded)
        let outputs = try runBatch(ids: ids, attentionMasks: masks, tokenTypeIds: types)

        let scoreNames = ["logits", "scores", "output", "similarity"]
        let outputName = scoreNames.first(where: { outputs[$0] != nil }) ?? outputs.keys.first
        guard let name = outputName, let value = outputs[name] else {
            throw ORTEngineError.outputReadFailed("未找到输出张量")
        }
        let (floats, _) = try readFloatTensor(value)
        let batchSize = pairs.count
        let scores = Array(floats.prefix(batchSize))
        let tokens = masks.reduce(0) { $0 + $1.reduce(0, +) }
        return (scores, tokens)
    }

    /// 读取模型输出：优先直接输出（sentence_embedding / pooler_output 等），
    /// 否则取 last_hidden_state / hidden_states 等三维输出，最后回退到首个输出。
    private func readFirstOutput(_ outputs: [String: ORTValue]) throws -> (floats: [Float], shape: [Int]) {
        let directNames = ["sentence_embedding", "pooler_output", "pooled_output", "embeddings", "embedding"]
        let pooledNames = ["last_hidden_state", "hidden_states", "last_hidden", "sequence_output"]
        let name = directNames.first(where: { outputs[$0] != nil })
            ?? pooledNames.first(where: { outputs[$0] != nil })
            ?? outputs.keys.first
        guard let n = name, let value = outputs[n] else {
            throw ORTEngineError.outputReadFailed("未找到输出张量")
        }
        return try readFloatTensor(value)
    }

    /// 将一个批次的编码结果 padding 到相同长度
    private func padBatch(_ batch: [EncodedText]) -> (ids: [[Int]], masks: [[Int]], types: [[Int]]) {
        let maxLen = batch.map { $0.ids.count }.max() ?? 1
        var ids: [[Int]] = []
        var masks: [[Int]] = []
        var types: [[Int]] = []
        for item in batch {
            let padCount = maxLen - item.ids.count
            ids.append(item.ids + [Int](repeating: 0, count: padCount))
            masks.append(item.attentionMask + [Int](repeating: 0, count: padCount))
            types.append(item.tokenTypeIds + [Int](repeating: 0, count: padCount))
        }
        return (ids, masks, types)
    }
}
