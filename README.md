# 多模型格式转换工具

把机器学习模型在 **HF（HuggingFace）**、**ONNX**、**MLX（Apple）**、**GGUF（llama.cpp）**
四种格式之间互相转换，并导出为可直接在 Apple 平台端侧运行的产物。图形界面（PyQt6）覆盖
macOS / iOS / visionOS 的 **Core ML `.mlpackage`**、**ONNX**、**MLX**、**GGUF** 四类目标，
覆盖 LLM（causal）、encoder 类（embedding / reranker）与多模态（VLM）模型。

> Python 包名 / 命令行为 `modconv`，macOS 应用名为 **多模型格式转换工具**，仓库名为 `modconv`。

## 它解决什么

- 想把下载到的 MLX / HF / GGUF 模型放到 iPhone / Mac 上跑，但每种格式的转换链路
  （权重命名、量化、分词器、状态算子）各不相同，手工对齐极易踩坑；
- 想在不同端侧后端之间横向比较（Core ML / ONNX Runtime / MLX / llama.cpp），
  需要把同一个 checkpoint 转成多种目标并验证保真度；
- 想把云端 encoder 模型（bge / jina / nomic embedding、reranker）搬到端侧做检索 /
  重排序，但 HF 与端侧 MLX/ONNX 的算子与命名差异大。

本工具把这些链路统一到一套 GUI + 后端探测 + 自动验证流程里，并针对 Apple 平台做了
大量工程化处理（见下文「多目标转换」与[设计文档](docs/design.md)）。

## 功能特性

- **四源 → 四目标矩阵**：HF / ONNX / MLX / GGUF 任一源，可转 Core ML / ONNX / MLX / GGUF
  （组合矩阵见下方「多目标转换」）。不支持的组合（如 GGUF→GGUF 重量化、encoder 的
  GGUF 腿）会明确报错，不静默降级。
- **后端自动探测与选择**：按源格式 + 目标格式 + 模型类型 + 依赖可用性，自动挑选
  真实转换后端；依赖缺失或架构不支持时**明确报错**，绝不静默降级为不可推理的产物。
- **Core ML：Stateful KV cache 模型**：KV cache 以 Core ML State 存储（iOS 18+ 目标），
  prefill 与 decode 共用同一模型，Swift 端无需重算前缀，自回归生成每步只前向 1 个 token。
  Qwen3.5 系的 GDN（linear attention）conv/rec 状态同样以图 I/O 张量逐步传递。
- **Core ML：动态序列长度**：输入 `input_ids [1, q]` + `cache_position [q]`，
  `q ∈ 1..max_seq_len`，一次转换覆盖任意上下文长度，无需按固定长度多次导出。
- **Core ML：最大上下文长度可配**：GUI 下拉框指定（默认「自动（≤512）」）。留空/选「自动」
  时按 `min(模型位置编码上限, 512)` 选取（`SEQ_LEN_CAP=512` 是 ANE 能稳定构建执行计划的上限）；
  也可选预设档位或输入自定义整数（支持 `k`/`m` 后缀，如 `8k`）。显式大于 512 的值会被采纳，
  但可能生成 GB 级的 KV cache，且在 ANE 上大概率 `-14`（执行计划构建失败），届时工具会
  自动逐级降长度直到可运行——因此默认走「自动」最稳妥。超过模型位置编码上限时自动截断。
- **Core ML：量化压缩**：默认对权重做 4-bit palettization（kmeans LUT）；也可选
  fp16 / 6-bit / 8-bit。压缩失败直接报错（不静默降级为全精度）。
- **统一量化档位**：跨目标 `quant_mode` ∈ `fp16` / `int8` / `int4`（Core ML 另支持
  4bit/6bit/8bit 调色板）。ONNX 的 embedding/reranker 走 **weight-only int8**（只量化
  权重、激活保持 fp32，避免 onnxruntime 动态量化把激活也量化掉精度）。
- **MLX 量化模型支持**：含 `quantization` 的 MLX 模型先用 mlx-lm 反量化再转换。
- **自动可用性与保真验证**：每次转换成功都会在产物上自动做验证（详见[设计文档](
  docs/design.md) §验证）：用目标原生运行时加载、跑一次前向/生成、检查有限值，并用
  **参考余弦**衡量保真（fp16≈1.0、int8>0.99、int4 嵌入>0.95），低于 0.9 告警。
  依赖缺失或无法构建参考时**跳过验证而非失败**，不阻塞转换结果。
- **Swift 调用示例**（Core ML 目标）：转换成功后在输出目录生成
  `<模型名>-swift-example.swift`，含 `makeState()` + 自回归生成循环，可直接拷入 Xcode。
  示例会**自动编译**传入的 `.mlpackage`（Core ML 只能加载编译后的模型）。
- **混合后端兜底**：依赖缺失或架构不支持时自动回退到打包后端（权重 1:1 拷贝），
  工具始终可用；但真实转换**过程**中失败会直接报错，不回退到打包产物（打包产物不是
  可推理模型）。
- **全程后台线程**：目录扫描与转换均在 QThread 中执行，大模型转换时界面不卡顿；
  支持随时取消（扫描也可取消，关闭窗口时先取消再等待线程退出）；切换到新源目录时
  会取消上一次已作废的扫描，不浪费 CPU/IO。
- **输入健壮性**：源目录输入框防抖扫描；转换前校验输出目录不位于源目录内；
  单文件读取失败跳过而非让整次转换失败；扫描后立即提示模型架构是否走真实转换。

## 分发包（macOS .app / .dmg）

不想折腾 Python 环境的话，用分发包：

1. 从 [Releases](https://github.com/arlenecc/modconv/releases) 或本地
   `packaging/build_macos.py` 产物拿到 `多模型格式转换工具-<版本>.dmg`，双击挂载；
2. 把 **多模型格式转换工具.app** 拖进「应用程序」；
3. 双击镜像里的 **安装依赖.command**（仅首次）：它在
   `~/Library/Application Support/多模型格式转换工具/venv` 创建独立虚拟环境并安装依赖，
   会询问是否一并装上模型转换依赖（torch / transformers / coremltools / onnx / mlx 等，
   约 2–4 GB）；
4. 双击 **多模型格式转换工具** 启动。未签名应用若被 Gatekeeper 拦截，
   在 Finder 里按住 Control 点击 App →「打开」。

### 为什么是「启动器形态」的 .app

转换功能依赖 torch / transformers / coremltools / onnx / mlx，合计 2–4 GB，且与 Python
版本强绑定。把它们全部打进 `.app` 会得到一个数 GB 的产物，且换 Python 版本即失效。
因此 `.app` **内嵌图标与源码**（`Contents/Resources/app`），运行时复用宿主 Python：
优先用一键安装脚本创建的独立虚拟环境，其次用 PATH 上已装好 PyQt6 的 `python3`，
都不可用时弹出图形提示并引导安装。日志写入 `~/Library/Logs/多模型格式转换工具.log`。

用环境变量可指定解释器（须已装 PyQt6）：

```bash
MODCONV_PYTHON=/path/to/python3 open -a 多模型格式转换工具
```

### 自行构建分发包

```bash
python3 packaging/make_icon.py       # 生成 packaging/AppIcon.icns（需 Pillow + numpy）
python3 packaging/build_macos.py     # 产出 dist/多模型格式转换工具.app 与 dist/多模型格式转换工具-<版本>.dmg
```

`build_macos.py` 会在图标缺失时自动调用 `make_icon.py`；`--skip-dmg` 可只出 `.app`。
图标设计为蓝紫渐变圆角底 + 白色转换箭头，脚本可复现（见 `packaging/make_icon.py`）。

## 环境要求

- macOS（Apple Silicon 推荐；MLX 反量化仅支持 Apple Silicon）
- Python ≥ 3.10（**推荐 3.13**：coremltools 官方轮子在 3.13 上自带原生序列化库，
  可正常产出 `.mlpackage`；更低版本可能触发降级）
- GUI：PyQt6
- 真实转换按目标按需安装依赖：
  - Core ML：`torch` / `transformers` / `coremltools` / `safetensors` / `numpy`
  - ONNX：`onnx` / `onnxruntime`
  - MLX（库路径）：`mlx` / `mlx-lm` / `mlx-embeddings`（embedding/reranker）/ `mlx-vlm`（VLM）
  - GGUF：`gguf`（写 GGUF）+ `transformers`（分词器/结构）
  - 量化反量化：`mlx-lm`（MLX 量化模型）

## 安装

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,convert]"
```

## 使用

```bash
modconv          # 或 python -m modconv
```

1. 选择**模型源目录 / 文件**（自动探测源格式：HF 目录、`.onnx` 单文件、`.gguf` 单文件、
   MLX 目录）；
2. 选择**输出目录**；
3. 选择**目标格式**（Core ML / ONNX / MLX / GGUF）；
4. 可选选择/填写**最大上下文长度**（仅 Core ML 目标；默认「自动（≤512）」）；
5. 可选选择**量化模式**（`fp16` / `int8` / `int4`，Core ML 另支持 4bit/6bit/8bit）；
6. 点击**开始转换**。转换阶段与进度实时显示在状态栏与日志区，可随时**取消**。

转换完成后输出目录包含对应目标产物：

```
<输出目录>/
├── <模型名>.mlpackage          # Core ML：stateful 模型（含权重）
├── <模型名>-swift-example.swift # Core ML：Swift 调用示例（自动编译）
├── <模型名>.onnx               # ONNX：单文件（含权重）
├── model.safetensors + config.json  # MLX：MLXEmbedders 命名（及量化陪衬）
└── <模型名>.gguf               # GGUF：Llama 布局
```

### Swift 侧调用（Core ML 概要）

```swift
let model = try MLModel(contentsOf: modelURL)
let state = try model.makeState()
// prefill：整段 prompt 一次前向，KV cache 写入 state
// decode：每步传 1 个 token + 起始位置，state 自动携带 KV cache
```

完整可运行示例见生成的 `<模型名>-swift-example.swift`（`generateWithConvertedModel`）。

## 多目标转换（四源 × 四目标）

GUI「目标格式」下拉提供 **Core ML / ONNX / MLX / GGUF**，按源格式与模型类型自动选择
转换链路与所用库。后端探测（`backend.py:probe_backend`）决定真实后端与任务，
`workers.py` 据 `BackendPlan` 分派到对应实现：

| 目标 \ 源 | HF | ONNX | MLX | GGUF |
| --- | --- | --- | --- | --- |
| **Core ML** | `coremltools` 真实转换（stateful KV / GDN 单步 decode） | 不支持 | 反量化后同 HF→CoreML | 不支持 |
| **ONNX** | `HfToOnnxBackend`（`torch.onnx.export`） | `OnnxToOnnxBackend`（weight-only int8） | `MlxToOnnxBackend`（归一化为 HF 后导出） | `GgufToOnnxBackend`（反量化+重建） |
| **MLX** | `HfToMlxBackend`（mlx-lm/embeddings/vlm） | `OnnxToMlxBackend`（自定义权重映射，仅 encoder） | `MlxToMlxBackend`（重命名+量化，纯 MLX） | `GgufToMlxBackend`（归一化为 HF 后走 mlx-lm） |
| **GGUF** | `GgufBackend`（写 Llama 布局） | `GgufBackend`（权重取自 .onnx，结构/分词器取自同目录） | `GgufBackend`（反量化后同 HF→GGUF） | 不支持（重量化） |

要点与限制：

- **转换完成后自动验证**：每次转换成功都会自动验证（详见[设计文档](docs/design.md) §验证）：
  用目标原生运行时加载产物（ONNX→onnxruntime、MLX→mlx、GGUF→llama_cpp、CoreML→coremltools），
  跑前向后用**参考余弦**衡量保真（fp16≈1.0、int8>0.99、int4 嵌入>0.95），低于 0.9 告警；
  依赖缺失或无法构建参考时跳过验证而非失败。
- **ONNX→MLX（自定义）**：仅支持 encoder 类 embedding/reranker（BERT/RoBERTa/XLM-R），
  需原始 HF 的 `config.json` 与 ONNX 同目录（提供结构信息）；decoder（LLM）不支持
  （请用 HF→MLX）。权重经 ONNX 图遍历恢复（含内联为 Constant 的 bias 与 LayerNorm），
  输出 `model.safetensors` + `config.json`，命名对齐 **MLXEmbedders**（`encoder.layers.N.
  attention.query_proj` 等），量化陪衬为**兄弟键**。
- **embedding / reranker（encoder 家族：bge / jina / nomic 等）格式间转换**：
  - HF→ONNX：按模型类型导出（embedding→last_hidden_state，reranker→logits）。
  - ONNX→MLX：按**节点名**解析角色（兼容 bge 的 `attention/self/query` 与 jina 的
    `mixer/Wqkv` **融合 QKV 自动拆分**两种方言），朝向按 Transpose 前置 / transB 判定
    （无 Transpose 的 `[in,out]` 自动转置为 HF `[out,in]`），输出 MLXEmbedders 命名，
    量化陪衬为兄弟键；config 补 `model_type`/`num_labels`，缺 embeddings.LayerNorm 时补占位。
  - MLX→MLX：HF roberta/bert 命名 → MLXEmbedders 命名 + 可选量化（纯 MLX，已精确复现
    bge-reranker-v2-m3 交付物）；源已是量化(uint32) 时仅重命名并沿用源量化配置。
  - ONNX→ONNX / ONNX 目标 int8：embedding/reranker 用 **weight-only 量化**（只量化
    MatMul/Gemm 权重、激活保持 fp32；fp16 源须先整体转 fp32）——nomic-embed-text-v1.5
    已验证余弦 >0.997。
- **HF→ONNX / VLM**：仅导出 `input_ids`（不带 attention_mask），以保证 ONNX→MLX 链路可
  无损转回权重。多模态 Qwen2/3-VL 系（含 Qwen3.5）的 M-RoPE `position_ids` 改为**显式
  ONNX 输入**：`get_rope_index` 用 `mm_token_type_ids.tolist()`+groupby 与
  `iter(image_grid_thw)` 无法被 `torch.onnx.export` 追踪，原会烤成哑输入布局的常量导致
  文本 token 输出随序列长度偏离；现导出期仅用模型自带 `get_rope_index` 算占位，运行时由
  消费端按真实布局重算传入（见 `onnx_backend:mrope_position_ids`）。Qwen3.5 系仍以固定
  `seq=1024`（chunk 线性注意力展开）导出，运行时右填充到 1024。
- **GGUF 目标**：覆盖 Llama 布局的 BPE causal 模型（llama / qwen2 / qwen3 / mistral）；
  MoE、SentencePiece 特有模型暂不支持。源可为 HF 目录、MLX 目录（含 int4/int8 量化权重，
  自动反量化）与单个 `.onnx` 文件（界面有「选 ONNX 文件」按钮；权重取自 .onnx 且需保留
  HF 风格权重名，`config.json` / `tokenizer.json` 取自其同目录）。
- **跨格式互转（归一化）**：`source_normalize.py` 把 MLX / GGUF 源**归一化为临时 HF 目录**
  （MLX 量化权重反量化、MLXEmbedders↔HF 命名互转、GGUF 张量反量化与形状还原、config/
  分词器由 GGUF 元数据与词表重建——分词 ID 实测一致），再复用既有 HF→{ONNX,MLX,GGUF}
  后端。因此新增 MLX→ONNX、GGUF→MLX、GGUF→ONNX、embedding/reranker 在 ONNX↔MLX 间互转。
  注意：GGUF→GGUF（重量化）暂不支持；**embedding 的 GGUF 腿不可用**——llama.cpp 只承载
  llama 布局 causal 模型，BERT 家族 encoder（bge/jina/nomic）无法写入 GGUF（decoder 型
  embedding 如 Qwen3-Embedding 可走 GGUF）。
- 两个已修复的兼容性问题：transformers 5.x 的 `masking_utils` 用 `torch.diff` 做
  packed-sequence 检测，ONNX 无对应算子导致 causal 导出报 `aten::diff`——导出期间屏蔽该
  检测（只传 input_ids、无 packed sequence，数值无影响）；MLX 量化权重在维度不整除
  group_size 时会回退到更小的 group_size，反量化改为**按 scales 形状推断**而非信任 config 字段。
- 目标相关的依赖缺失时，界面会明确提示（不静默降级）。体积预估按各目标口径动态调整。

量化档位随目标自适应：Core ML 为 fp16/4-bit/6-bit/8-bit（palettization）；
ONNX/MLX/GGUF 为 fp16/int8/int4。

## 支持的模型架构

`llama` / `mistral` / `qwen2` / `qwen3` / `qwen3_5` / `qwen3_5_text` / `qwen3_5_moe` /
`qwen3_5_moe_text` / `gemma` / `gemma2` / `phi3`。

Qwen3.5 全系（dense 与 MoE，含纯文本 `qwen3_5_text` / `qwen3_5_moe_text` 检查点）均支持其
混合注意力架构：GDN（linear attention）与 full attention 交替的层。Causal / embedding /
reranker / VLM 四类模型类型均按后端自动识别。

## 开发与测试

```bash
.venv/bin/python3.13 -m pytest tests/ -q          # 全量测试（含真实转换集成测试）
QT_QPA_PLATFORM=offscreen .venv/bin/python3.13 -m modconv   # 无显示环境冒烟
```

单元测试以桩依赖隔离重量级库；集成测试用内置 tiny 模型走完整转换链路
（导出 → 转换 → 压缩 → 原生 predict parity 校验）；另有 Swift 集成测试用 `swiftc`
真实编译并运行生成的调用示例，与 Python 侧贪心解码逐 token 比对。

## 项目结构

```
modconv/             Python 包（GUI + 转换核心）
  app.py / __main__.py 应用入口（QApplication + MainWindow）
  main_window.py     PyQt6 主窗口
  workers.py         QThread 后台线程（扫描 / 转换）+ 后端分派
  backend.py         后端探测与架构白名单（4 目标：coreml/onnx/mlx/gguf）
  real_backend.py    HF→Core ML（torch.export + coremltools：stateful KV / GDN 单步 decode）
  onnx_backend.py    HF/ONNX/MLX/GGUF → ONNX（torch.onnx.export + onnxruntime 量化）
  mlx_backend.py     ONNX→MLX（encoder 权重映射：节点名角色/朝向/融合QKV/MLXEmbedders命名）
  mlx_rekey_backend.py MLX→MLX（HF 命名→MLXEmbedders 命名 + 量化，纯 MLX）
  mlx_libs_backend.py HF→MLX（mlx-lm / mlx-embeddings / mlx-vlm）
  gguf_to_mlx_backend.py GGUF→MLX（归一化为 HF 后走 mlx-lm）
  gguf_backend.py    HF/MLX/ONNX→GGUF（gguf 写 Llama 布局）
  source_normalize.py MLX/GGUF 源 → 临时 HF 目录（反量化/命名互转/config/分词器重建）
  verify.py          转换后产物自动验证（加载/前向/保真余弦）
  kv_wrapper.py      KV cache / GDN 状态的 Core ML State 模块（Core ML 目标）
  converter.py       打包后端（1:1 拷贝兜底）+ 共享的验证结果数据结构
  scanner.py         目录扫描     estimator.py  输出体积预估
  utils.py           通用工具（体积格式化 / 模型类型识别 / 混合注意力层类型名单）
examples/swift-gui/  SwiftPM 包装（便于 Xcode 调试 GUI 源码）
packaging/           macOS 分发包构建
docs/design.md       设计文档
tests/               单元测试 + 真实转换/Swift 集成测试
tools/               独立转换脚本（复刻核心逻辑，便于无 GUI 批量/复现交付物）
```

## 文档

- 软件设计文档：[docs/design.md](docs/design.md)

## 许可证

GPL-3.0，见 [LICENSE](LICENSE)。
