# 多模型格式转换工具 软件设计文档

版本：0.2.0 ｜ 更新日期：2026-10-19

把机器学习模型在 **HF / ONNX / MLX / GGUF** 四种格式之间互相转换，并导出为 Apple 平台
可直接运行的产物（Core ML `.mlpackage`、ONNX、MLX、GGUF）。Python 包与命令行名为
`modconv`，macOS 应用名为 **多模型格式转换工具**，仓库名为 `modconv`。

> 本项目最初只做「MLX → Core ML」转换，现已扩展为四源 × 四目标。

## 1. 系统概览

```
┌────────────────────────── MainWindow (PyQt6) ──────────────────────────┐
│  配置区：源目录/文件 · 输出目录 · 目标格式 · 上下文长度 · 量化模式          │
│  显示区：统计行 + 开始/取消 + 日志；状态栏：总进度                        │
└──────────────┬─────────────────────────────┬───────────────────────────┘
               │ QThread                     │ QThread
        ┌──────▼──────┐              ┌───────▼─────────┐
        │  ScanWorker │              │ ConversionWorker │
        └──────┬──────┘              └───────┬─────────┘
               │ scan_directory              │ probe_backend → BackendPlan → 分派
               ▼                             ▼
        ┌─────────────┐      ┌──────────────────────────────────────────┐
        │  scanner.py │      │ 真实后端（按目标/源分派）                   │
        └─────────────┘      │  real_backend (Core ML)                   │
                             │  onnx_backend  (HF/ONNX/MLX/GGUF → ONNX)  │
                             │  mlx_backend / mlx_libs / mlx_rekey       │
                             │  gguf_backend / gguf_to_mlx               │
                             │  source_normalize (MLX/GGUF → 临时 HF)     │
                             └───────────────────┬──────────────────────┘
                                                 │ 转换成功
                                                 ▼
                                        ┌────────────────┐
                                        │  verify.py     │ 加载+前向+保真余弦
                                        └────────────────┘
```

分发包（§15）不改动上述架构：`.app` 只是一个带图标的启动器，负责定位 Python 解释器并把
`Contents/Resources/app` 加入 `PYTHONPATH` 后执行 `python -m modconv`。

## 2. 模块职责

| 模块 | 职责 |
|---|---|
| `app.py` / `__main__.py` | 应用入口（QApplication + MainWindow） |
| `main_window.py` | 界面构建、输入校验、信号接线、线程生命周期（关闭保护） |
| `workers.py` | `ScanWorker` / `ConversionWorker`（QThread）；`ConversionWorker._run_backend` 按 `BackendPlan` 分派后端；转换成功后调 `verify` |
| `backend.py` | 后端探测：源格式判定、架构白名单、依赖探测、MLX 量化检测，产出 `BackendPlan`（含 `name` / `task` / `target` / `available`） |
| `converter.py` | 打包后端 `ModelConverter`：流式 1:1 拷贝兜底（不可直接推理）；共享 `ConversionResult` / `VerificationResult` 数据结构 |
| `real_backend.py` | **Core ML** 真实转换：权重布局归一化 → `torch.export` → coremltools（stateful mlprogram）→ 量化压缩 → 元数据/Swift 示例 |
| `kv_wrapper.py` | `KvCacheWrapper`：把 HF 模型包装为 (input_ids, cache_position) → logits 的状态模块（Core ML 目标） |
| `onnx_backend.py` | **ONNX** 后端：`HfToOnnxBackend` / `OnnxToOnnxBackend` / `MlxToOnnxBackend` / `GgufToOnnxBackend`；含 VLM M-RoPE 处理、causal/encoder 分派、onnxruntime 量化 |
| `mlx_backend.py` | **ONNX→MLX**（自定义权重映射，仅 encoder：节点名角色/朝向/融合 QKV 拆分/MLXEmbedders 命名/兄弟键量化） |
| `mlx_rekey_backend.py` | **MLX→MLX**（HF 命名→MLXEmbedders 命名 + 量化，纯 MLX） |
| `mlx_libs_backend.py` | **HF→MLX**（`mlx-lm` / `mlx-embeddings` / `mlx-vlm`，按模型类型分派） |
| `gguf_to_mlx_backend.py` | **GGUF→MLX**（归一化为 HF 后走 `mlx-lm`） |
| `gguf_backend.py` | **HF/MLX/ONNX→GGUF**（gguf 写 Llama 布局） |
| `source_normalize.py` | 把 MLX / GGUF 源**归一化为临时 HF 目录**（反量化/命名互转/config/分词器重建），供既有 HF→{ONNX,MLX,GGUF} 后端复用 |
| `verify.py` | 转换后产物自动验证（按目标加载/前向/保真余弦；缺依赖则跳过） |
| `scanner.py` / `estimator.py` / `utils.py` | 目录扫描、输出预估（打包/真实两种口径）、格式化与模型类型识别 |
| `packaging/make_icon.py` / `build_macos.py` | 构建期：图标绘制与 `.app`/`.dmg` 组装 |

## 3. 后端探测与分派

`backend.py` 是「源格式 + 目标格式 + 模型类型 + 依赖可用性」的决策中心：

- **源格式判定** `detect_source_format`：单文件按扩展名（`.onnx` / `.gguf`）；目录按
  `config.json` + 量化字段 / MLXEmbedders 命名判定为 `hf` 或 `mlx`，无 config 时看是否含
  `.onnx` / `.gguf`。
- **目标** `TARGET_FORMATS = ("coreml", "onnx", "mlx", "gguf")`；**源** `hf / onnx / mlx / gguf`。
- **架构白名单** `SUPPORTED_MODEL_TYPES`（Core ML 目标）：`llama / mistral / qwen2 / qwen3 /
  qwen3_5 / qwen3_5_text / qwen3_5_moe / qwen3_5_moe_text / gemma / gemma2 / phi3`。
- **依赖探测**：各目标的依赖集合（torch/transformers/coremltools、onnx/onnxruntime、
  mlx/mlx_lm/mlx_embeddings/mlx_vlm、gguf）用 `find_spec` 轻量探测并缓存，供 UI 预估与
  架构提示复用（避免每次扫描触发重型导入卡界面）。
- **`probe_backend(source, requested, target)`** 返回 `BackendPlan`，其中 `name ∈
  {coremltools, onnx, mlx, mlx_libs, mlx_rekey, gguf, package}`、`task` 描述具体链路
  （如 `hf->onnx` / `mlx->onnx` / `onnx->mlx` / `gguf->mlx` / `hf->gguf`）。依赖缺失或
  源-目标不匹配时 `available=False` 且给出 `reason`——调用方据此**明确报错而非静默降级**。

`workers.py:ConversionWorker._run_backend` 据 `BackendPlan.name` + `task` 分派：

- `package` → `ModelConverter`（1:1 拷贝兜底，不可直接推理）；
- `coremltools` → `real_backend.CoremltoolsBackend`；
- `onnx` → 按 task 分派 `OnnxToOnnxBackend` / `MlxToOnnxBackend` / `GgufToOnnxBackend` /
  `HfToOnnxBackend`；
- `mlx` → `OnnxToMlxBackend`（仅 ONNX 源）；
- `mlx_rekey` → `MlxToMlxBackend`；
- `mlx_libs` → `GgufToMlxBackend`（gguf→mlx）或 `HfToMlxBackend`；
- `gguf` → `GgufBackend`。

真实转换（非 package）失败**直接报错**，禁止静默回退到打包产物。转换成功且非打包时，
`ConversionWorker.run` 的 `finally` 中自动调用 `verify_conversion` 并写入结果的
`verification` 字段（验证异常不影响转换结果）。

统一量化模式 `quant_mode ∈ {fp16, int8, int4}`（Core ML 目标另映射到 4/6/8-bit 调色板）。
`model_kind ∈ {auto, causal, embedding, reranker}` 仅作 HF→ONNX 的模型类型提示。

## 4. Core ML 真实转换（real_backend + kv_wrapper）

### 4.1 流程

`CoremltoolsBackend.run` 阶段：

0. **权重布局归一化**（按需，见 §4.4）：checkpoint 权重名/形状与架构期望不符时另存一份
   纯 HF 布局的临时副本再加载；中间产物写在输出目录旁，`finally` 中清理；
1. **反量化**（可选）：MLX 量化模型经 `mlx_lm.convert(dequantize=True)` 输出到临时目录；
2. **加载配置与权重**：`AutoConfig` + `AutoModelForCausalLM`（eager 注意力 + eager 专家）。
   最大上下文长度 = 用户输入（先截断到 `max_position_embeddings`、再抬到架构下限，混合注意
   力架构为 2）或自动选取（`min(max_position_embeddings, 512)`）。MoE 必须指定
   `experts_implementation="eager"`（`torch.histc` 算子 coremltools 前端不支持）；反量化
   中间产物写在**输出目录旁**，避免数十 GB 文件撑爆启动盘；
3. **torch.export**：`KvCacheWrapper` 包装为**单步 decode** 后导出（详见 §4.2）：图每次只
   推进一个 token，形状全静态（input_ids `(1,1)`、cache_position `(1,)`）；
4. **coremltools 转换**：`convert_to="mlprogram"`，`minimum_deployment_target=iOS18`，
   `compute_precision=FLOAT16`，`states=` **只一个** `StateType`（合并的 KV 缓存）。混合架构
   的 GDN conv/rec 状态走图的输入/输出张量（理由见 §4.3）；转换完成后立即 `del` torch 侧
   大对象并 `gc.collect()`，降低后续峰值内存；
5. **后处理与保存**：量化压缩（`OpPalettizerConfig(nbits,kmeans)`，失败直接抛异常）→ spec
   重命名输出为 `logits` + 写元数据 → `MLModel(spec, weights_dir=…)` 重建（mlprogram 不能
   仅从 spec 重建）→ 保存 `.mlpackage` + 生成 Swift 示例。

**执行计划构建失败（-14）的自动收敛**：ANE 能否构建执行计划既无文档也无先验公式，只能实测。
Qwen3.5-0.8B 在**旧设计**（静态 `max_seq_len` 大图 + 3 个 StateType）下实测：4-bit 量化后
的图在 512/256/128 都 -14，未量化时这些长度都能过；fp16 加载把权重内存减半也救不了量化场景。
改为单步 decode 图后算子数下降两个数量级（-14 是否仍出现待重测，降级机制保留作兜底）。
`OpPalettizerConfig` 只接受 `per_tensor` / `per_grouped_channel`，`per_block` 粒度的线性
量化（int4/int8）实测同样 -14。因此"追踪 → 转换 → 量化 → 重建"整段可重试，降级候选按实测
影响力排序（`_fallback_plans`）：① 原样（量化 + 原长度）；② 量化 + 长度减半；③ **取消量化
回到原长度**（4-bit 图才是 -14 主因，宁可体积更大）；④ 逐级降长度（下限 64）。所有降级都写
进日志并附在结果消息里，不静默发生。

每构造一个 `MLModel` 都会构建一次执行计划（大模型十几分钟），因此：`ct.convert` 返回对象
本身即判据；量化后重建同样作为判据；而写元数据后的"重建"**只为让改过的 spec 落盘**，显式传
`compute_units=CPU_ONLY` 跳过 ANE 编译——省掉十几分钟，也避免内存紧张引起的误报。

**临时包必须显式删除**（`_drop_package`）：coremltools 只在本进程正常退出时（atexit）清理
系统临时目录下的完整包（大模型上 GB 级）；GUI 长驻进程连续转换会堆积。重建完成、重试切换、
保存落盘后都立即删除旧包（只删标记为 `is_temp_package` 的包）。

任何阶段失败抛异常，`ConversionWorker` 捕获后**直接向用户报错**——不回退打包后端。coremltools
遇到编译不过的模型并不抛异常，只发 `RuntimeWarning: -14` 就照常返回，故重建 `MLModel` 时
`warnings.catch_warnings` 捕获该告警：命中即**判定失败**（产物照常落盘便于排查）。转换期
int64→int32 常数降级的 `might lose precision` 告警（大模型可达数万条）在 coremltools logger 上
挂过滤规则丢弃。

### 4.2 Stateful KV cache 设计（kv_wrapper.py）

- **State 载体**：每层 K/V 注册为 named buffer `cache_{i}_k/v`，形状
  `(1, n_kv_heads, max_seq_len, head_dim)`；coremltools 按 buffer 名匹配 `StateType`。
- **buffer 固定 fp32**：coremltools 前端对 `read_state` 无条件插入 fp32 cast，in-place scatter
  的 data 恒为 fp32，updates 也必须 fp32；写回 state 时前端自动 cast 回 fp16（Core ML State
  仅支持 fp16）。掩码精度独立跟随模型原精度。
- **写入槽位**：用 `arange(kv_length) + cumulative_length` 定位写入槽位，游标每次 forward 重
  绑定为 `cache_position[0]`（输入派生），不做 in-place 累加；导出图中唯一的 mutation 是对 KV
  buffer 的 rank-4 `scatter_`（前端映射为 `scatter_along_axis`，支持动态长度）。
- **注意力掩码**：wrapper 内部按 `cache_position` 自算静态形状 4D 因果掩码
  （`slots > cache_position` 广播比较 → 布尔 cast 乘 fp16 min -65504），绕开 HF 动态掩码机器
  （其运行时 tile 会破坏 ANE）；掩码值取 fp16 有限最小值，避免 fp32 min 在 fp16 计算中溢出。
- **导出要点**：单步输入形状全静态（无动态维度声明）；导出包在 `torch.no_grad()`；分解表去掉
  `aten.matmul` 分解并保留 `alias` 恒等（matmul 分解产生的 expand 会让 coremltools 无法静态
  证明 tile reps=1 而 -14）。

### 4.3 Qwen3.5 全系（dense / MoE）：结构特征补丁 + 单步 GDN decode

- **补丁按结构特征识别而非写死类名**：transformers 由同一份 modular 源码为各变体生成具名类
  （dense 的 `Qwen3_5GatedDeltaNet` / MoE 的 `Qwen3_5MoeGatedDeltaNet`、
  `Qwen3_5TextRotaryEmbedding` / `Qwen3_5MoeTextRotaryEmbedding`），写死类名会让 MoE 失败。
  GDN 前向所需内核函数在打补丁时从 mixer 所属模块解析并挂到实例上（动态导入无法被 torch.export 追踪）。
- **MoE 专家层**：两条专家路径都不可导出（eager 在 Python 层遍历数据相关的 `expert_hit`；
  `grouped_mm`/`batched_mm` 用 `torch.histc`）。wrapper 替换为静态批量前向：把 (token, top-k
  槽位) 展平成 S=T×k 行 → 按 expert id gather 权重 → batched matmul → reshape 回 (T,k,H) 求和。
  与原版逐专家累加**逐位等价**（未选中专家的路由权重恒为 0）。
- **GDN 单步 decode 取代分块内核**：原实现 pad 到静态 `max_seq_len` 后走 `chunk_gated_delta_rule`
  ——构造 (chunk,chunk) 衰减矩阵（`cumsum`→pairwise `exp`→63 步 Neumann 迭代求解三角系统），
  fp16 下精度崩（cos 仅 0.41）且全图 3.4 万算子（真实模型 30 分钟）。单步下 delta rule 退化为
  一次秩一更新（与 `ref/CoreML-LLM` 的 `MLKVLinearAttnDecodeStep` 同构）：

      rec    = rec * exp(g)              # 衰减
      kv_mem = (rec * k).sum(-2)         # 读记忆
      delta  = (v - kv_mem) * beta       # delta 修正
      rec    = rec + k ⊗ delta           # 秩一更新
      out    = (rec * q).sum(-2)         # 读出

  无 cumsum / pairwise 矩阵 / 三角求解：fp16 下数值良态，算子数每层从数百降到十余个。已验证
  逐步 prefill 与 HF 原生整段 prefill **逐位一致**（cos=1.000000，差 ~6e-8）。prefill 改为
  **循环调用同一张 decode 图**（recurrent prefill）。
- **GDN 状态走图 I/O（只留一个 MLState）**：conv/rec 原本也做成 `ct.StateType`（与 KV 共 3 个）。
  `ref/CoreML-LLM` 实测：每个 chunk 超过 1~2 个 StateType 时 Core ML 的 CPU runtime 会误编译
  `slice_update`（输出 garbage/NaN），ANE 侧报 Error 11——正好解释"转换成功、CPU 预测全 NaN"。
  改为只把最大的 KV 缓存放进 MLState，conv/rec 作为图的输入/输出张量逐步传递
  （`new_conv_states` / `new_recurrent_states`，Swift 侧字典持有）。

### 4.4 权重布局归一化（加载前的强制校验）

MLX 生态 checkpoint 未必是 transformers 能直接加载的布局，三类差异必须修正：

1. **多模态（复合）前缀**：MLX 把文本模型放在 `language_model.` 之下（带 `vision_tower.*`），
   `AutoModelForCausalLM` 期望 `model.layers.*`；
2. **Conv1d 权重轴序**：MLX 的 `Conv1d` 是 `(out, kernel, in)`，PyTorch 是 `(out, in, kernel)`；
3. **RMSNorm 权重折叠**（Qwen3.5 系）：HF 的 `Qwen3_5RMSNorm` 是零中心约定 `1.0 + weight`，
   mlx-lm 的 `sanitize` 在 convert 时给 norm 权重预加 1.0 折叠；喂给 HF 会**双重 +1**（`2+gain`）
   导致输出乱码。修正：写回 HF 布局时对 `input_layernorm` / `post_attention_layernorm` /
   `model.norm` / `q_norm` / `k_norm` 减 1.0；GDN 内部的 `linear_attn.norm` 是 `RMSNormGated`
   （HF 侧本不加 +1），必须原样保留。判定信号与 mlx-lm 互为镜像：conv1d 需转置 ⟺ norm 权重是 `1+gain`。

关键在于：名字对不上时 transformers 不报错，而是把所有权重"新建初始化"——转换"成功"却产出
随机权重模型。因此加载前必须先校验：`_expected_state_shapes`（meta 设备搭骨架取期望名字与
形状，0.8B 约 0.2s）与 `_checkpoint_tensor_shapes`（只读 safetensors 头部）比对 →
`_resolve_weight_layout` 选前缀并按"期望形状 = checkpoint 形状交换后两维"判定转置张量；重合
比例低于阈值返回 `None` → **中止转换并报错**。需要归一化时 `_write_text_checkpoint` 逐分片另存
（内存约一个分片），只留文本权重（丢弃 `vision_tower.*`），Qwen3.5 系另还原 RMSNorm 折叠。
布局已匹配时整套检查为零成本（不读权重数据、不落盘）。

## 5. ONNX 后端（onnx_backend.py）

四类 `run` 后端共用同一 `_prepare_vlm` / `_run` 导出骨架，按源分派：

| 类 | 源 | 实现 |
|---|---|---|
| `HfToOnnxBackend` | HF | `torch.onnx.export` + onnxruntime 量化 |
| `OnnxToOnnxBackend` | ONNX | weight-only int8（embeddings/reranker 友好） |
| `MlxToOnnxBackend` | MLX | 归一化为 HF 目录（反量化+命名映射）后 `torch.onnx.export` |
| `GgufToOnnxBackend` | GGUF | llama 布局反量化 + 重建 config/tokenizer 后导出 |

通用要点：

- **模型类型分派**：causal / embedding（→last_hidden_state）/ reranker（→logits）/ vlm。
- **HF→ONNX 仅导出 `input_ids`**（不带 attention_mask）：保证 ONNX→MLX 链路可无损转回权重
  （避免 GreaterOrEqual 等 onnx2torch 不支持的算子）。
- **导出期算子补丁**：
  - `_patch_transformers_for_onnx_export`：屏蔽 transformers 5.x `masking_utils` 的
    `torch.diff` packed-sequence 检测（ONNX 无对应算子，`aten::diff` 报错）；
  - `_patch_causal_conv1d_for_onnx_export`：把 `torch.linalg.solve_triangular` 替换为手写
    可 trace 的**单位下三角前代求解**（GDN chunk 解三角系统用）；实测与 `solve_triangular`
    cosine=1.0000，且避免了原 `is_torchdynamo_exporting=True` 走错前向（cos 仅 0.945）；
  - `_patch_mrope_cache`：对 M-RoPE 模型强制每次重算（已被显式 `position_ids` 输入取代，留作保险）。
- **量化**：causal 用 onnxruntime `quantize_dynamic`（LLM 对激活量化不敏感）；embeddings/reranker
  用 **weight-only int8**（手写 `MatMul`/`Gemm` 权重 int8 + `DequantizeLinear`，激活全程 fp32——
  onnxruntime 动态量化会量化激活使 embedding 掉精度，余弦只到 0.985）；int4 由
  `MatMul4BitsQuantizer` best-effort，缺失时回退 int8。

### 5.1 VLM / M-RoPE（Qwen2/3-VL 系，含 Qwen3.5）

**根因**：transformers 的 `get_rope_index` 用 `mm_token_type_ids.tolist()` + `itertools.groupby`
与 `iter(image_grid_thw)`/`next()`，对张量均非 trace 友好，被 `torch.onnx.export` 追踪时会把
`position_ids` **烤成哑输入布局的常量**。结果：只有真实输入的图文布局（图像 token 位置/数量/网格）
与导出哑输入完全一致时输出才对，否则文本 token logits 随序列长度逐渐偏离（实测逐位置 cosine
中值 ~0.81，图像 token 正确、文本 token 退化）。

**修复**：`_vlm_export_spec` 对 `_MROPE_FAMILIES`（qwen2_vl / qwen2.5_vl / qwen3_vl /
qwen3_5 等）把 **`position_ids` 作为显式 ONNX 输入**——导出期用模型自带 `get_rope_index` 仅算
哑样本占位（形状 `(3,B,S)`），并**移除 `mm_token_type_ids` 图输入**（模型收到 `position_ids`
后跳过内部的 `compute_3d_position_ids`）。消费端（onnxruntime / Swift）运行前必须用同一
`get_rope_index` 按真实布局算好 `position_ids` 传入，否则 ONNX 报错。已新增消费端工具
`mrope_position_ids(model, input_ids, mm_token_type_ids, image_grid_thw, attention_mask)`。

**Qwen3.5 定长**：混合线性注意力用 chunk 顺序扫描，旧版 `torch.onnx.export` 无法表达动态循环，
故把序列长度固定为 chunk_size(64) 的倍数——`fixed_seq=True`，`seq=1024`。导出时扫描展开为固定
16 个 chunk；运行时消费端把输入右填充到 1024（右侧 pad 对因果注意力/因果递归安全，不影响前置
真实 token）。`dynamic_axes` 仅 batch 维动态。

## 6. MLX 后端（mlx_backend / mlx_rekey_backend / mlx_libs_backend / gguf_to_mlx_backend）

- **ONNX→MLX（自定义，`mlx_backend.OnnxToMlxBackend`）**：仅 encoder 类 embedding/reranker
  （BERT/RoBERTa/XLM-R）。直接读 ONNX initializers（不依赖 onnx2torch）映射到 MLX：
  - 按**节点名**解析角色，兼容 bge 的 `attention/self/query` 与 jina 的 `mixer/Wqkv` **融合
    QKV 自动拆分**两种方言；
  - 朝向按 Transpose 前置 / `transB` 判定（无 Transpose 的 `[in,out]` 自动转置为 HF `[out,in]`）；
  - 输出 **MLXEmbedders 命名**（`encoder.layers.N.attention.query_proj` 等）；
  - 量化陪衬为**兄弟键**（权重键去 `.weight` 后追加 `.scales`/`.biases`，如
    `…query_proj.scales`——绝不能是 `…query_proj.weight.scales`，否则 Swift 加载器找不到陪衬键报错）；
  - config 补 `model_type` / `num_labels`，缺 `embeddings.LayerNorm` 时补 ones/zeros 占位。
  decoder（LLM）不支持，给出明确报错（请走 HF→MLX）。
- **MLX→MLX（`mlx_rekey_backend.MlxToMlxBackend`，纯 MLX）**：HF roberta/bert 命名 →
  MLXEmbedders 命名 + 可选量化（复刻 `tools/convert_mlx_rekey_quantize.py`，已精确复现
  bge-reranker-v2-m3 交付物：重键 + config 补 `quantization{bits,group_size,mode}`）。源已是
  量化(uint32) 时仅重命名并沿用源量化配置。
- **HF→MLX（`mlx_libs_backend.HfToMlxBackend`）**：按模型类型分派 `mlx-lm`（causal）/
  `mlx-embeddings`（embedding/reranker）/ `mlx-vlm`（VLM），量化由库原生支持（int4/int8）。
- **GGUF→MLX（`gguf_to_mlx_backend.GgufToMlxBackend`）**：GGUF 源经 `source_normalize` 归一化
  为 HF 目录后走 `mlx-lm`。

## 7. GGUF 后端（gguf_backend.py）

- **HF / MLX / ONNX → GGUF**：用 `gguf` 写 **Llama 布局**（llama / qwen2 / qwen3 / mistral）；
  MoE、SentencePiece 特有模型暂不支持。MLX 源含 int4/int8 量化权重时自动反量化。
- **ONNX 文件源**：权重取自 `.onnx`（需保留 HF 风格权重名），结构与分词器取自同目录的
  `config.json` / `tokenizer.json`（界面有「选 ONNX 文件」按钮，`backend.resolve_sidecar_dir`
  逐级向上定位伴随目录）。
- **GGUF→GGUF（重量化）暂不支持**：会明确报错，提示先转为 HF/MLX/ONNX。

## 8. 源归一化（source_normalize.py）

把 **MLX / GGUF 源归一化为临时 HF 目录**，再复用既有 HF→{ONNX,MLX,GGUF} 后端，从而支撑跨格式
互转（MLX→ONNX、GGUF→MLX/ONNX、embedding/reranker 在 ONNX↔MLX 间互转）：

- **MLX → HF**：MLX 量化权重反量化（按 scales 形状推断 group_size，不信任 config 字段）；
  MLXEmbedders ↔ HF 命名互转；仅保留文本权重（丢弃 `vision_tower.*`）。
- **GGUF → HF**：GGUF 张量反量化与形状还原；config / tokenizer 由 GGUF 元数据与词表重建
  （分词 ID 实测一致）。
- 临时目录写在输出目录旁，`finally` 中清理（避免 GB 级中间文件撑爆启动盘）。

> 注意：**embedding 的 GGUF 腿不可用**——llama.cpp 只承载 llama 布局 causal 模型，BERT 家族
> encoder（bge/jina/nomic）无法写入 GGUF（decoder 型 embedding 如 Qwen3-Embedding 可走 GGUF）。

## 9. 自动验证（verify.py）

`verify_conversion(target, output_path, source, hooks, ...)` 按目标分派，原则：

- 统一入口、按目标格式（onnx / mlx / gguf / coreml）分派；
- **依赖缺失 / 无法构建参考时跳过（skipped）而非失败**，不阻塞转换结果；内部异常全部捕获，
  绝不抛到转换主流程之外；
- **保真用参考余弦衡量**：相同输入下，转换产物与参考（源 ONNX / 源 HF）输出余弦；正常量化应
  在 0.9 以上（fp16≈1.0，int8>0.99，int4 嵌入>0.95），远低于此说明有额外损失（朝向/命名/数值错误）。

`ConversionWorker.run` 在转换成功且非打包后端时自动调用，结果写入 `ConversionResult.verification`。

## 10. 线程模型与生命周期

- 所有耗时操作（扫描 / 转换）在 QThread 中执行，通过信号跨线程更新 UI；
- `ConversionWorker` 默认 `backend="auto"`：`probe_backend` 决定真实/打包后端，真实转换失败
  **直接报错**（不回退）；`conversion_finished` **无条件发出**——探测阶段抛异常若漏掉它，主线程
  永远不会调用 `_set_running(False)`，GUI 永久卡在禁用态；
- `ScanWorker.cancel()` 从扫描回调抛内部异常提前退出（不发 `scan_finished`）；关闭窗口时先取消
  再等待，否则大目录扫描远超等待预算，线程会在对象销毁后继续运行而崩溃；
- 扫描线程结束后立即从 `MainWindow._scan_workers` 移除；源目录输入框 `editingFinished` 经 400ms
  防抖才触发扫描；任何"放弃扫描"分支都递增 `_scan_gen`，避免旧扫描结果覆盖已失效状态；
- 关闭等待用 `_wait_for_thread`（小步 `wait` + `processEvents`）替代单次 `wait(30s)`，避免主线程阻塞；
- **关闭保护**（`closeEvent`）：转换运行中先 `cancel()` 并等待至多 30s，超时则拒绝关闭；
- **线程引用模型**：`ScanWorker`/`ConversionWorker` 继承 `_TrackedThread`，`start()` 时登记到模块级
  集合、线程真正结束后注销。QThread 运行中被销毁会触发 Qt `qFatal` → 进程 abort，Python 无法捕获；
- 转换线程分两层持有（`_conv_workers` 全部未退出线程 + `_conv_worker` 当前这次），避免关闭时销毁
  仍在运行的线程或重复启动两个转换线程；新扫描开始前先取消在飞旧扫描。

## 11. 内存与性能

- 扫描/打包拷贝均为流式（扫描只 stat；打包 4 MiB 块），内存与文件大小无关；
- 扫描用 `os.scandir` + 显式栈遍历（`DirEntry` 自带 d_type，通常无需系统调用）；文件名排序保证
  结果与文件系统返回顺序无关；进度按 16 MiB 字节间隔 / 每 256 文件节流，避免事件洪泛；
- 真实转换：量化模型反量化后以 fp16 加载（内存减半）；带 `low_cpu_mem_usage` 就地载入（省一倍
  峰值内存）；追踪结束先丢弃 wrapper + HF 模型，`ct.convert` 完成立即释放导出图（数十 GB）再进入
  量化/保存；4-bit 压缩后再释放压缩前模型对象及其临时权重目录；
- Core ML 4-bit palettization 受 coremltools `weight_threshold`（2048 元素）约束，小权重自动跳过；
- 阶段进度以 `file_progress`（relpath = "阶段 i/total"）上报，GUI 识别该前缀显示阶段文案并同步进度条。

## 12. 测试策略

- **单元测试**（桩依赖隔离 torch/coremltools/onnx/mlx/gguf）：后端探测与 `BackendPlan` 描述、
  最大上下文选取、dtype/quant_mode 兼容、转换主流程 kwargs 断言、量化/取消/失败报错（不回退）、
  GUI 交互（输入校验、阶段文案、closeEvent 保护、信号透传）、**VLM 导出规格**（`_vlm_export_spec`：
  qwen3_5 定长 1024、`position_ids` 为显式输入且非烤死常量、移除 `mm_token_type_ids`）、
  跨格式互转与验证；
- **集成测试**（真实依赖，内置 tiny 模型，不联网）：KV wrapper 与 HF eager parity、完整转换链路
  产出 `.mlpackage`、4-bit LUT proto 断言、原生 `predict` prefill/decode parity、反量化临时目录
  清理；Qwen3.5 dense（4 变体）与 MoE 端到端；Core ML 与 torch 参考的数值一致性（最大误差 /
  余弦 / argmax 一致率三层断言）；ONNX / MLX / GGUF 各链路的 round-trip 与保真余弦；
- **Swift 集成测试**（需 `xcrun swiftc`）：生成的调用示例零警告通过类型检查；编译并运行自回归
  生成，与 Python 侧贪心解码逐 token 比对；
- 边界与失败路径：探测抛异常也必须发 `conversion_finished`、`max_seq_len=1` 完整转换、越界相对
  路径被拒、缺失源文件被跳过、扫描取消不发结果；GUI 测试在 `QT_QPA_PLATFORM=offscreen` 下运行。

## 13. 已知限制

- 支持架构白名单（Core ML 目标）：llama / mistral / qwen2 / qwen3 / qwen3_5 / qwen3_5_text /
  qwen3_5_moe / qwen3_5_moe_text / gemma / gemma2 / phi3；
- `SEQ_LEN_CAP = 512`：Core ML 自动选取上限（KV buffer 体积随长度线性增长）；显式大长度会生成
  GB 级 KV cache 且在 ANE 上大概率 `-14`，触发自动逐级降长度重试（每次重做 ANE 编译极耗时）；
- 非量化模型以 fp32 加载（转换目标 fp16，保证单次舍入）；
- 打包后端输出为权重 1:1 拷贝（含 Manifest），不能直接推理，仅作依赖缺失时的兜底产物；
- 输出体积预估（GUI「预计输出」）按后端口径估算，系统性偏低（未计入 LUT / Manifest 固定开销），
  仅供粗略参考；
- 多模态（VL）checkpoint 的 Core ML 目标只转换其**文本 LM**部分（`vision_tower.*` 丢弃），产出
  纯文本模型；VLM 的 ONNX 目标保留视觉塔但 `position_ids` 须由消费端按真实布局提供；
- 权重布局不匹配时需额外写出与源权重等大的临时副本，占用同等磁盘（写在输出目录旁）；
- ONNX→MLX 仅支持 encoder；GGUF 目标仅 Llama 布局 causal；GGUF→GGUF 重量化不支持；
  embedding 的 GGUF 腿不可用；
- 分发包是**启动器形态**的 `.app`，不内嵌 torch/transformers/coremltools/onnx/mlx（合计 2–4 GB
  且与 Python 版本强绑定），需宿主 Python 环境（见 §15）。

## 14. 项目结构

```
modconv/             Python 包（GUI + 转换核心）
  app.py / __main__.py 应用入口
  main_window.py     PyQt6 主窗口
  workers.py         QThread 后台线程 + 后端分派
  backend.py         后端探测与架构白名单（4 目标）
  real_backend.py    HF→Core ML（torch.export + coremltools）
  onnx_backend.py    HF/ONNX/MLX/GGUF → ONNX
  mlx_backend.py     ONNX→MLX（encoder 自定义映射）
  mlx_rekey_backend.py MLX→MLX（重命名 + 量化，纯 MLX）
  mlx_libs_backend.py HF→MLX（mlx-lm/embeddings/vlm）
  gguf_to_mlx_backend.py GGUF→MLX
  gguf_backend.py    HF/MLX/ONNX→GGUF
  source_normalize.py MLX/GGUF → 临时 HF 目录
  verify.py          转换后产物自动验证
  kv_wrapper.py      KV cache / GDN 状态的 Core ML State 模块
  converter.py       打包后端（1:1 拷贝兜底）
  scanner.py / estimator.py / utils.py 扫描 / 预估 / 工具
examples/swift-gui/  SwiftPM 包装（Xcode 调试 GUI 源码）
packaging/           macOS 分发包构建
tools/               独立转换脚本（复刻核心逻辑，便于无 GUI 批量/复现交付物）
tests/               单元测试 + 真实转换/Swift 集成测试
docs/design.md       设计文档
```

## 15. 分发包构建

```
packaging/make_icon.py      1024×1024 绘制（Pillow + numpy）→ 各尺寸 PNG → iconutil → AppIcon.icns
packaging/build_macos.py    pyproject 取版本 → 组装 .app → ditto/hdiutil → .dmg
```

`.app` 结构（`build_macos.py` 无第三方依赖，仅用标准库 + `ditto`/`hdiutil`）：

```
多模型格式转换工具.app/Contents/
├── Info.plist                        CFBundleIconFile=AppIcon、BundleID、最低 12.0
├── PkgInfo
├── MacOS/多模型格式转换工具                  启动器（bash，可执行）
└── Resources/
    ├── AppIcon.icns                  图标
    ├── install_dependencies.command  一键装依赖（App 内 + DMG 根目录各一份）
    └── app/                          内嵌源码：modconv/ + pyproject.toml + README + LICENSE
```

启动器解析顺序：`$MODCONV_PYTHON` → `~/Library/Application Support/多模型格式转换工具/venv` → PATH 上
已装 PyQt6 的 `python3`（3.13→3.10 依次尝试）。全部不可用时弹 `osascript` 对话框引导运行安装脚本。
启动前把 `Contents/Resources/app` 前置到 `PYTHONPATH`，`cd "$HOME"` 后执行 `python -m modconv`，
输出重定向到 `~/Library/Logs/多模型格式转换工具.log`；非零退出码时弹错误框并 `open -R` 日志文件。

安装脚本在与 App 同级、App 内两种位置都能定位源码，在独立虚拟环境里
`pip install "<源码目录>[convert]"`（extras 可选）。

DMG 内容：`多模型格式转换工具.app`、`安装依赖.command`、`使用说明.txt`、`README.md`、`LICENSE` 与指向
`/Applications` 的软链接；`hdiutil ... -fs HFS+ -format UDZO -imagekey zlib-level=9`。

构建产物（`dist/`、`*.dmg`、`icon-preview-512.png`）已被 `.gitignore` 排除。
