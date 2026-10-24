"""HF→ONNX 转换后端（不依赖 Qt，可独立测试）。

用 torch.onnx.export 把 HF 模型导出为 ONNX，按模型类型分别处理：
  - causal    文本生成（input_ids → logits）
  - embedding 基座（input_ids → last_hidden_state；下游自行池化）
  - reranker  序列分类头（input_ids → logits）
  - vlm       多模态（input_ids + pixel_values[+ image_grid_thw 等] → logits），
              通过 inspect 探测 forward 签名自动决定导出的视觉输入集合。
              M-RoPE 模型（Qwen2-VL 系）额外把 position_ids 作为显式输入：
              torch.onnx.export 无法追踪 get_rope_index 内的
              mm_token_type_ids.tolist()+groupby 与 iter(image_grid_thw)，故
              position_ids 不能烤进图；导出期仅用 get_rope_index 算哑样本占位，
              运行时由消费端（见 mrope_position_ids）按真实布局重算后传入。
量化（onnxruntime）：
  - int8  embedding/reranker 用**手写 weight-only int8**（MatMul/Gemm 权重 per-channel
    int8 + DequantizeLinear 还原 fp32 再计算，激活全程 fp32）——onnxruntime 的
    quantize_dynamic 会把激活也量化（embedding 对激活量化敏感，余弦只到 0.985），
    且对 fp16 源不会真正量化权重（已验证于 nomic-embed-text-v1.5）；causal 仍用
    quantize_dynamic（LLM 对激活量化不敏感，且已被验证稳定）
  - int4  best-effort（MatMul4BitsQuantizer；若当前 onnxruntime 未提供则回退 int8 并提示）

另含 OnnxToOnnxBackend：ONNX(fp16) → ONNX(int8 weight-only)，同目录提供
config.json / tokenizer（已验证于 nomic-embed-text-v1.5-onnx-fp16 → -int8）。

注：导出仅用 input_ids（不带 attention_mask），避免模型内部 mask 比较产生的
GreaterOrEqual 等 onnx2torch 暂不支持的算子，从而保证 ONNX→MLX 链路可无损转回权重。
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import shutil
import time
from pathlib import Path

import torch
from typing import Optional
from modconv.converter import ConversionHooks, ConversionResult
from modconv.backend import resolve_sidecar_dir
from modconv.scanner import FileInfo

# protobuf 单条消息上限 2GB：超过后 onnx.save 会抛
# "EncodeError: Failed to serialize proto"，必须改走外部数据（见 _save_onnx）。
# 留 200MB 余量，避免估算误差导致落盘时才失败。
_PROTOBUF_SIZE_LIMIT = (2 << 30) - (200 << 20)

# TensorProto 各 dtype 的每元素字节数（用于估算模型大小，见 _onnx_initializer_bytes）
_TENSOR_DTYPE_BYTES = {
    1: 4,    # FLOAT
    2: 8,    # UINT8 -> 实际 1，见下方修正
    3: 1,    # INT8
    4: 2,    # UINT16
    5: 2,    # INT16
    6: 4,    # INT32
    7: 8,    # INT64
    8: 1,    # STRING（按 1 估，权重不会是 string）
    9: 1,    # BOOL
    10: 2,   # FLOAT16
    11: 8,   # DOUBLE
    12: 4,   # UINT32
    13: 8,   # UINT64
    14: 4,   # COMPLEX64
    15: 8,   # COMPLEX128
    16: 2,   # BFLOAT16
}
_TENSOR_DTYPE_BYTES[2] = 1  # UINT8


def _is_bf16_config(config) -> bool:
    """配置声明的权重 dtype 是否为 bfloat16（含复合配置的 text_config）。"""
    for src in (config, getattr(config, "text_config", None)):
        if src is None:
            continue
        for attr in ("dtype", "torch_dtype"):
            val = getattr(src, attr, None)
            if val is not None and "bfloat16" in str(val):
                return True
    return False


# 折叠权重常量链时支持的算子及其 numpy dtype（Cast 的 to 属性）
_CAST_TO_NUMPY = {
    1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64",
    9: "bool", 10: "float16", 11: "float64", 12: "uint32",
}


def fold_const_weight_nodes(model, max_rounds: int = 3) -> int:
    """把权重上的常量链（``Transpose`` / ``Cast``）折叠进 initializer。

    与 ``fold_transposed_initializers`` 同目的，但覆盖完整链路：
    ``initializer → Transpose → Cast → MatMul``（PyTorch 导出 nn.Linear 的常见形态）。
    只折 Transpose 不够——中间那层 Cast 仍会让量化器判定
    "MatMul doesn't have const weight" 而跳过（实测 Qwen3.5-2B：只折 Transpose 时
    187 个权重里只有 102 个被量化，其余 85 个卡在 Cast 上）。
    """
    import numpy as np
    from onnx import numpy_helper

    total = 0
    for _ in range(max_rounds):
        inits = {i.name: i for i in model.graph.initializer}
        consumers = {}
        for node in model.graph.node:
            for name in node.input:
                consumers[name] = consumers.get(name, 0) + 1

        folded, drop_nodes, drop_inits = [], [], set()
        for node in model.graph.node:
            src = node.input[0] if node.input else None
            if src is None or src not in inits or len(node.input) != 1:
                continue
            try:
                arr = numpy_helper.to_array(inits[src])
                if node.op_type == "Transpose" and node.attribute:
                    perm = tuple(int(v) for v in node.attribute[0].ints)
                    out = np.ascontiguousarray(np.transpose(arr, perm))
                elif node.op_type == "Cast":
                    to = int(node.attribute[0].i)
                    if to not in _CAST_TO_NUMPY:
                        continue  # bf16 等 numpy 无直接对应的类型：保持原节点
                    out = np.ascontiguousarray(arr.astype(_CAST_TO_NUMPY[to]))
                else:
                    continue
                folded.append(numpy_helper.from_array(out, node.output[0]))
            except Exception:
                continue
            drop_nodes.append(node)
            if consumers.get(src, 0) <= 1:
                drop_inits.add(src)

        if not folded:
            break
        keep = [n for n in model.graph.node if n not in drop_nodes]
        keep_i = [i for i in model.graph.initializer if i.name not in drop_inits]
        model.graph.ClearField("node")
        model.graph.node.extend(keep)
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(keep_i)
        model.graph.initializer.extend(folded)
        total += len(folded)
    return total


def fold_transposed_initializers(model) -> int:
    """把 ``Transpose(initializer)`` 折叠成 initializer，让权重量化器能识别常量权重。

    ``torch.onnx.export`` 导出 ``nn.Linear`` 时产出 ``Transpose(W) → MatMul(x, Wᵀ)``。
    本工具为保留 HF 权重名（供 ONNX→MLX 反查）而使用 ``do_constant_folding=False``，
    于是该 Transpose 没被折叠，MatMul 的 B 输入是**动态张量**——onnxruntime 的权重
    量化器（int4 MatMulNBits、int8 dynamic）会逐个判定
    "MatMul doesn't have const weight" 而**跳过全部 MatMul**，产物实际仍是 fp32，
    体积与精度都没变（实测 Qwen3.5-2B：3685 个 MatMul 全部被跳过）。

    折叠后新 initializer 直接沿用原 Transpose 的输出名，消费节点无需改名；原
    initializer 若只剩这一个消费者则一并删除（否则权重会存两份，体积翻倍）。
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    # 统计每个 initializer 被多少个节点消费（含即将删除的 Transpose）
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers[name] = consumers.get(name, 0) + 1

    folded = []
    drop_nodes = []
    drop_inits = set()
    for node in model.graph.node:
        if node.op_type != "Transpose" or len(node.input) != 1:
            continue
        src = node.input[0]
        if src not in inits or not node.attribute:
            continue
        try:
            perm = tuple(int(v) for v in node.attribute[0].ints)
            arr = numpy_helper.to_array(inits[src])
            out = np.ascontiguousarray(np.transpose(arr, perm))
            folded.append(numpy_helper.from_array(out, node.output[0]))
        except Exception:
            continue
        drop_nodes.append(node)
        if consumers.get(src, 0) <= 1:
            drop_inits.add(src)

    if not folded:
        return 0
    keep_nodes = [n for n in model.graph.node if n not in drop_nodes]
    keep_inits = [i for i in model.graph.initializer if i.name not in drop_inits]
    model.graph.ClearField("node")
    model.graph.node.extend(keep_nodes)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(keep_inits)
    model.graph.initializer.extend(folded)
    return len(folded)


def _onnx_initializer_bytes(model) -> int:
    """按 initializer 的 dims×dtype 估算模型字节数。

    不用 ``model.ByteSize()``：它在超过 protobuf 2GB 上限时会直接抛 EncodeError，
    没法用来预判（实测 2B 的 Qwen3.5 就是如此）。
    """
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= int(d)
        total += n * _TENSOR_DTYPE_BYTES.get(init.data_type, 4)
    return total


class HfToOnnxBackend:
    """把 HF 模型导出为 ONNX，可选 int8 / int4 量化。

    quant_mode: "fp16"（不量化，权重存 fp16）/ "int8" / "int4"。
    model_kind: "auto" 自动判定 / "causal" / "embedding" / "reranker"。
    """

    _TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.txt",
                        "vocab.json", "merges.txt",
                        # 新版 HF 把对话模板单独放在 chat_template.jinja；多模态还需要
                        # processor_config.json（图像预处理）。缺了它们产物不能直接用。
                        "chat_template.jinja", "processor_config.json")

    def __init__(self, source: Path, output_dir: Path, *,
                 quant_mode: str = "fp16", model_kind: str = "auto",
                 opset: int = 17, output_name: Optional[str] = None):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode
        self.model_kind = model_kind
        self.opset = opset
        # 产物文件名。MLX/GGUF→ONNX 走的是"归一化成临时 HF 目录"的路径，若按
        # source.name 命名会得到一个临时目录名（如 modconv_mlx2hf_xxxx.onnx）。
        self.output_name = output_name

    @staticmethod
    def _import_hf():
        if (importlib.util.find_spec("torch") is None
                or importlib.util.find_spec("transformers") is None
                or importlib.util.find_spec("onnx") is None
                or importlib.util.find_spec("onnxruntime") is None):
            raise RuntimeError(
                "HF→ONNX 需要 torch / transformers / onnx / onnxruntime，请先安装")
        import transformers
        return transformers

    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            transformers = self._import_hf()
            return self._run(transformers, hooks, finish)
        except Exception as exc:
            msg = f"HF→ONNX 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)

    def _resolve_kind(self, config) -> str:
        if self.model_kind != "auto":
            return self.model_kind
        archs = " ".join(list(getattr(config, "architectures", []) or []))
        mt = getattr(config, "model_type", "") or ""
        if self._is_vlm(config, archs, mt):
            return "vlm"
        if "ForCausalLM" in archs or mt in {
                "llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "phi3",
                "qwen3_5"}:
            return "causal"
        if "ForSequenceClassification" in archs or getattr(config, "num_labels", 0):
            return "reranker"
        return "embedding"

    @staticmethod
    def _is_vlm(config, archs: str, mt: str) -> bool:
        if any(k in archs for k in ("ConditionalGeneration", "ImageTextToText",
                                    "VisionTextDualEncoder", "ForVision2Seq",
                                    "VlmForConditionalGeneration")):
            return True
        if mt in _VLM_MODEL_TYPES:
            return True
        # config 带 vision_config 且非纯文本生成模型（如某些图文 dual-encoder）
        if hasattr(config, "vision_config") and "ForCausalLM" not in archs:
            return True
        return False

    def _run(self, transformers, hooks, finish):
        cfg = transformers.AutoConfig.from_pretrained(str(self.source))
        kind = self._resolve_kind(cfg)
        hooks.on_log(f"检测到模型类型: {kind}（model_type={getattr(cfg, 'model_type', '?')}）")

        if kind == "vlm":
            model, wrapper, output_name, arg_names, dummies, fixed_seq = self._prepare_vlm(
                transformers, cfg)
        elif kind == "causal":
            model = transformers.AutoModelForCausalLM.from_pretrained(str(self.source))
            wrapper, arg_names, dummies = (
                _CausalWrap(model), ["input_ids"],
                {"input_ids": torch.ones(1, 8, dtype=torch.long)})
            output_name = "logits"
        elif kind == "reranker":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self.source))
            wrapper, arg_names, dummies = (
                _RankWrap(model), ["input_ids"],
                {"input_ids": torch.ones(1, 8, dtype=torch.long)})
            output_name = "logits"
        else:
            model = transformers.AutoModel.from_pretrained(str(self.source))
            wrapper, arg_names, dummies = (
                _EmbWrap(model), ["input_ids"],
                {"input_ids": torch.ones(1, 8, dtype=torch.long)})
            output_name = "last_hidden_state"
        model.eval()

        fp16 = (self.quant_mode == "fp16")
        # 源权重是 bfloat16 时**必须**转 fp16 再导出：onnxruntime 不支持 bf16 的
        # Conv 等算子，按原 dtype 导出的图加载即失败
        # （InvalidGraph: Type 'tensor(bfloat16)' of input parameter ... of Conv）。
        # Qwen3.5/Qwen3 等权重默认就是 bf16，而 bf16 只在 fp16 档位下才被转走，
        # 量化档位原本会直接导出 bf16 → 产物无法加载。
        if _is_bf16_config(cfg):
            hooks.on_log("检测到 bfloat16 权重，导出前转为 fp16"
                         "（onnxruntime 不支持 bf16 算子）")
            fp16 = True
        if fp16:
            model.half()
            if kind != "vlm":
                wrapper = wrapper.half()
            else:
                # VLM 包装器本身无参数，改为把浮点哑输入降到 fp16（与半精度模型匹配）
                dummies = {k: (v.half() if v.is_floating_point() else v)
                           for k, v in dummies.items()}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_name or self.source.name
        onnx_path = self.output_dir / f"{stem}.onnx"

        hooks.on_log(f"正在导出 ONNX（fp16={fp16}, opset={self.opset}）…")
        dynamic_axes = {output_name: {0: "batch", 1: "seq"}}
        if kind == "vlm":
            for n in arg_names:
                if n == "position_ids":
                    # M-RoPE 的 position_ids 形状是 (3, B, S)：第 0 维是 t/h/w 三个
                    # 分量，恒为 3。若与其它输入一样把第 0 维声明成 batch 动态维，
                    # ORT 会把它报成 (position_ids_batch, 1, S)，消费端极易误传
                    # (B, 1, S) 而算出错误结果——故第 0 维保持静态。
                    axes = {1: f"{n}_batch"}
                    if not fixed_seq:
                        axes[2] = f"{n}_seq"
                    dynamic_axes[n] = axes
                elif fixed_seq:
                    # 定长导出（qwen3_5 等 chunk 线性注意力）：仅 batch 维动态，seq 维
                    # 固定为导出长度（chunk 顺序扫描已展开为该长度），运行时消费端右
                    # 填充到该长度。
                    dynamic_axes[n] = {0: f"{n}_batch"}
                else:
                    dynamic_axes[n] = {i: f"{n}_{i}"
                                       for i in range(getattr(dummies[n], "ndim", 0))}
            if fixed_seq:
                dynamic_axes[output_name] = {0: "batch"}
        else:
            dynamic_axes["input_ids"] = {0: "batch", 1: "seq"}
        # 仅传各模型实际需要的输入（causal/embedding/reranker 仅 input_ids；vlm 含
        # pixel_values 等视觉输入）。不传 attention_mask，避免其比较算子触发 onnx2torch
        # 不支持的 op。do_constant_folding=False：保留原始权重名，便于 ONNX→MLX 映射。
        with torch.no_grad(), _patch_transformers_for_onnx_export(), \
                _patch_causal_conv1d_for_onnx_export(), _patch_mrope_cache(model):
            torch.onnx.export(
                wrapper, tuple(dummies[a] for a in arg_names), str(onnx_path),
                input_names=list(arg_names), output_names=[output_name],
                dynamic_axes=dynamic_axes, opset_version=self.opset,
                do_constant_folding=False)

        # 大模型会被 torch 拆成"每个张量一个文件"（几百个散文件），统一合并成
        # <name>.onnx + <name>.onnx.data 两个文件，避免产物目录散乱。
        self._consolidate_external_data(onnx_path, hooks)

        if self.quant_mode == "int8":
            onnx_path = (self._quantize_int8_weight_only(onnx_path, hooks)
                         if kind in ("embedding", "reranker")
                         else self._quantize_int8(onnx_path, hooks))
        elif self.quant_mode == "int4":
            onnx_path = self._quantize_int4(onnx_path, hooks)

        copied = 0
        # config.json 一并拷贝：使 ONNX 产物自包含，可直接作为 ONNX→GGUF / ONNX→MLX 的源
        cfg_src = self.source / "config.json"
        if cfg_src.is_file():
            shutil.copy(cfg_src, self.output_dir / "config.json")
            copied += 1
        for fname in self._TOKENIZER_FILES:
            src = self.source / fname
            if src.is_file():
                shutil.copy(src, self.output_dir / fname)
                copied += 1
        if copied:
            hooks.on_log(f"已拷贝 {copied} 个 config/tokenizer 文件")

        total = sum(p.stat().st_size for p in self.output_dir.glob("*"))
        msg = f"HF→ONNX 完成: {onnx_path.name}（{kind}, {self.quant_mode}）"
        hooks.on_log(msg)
        return finish(True, False, msg, onnx_path, 1, total)

    def _prepare_vlm(self, transformers, config):
        """加载多模态模型并构造导出包装与哑输入。

        优先用 AutoModelForImageTextToText（transformers>=4.45 统一多模态入口），
        失败则回退 AutoModelForCausalLM（多数 VLM 是其子类，可正常返回 logits）。
        """
        AutoVlm = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoVlm is not None:
            try:
                model = AutoVlm.from_pretrained(str(self.source))
            except Exception:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    str(self.source))
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(self.source))
        arg_names, dummies, fixed_seq = _vlm_export_spec(model, config)
        wrapper = _VlmWrap(model, arg_names)
        return model, wrapper, "logits", arg_names, dummies, fixed_seq

    @staticmethod
    def _save_onnx(model, path: Path, hooks) -> None:
        """保存 ONNX；超过 protobuf 2GB 上限时改用外部数据。

        ``onnx.save`` 默认把权重内联进单个 protobuf，而 protobuf 单条消息上限是 2GB：
        大模型落盘会抛 ``EncodeError: Failed to serialize proto``。注意**不能**用
        ``model.ByteSize()`` 预判——它同样会触碰该上限而直接抛异常（实测 2B 模型），
        故按 initializer 的 dims×dtype 估算；估算偏小导致内联失败时再兜底重试。
        """
        import onnx

        def save_external():
            hooks.on_log("模型超过 protobuf 2GB 上限，改用外部数据保存"
                         "（.onnx + .onnx.data）")
            onnx.save(model, str(path), save_as_external_data=True,
                      all_tensors_to_one_file=True,
                      location=path.name + ".data")

        if _onnx_initializer_bytes(model) > _PROTOBUF_SIZE_LIMIT:
            save_external()
            return
        try:
            onnx.save(model, str(path))
        except Exception:
            save_external()

    @staticmethod
    def _consolidate_external_data(onnx_path: Path, hooks) -> None:
        """把 torch 导出的"每个张量一个文件"外部数据合并为单个 ``.onnx.data``。

        ``torch.onnx.export`` 对超过 protobuf 2GB 的模型默认把每个 initializer 写成
        独立文件、文件名即张量名（2B 模型实测 404 个），产物目录非常散乱且容易漏拷。
        ONNX 只规定"超限必须外部化"，并不要求一个张量一个文件，故导出后统一合并为
        ``<name>.onnx`` + ``<name>.onnx.data`` 两个文件。

        代价是要把模型整体读进内存一次（大模型约等于一份权重），失败时保留原分散
        布局并告警，不会让转换失败。
        """
        import onnx

        def _locations(model):
            """图里所有外部数据文件的相对路径名（initializer 与 Constant 属性）。"""
            names = set()
            for init in model.graph.initializer:
                if init.data_location == onnx.TensorProto.EXTERNAL:
                    names |= _external_location(init)
            for node in model.graph.node:
                if node.op_type != "Constant":
                    continue
                for attr in node.attribute:
                    if attr.t.data_location == onnx.TensorProto.EXTERNAL:
                        names |= _external_location(attr.t)
            return {n for n in names if n}

        def _external_location(tensor):
            out = set()
            for entry in tensor.external_data:
                if entry.key != "location":
                    continue
                val = entry.value
                out.add(val.decode() if isinstance(val, bytes) else str(val))
            return out

        skeleton = onnx.load(str(onnx_path), load_external_data=False)
        old = _locations(skeleton)
        if not old:
            return  # 本来就内联在单个 .onnx 里
        del skeleton
        hooks.on_log(f"合并 {len(old)} 个外部权重文件为单个 .onnx.data"
                     "（需把模型读入内存一次，大模型较慢）…")
        location = onnx_path.name + ".data"
        try:
            full = onnx.load(str(onnx_path))   # 读回全部外部数据
            onnx.save(full, str(onnx_path), save_as_external_data=True,
                      all_tensors_to_one_file=True, location=location,
                      convert_attribute=True)
            del full
        except Exception as exc:
            hooks.on_log(f"[warn] 合并外部数据失败，保留原分散布局: {exc}")
            return
        removed = 0
        for name in old:
            try:
                p = onnx_path.parent / name
                if p.is_file():
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        hooks.on_log(f"已合并外部权重：{onnx_path.name} + {location}"
                     f"（删除 {removed} 个散文件）")

    def _load_for_quantization(self, onnx_path: Path, hooks):
        """读入待量化的 ONNX 并折叠常量（见 fold_transposed_initializers）。"""
        import onnx
        model = onnx.load(str(onnx_path))
        n = fold_const_weight_nodes(model)
        if n:
            hooks.on_log(f"已折叠 {n} 个权重常量链（Transpose/Cast），使量化器能"
                         "识别常量权重（do_constant_folding=False 会保留这些节点）")
        return model

    def _quantize_int8(self, onnx_path: Path, hooks) -> Path:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qpath = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        model = self._load_for_quantization(onnx_path, hooks)
        quantize_dynamic(model, str(qpath),
                         weight_type=QuantType.QInt8, per_channel=True,
                         use_external_data_format=True)
        hooks.on_log(f"已做 int8 动态量化: {qpath.name}")
        return qpath

    def _quantize_int8_weight_only(self, onnx_path: Path, hooks) -> Path:
        """手写 weight-only int8（复刻 tools/convert_onnx_fp16_to_int8.py，已验证）。

        只量化 MatMul/Gemm 权重（per-channel int8），插入 DequantizeLinear 还原
        fp32 再计算，激活全程 fp32——embedding 对激活量化敏感，动态量化会掉精度；
        且 fp16 源权重须先整体转 fp32（初始化器/Constant/类型声明/Cast 的 to 全要改）。
        """
        import onnx
        qpath = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        model = self._load_for_quantization(onnx_path, hooks)
        if any(i.data_type == onnx.TensorProto.FLOAT16 for i in model.graph.initializer):
            hooks.on_log("检测到 fp16 权重，先整体转 fp32 再做 weight-only int8")
            model = fp16_to_fp32(model)
        model = weight_only_quantize(model, per_channel=True)
        self._save_onnx(model, qpath, hooks)
        hooks.on_log(f"已做 weight-only int8 量化（激活保持 fp32）: {qpath.name}")
        return qpath

    def _quantize_int4(self, onnx_path: Path, hooks) -> Path:
        """int4 权重-only 量化（MatMulNBits）。

        onnxruntime 的模块名随版本变过：1.22~1.30 是
        ``matmul_4bits_quantizer.MatMul4BitsQuantizer``，1.31 起改为
        ``matmul_nbits_quantizer.MatMulNBitsQuantizer``（且依赖 onnx_ir）。只认旧名会
        在较新的 onnxruntime 上"int4 不可用 → 静默回退 int8"，用户拿到的并不是 int4。
        """
        import onnx
        quantizer = None
        label = ""
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import (
                MatMulNBitsQuantizer)
            quantizer, label = MatMulNBitsQuantizer, "MatMulNBitsQuantizer"
        except Exception:
            try:
                from onnxruntime.quantization.matmul_4bits_quantizer import (
                    MatMul4BitsQuantizer)
                quantizer, label = MatMul4BitsQuantizer, "MatMul4BitsQuantizer"
            except Exception as exc:
                hooks.on_log(
                    f"int4 量化不可用（{exc}；onnxruntime≥1.31 需安装 onnx_ir），回退 int8")
                return self._quantize_int8(onnx_path, hooks)
        model = self._load_for_quantization(onnx_path, hooks)
        try:
            quant = quantizer(model, bits=4, block_size=32, accuracy_level=3)
        except TypeError:
            # 旧版签名：MatMul4BitsQuantizer(model, block_size=..., is_weight_int4=...)
            quant = quantizer(model, block_size=32, is_weight_int4=True,
                              accuracy_level=3)
        quant.process()
        # onnxruntime 1.31 起 quant.model 是 ONNXModel 包装（真正的 proto 在其 .model
        # 属性里），旧版直接是 ModelProto——两种都要兼容。
        result = getattr(quant, "model", None)
        model = result if isinstance(result, onnx.ModelProto) \
            else getattr(result, "model", result)
        if not isinstance(model, onnx.ModelProto):
            raise RuntimeError(f"{label} 未产出有效的 ONNX 模型")
        # 量化器对不识别的图会逐个跳过而不报错（产物仍是浮点），必须显式核对
        ops = [n.op_type for n in model.graph.node]
        done = sum(1 for o in ops if o == "MatMulNBits")
        left = sum(1 for o in ops if o == "MatMul")
        if done == 0:
            hooks.on_log(f"[warn] 没有任何 MatMul 被量化（仍有 {left} 个 MatMul），"
                         "产物并非 int4")
        else:
            hooks.on_log(f"int4 量化：{done} 个 MatMulNBits，剩余 {left} 个 MatMul")
        qpath = onnx_path.with_name(onnx_path.stem + "_int4.onnx")
        self._save_onnx(model, qpath, hooks)
        hooks.on_log(f"已用 {label} 做 int4 量化: {qpath.name}")
        return qpath


@contextlib.contextmanager
def _patch_transformers_for_onnx_export():
    """导出期间屏蔽 transformers 的 packed-sequence 检测（torch.diff）。

    transformers 5.x 的 ``masking_utils.find_packed_sequence_indices`` 用
    ``torch.diff`` 从 position_ids 推导序列边界，而 ``torch.diff`` 无对应 ONNX
    算子——opset 17 导出 causal 模型会直接报
    ``UnsupportedOperatorError: aten::diff``（eager/sdpa 均触发）。

    本工具导出只传 ``input_ids``（无 attention_mask、无 packed sequence），该检测
    结果恒为 None，故导出期间屏蔽；已验证屏蔽后 ONNX 与 HF 前向余弦为 1.0。
    """
    mu = orig = None
    try:
        import transformers.masking_utils as _mu
        mu, orig = _mu, _mu.find_packed_sequence_indices
        _mu.find_packed_sequence_indices = lambda position_ids: None
    except Exception:
        mu = None
    try:
        yield
    finally:
        if mu is not None and orig is not None:
            mu.find_packed_sequence_indices = orig


@contextlib.contextmanager
def _patch_causal_conv1d_for_onnx_export():
    """让混合线性注意力（Qwen3.5 等）在 ONNX 追踪下走可 trace 的纯 PyTorch 路径。

    两处干扰：
    1. ``causal_conv1d_fn`` 若装了 CUDA 内核包会走内核路径，在 CPU 追踪下 padding
       计算损坏（``padding=[]``）。这里替换为等价纯 torch 实现。
    2. ``torch_chunk_gated_delta_rule`` 调用 ``torch.linalg.solve_triangular``
       （ONNX opset17 不支持），而 transformers 自带的 ``is_torchdynamo_exporting``
       回退分支（前向代入）与 ``linalg_solve_triangular`` 数学不等价（实测 logits
       余弦仅 ~0.94）。这里把 ``torch.linalg.solve_triangular`` 替换成手写的可 trace
       单位下三角前向代入求解（与 ``solve_triangular`` 数值完全一致，已验证 cosine=1.0）。
       改动仅作用于导出期间，不影响全局 torch。
    """
    mod = None
    orig_conv = None
    orig_solve = None
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod  # noqa
        from transformers.activations import ACT2FN as _ACT2FN
    except Exception:
        yield
        return
    orig_conv = getattr(mod, "causal_conv1d_fn", None)

    def _pure(hidden_states, weight, bias=None, activation=None, **kwargs):
        import torch.nn.functional as _F
        _, hidden_size, seq_len = hidden_states.shape
        padding = int(weight.shape[-1]) - 1
        out = _F.conv1d(
            hidden_states.to(weight.dtype),
            weight=weight.unsqueeze(1),
            bias=bias,
            padding=padding,
            groups=hidden_size,
        )[:, :, :seq_len]
        if activation is not None:
            out = _ACT2FN[activation](out)
        return out.to(hidden_states.dtype)

    def _solve_unit_lower(a, b, *, upper=False, unitriangular=False, **_):
        # 与 torch.linalg.solve_triangular(upper=False, unitriangular=True) 等价，
        # 但用常量展开的前向代入实现（ONNX 可 trace）。a: [..., n, n] 单位下三角。
        n = a.shape[-1]
        parts = [b[..., 0:1, :]]
        for i in range(1, n):
            ai = a[..., i:i + 1, :i]
            acc = torch.matmul(ai, torch.cat(parts, dim=-2))
            parts.append(b[..., i:i + 1, :] - acc)
        return torch.cat(parts, dim=-2)

    try:
        import torch as _torch
        orig_solve = _torch.linalg.solve_triangular
        mod.causal_conv1d_fn = _pure
        _torch.linalg.solve_triangular = _solve_unit_lower
        yield
    finally:
        if orig_conv is not None:
            mod.causal_conv1d_fn = orig_conv
        if orig_solve is not None:
            import torch as _torch
            _torch.linalg.solve_triangular = orig_solve


@contextlib.contextmanager
def _patch_mrope_cache(model):
    """Qwen2-VL 系用 ``self.rope_deltas`` 缓存 M-RoPE 位置；``torch.onnx.export`` 导出时会
    多次调用 forward，第二次 ``self.rope_deltas`` 已非 None 会命中缓存分支并跳过
    ``mm_token_type_ids``，导致该输入被剪枝、M-RoPE 计算错误。这里强制每次重算，
    确保 ``mm_token_type_ids`` 进入计算图。无该方法的模型（纯文本/causal）自动跳过。
    """
    cpi = getattr(type(model), "compute_3d_position_ids", None)
    orig = getattr(model, "compute_3d_position_ids", None)
    if cpi is None or orig is None:
        yield
        return

    def _cpi(self, *a, **k):
        self.rope_deltas = None
        return cpi(self, *a, **k)

    model.compute_3d_position_ids = _cpi.__get__(model)
    try:
        yield
    finally:
        model.compute_3d_position_ids = orig


class _CausalWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids=input_ids, use_cache=False).logits


class _EmbWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        # 返回 last_hidden_state（不在图内做均值池化）：下游按需自行池化。
        return self.m(input_ids=input_ids).last_hidden_state


class _RankWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids=input_ids).logits


# --------------------------------------------------------------------------- #
# VLM（多模态）导出：文本 + 视觉塔联合导出
# --------------------------------------------------------------------------- #
# 已知多模态模型 model_type（识别用）。也可用 architectures 含 ConditionalGeneration
# / ImageTextToText / VisionTextDualEncoder 等判定。
_VLM_MODEL_TYPES = {
    "llava", "llava_llama", "llava_mistral", "llava_next", "llava_next_interleave",
    "qwen2_vl", "qwen2.5_vl", "qwen3_vl", "qwen_vl",
    "gemma3", "idefics3", "smolvlm", "mllama", "pixtral", "mistral3",
    "minicpm-v", "minicpm_v", "internvl_chat", "phi3_v", "ovis", "lighton",
    "molmo", "glm4v", "deepseek_vl", "florence2", "florence",
    "qwen3_5", "qwen3.5", "qwen3_5_vl",
}

# forward 中可能参与导出的输入候选（顺序无关，按模型实际签名筛选）
_VLM_INPUT_CANDIDATES = (
    "input_ids", "attention_mask", "pixel_values", "pixel_attention_mask",
    "image_grid_thw", "pixel_values_videos", "video_grid_thw",
    "aspect_ratio_ids", "aspect_ratio_mask", "mm_token_type_ids",
)

# M-RoPE（多模态旋转位置编码）模型族：position_ids 由 get_rope_index 用非 trace 的
# mm_token_type_ids.tolist()+itertools.groupby 与 iter(image_grid_thw) 计算，无法被
# torch.onnx.export 追踪（会被烤成常量），故导出时把 position_ids 作为显式输入，运行时
# 由消费端按真实布局重新计算后传入。
_MROPE_FAMILIES = {
    "qwen2_vl", "qwen2.5_vl", "qwen3_vl", "qwen3_5",
    "qwen3.5", "qwen3_5_vl", "qwen2-vl",
}


class _VlmWrap(torch.nn.Module):
    """把任意 VLM 的 forward 包成固定签名（按 arg_names 顺序传入），输出 logits。

    视觉塔与文本塔一起导出，下游 ONNX→MLX / ONNX→GGUF 才能拿到完整权重。
    """

    def __init__(self, m, arg_names):
        super().__init__()
        self.m = m
        self._arg_names = list(arg_names)

    def forward(self, *inputs):
        kwargs = dict(zip(self._arg_names, inputs))
        out = self.m(**kwargs)
        return getattr(out, "logits", out)


def _vlm_grid_thw(config, mt, video=False):
    """Qwen2-VL 系以 merge_size(=2) 为单位的 grid；其它模型不强制。"""
    if mt in ("qwen2_vl", "qwen2.5_vl", "qwen3_vl", "qwen3_5"):
        g = 4  # 4x4（merge 单位）=> 8x8 patch
        return torch.tensor([[1, g, g]], dtype=torch.long)
    return torch.zeros(1, 3, dtype=torch.long)


def _vlm_pixel_dummy(config, mt, video=False):
    """构造 pixel_values 哑输入。Qwen2-VL 为扁平 (N, 3*tps*ps*ps) 且 N 须与 grid_thw 一致；
    其余标准视觉塔为 (N, 3, image_size, image_size)。"""
    if mt in ("qwen2_vl", "qwen2.5_vl", "qwen3_vl", "qwen3_5"):
        g = 4
        n = 1 * g * g  # t*h*w（单图）
        vc = getattr(config, "vision_config", None)
        tps = getattr(vc, "temporal_patch_size", 2) if vc else 2
        ps = getattr(vc, "patch_size", 14) if vc else 14
        feat = 3 * tps * ps * ps
        return torch.zeros(n, feat, dtype=torch.float32)
    vc = getattr(config, "vision_config", None)
    img = getattr(vc, "image_size", 224) if vc else 224
    return torch.zeros(1, 3, img, img, dtype=torch.float32)


def _vlm_pixel_mask_dummy(config, mt):
    vc = getattr(config, "vision_config", None)
    img = getattr(vc, "image_size", 224) if vc else 224
    return torch.ones(1, img, img, dtype=torch.long)


def _vlm_export_spec(model, config):
    """探测模型 forward 接受的输入，构造导出用的哑输入。

    返回 (arg_names, dummies)。只纳入候选集内的输入，避免传入模型不接受的参数导致
    transformers 内部断言失败；位置无关，按固定顺序打包。
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (ValueError, TypeError):
        params = {}
    # 仅导出图文（排除视频输入）：避免 Qwen2-VL 系对 video token 的额外断言，
    # 且图文 ONNX 已覆盖文本+视觉塔的完整权重。后续需要视频时可单独扩展。
    _SKIP = ("pixel_values_videos", "video_grid_thw")
    mt = getattr(config, "model_type", "") or ""
    accepted = [n for n in _VLM_INPUT_CANDIDATES if n in params and n not in _SKIP]
    if "input_ids" not in accepted:
        accepted.insert(0, "input_ids")
    # M-RoPE 模型（Qwen2-VL 系）：position_ids 由 get_rope_index 用非 trace 的
    # mm_token_type_ids.tolist()+itertools.groupby 与 iter(image_grid_thw) 计算，无法被
    # torch.onnx.export 追踪，会被烤成常量 → 只有与哑输入布局一致的输入才正确。故改为把
    # position_ids 作为显式 ONNX 输入：导出期用模型自带 get_rope_index 仅算出哑样本的
    # position_ids 作形状占位，运行时由消费端按真实布局重新计算后传入，使 ONNX 对任意图文
    # 布局都正确。mm_token_type_ids 因此不再是图输入（position_ids 已提供，模型会跳过其内部
    # 的 compute_3d_position_ids）。
    if mt in _MROPE_FAMILIES:
        if "mm_token_type_ids" in accepted:
            accepted.remove("mm_token_type_ids")
        if "position_ids" not in accepted:
            accepted.append("position_ids")
    else:
        # 非 M-RoPE 的 qwen2-vl 系 forward 被 @force_accelerate_hooks 包装，inspect.signature
        # 暴露为 (*args, **kwargs) 导致 mm_token_type_ids 被签名探查过滤；但它是必需输入（缺省
        # 会 ValueError），故按模型族强制补回（仅当未走 position_ids 路径时）。
        if (mt in ("qwen2_vl", "qwen2.5_vl", "qwen3_vl")
                and "mm_token_type_ids" in _VLM_INPUT_CANDIDATES
                and "mm_token_type_ids" not in accepted):
            accepted.append("mm_token_type_ids")
    # qwen3_5 等混合线性注意力模型用 chunk 顺序扫描（跨 chunk 传递递归状态），
    # 旧版 torch.onnx.export 无法表达动态循环，故把序列长度固定为 chunk_size(64) 的倍数，
    # 导出时该扫描被展开为固定 chunk 数；运行时消费端把输入右填充到该长度即可
    # （右侧 pad 对因果注意力/因果递归安全，不影响前置真实 token 输出）。
    fixed_seq = mt in ("qwen3_5",)
    dummies = {}
    seq = 1024 if fixed_seq else 8
    # 图像占位 token 数：须与 pixel_values 经视觉塔后的视觉 token 数一致，否则
    # Qwen2-VL 系 forward 内的 get_placeholder_mask 断言失败
    # （n_image_tokens * hidden == image_features.numel()）。
    n_image_tokens = _vlm_n_image_tokens(config, mt)
    image_token_id = _vlm_image_token_id(config)

    # 构造 input_ids / attention_mask / mm_token_type_ids 哑样本（供 M-RoPE position_ids
    # 计算；position_ids 会在下方循环中覆盖写入 dummies）。
    if n_image_tokens and image_token_id is not None:
        toks = [image_token_id] * n_image_tokens + [1] * max(0, seq - n_image_tokens)
    else:
        toks = [1] * seq
    input_ids_dummy = torch.tensor(toks, dtype=torch.long).unsqueeze(0)
    attn_dummy = torch.ones(1, seq, dtype=torch.long)
    mm_dummy = ((input_ids_dummy == image_token_id).long()
                if image_token_id is not None else None)

    for name in accepted:
        if name == "input_ids":
            dummies[name] = input_ids_dummy
        elif name == "attention_mask":
            dummies[name] = attn_dummy
        elif name == "pixel_values":
            dummies[name] = _vlm_pixel_dummy(config, mt)
        elif name == "pixel_attention_mask":
            dummies[name] = _vlm_pixel_mask_dummy(config, mt)
        elif name == "image_grid_thw":
            dummies[name] = _vlm_grid_thw(config, mt)
        elif name in ("aspect_ratio_ids", "aspect_ratio_mask"):
            dummies[name] = torch.zeros(1, dtype=torch.long)
        elif name == "mm_token_type_ids":
            # 与 input_ids 中图像占位 token 位置对齐：图像 token→1，其余→0
            dummies[name] = mm_dummy if mm_dummy is not None else torch.zeros_like(input_ids_dummy)
        elif name == "position_ids":
            dummies[name] = _compute_mrope_position_ids(
                model, input_ids_dummy, mm_dummy,
                _vlm_grid_thw(config, mt), attn_dummy, mt)
    return accepted, dummies, fixed_seq


def _vlm_n_image_tokens(config, mt):
    """由 image_grid_thw 哑输入推算视觉 token 数（经 spatial_merge 后）。

    与 _vlm_grid_thw 保持一致的 grid（单图 [[1,4,4]]，merge 单位）。
    """
    if mt in ("qwen2_vl", "qwen2.5_vl", "qwen3_vl", "qwen3_5"):
        g = 4
        prod = 1 * g * g
        sm = getattr(getattr(config, "vision_config", None), "spatial_merge_size", 2)
        if not sm:
            sm = 2
        return prod // (sm * sm)
    return 0


def _vlm_image_token_id(config):
    """取图像占位 token id（config 或嵌套 text_config 的 image_token_id）。"""
    tid = getattr(config, "image_token_id", None)
    if tid is None:
        tc = getattr(config, "text_config", None)
        tid = getattr(tc, "image_token_id", None)
    return tid


def _compute_mrope_position_ids(model, input_ids, mm_token_type_ids,
                                image_grid_thw, attention_mask, mt):
    """M-RoPE 3D position_ids（仅作导出期形状占位；运行时须按真实输入重算后传入）。

    torch.onnx.export 无法追踪 get_rope_index 内的 mm_token_type_ids.tolist()+
    itertools.groupby 与 iter(image_grid_thw)，故导出期只调用模型自带 get_rope_index
    算出哑样本的 position_ids 用于确定图输入形状；运行时消费端须用同一函数对真实输入
    计算后作为 position_ids 传入 ONNX。无多模态（mm_token_type_ids=None）时退化为顺序
    position（各维相同）。合成/无 get_rope_index 的模型或任何异常下返回零占位张量
    （仅用于保证导出管线形状，运行值由消费端提供）。
    """
    if mm_token_type_ids is None:
        L = input_ids.shape[1]
        ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(3, -1)
        return ids.unsqueeze(1).to(input_ids.dtype)  # (3, 1, L)
    try:
        sub = getattr(model, "model", model)
        fn = getattr(sub, "get_rope_index", None)
        if fn is None:
            return torch.zeros(3, input_ids.shape[0], input_ids.shape[1],
                               dtype=input_ids.dtype)
        pos, _ = fn(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        return pos
    except Exception:
        return torch.zeros(3, input_ids.shape[0], input_ids.shape[1],
                           dtype=input_ids.dtype)


def mrope_position_ids(model, input_ids, mm_token_type_ids, image_grid_thw,
                       attention_mask=None):
    """消费端工具：用 HF 模型自带 get_rope_index 计算真实输入的 M-RoPE position_ids。

    导出后的 ONNX 依赖调用方提供 position_ids 输入（见 _compute_mrope_position_ids）。
    调用方（onnxruntime / Swift 等）应在运行前对真实 input_ids + mm_token_type_ids +
    image_grid_thw 调用本函数，再把返回的 (3, B, S) 张量作为 position_ids 喂给 ONNX。
    """
    return _compute_mrope_position_ids(
        model, input_ids, mm_token_type_ids, image_grid_thw, attention_mask, None)


# --------------------------------------------------------------------------- #
# weight-only int8（复刻 tools/convert_onnx_fp16_to_int8.py，nomic-embed 已验证）
# --------------------------------------------------------------------------- #
def fp16_to_fp32(model):
    """把 fp16 ONNX 整体转 fp32：初始化器 / Constant / 类型声明 / Cast 的 to 都要改。

    （只改张量声明会破坏图内 Cast 节点；onnxruntime 的 quant_pre_process
    不会真正转换 fp16 权重。）
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))

    def fix(tp):
        if tp.elem_type == TensorProto.FLOAT16:
            tp.elem_type = TensorProto.FLOAT

    for i in model.graph.input:
        fix(i.type.tensor_type)
    for o in model.graph.output:
        fix(o.type.tensor_type)
    for v in model.graph.value_info:
        fix(v.type.tensor_type)
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    return model


def weight_only_quantize(model, per_channel: bool = True):
    """weight-only int8：只把 MatMul/Gemm 权重存为 int8（per-channel），
    插入 DequantizeLinear 还原 fp32 再计算；激活保持 fp32。

    量化轴 = 权重输出维：MatMul 输出在最后一维（axis=1）；Gemm transB=1 时
    输出在 dim0（axis=0）。"""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replaced, new_inits = {}, []

    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        w_name = next((i for i in n.input if i in inits
                       and inits[i].data_type == onnx.TensorProto.FLOAT), None)
        if w_name is None:
            continue
        W = numpy_helper.to_array(inits[w_name]).astype(np.float32)
        if W.ndim != 2:
            continue
        if n.op_type == "MatMul":
            axis = 1
        else:
            trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
            axis = 0 if trans_b == 1 else 1
        # 沿 axis 维做 per-channel：scale/zp 的形状必须等于 W.shape[axis]，
        # 且与 DequantizeLinear 的 axis 一致（否则 onnxruntime 报
        # "x_zero_point must be 1D tensor with size ..."）。
        out_dim = W.shape[axis]

        if per_channel:
            # 对"另一维"取 max → 得到每个 axis 通道一个 scale
            maxabs = np.max(np.abs(W), axis=1 - axis)
            maxabs = np.where(maxabs == 0, 1.0, maxabs)
            scale = (maxabs / 127.0).astype(np.float32)
            q = (np.round(W / scale[None, :]).clip(-127, 127) if axis == 1
                 else np.round(W / scale[:, None]).clip(-127, 127)).astype(np.int8)
        else:
            s = np.array([np.max(np.abs(W)) / 127.0], dtype=np.float32)
            q = np.round(W / s).clip(-127, 127).astype(np.int8)
            scale, axis, out_dim = s, 0, 1
        zp = np.zeros(out_dim, dtype=np.int8)

        wq, sc, zp_n = w_name + "_wq", w_name + "_ws", w_name + "_wz"
        new_inits += [numpy_helper.from_array(q, wq),
                      numpy_helper.from_array(scale, sc),
                      numpy_helper.from_array(zp, zp_n)]
        replaced[n.name] = (wq, sc, zp_n, axis, w_name)

    old_w = {v[4] for v in replaced.values()}
    keep = [i for i in g.initializer if i.name not in old_w]
    g.ClearField("initializer")
    g.initializer.extend(keep)
    g.initializer.extend(new_inits)

    old_nodes = list(g.node)
    g.ClearField("node")
    for n in old_nodes:
        if n.name in replaced:
            wq, sc, zp_n, axis, oldw = replaced[n.name]
            idx = list(n.input).index(oldw)
            deq_out = oldw + "_deq"
            deq = helper.make_node(
                "DequantizeLinear", [wq, sc, zp_n], [deq_out], axis=axis,
                name=n.name + "_deq")
            n.input[idx] = deq_out
            g.node.append(deq)
            g.node.append(n)
        else:
            g.node.append(n)
    onnx.checker.check_model(model)
    return model


class OnnxToOnnxBackend:
    """ONNX(fp16/fp32) → ONNX(int8 weight-only)。

    复刻 tools/convert_onnx_fp16_to_int8.py（nomic-embed-text-v1.5 已验证，
    fp16 源 vs int8 embedding 余弦 >0.997、检索排序正确）。
    config.json / tokenizer 从 ONNX 同目录拷贝。quant_mode="fp16" 时仅原样拷贝。
    """

    _TOKENIZER_FILES = HfToOnnxBackend._TOKENIZER_FILES

    def __init__(self, source: Path, output_dir: Path, *,
                 quant_mode: str = "int8", per_channel: bool = True):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode
        self.per_channel = per_channel

    @staticmethod
    def _import_deps():
        if (importlib.util.find_spec("onnx") is None
                or importlib.util.find_spec("onnxruntime") is None):
            raise RuntimeError("ONNX→ONNX 需要 onnx / onnxruntime，请先安装")
        import onnx
        return onnx

    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            onnx = self._import_deps()
            return self._run(onnx, hooks, finish)
        except Exception as exc:
            msg = f"ONNX→ONNX 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)

    def _run(self, onnx, hooks, finish):
        import shutil
        # 伴随目录：config/tokenizer 可能在 onnx 文件同级或其父目录（Optimum 布局）
        src_dir = resolve_sidecar_dir(self.source)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "model.onnx"
        model = onnx.load(str(self.source))
        has_external = any(i.data_location == onnx.TensorProto.EXTERNAL
                           for i in model.graph.initializer)

        if self.quant_mode == "fp16" and not has_external:
            shutil.copy(self.source, out_path)
            hooks.on_log("fp16 模式：原样拷贝 ONNX")
        else:
            if has_external:
                hooks.on_log("检测到外部数据（.onnx_data），将内联为单文件 ONNX")
            if any(i.data_type == onnx.TensorProto.FLOAT16
                   for i in model.graph.initializer):
                hooks.on_log("检测到 fp16 权重，先整体转 fp32 再做 weight-only int8")
                model = fp16_to_fp32(model)
            if self.quant_mode != "fp16":
                model = weight_only_quantize(model, per_channel=self.per_channel)
                hooks.on_log(f"已做 weight-only int8 量化: {out_path.name}")
            onnx.save(model, str(out_path))

        # config.json 补 quantization 说明 + 拷 tokenizer
        cfg_path = src_dir / "config.json"
        cfg = {}
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            shutil.copy(cfg_path, self.output_dir / "config.json")
        if self.quant_mode != "fp16":
            cfg["quantization"] = {
                "backend": "weight-only", "weight_type": "int8",
                "per_channel": bool(self.per_channel),
                "op_types_to_quantize": ["MatMul", "Gemm"]}
            (self.output_dir / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        copied = 0
        for fname in self._TOKENIZER_FILES:
            sp = src_dir / fname
            if sp.is_file():
                shutil.copy(sp, self.output_dir / fname)
                copied += 1
        if copied:
            hooks.on_log(f"已拷贝 {copied} 个 tokenizer 文件")

        total = sum(p.stat().st_size for p in self.output_dir.glob("*"))
        msg = f"ONNX→ONNX 完成: {out_path.name}（{self.quant_mode}）"
        hooks.on_log(msg)
        return finish(True, False, msg, out_path, 1, total)


class _NormalizedSourceBackend:
    """把 MLX / GGUF 源归一化为 HF 目录后复用 HfToOnnxBackend 的混入基类。

    子类提供 `_normalize(source, hooks)` 上下文管理器（产出 HF 目录）。
    """

    def __init__(self, source: Path, output_dir: Path, *,
                 quant_mode: str = "fp16", model_kind: str = "auto"):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode
        self.model_kind = model_kind

    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            # 产物名取自**原始源**（MLX 目录名 / GGUF 文件名），而不是归一化出来的
            # 临时 HF 目录名——后者形如 modconv_mlx2hf_xxxx，对用户毫无意义。
            stem = self.source.stem if self.source.is_file() else self.source.name
            with self._normalize(self.source, hooks) as hf_dir:
                hooks.on_log(f"已归一化为 HF 目录: {hf_dir}")
                return HfToOnnxBackend(
                    hf_dir, self.output_dir, quant_mode=self.quant_mode,
                    model_kind=self.model_kind, output_name=stem).run(hooks)
        except Exception as exc:
            msg = f"{self._label}→ONNX 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)


class MlxToOnnxBackend(_NormalizedSourceBackend):
    """MLX 目录 → ONNX（causal / encoder 均可）。

    causal：mlx-lm 产物键名本就是 HF 风格；encoder：MLXEmbedders 命名反向映射回
    bert/roberta 命名。含量化权重时先反量化。
    """
    _label = "MLX"

    @contextlib.contextmanager
    def _normalize(self, source, hooks):
        from modconv.source_normalize import mlx_to_hf_dir_ctx
        with mlx_to_hf_dir_ctx(source, hooks) as hf:
            yield hf


class GgufToOnnxBackend(_NormalizedSourceBackend):
    """GGUF 文件 → ONNX（llama 布局 causal：llama/qwen2/qwen3/mistral）。"""

    _label = "GGUF"

    @contextlib.contextmanager
    def _normalize(self, source, hooks):
        from modconv.source_normalize import gguf_to_hf_dir_ctx
        with gguf_to_hf_dir_ctx(source, hooks) as hf:
            yield hf
