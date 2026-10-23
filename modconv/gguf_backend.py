"""多源 → GGUF 转换后端（用 gguf 库直接写 GGUF）。

支持 Llama 布局的 causal LLM（llama / qwen2 / qwen3 / mistral），从以下三种源转换：
  - HF 目录（config.json + model.safetensors + tokenizer.json）
  - MLX 目录（mlx-lm / mlx-embeddings 产出的 safetensors + config.json；含 int4/int8
    量化权重时自动反量化）
  - ONNX 文件（权重名保留为 HF 风格，同目录需有 config.json + tokenizer.json）

量化：fp16 / int8(Q8_0) / int4(Q4_0)。产出 llama.cpp / llama-cpp-python 可直接加载的
.gguf 文件。

范围说明：仅覆盖 Llama 布局的 BPE causal 模型；MoE、SentencePiece 特有模型、
embedding/reranker 暂不支持（会给出明确报错）。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from modconv.converter import ConversionHooks, ConversionResult
from modconv.backend import resolve_sidecar_dir

# 支持的 HF model_type
_LLAMA_LAYOUT = {"llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen3_5_text",
                 "qwen3_5", "mistral", "mixtral"}

# HF model_type → GGUF general.architecture。llama.cpp 把 Qwen3.5 写作 "qwen35"
# （见实测 GGUF：general.architecture=qwen35，元数据前缀同为 qwen35.*），写成
# qwen3_5 会让它按未知架构解析而加载失败。其余架构名与 HF model_type 一致。
_GGUF_ARCH_NAME = {"qwen3_5": "qwen35", "qwen3_5_text": "qwen35"}

# Qwen3.5 零中心 RMSNorm 的权重后缀（HF 存原始 gain，llama.cpp 的 ggml_rms_norm
# 不加 +1，故写 GGUF 前必须折叠成 1+gain）。GDN 内的 linear_attn.norm 是
# Qwen3_5RMSNormGated，两侧都不加 +1，不在此列。
_FOLDED_NORM_GGUF_NAMES = (
    "output_norm.weight",
    ".attn_norm.weight",
    ".post_attention_norm.weight",
    ".attn_q_norm.weight",
    ".attn_k_norm.weight",
)


def _struct_cfg(cfg: dict) -> dict:
    """取结构字段所在的配置层。

    Qwen3.5 等复合（多模态）config 把 hidden_size / num_hidden_layers / layer_types
    等全部放在 text_config 下，顶层只有 vision 相关字段——直接读顶层会 KeyError。
    """
    out = dict(cfg)
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        for k, v in tc.items():
            out.setdefault(k, v)
    return out


def _strip_hf_prefix(name: str) -> str:
    """去掉 HF 权重名的包装前缀，露出 layers.N.* / embed_tokens.weight 等主体。

    多模态 checkpoint 把文本模型放在 model.language_model.* 之下；mlx-vlm 产物则是
    language_model.model.*。顺序敏感：更具体的前缀先匹配。
    """
    for p in ("model.language_model.", "language_model.model.", "language_model.",
              "model."):
        if name.startswith(p):
            return name[len(p):]
    return name

# ONNX 初始化器反归一化后，仅接受以下 HF 命名（Llama 布局），其余视为图常量丢弃
_ONNX_ACCEPT_RE = re.compile(
    r"^(model\.embed_tokens\.weight|model\.norm\.weight|lm_head\.weight|"
    r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight|"
    r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.(weight|bias)|"
    r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj\.(weight|bias))$"
)


class GgufBackend:
    """把 HF/MLX/ONNX 的 Llama 布局模型转为 GGUF，可选 int4/int8/fp16。

    quant_mode: "fp16" / "int8" / "int4"。
    source 可以是目录（HF/MLX）或单个 .onnx 文件。
    """

    def __init__(self, source: Path, output_dir: Path, *, quant_mode: str = "fp16"):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode

    # ------------------------------------------------------------------
    @staticmethod
    def _add_qwen35_metadata(writer, scfg: dict, arch: str, hooks) -> None:
        """写入 Qwen3.5（混合线性/全注意力）llama.cpp 必需的元数据。

        键名与实测参考 GGUF（general.architecture=qwen35）逐条对齐：缺了
        full_attention_interval / ssm.* 就无法还原混合注意力结构，缺了
        rope.dimension_sections 则 M-RoPE 位置编码错误。
        """
        interval = int(scfg.get("full_attention_interval", 4) or 4)
        writer.add_uint32(f"{arch}.full_attention_interval", interval)
        lin_kh = int(scfg.get("linear_num_key_heads", 16) or 16)
        lin_vh = int(scfg.get("linear_num_value_heads", 16) or 16)
        lin_kd = int(scfg.get("linear_key_head_dim", 128) or 128)
        lin_vd = int(scfg.get("linear_value_head_dim", 128) or 128)
        writer.add_uint32(f"{arch}.ssm.conv_kernel",
                          int(scfg.get("linear_conv_kernel_dim", 4) or 4))
        writer.add_uint32(f"{arch}.ssm.inner_size", lin_vh * lin_vd)
        writer.add_uint32(f"{arch}.ssm.state_size", lin_vd)
        writer.add_uint32(f"{arch}.ssm.group_count", lin_kh)
        writer.add_uint32(f"{arch}.ssm.time_step_rank", lin_vh)
        # M-RoPE 分段：参考 GGUF 为 4 段（末段补 0），llama.cpp 按该长度读取
        sections = None
        rope_params = scfg.get("rope_parameters")
        if isinstance(rope_params, dict):
            sections = rope_params.get("mrope_section")
        sections = sections or scfg.get("mrope_section") or [11, 11, 10]
        try:
            sections = [int(x) for x in sections]
        except (TypeError, ValueError):
            sections = [11, 11, 10]
        sections = (sections + [0] * 4)[:4]
        writer.add_array(f"{arch}.rope.dimension_sections", sections)
        hooks.on_log(f"Qwen3.5 元数据：全注意力间隔 {interval}，"
                     f"线性注意力 {lin_vh}×{lin_vd}（k {lin_kh}×{lin_kd}），"
                     f"mrope {sections}")

    @staticmethod
    def _is_onnx_source(source: Path) -> bool:
        return source.is_file() and source.suffix.lower() == ".onnx"

    @staticmethod
    def _load_safetensors(src_dir: Path, hooks, cfg: dict) -> Dict[str, np.ndarray]:
        """读取 HF/MLX 目录的 safetensors；若含 MLX 量化权重（.scales/.biases 兄弟键）
        则自动反量化为 fp16。"""
        from safetensors import safe_open
        index = src_dir / "model.safetensors.index.json"
        files: List[str]
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            files = ["model.safetensors"]
        tensors: Dict[str, np.ndarray] = {}
        for fname in files:
            fpath = src_dir / fname
            if not fpath.is_file():
                continue
            try:
                with safe_open(str(fpath), framework="numpy") as sf:
                    for k in sf.keys():
                        tensors[k] = sf.get_tensor(k)
            except TypeError:
                # numpy 不支持的类型（如 bfloat16）：回退 torch 读取并转 fp16
                import torch
                with safe_open(str(fpath), framework="torch") as sf:
                    for k in sf.keys():
                        t = sf.get_tensor(k)
                        if t.dtype == torch.bfloat16:
                            t = t.to(torch.float16)
                        tensors[k] = t.detach().cpu().numpy()
            hooks.on_log(f"已读取权重分片: {fname}")
        if not tensors:
            raise RuntimeError("未找到 safetensors 权重文件")

        # MLX 量化权重：weight(uint32) + .scales + .biases → 反量化
        qcfg = cfg.get("quantization") or {}
        if any(k.endswith(".scales") for k in tensors):
            tensors = GgufBackend._dequantize_mlx(tensors, cfg, hooks)
        return tensors

    @staticmethod
    def _dequantize_mlx(tensors: Dict[str, np.ndarray], cfg: dict, hooks) -> Dict[str, np.ndarray]:
        import mlx.core as mx
        from modconv.source_normalize import infer_group_size
        qcfg = cfg.get("quantization") or {}
        bits = int(qcfg.get("bits", 4))
        cfg_gs = int(qcfg.get("group_size", 64))
        mode = qcfg.get("mode", "affine")
        out = dict(tensors)
        for k in list(out):
            if not k.endswith(".scales"):
                continue
            base = k[: -len(".scales")]
            wkey = base + ".weight"
            bkey = base + ".biases"
            if wkey in out:
                # group_size 按打包权重/scales 形状推断（维度不整除时会回退，config 值未必准确）
                group_size = (infer_group_size(np.asarray(out[wkey]).shape,
                                               np.asarray(out[k]).shape, bits) or cfg_gs)
                w = mx.array(out[wkey])
                s = mx.array(out[k])
                b = mx.array(out[bkey]) if bkey in out else None
                # mx.dequantize(w, scales, biases, group_size, bits, mode)
                # 注意：mx.eval() 是就地求值且返回 None，须用 np.array(mx.array) 直转
                dq = mx.dequantize(w, s, b, group_size, bits, mode)
                out[wkey] = np.array(dq).astype(np.float16)
                del out[k]
                if bkey in out:
                    del out[bkey]
                hooks.on_log(f"反量化 MLX 张量: {wkey} (bits={bits}, gs={group_size})")
        return out

    @staticmethod
    def _normalize_onnx_name(name: str) -> str:
        """把 ONNX 初始化器名归一化为 HF 名（去掉 m./module. 等包装前缀）。"""
        n = name.strip()
        while n.startswith(("m.", "module.")):
            n = n[2:]
        if not (n.startswith("model.") or n.startswith("lm_head")):
            idx = n.find("model.")
            jdx = n.find("lm_head")
            if idx != -1:
                n = n[idx:]
            elif jdx != -1:
                n = n[jdx:]
        return n

    @staticmethod
    def _load_onnx_weights(onnx_path: Path, src_dir: Path, hooks) -> Dict[str, np.ndarray]:
        """从 ONNX 文件读取权重，归一化为 HF 命名（仅保留 Llama 布局张量）。"""
        import onnx
        from onnx import numpy_helper
        model = onnx.load(str(onnx_path))
        raw: Dict[str, np.ndarray] = {
            i.name: numpy_helper.to_array(i) for i in model.graph.initializer
        }
        hooks.on_log(f"ONNX 初始化器数量: {len(raw)}")
        kept: Dict[str, np.ndarray] = {}
        skipped = 0
        for oname, arr in raw.items():
            hf = GgufBackend._normalize_onnx_name(oname)
            if not _ONNX_ACCEPT_RE.match(hf):
                skipped += 1
                continue
            kept[hf] = arr
        if skipped:
            hooks.on_log(f"已忽略 {skipped} 个非权重/非 Llama 布局的 ONNX 常量")
        if not kept:
            raise RuntimeError("ONNX 中未发现可识别的 Llama 布局权重（需保留 HF 风格权重名）")
        return kept

    @staticmethod
    def _load_tokenizer(src_dir: Path):
        """从 tokenizer.json 解析 BPE 分词信息。返回 (model_name, tokens, types, merges, scores)。"""
        tj = src_dir / "tokenizer.json"
        if not tj.is_file():
            return None
        data = json.loads(tj.read_text(encoding="utf-8"))
        model = data.get("model", {})
        vocab = model.get("vocab") or {}
        merges = model.get("merges") or []
        added = data.get("added_tokens") or []

        max_id = -1
        tok_by_id: Dict[int, str] = {}
        for tok, tid in vocab.items():
            tok_by_id[int(tid)] = tok
            max_id = max(max_id, int(tid))
        special_ids = set()
        for at in added:
            tid = int(at["id"])
            tok_by_id[tid] = at["content"]
            max_id = max(max_id, tid)
            if at.get("special"):
                special_ids.add(tid)

        tokens = [tok_by_id.get(i, "") for i in range(max_id + 1)]
        # TokenType: NORMAL=1, CONTROL=3, USER_DEFINED=4
        types = [3 if i in special_ids else 1 for i in range(max_id + 1)]
        scores = [0.0] * (max_id + 1)
        # pre_tokenizer 为 ByteLevel → llama.cpp 用 "gpt2" 模型名（byte-level BPE）
        pre = (data.get("pre_tokenizer") or {})
        pre_type = pre.get("type") if isinstance(pre, dict) else None
        model_name = "gpt2" if (pre_type == "ByteLevel" or merges) else "llama"
        return model_name, tokens, types, merges, scores

    # ------------------------------------------------------------------
    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            return self._run(hooks, finish)
        except Exception as exc:
            msg = f"→GGUF 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)

    def _run(self, hooks, finish):
        is_onnx = self._is_onnx_source(self.source)
        # onnx 单文件源：config/tokenizer 可能在同级或其父目录（Optimum 布局）
        src_dir = resolve_sidecar_dir(self.source) if is_onnx else self.source

        cfg_path = src_dir / "config.json"
        if not cfg_path.is_file():
            raise RuntimeError(f"源目录缺少 config.json（{src_dir}）")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        mt = str(cfg.get("model_type", "") or "")
        if mt not in _LLAMA_LAYOUT:
            raise RuntimeError(
                f"GGUF 目前仅支持 Llama 布局（llama/qwen2/qwen3/mistral），"
                f"当前 model_type='{mt}' 暂不支持")

        if is_onnx:
            tensors = self._load_onnx_weights(self.source, src_dir, hooks)
            src_label = "ONNX"
        else:
            tensors = self._load_safetensors(src_dir, hooks, cfg)
            src_label = "MLX" if any(k.endswith(".scales") for k in tensors) else "HF"

        from gguf import GGUFWriter
        from gguf.quants import GGMLQuantizationType, quantize as gguf_quantize

        # 复合（多模态）config 的结构字段在 text_config 下
        scfg = _struct_cfg(cfg)
        n_layers = int(scfg["num_hidden_layers"])
        hidden = int(scfg["hidden_size"])
        n_heads = int(scfg["num_attention_heads"])
        n_kv = int(scfg.get("num_key_value_heads", n_heads))
        inter = int(scfg.get("intermediate_size", 0))
        # rope_theta 在 Qwen3/3.5 上位于 rope_parameters 内（顶层没有），只读顶层会
        # 拿到默认值 10000 而非实际的 1e7——RoPE 频率错误会让输出完全不可用。
        _rp = scfg.get("rope_parameters")
        _rp = _rp if isinstance(_rp, dict) else {}
        rope_theta = float(scfg.get("rope_theta") or _rp.get("rope_theta") or 10000.0)
        rms_eps = float(scfg.get("rms_norm_eps", 1e-5))
        ctx = int(scfg.get("max_position_embeddings", 4096))

        quant_map = {"fp16": GGMLQuantizationType.F16,
                     "int8": GGMLQuantizationType.Q8_0,
                     "int4": GGMLQuantizationType.Q4_0}
        file_type = quant_map[self.quant_mode]
        quantize_enabled = self.quant_mode in ("int8", "int4")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_name = (self.source.stem if is_onnx else self.source.name)
        out_path = self.output_dir / f"{out_name}.gguf"
        # 架构名必须是 llama.cpp 侧的架构名（Qwen3.5 → "qwen35"），否则它按错误布局
        # 解析元数据与张量名而加载失败。
        gguf_arch = _GGUF_ARCH_NAME.get(mt, mt)
        writer = GGUFWriter(str(out_path), gguf_arch)
        is_qwen35 = gguf_arch == "qwen35"

        # ---- 元数据 ----
        # head_dim 优先取 config（Qwen3 等 QKV 维度可与 hidden/n_heads 不一致）
        head_dim = int(scfg.get("head_dim", hidden // n_heads))
        writer.add_context_length(ctx)
        writer.add_embedding_length(hidden)
        writer.add_block_count(n_layers)
        writer.add_feed_forward_length(inter)
        writer.add_head_count(n_heads)
        writer.add_head_count_kv(n_kv)
        # Qwen3.5 用**部分旋转**（partial_rotary_factor=0.25）：rope.dimension_count
        # 是 head_dim*partial（实测参考 GGUF 为 64），而 attention.key/value_length
        # 才是完整 head_dim（256）。
        rope_dim = head_dim
        if is_qwen35:
            partial = float(scfg.get("partial_rotary_factor")
                            or _rp.get("partial_rotary_factor") or 0.25)
            rope_dim = int(round(head_dim * partial))
            writer.add_uint32(f"{gguf_arch}.attention.key_length", head_dim)
            writer.add_uint32(f"{gguf_arch}.attention.value_length", head_dim)
        writer.add_rope_dimension_count(rope_dim)
        # Qwen3 等需要显式 attention.key_length/value_length（head_dim）
        if hasattr(writer, "add_key_length") and not is_qwen35:
            writer.add_key_length(head_dim)
        if hasattr(writer, "add_value_length") and not is_qwen35:
            writer.add_value_length(head_dim)
        writer.add_rope_freq_base(rope_theta)
        writer.add_layer_norm_rms_eps(rms_eps)
        writer.add_file_type(file_type)
        if "tie_word_embeddings" in scfg:
            writer.add_bool(f"{gguf_arch}.tie_word_embeddings",
                            bool(scfg["tie_word_embeddings"]))
        if is_qwen35:
            self._add_qwen35_metadata(writer, scfg, gguf_arch, hooks)

        # ---- 分词器 ----
        tok = self._load_tokenizer(src_dir)
        if tok is None:
            raise RuntimeError("缺少 tokenizer.json，无法导出 GGUF 分词器")
        model_name, tokens, types, merges, scores = tok
        n_vocab = len(tokens)  # GGUF 的 n_vocab 取自 token 列表长度
        writer.add_vocab_size(n_vocab)
        writer.add_tokenizer_model(model_name)
        writer.add_token_list(tokens)
        writer.add_token_types(types)
        writer.add_token_scores(scores)
        if merges:
            # llama.cpp 需要扁平 ARRAY<STRING>（每条 merge 为 "partA partB"）；
            # gguf 0.19 的 add_token_merges 会写成嵌套数组，故这里直接用 add_array。
            from gguf.constants import Keys
            flat = [" ".join(m) if isinstance(m, (list, tuple)) else str(m) for m in merges]
            writer.add_array(Keys.Tokenizer.MERGES, flat)
        writer.add_tokenizer_pre("default")
        for tid_key, tid in (("bos", cfg.get("bos_token_id")),
                             ("eos", cfg.get("eos_token_id")),
                             ("unk", cfg.get("unk_token_id")),
                             ("pad", cfg.get("pad_token_id"))):
            if isinstance(tid, int) and 0 <= tid < n_vocab:
                getattr(writer, f"add_{tid_key}_token_id")(tid)
        hooks.on_log(f"分词器导出完成：{len(tokens)} tokens, {len(merges)} merges")

        # ---- 张量映射（HF/MLX/ONNX 统一：llama → GGUF llama）----
        def gname(hf_name: str) -> Optional[str]:
            name = _strip_hf_prefix(hf_name)
            repl = {
                "embed_tokens.weight": "token_embd.weight",
                "norm.weight": "output_norm.weight",
            }
            if name in repl:
                return repl[name]
            for i in range(n_layers):
                p = f"layers.{i}."
                if name.startswith(p):
                    rest = name[len(p):]
                    m = {
                        "input_layernorm.weight": "attn_norm.weight",
                        # Qwen3.5 用 post_attention_norm（llama.cpp 侧的名字），
                        # 其余架构沿用 llama 的 ffn_norm
                        "post_attention_layernorm.weight":
                            "post_attention_norm.weight" if is_qwen35 else "ffn_norm.weight",
                        "self_attn.q_proj.weight": "attn_q.weight",
                        "self_attn.k_proj.weight": "attn_k.weight",
                        "self_attn.v_proj.weight": "attn_v.weight",
                        "self_attn.o_proj.weight": "attn_output.weight",
                        # Qwen3 等带 QK-Norm
                        "self_attn.q_norm.weight": "attn_q_norm.weight",
                        "self_attn.k_norm.weight": "attn_k_norm.weight",
                        "mlp.gate_proj.weight": "ffn_gate.weight",
                        "mlp.up_proj.weight": "ffn_up.weight",
                        "mlp.down_proj.weight": "ffn_down.weight",
                        # Qwen2 等带 QKV/MLP bias
                        "self_attn.q_proj.bias": "attn_q.bias",
                        "self_attn.k_proj.bias": "attn_k.bias",
                        "self_attn.v_proj.bias": "attn_v.bias",
                        "self_attn.o_proj.bias": "attn_output.bias",
                        "mlp.gate_proj.bias": "ffn_gate.bias",
                        "mlp.up_proj.bias": "ffn_up.bias",
                        "mlp.down_proj.bias": "ffn_down.bias",
                    }
                    if is_qwen35:
                        # Qwen3.5 线性注意力层（GatedDeltaNet）的 HF 名 → llama.cpp 名
                        m.update({
                            "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
                            "linear_attn.in_proj_z.weight": "attn_gate.weight",
                            "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
                            "linear_attn.in_proj_b.weight": "ssm_beta.weight",
                            "linear_attn.out_proj.weight": "ssm_out.weight",
                            "linear_attn.conv1d.weight": "ssm_conv1d.weight",
                            "linear_attn.A_log": "ssm_a",
                            "linear_attn.dt_bias": "ssm_dt.bias",
                            "linear_attn.norm.weight": "ssm_norm.weight",
                        })
                    if rest in m:
                        return f"blk.{i}.{m[rest]}"
            return None  # 未识别的权重丢弃

        mapped: Dict[str, np.ndarray] = {}
        for hf_name, arr in tensors.items():
            if hf_name == "lm_head.weight":
                mapped["output.weight"] = arr
                continue
            gg = gname(hf_name)
            if gg is None:
                continue
            # GDN 的 depthwise conv1d：HF 是 (C, 1, K)，GGUF/ggml 存 (C, K)
            if gg.endswith("ssm_conv1d.weight") and arr.ndim == 3:
                arr = np.ascontiguousarray(arr.reshape(arr.shape[0], arr.shape[2]))
            mapped[gg] = arr
        if is_qwen35:
            # HF 的零中心 RMSNorm 存原始 gain，llama.cpp 的 ggml_rms_norm 不加 +1，
            # 故写 GGUF 前折叠成 1+gain（与 GGUF→MLX 时的还原互为逆操作）。
            n = 0
            for k, v in list(mapped.items()):
                if v.ndim != 1:
                    continue
                hit = (k == _FOLDED_NORM_GGUF_NAMES[0]
                       or k.endswith(_FOLDED_NORM_GGUF_NAMES[1:]))
                if hit:
                    mapped[k] = (v.astype(np.float32) + 1.0).astype(np.float16)
                    n += 1
            hooks.on_log(f"已折叠 {n} 个 Qwen3.5 RMSNorm 权重为 1+gain")
        # 权重绑定：无 lm_head 时用 token_embd 作为 output
        if "output.weight" not in mapped and "token_embd.weight" in mapped:
            mapped["output.weight"] = mapped["token_embd.weight"]
            hooks.on_log("未检测到 lm_head，按权重绑定用 token_embd 作为输出")
        # 依据实际 token 数裁剪嵌入/输出行，保证与 n_vocab 一致
        for key in ("token_embd.weight", "output.weight"):
            if key in mapped and mapped[key].shape[0] > n_vocab:
                mapped[key] = mapped[key][:n_vocab]
        # 注：HF 的 [vocab, n_embd] 经 GGUF/ggml 的反转维度约定即为 llama.cpp 期望的
        # [n_embd, n_vocab]，无需手动转置。

        def write_tensor(name: str, arr: np.ndarray):
            a = np.ascontiguousarray(arr)
            # Q4_0/Q8_0 块大小为 32，行尾须整除；否则该张量保持 f16（如非对齐的词表嵌入）
            if quantize_enabled and a.ndim == 2 and a.shape[-1] % 32 == 0:
                q = gguf_quantize(a.astype(np.float32), file_type)
                writer.add_tensor(name, q, raw_dtype=file_type)
            else:
                # 量化模型以 f32 计算；1D（RMSNorm/QK-Norm）等权重若存为 f16 会与 f32
                # 张量做逐元素 binary_op 触发 ggml 不支持的类型报错 → 量化模式下保持 f32
                writer.add_tensor(name,
                                 a.astype(np.float32) if quantize_enabled
                                 else a.astype(np.float16))

        for name in sorted(mapped):
            write_tensor(name, mapped[name])
            hooks.on_log(f"写入张量: {name} {tuple(mapped[name].shape)}")

        # gguf>=0.19：close() 仅关闭句柄，需显式分步写文件
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        total = out_path.stat().st_size
        msg = f"{src_label}→GGUF 完成: {out_path.name}（{mt}, {self.quant_mode}, {total} bytes）"
        hooks.on_log(msg)
        return finish(True, False, msg, out_path, 1, total)
