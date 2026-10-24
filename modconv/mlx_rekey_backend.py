"""MLX→MLX 转换后端：重命名（HF roberta/bert 命名 → MLXEmbedders 命名）+ 可选量化。

复刻 tools/convert_mlx_rekey_quantize.py（已验证精确复现 bge-reranker-v2-m3-mlx
fp16/int8/int4 交付物）与 jina 工具的补占位逻辑：

- 重命名：剥离 `roberta.` / `bert.` 前缀；`encoder.layer.`→`encoder.layers.`；
  `attention.self.{query,key,value}.`→`attention.{query,key,value}_proj.`；
  `attention.output.dense.`→`attention.out_proj.`；`attention.output.LayerNorm.`→`ln1.`；
  `output.LayerNorm.`→`ln2.`；`intermediate.dense.`→`linear1.`；`output.dense.`→`linear2.`；
  `embeddings.LayerNorm.`→`embeddings.norm.`；丢弃 `pooler.*`。
  （替换顺序敏感：更具体的子串先替换。对 `.scales`/`.biases` 兄弟键同样适用。）
- 量化：2D 权重 mx.quantize（affine），陪衬键为**兄弟键**；源已是量化(uint32) 则仅重命名。
- config.json：拷贝源配置；量化时写 quantization 字段，fp16 时移除；缺 model_type 补
  xlm-roberta（jina 源只有 auto_map）。
- jina 类源无 embeddings.LayerNorm：补 embeddings.norm ones/zeros 占位（加载器
  verify:[.all] 要求齐备；经第一层 ln1 抵消不破坏模型）。

纯 MLX 实现（不依赖 transformers/torch）。不依赖 Qt，可独立测试。
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from modconv.converter import ConversionHooks, ConversionResult
from modconv.utils import is_appledouble, list_model_files

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "vocab.txt",
                    "vocab.json", "merges.txt")


def rekey(k: str) -> Optional[str]:
    """HF roberta/bert 键 → MLXEmbedders 键。返回 None 表示丢弃。"""
    if "pooler" in k:
        return None
    if k.startswith("roberta."):
        k = k[len("roberta."):]
    elif k.startswith("bert."):
        k = k[len("bert."):]
    k = k.replace("encoder.layer.", "encoder.layers.")
    # 顺序敏感：更具体的子串先替换（output.LayerNorm 会误伤 attention.output.LayerNorm）
    k = k.replace("attention.output.dense.", "attention.out_proj.")
    k = re.sub(r"attention\.self\.(query|key|value)\.", r"attention.\1_proj.", k)
    k = k.replace("attention.output.LayerNorm.", "ln1.")
    k = k.replace("output.LayerNorm.", "ln2.")
    k = k.replace("intermediate.dense.", "linear1.")
    k = k.replace("output.dense.", "linear2.")
    k = k.replace("embeddings.LayerNorm.", "embeddings.norm.")
    return k


class MlxToMlxBackend:
    """把 MLX 权重目录（HF 命名或 MLXEmbedders 命名，fp16 或已量化）转换为
    MLXEmbedders 命名 + 可选 int4/int8 量化。

    quant_mode: "fp16"（仅重命名）/ "int4" / "int8"。
    """

    def __init__(self, source: Path, output_dir: Path, *,
                 quant_mode: str = "fp16", group_size: int = 64):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode
        self.group_size = group_size

    @staticmethod
    def _import_mlx():
        import mlx.core as mx
        return mx

    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            mx = self._import_mlx()
            return self._run(mx, hooks, finish)
        except Exception as exc:
            msg = f"MLX→MLX 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)

    # ------------------------------------------------------------------
    def _load_weights(self, mx) -> Tuple[Dict, Optional[dict]]:
        """读取目录内全部 safetensors 权重 + 源 config（无则 None）。"""
        files = list_model_files(self.source, "*.safetensors")
        if not files:
            raise RuntimeError(f"源目录无 safetensors 权重: {self.source}")
        raw: Dict = {}
        for f in files:
            raw.update(mx.load(str(f)))
        cfg_path = self.source / "config.json"
        cfg = None
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except ValueError:
                cfg = None
        return raw, cfg

    def _run(self, mx, hooks, finish):
        raw, src_cfg = self._load_weights(mx)
        bits = {"int4": 4, "int8": 8}.get(self.quant_mode)  # None → 仅重命名

        # 源已是量化（存在 uint32 权重）：仅重命名，沿用源 quantization 配置
        already_quant = any(getattr(v, "dtype", None) == getattr(mx, "uint32", None)
                            for k, v in raw.items() if k.endswith(".weight"))
        src_qcfg = (src_cfg or {}).get("quantization") or {}

        state: Dict = {}
        n_quant = 0
        for k, v in raw.items():
            nk = rekey(k)
            if nk is None:
                continue
            is_w = nk.endswith(".weight")
            if v.dtype == mx.uint32:
                state[nk] = v  # 已量化：仅重命名
                continue
            if bits and is_w and v.ndim == 2:
                a = v.astype(mx.float16) if v.dtype != mx.float16 else v
                done = False
                for gs in (self.group_size, 32, 16, 8):
                    if int(v.shape[-1]) % gs:
                        continue
                    wq, sc, bi = mx.quantize(a, gs, bits, mode="affine")
                    if gs != self.group_size:
                        hooks.on_log(f"group_size 回退为 {gs}（维度 {v.shape}）")
                    base = nk[:-len(".weight")]
                    state[f"{base}.weight"] = wq
                    state[f"{base}.scales"] = sc
                    state[f"{base}.biases"] = bi
                    n_quant += 1
                    done = True
                    break
                if not done:
                    hooks.on_log(f"无法量化 {nk}（维度 {v.shape}），保持 fp16")
                    state[nk] = v.astype(mx.float16)
            else:
                state[nk] = v.astype(mx.float16) if v.dtype != mx.float16 else v

        hidden = int(state["embeddings.word_embeddings.weight"].shape[-1])
        # jina 类源无 embeddings.LayerNorm → 补占位（加载器 verify:[.all] 要求齐备）
        if "embeddings.norm.weight" not in state:
            state["embeddings.norm.weight"] = mx.ones((hidden,), dtype=mx.float16)
            state["embeddings.norm.bias"] = mx.zeros((hidden,), dtype=mx.float16)

        # ---- config.json ----
        cfg = dict(src_cfg) if src_cfg else {}
        if already_quant:
            # 沿用源量化配置
            if src_qcfg:
                cfg["quantization"] = src_qcfg
        elif bits:
            cfg["quantization"] = {"bits": bits, "group_size": self.group_size,
                                   "mode": "affine"}
        else:
            cfg.pop("quantization", None)
        cfg.setdefault("model_type", "xlm-roberta")

        # ---- 写出 ----
        self.output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.output_dir / "model.safetensors"
        mx.save_safetensors(str(weights_path), state)
        (self.output_dir / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        copied = 0
        for fname in _TOKENIZER_FILES:
            sp = self.source / fname
            if sp.is_file():
                shutil.copy(sp, self.output_dir / fname)
                copied += 1
        if (self.source / "adapters").is_dir():
            shutil.copytree(self.source / "adapters", self.output_dir / "adapters",
                            dirs_exist_ok=True)
        # sentence-transformers 的池化配置在 <N>_Pooling/ 子目录里（nomic 为 1_Pooling/
        # 的 mean pooling）。MLXEmbedders/Swift 侧会读它决定池化方式，丢了可能退化成
        # 默认池化而改变嵌入语义，故一并拷贝。
        for sub in sorted(self.source.glob("*_Pooling")):
            if sub.is_dir():
                shutil.copytree(sub, self.output_dir / sub.name, dirs_exist_ok=True)
                hooks.on_log(f"已拷贝池化配置目录: {sub.name}")
        if copied:
            hooks.on_log(f"已拷贝 {copied} 个 tokenizer 文件")

        total = sum(p.stat().st_size for p in self.output_dir.glob("*"))
        qinfo = "沿用源量化" if already_quant else (
            f"int{bits}（{n_quant} 个 2D 权重）" if bits else "fp16 仅重命名")
        msg = f"MLX→MLX 完成: {weights_path.name}（{qinfo}，共 {len(state)} 张量）"
        hooks.on_log(msg)
        return finish(True, False, msg, weights_path, 1, total)
