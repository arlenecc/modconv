"""HF→MLX 转换后端（经 mlx-lm / mlx-embeddings / mlx-vlm）。

直接用官方 MLX 转换库把 HF 模型转为 MLX 数组格式，并按需 int4/int8 量化：
  - causal(LLM)          → mlx_lm.convert
  - embedding / reranker  → mlx_embeddings.convert
  - vlm(视觉语言)         → mlx_vlm convert

这些库同时支持转换与量化（-q/--q-bits/--q-group-size），是 HF→MLX 的健壮路径。
实现上以子进程调用各库 CLI（用当前解释器 sys.executable），实时把日志转发给 UI，
并支持取消。不依赖 Qt，可独立测试。
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from modconv.converter import ConversionHooks, ConversionResult

# 模型类型 → 库 的归类
_CAUSAL_TYPES = {"llama", "mistral", "mixtral", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
                 "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
                 "gemma", "gemma2", "gemma3", "phi3", "phi", "stablelm", "starcoder2",
                 "cohere", "commandr", "falcon", "internlm2", "baichuan", "yi"}
_VLM_TYPES = {"qwen2_vl", "qwen2_5_vl", "qwen2vl", "llava", "llava_next", "idefics",
              "idefics2", "idefics3", "paligemma", "molmo", "mllama", "pixtral"}

# 转换完成后需要从源目录还原/补齐的 tokenizer、processor 侧车文件。
# 见 _restore_tokenizer_files：转换库会用 transformers 重新序列化
# tokenizer_config.json，把 tokenizer_class 改写成模型专属类。
_SIDECAR_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja", "vocab.json", "vocab.txt",
    "merges.txt", "processor_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json",
)


class HfToMlxBackend:
    """把 HF 模型转换为 MLX 格式，可选 int4/int8 量化。

    quant_mode: "fp16"（不量化）/ "int8" / "int4"。
    """

    def __init__(self, source: Path, output_dir: Path, *,
                 quant_mode: str = "fp16", group_size: int = 64,
                 dtype: str = "float16"):
        self.source = Path(source).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser()
        self.quant_mode = quant_mode
        self.group_size = group_size
        self.dtype = dtype

    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(source: Path) -> dict:
        cfg_path = source / "config.json"
        if not cfg_path.is_file():
            raise RuntimeError("源目录缺少 config.json，无法判定模型类型")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {}

    @classmethod
    def select_lib(cls, cfg: dict):
        """按模型类型选择转换库。返回 (lib_pkg, module, subcmd)。"""
        archs = " ".join(list(cfg.get("architectures", []) or []))
        mt = str(cfg.get("model_type", "") or "")
        text_cfg = cfg.get("text_config", {}) if isinstance(cfg.get("text_config"), dict) else {}
        archs = archs + " " + " ".join(list(text_cfg.get("architectures", []) or []))
        is_vlm = ("vision_config" in cfg or mt in _VLM_TYPES
                  or any(t in archs for t in ("VL", "VisionLanguage", "ForConditionalGeneration")))
        if is_vlm:
            return "mlx_vlm", "mlx_vlm", ["convert"]
        if "ForCausalLM" in archs or mt in _CAUSAL_TYPES:
            return "mlx_lm", "mlx_lm", ["convert"]
        # embedding / reranker（encoder，默认）：mlx_embeddings 仅提供 .convert 模块
        return "mlx_embeddings", "mlx_embeddings.convert", []

    # ------------------------------------------------------------------
    def _restore_tokenizer_files(self, hooks) -> None:
        """用源目录的 tokenizer 配置修正转换库重写后的产物（关键，否则模型加载不了）。

        mlx-lm / mlx-vlm 的 `convert` 会经 transformers 重新序列化
        tokenizer_config.json，带来两处破坏：
          1. tokenizer_class 被写成**模型专属类**（Qwen3.5 → `Qwen3_5Tokenizer`），
             而 mlx-swift-lm 只接受 `TokenizersBackend` 等少数几种，加载时直接抛
             "The tokenizer type 'Qwen3_5Tokenizer' is not supported."——产物等于
             不可用（Qwen3.5-2B 实测：改回 TokenizersBackend 后立刻正常生成）；
          2. chat_template 与 added_tokens_decoder 被丢掉，靠 chat_template 构造
             提示词的消费端会退化成裸文本输入。
        源目录的 HF 原版配置才是 mlx-swift-lm / mlx-lm 期望的形式，故以它为准；
        库产物里源目录没有的键（backend / is_local 等运行期字段）保留。
        """
        src = self.source if self.source.is_dir() else self.source.parent
        out_cfg = self.output_dir / "tokenizer_config.json"
        src_cfg = src / "tokenizer_config.json"
        if src_cfg.is_file() and out_cfg.is_file():
            try:
                generated = json.loads(out_cfg.read_text(encoding="utf-8"))
                original = json.loads(src_cfg.read_text(encoding="utf-8"))
                if isinstance(generated, dict) and isinstance(original, dict):
                    # 源目录优先；库产物独有的键（运行期字段）补齐保留
                    merged = {**generated, **original}
                    out_cfg.write_text(
                        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
                    if generated.get("tokenizer_class") != original.get("tokenizer_class"):
                        hooks.on_log(
                            f"已还原 tokenizer_class: {generated.get('tokenizer_class')} → "
                            f"{original.get('tokenizer_class')}"
                            "（转换库重写为模型专属类，mlx-swift-lm 会拒绝加载）")
            except (OSError, ValueError) as exc:
                hooks.on_log(f"[warn] 还原 tokenizer_config.json 失败: {exc}")
        elif src_cfg.is_file():
            shutil.copy(src_cfg, out_cfg)
        # 转换库没带过来的其余 tokenizer / processor 侧车文件，按需补齐
        copied = []
        for name in _SIDECAR_FILES:
            sp, dp = src / name, self.output_dir / name
            if sp.is_file() and not dp.exists():
                shutil.copy(sp, dp)
                copied.append(name)
        if copied:
            hooks.on_log(f"已补齐 {len(copied)} 个 tokenizer/processor 文件: "
                         + ", ".join(copied))

    @staticmethod
    def _cfg_dims(cfg: dict) -> tuple:
        """取 (hidden_size, intermediate_size)。

        多模态的复合配置（Qwen3.5 / Qwen3-VL 等）把结构字段放在 text_config 下，
        顶层只有 vision/audio 相关字段——直接读顶层会全部取到 0，group_size 就退化成
        默认值，恰好能整除时看不出问题，不能整除时才会随机挑错。
        """
        text_cfg = cfg.get("text_config")
        for src in (cfg, text_cfg if isinstance(text_cfg, dict) else {}):
            hidden = int(src.get("hidden_size") or src.get("d_model") or 0)
            inter = int(src.get("intermediate_size") or src.get("ffn_dim") or 0)
            if hidden or inter:
                return hidden, inter
        return 0, 0

    def _effective_group(self, cfg: dict) -> int:
        """按 hidden/intermediate 维度选能整除的量化 group_size（默认尽量 64）。"""
        hidden, inter = self._cfg_dims(cfg)
        for g in (64, 32, 16, 8):
            if (not hidden or hidden % g == 0) and (not inter or inter % g == 0):
                return g
        return 8

    def _build_cmd(self, lib_pkg, module, subcmd) -> list:
        cmd = [sys.executable, "-m", module, *subcmd,
               "--hf-path", str(self.source), "--mlx-path", str(self.output_dir),
               "--dtype", self.dtype]
        if self.quant_mode in ("int4", "int8"):
            bits = "8" if self.quant_mode == "int8" else "4"
            # mlx_vlm convert 同样接受 --q-group-size：不传就退回库自身的默认值，
            # 与界面上选的档位（_effective_group 按维度收敛的结果）不一致。
            cmd += ["-q", "--q-bits", bits, "--q-group-size", str(self.group_size)]
        return cmd

    # ------------------------------------------------------------------
    def run(self, hooks: ConversionHooks) -> ConversionResult:
        start = time.monotonic()

        def finish(success, cancelled, message, output=None, files=0, bytes_=0):
            return ConversionResult(success, cancelled, message, output,
                                    files, bytes_, time.monotonic() - start)

        try:
            cfg = self._load_config(self.source)
            lib_pkg, module, subcmd = self.select_lib(cfg)
            if self.quant_mode in ("int4", "int8"):
                self.group_size = self._effective_group(cfg)
            hooks.on_log(f"选用转换库: {lib_pkg}（model_type={cfg.get('model_type', '?')}）")
            if importlib.util.find_spec(lib_pkg) is None:
                raise RuntimeError(f"未安装 {lib_pkg}，无法执行 HF→MLX（此类型模型）")
            # 这些库要求输出目录不存在；预先清理并确保父目录存在
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)
            cmd = self._build_cmd(lib_pkg, module, subcmd)
            hooks.on_log("执行: " + " ".join(cmd))

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            cancelled = False
            assert proc.stdout is not None
            for line in proc.stdout:
                text = line.rstrip()
                if text:
                    hooks.on_log(text)
                if hooks.should_cancel():
                    proc.terminate()
                    cancelled = True
                    break
            rc = proc.wait()

            if cancelled:
                return finish(False, True, "已取消 HF→MLX")
            if rc != 0:
                return finish(False, False, f"{lib_pkg} 转换失败（exit={rc}）")

            # 校验产物存在权重
            has_weights = any(
                p.suffix in (".safetensors", ".npz") for p in self.output_dir.glob("*"))
            if not has_weights:
                return finish(False, False, "转换结束但未找到 MLX 权重文件")

            # 转换库会重写 tokenizer_config.json，必须还原（否则 mlx-swift-lm 加载失败）
            self._restore_tokenizer_files(hooks)
            if not (self.output_dir / "tokenizer.json").is_file():
                hooks.on_log("[warn] 产物缺少 tokenizer.json，可能无法直接推理")

            total = sum(p.stat().st_size for p in self.output_dir.glob("*"))
            msg = f"HF→MLX 完成: {lib_pkg}（{self.quant_mode}）"
            hooks.on_log(msg)
            return finish(True, False, msg, self.output_dir, -1, total)
        except Exception as exc:
            msg = f"HF→MLX 失败: {type(exc).__name__}: {exc}"
            try:
                hooks.on_log(msg)
            except Exception:
                pass
            return finish(False, False, msg)
