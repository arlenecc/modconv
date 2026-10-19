#!/usr/bin/env python3
"""生成多模型格式转换工具 的 macOS 应用图标（AppIcon.icns）。

设计：圆角方形底 + 蓝紫渐变 + 居中的白色粗箭头（"转换"语义），
外加轻微投影。渐变用 numpy 生成、箭头的圆角/抗锯齿由 PIL 处理，
统一先在 1024×1024 上绘制，再按 LANCZOS 缩放到各尺寸，保证小图标也干净。

用法：
    python3 packaging/make_icon.py            # 生成 packaging/AppIcon.icns
    python3 packaging/make_icon.py --preview  # 额外导出 512 PNG 便于查看

依赖：Pillow、numpy（仅构建期需要，运行 App 不需要）。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as exc:  # pragma: no cover - 构建期环境问题
    raise SystemExit(
        f"生成图标需要 Pillow 与 numpy（pip install pillow numpy）: {exc}")

HERE = Path(__file__).resolve().parent
OUT_ICNS = HERE / "AppIcon.icns"

# Apple HIG 推荐的圆角比例（连续圆角）
CORNER_RATIO = 0.2237
# 渐变端点（左上 → 右下）
GRADIENT_TOP = (0x2E, 0x6B, 0xF6)
GRADIENT_BOTTOM = (0x7A, 0x3C, 0xE8)


def _rounded_mask(size: int) -> Image.Image:
    radius = int(round(size * CORNER_RATIO))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def _gradient(size: int) -> Image.Image:
    """对角线性渐变（左上亮 → 右下暗）。"""
    t = np.linspace(0.0, 1.0, size, dtype=np.float32)
    # 对角线参数：以 (x+y)/2 归一化，避免只有角落最深
    grid = (t[None, :] + t[:, None]) * 0.5
    grid = grid[..., None]
    top = np.array(GRADIENT_TOP, dtype=np.float32)
    bottom = np.array(GRADIENT_BOTTOM, dtype=np.float32)
    rgb = top * (1.0 - grid) + bottom * grid
    rgba = np.concatenate(
        [rgb, np.full((size, size, 1), 255.0, dtype=np.float32)], axis=-1)
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def _arrow_layer(size: int) -> Image.Image:
    """白色粗箭头（圆角箭杆 + 三角箭头），带柔和投影与顶部高光。"""
    s = float(size)
    cy = s * 0.5
    thickness = s * 0.145
    left = s * 0.225
    right = s * 0.775
    head_len = s * 0.245
    head_half = s * 0.195

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # 箭杆：圆角矩形，右端伸入箭头内部以免出现接缝
    shaft_end = right - head_len * 0.55
    d.rounded_rectangle((left, cy - thickness / 2, shaft_end, cy + thickness / 2),
                        radius=thickness / 2, fill=(255, 255, 255, 255))
    # 箭头：三角形（顶点略钝化在缩放到小尺寸时由 LANCZOS 自然完成）
    d.polygon([(right, cy),
               (right - head_len, cy - head_half),
               (right - head_len, cy + head_half)],
              fill=(255, 255, 255, 255))

    # 顶部高光：让箭头有轻微立体感
    gloss = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(gloss).rectangle((0, 0, size, int(cy)), fill=(255, 255, 255, 26))
    layer = Image.alpha_composite(layer, Image.composite(
        gloss, Image.new("RGBA", (size, size), (0, 0, 0, 0)),
        layer.getchannel("A")))

    # 投影：向下偏移后高斯模糊，叠在箭头下方
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow.paste(layer, (0, int(round(s * 0.022))), layer)
    shadow = shadow.filter(ImageFilter.GaussianBlur(s * 0.018))
    shadow.putalpha(shadow.getchannel("A").point(lambda v: int(v * 0.38)))

    return Image.alpha_composite(shadow, layer)


def render(size: int) -> Image.Image:
    base = _gradient(size)
    base = Image.alpha_composite(base, _arrow_layer(size))
    base.putalpha(_rounded_mask(size))
    return base


# (iconset 文件名, 逻辑边长, 实际像素边长)
ICONSET_SIZES = [
    ("icon_16x16.png", 16, 16),
    ("icon_16x16@2x.png", 16, 32),
    ("icon_32x32.png", 32, 32),
    ("icon_32x32@2x.png", 32, 64),
    ("icon_128x128.png", 128, 128),
    ("icon_128x128@2x.png", 128, 256),
    ("icon_256x256.png", 256, 256),
    ("icon_256x256@2x.png", 256, 512),
    ("icon_512x512.png", 512, 512),
    ("icon_512x512@2x.png", 512, 1024),
]


def build_iconset(master: Image.Image, iconset_dir: Path) -> None:
    iconset_dir.mkdir(parents=True, exist_ok=True)
    for name, _logical, px in ICONSET_SIZES:
        master.resize((px, px), Image.LANCZOS).save(iconset_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="额外输出 packaging/icon-preview-512.png")
    args = parser.parse_args()

    master = render(1024)
    if args.preview:
        master.resize((512, 512), Image.LANCZOS).save(HERE / "icon-preview-512.png")

    if shutil.which("iconutil") is None:
        raise SystemExit("需要 macOS 的 iconutil 才能生成 .icns")

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        build_iconset(master, iconset)
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                        "-o", str(OUT_ICNS)], check=True)
    print(f"已生成 {OUT_ICNS} ({OUT_ICNS.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
