#!/usr/bin/env python3
"""构建 多模型格式转换工具 的 macOS 分发包（.app + .dmg）。

产物：
    dist/多模型格式转换工具.app            带图标的可双击应用包
    dist/多模型格式转换工具-<版本>.dmg     分发镜像（含应用、依赖安装脚本、说明与许可证）

为什么是"启动器形态"的 .app：
    转换功能依赖 torch / transformers / coremltools，三者合计 2–4 GB；把它们全部
    打进 .app 会得到一个数 GB、且与宿主 Python 版本强绑定的产物。因此 .app 内嵌
    源码与图标，运行时复用宿主 Python 环境（优先用一键安装脚本创建在
    ~/Library/Application Support/多模型格式转换工具/venv 下的独立虚拟环境），
    缺失依赖时给出图形化提示并引导安装。详见 README「分发包」一节。

用法：
    python3 packaging/build_macos.py            # 图标缺失时自动调用 make_icon.py
    python3 packaging/build_macos.py --skip-dmg # 只出 .app
"""
from __future__ import annotations

import argparse
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = ROOT / "dist"
APP_NAME = "多模型格式转换工具"
BUNDLE_ID = "com.modconv.app"
MIN_MACOS = "12.0"

# 不进入分发包的目录/文件
SKIP_NAMES = {"__pycache__", ".DS_Store", ".pytest_cache", ".git", ".venv"}


def read_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("version") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"\'')
    raise SystemExit("无法从 pyproject.toml 读取版本号")


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP_NAMES))


LAUNCHER = r'''#!/bin/bash
# 多模型格式转换工具 启动器：定位 Python 环境并启动 GUI。
# 依赖缺失时给出图形提示，引导运行"安装依赖"脚本。
set -u

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RES_DIR="$APP_ROOT/Contents/Resources"
SRC_DIR="$RES_DIR/app"
INSTALLER="$RES_DIR/install_dependencies.command"
SUPPORT_DIR="$HOME/Library/Application Support/多模型格式转换工具"
LOG_FILE="$HOME/Library/Logs/多模型格式转换工具.log"

mkdir -p "$HOME/Library/Logs" "$SUPPORT_DIR" 2>/dev/null

dialog() {
    osascript -e "display dialog \"$1\" buttons {\"$2\"} default button \"$2\" \
with title \"多模型格式转换工具\" with icon ${3:-caution}" >/dev/null 2>&1
}

has_pyqt6() {
    [ -n "${1:-}" ] && [ -x "$1" ] && "$1" -c 'import PyQt6' >/dev/null 2>&1
}

PY="${MODCONV_PYTHON:-}"

# 1) 一键安装脚本创建的独立虚拟环境（推荐）
if [ -z "$PY" ] && [ -x "$SUPPORT_DIR/venv/bin/python3" ]; then
    PY="$SUPPORT_DIR/venv/bin/python3"
fi

# 2) 退回 PATH 上的 python3（须已装 PyQt6）
if [ -z "$PY" ]; then
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        p="$(command -v "$candidate" 2>/dev/null || true)"
        if has_pyqt6 "$p"; then PY="$p"; break; fi
    done
fi

if ! has_pyqt6 "$PY"; then
    ANSWER="$(osascript -e 'return button returned of (display dialog "多模型格式转换工具 需要先安装 Python 依赖（PyQt6 等）。\n\n点“安装依赖”会在“终端”里运行安装脚本，期间需要联网，请勿关闭终端窗口。" buttons {"退出", "安装依赖"} default button "安装依赖" with title "多模型格式转换工具" with icon caution)' 2>/dev/null || echo "退出")"
    if [ "$ANSWER" = "安装依赖" ]; then
        open -a Terminal "$INSTALLER"
    fi
    exit 0
fi

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$HOME" || cd /

"$PY" -m modconv >>"$LOG_FILE" 2>&1
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    dialog "多模型格式转换工具 启动失败（退出码 $STATUS）。\n日志已保存到：\n$LOG_FILE" "好" stop
    open -R "$LOG_FILE" 2>/dev/null
fi
'''


INSTALLER = r'''#!/bin/bash
# 为 多模型格式转换工具 创建独立虚拟环境并安装依赖。
# 可在 App 包内运行，也可从 DMG 根目录双击运行。
set -u

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SELF_DIR/../Contents/Resources/app" ]; then
    SRC_DIR="$(cd "$SELF_DIR/../Contents/Resources/app" && pwd)"
elif [ -d "$SELF_DIR/多模型格式转换工具.app/Contents/Resources/app" ]; then
    SRC_DIR="$SELF_DIR/多模型格式转换工具.app/Contents/Resources/app"
elif [ -d "$SELF_DIR/../Resources/app" ]; then
    SRC_DIR="$(cd "$SELF_DIR/../Resources/app" && pwd)"
else
    echo "找不到 多模型格式转换工具 源码目录（Contents/Resources/app）。"
    echo "请把本脚本与 多模型格式转换工具.app 放在同一目录下再运行。"
    exit 1
fi

VENV="$HOME/Library/Application Support/多模型格式转换工具/venv"
PY="$(command -v python3 || true)"

echo "多模型格式转换工具 依赖安装"
echo "  源码目录: $SRC_DIR"
echo "  虚拟环境: $VENV"
echo

if [ -z "$PY" ]; then
    echo "未找到 python3。请先安装 Python 3.10 或更高版本："
    echo "  https://www.python.org/downloads/"
    exit 1
fi

echo "使用的 Python: $PY ($("$PY" -V 2>&1))"
echo

read -r -p "是否同时安装模型转换依赖（torch / transformers / coremltools，约 2-4 GB）？[Y/n] " WANT_CONVERT
case "${WANT_CONVERT:-Y}" in
    [Nn]*) EXTRA="" ;;
    *)     EXTRA="[convert]" ;;
esac

set -e
if [ ! -x "$VENV/bin/python3" ]; then
    echo "==> 创建虚拟环境"
    "$PY" -m venv "$VENV"
fi

echo "==> 升级 pip"
"$VENV/bin/python3" -m pip install --upgrade pip

echo "==> 安装 多模型格式转换工具$EXTRA"
"$VENV/bin/python3" -m pip install "${SRC_DIR}${EXTRA}"

echo
echo "==> 校验"
"$VENV/bin/python3" -c "import PyQt6, modconv; print('  PyQt6 + modconv 就绪')"
if [ -n "$EXTRA" ]; then
    "$VENV/bin/python3" -c "import torch, transformers, coremltools; print('  转换依赖就绪')" || \
        echo "  （转换依赖未就绪，仍可使用界面，但无法执行真实转换）"
fi

echo
echo "安装完成。回到“应用程序”双击 多模型格式转换工具 即可启动。"
read -r -p "按回车关闭本窗口。" _
'''


USAGE_TXT = """多模型格式转换工具 安装与使用
=========================

1. 把 多模型格式转换工具.app 拖到左边的“应用程序”文件夹。

2. 双击“安装依赖.command”（首次使用必需）
   - 会为 多模型格式转换工具 创建独立虚拟环境：
     ~/Library/Application Support/多模型格式转换工具/venv
   - 询问是否安装模型转换依赖（torch / transformers / coremltools，约 2-4 GB）
     只做界面体验可以选 n；要做真实转换请选 Y。
   - 安装需要联网，请耐心等待，勿关闭终端窗口。

3. 双击 多模型格式转换工具 启动（若提示已损坏或来自未验证的开发者，
   请在 Finder 里按住 Control 点击 App →“打开”）。

分发包是“启动器形态”的 .app：内嵌图标与源码，运行时复用宿主 Python
环境。之所以不把 torch/transformers/coremltools 打进 .app，是因为它们
合计 2-4 GB，且会与 Python 版本强绑定。

环境变量
--------
MODCONV_PYTHON  指定启动用的 Python 解释器（须已安装 PyQt6）

日志
----
~/Library/Logs/多模型格式转换工具.log

完整文档见 README.md。
"""


def write_plist(contents: Path, version: str) -> None:
    plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": MIN_MACOS,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "GPL-3.0. MLX → Core ML 模型转换工具。",
    }
    with (contents / "Info.plist").open("wb") as fh:
        plistlib.dump(plist, fh)


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def ensure_icon(icon: Path) -> None:
    if icon.is_file():
        return
    print("图标不存在，调用 packaging/make_icon.py 生成…")
    subprocess.run([sys.executable, str(HERE / "make_icon.py")], check=True)


def build_app(version: str, icon: Path) -> Path:
    app = DIST / f"{APP_NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    contents = app / "Contents"
    resources = contents / "Resources"
    macos = contents / "MacOS"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    copy_tree(ROOT / "modconv", resources / "app" / "modconv")
    shutil.copy2(ROOT / "pyproject.toml", resources / "app" / "pyproject.toml")
    shutil.copy2(ROOT / "README.md", resources / "app" / "README.md")
    license_src = ROOT / "LICENSE"
    if license_src.is_file():
        shutil.copy2(license_src, resources / "app" / "LICENSE")
    shutil.copy2(icon, resources / "AppIcon.icns")
    write_executable(resources / "install_dependencies.command", INSTALLER)
    (contents / "PkgInfo").write_text("APPL????", encoding="ascii")
    write_plist(contents, version)
    write_executable(macos / APP_NAME, LAUNCHER)
    return app


def build_dmg(app: Path, version: str) -> Path:
    dmg = DIST / f"{APP_NAME}-{version}.dmg"
    with tempfile.TemporaryDirectory(prefix="modconv-dmg-") as tmp:
        stage = Path(tmp) / APP_NAME
        stage.mkdir()
        subprocess.run(["ditto", str(app), str(stage / app.name)], check=True)
        (stage / "使用说明.txt").write_text(USAGE_TXT, encoding="utf-8")
        shutil.copy2(HERE.parent / "README.md", stage / "README.md")
        license_src = HERE.parent / "LICENSE"
        if license_src.is_file():
            shutil.copy2(license_src, stage / "LICENSE")
        installer = stage / "安装依赖.command"
        installer.write_text(INSTALLER, encoding="utf-8")
        installer.chmod(0o755)
        # 便于把 App 拖进"应用程序"（DMG 惯用做法）
        (stage / "Applications").symlink_to("/Applications")

        if dmg.exists():
            dmg.unlink()
        subprocess.run([
            "hdiutil", "create",
            "-volname", f"{APP_NAME} {version}",
            "-srcfolder", str(stage),
            "-fs", "HFS+",
            "-format", "UDZO",
            "-imagekey", "zlib-level=9",
            "-ov",
            str(dmg),
        ], check=True)
    return dmg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dmg", action="store_true", help="只构建 .app")
    args = parser.parse_args()

    if sys.platform != "darwin":
        raise SystemExit("分发包只能在 macOS 上构建")

    version = read_version()
    DIST.mkdir(exist_ok=True)
    icon = HERE / "AppIcon.icns"
    ensure_icon(icon)

    app = build_app(version, icon)
    print(f"已生成 {app}")
    if args.skip_dmg:
        return 0

    dmg = build_dmg(app, version)
    size_mb = dmg.stat().st_size / (1024 * 1024)
    print(f"已生成 {dmg} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
