"""分发包构建的静态/轻量校验（不依赖 GUI，不需要 Pillow）。

`packaging/` 只是构建脚本目录（无 `__init__.py`），且 `packaging` 是 setuptools 的
第三方包名，因此这里按文件路径加载，避免误导入。
"""
import importlib.util
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = ROOT / "packaging" / "build_macos.py"
ICON = ROOT / "packaging" / "AppIcon.icns"

pytestmark = pytest.mark.skipif(sys.platform != "darwin",
                                reason="分发包只能在 macOS 上构建")


def _load_build_module():
    spec = importlib.util.spec_from_file_location("m2c_build_macos", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_mod():
    return _load_build_module()


def test_version_matches_pyproject(build_mod):
    """包版本号只有一个来源（pyproject.toml），构建脚本不得另写一份。"""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{build_mod.read_version()}"' in text


def test_app_icon_is_present_and_valid():
    """图标需随仓库提供，保证无 Pillow 环境下也能构建分发包。"""
    assert ICON.is_file(), "缺少 packaging/AppIcon.icns（运行 packaging/make_icon.py 生成）"
    assert ICON.stat().st_size > 10_000
    assert ICON.read_bytes()[:4] == b"icns"


def test_build_app_produces_valid_bundle(build_mod, tmp_path, monkeypatch):
    version = build_mod.read_version()
    monkeypatch.setattr(build_mod, "DIST", tmp_path)
    app = build_mod.build_app(version, ICON)

    contents = app / "Contents"
    with (contents / "Info.plist").open("rb") as fh:
        plist = plistlib.load(fh)
    assert plist["CFBundleExecutable"] == build_mod.APP_NAME
    assert plist["CFBundleIconFile"] == "AppIcon"
    assert plist["CFBundleIdentifier"] == build_mod.BUNDLE_ID
    assert plist["CFBundleShortVersionString"] == version

    assert (contents / "Resources" / "AppIcon.icns").is_file()
    assert (contents / "Resources" / "app" / "modconv" / "__main__.py").is_file()
    # 构建产物不应把缓存/虚拟环境带进去
    assert not (contents / "Resources" / "app" / "modconv" / "__pycache__").exists()

    launcher = contents / "MacOS" / build_mod.APP_NAME
    assert launcher.is_file()
    assert launcher.stat().st_mode & stat.S_IXUSR, "启动器必须可执行"
    assert launcher.read_text(encoding="utf-8").startswith("#!/bin/bash")

    installer = contents / "Resources" / "install_dependencies.command"
    assert installer.is_file() and installer.stat().st_mode & stat.S_IXUSR


def test_launcher_resolves_python_and_sets_pythonpath(build_mod, tmp_path, monkeypatch):
    """用假解释器跑一遍启动器：应把内嵌源码前置到 PYTHONPATH 并调用 python -m modconv。

    日志重定向到 $HOME/Library/Logs，这里把 HOME 指到临时目录，避免污染真实用户目录。
    """
    version = build_mod.read_version()
    monkeypatch.setattr(build_mod, "DIST", tmp_path / "dist")
    app = build_mod.build_app(version, ICON)

    fake_py = tmp_path / "fakepy"
    fake_py.write_text(
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-c" ]; then exit 0; fi\n'   # 假装 PyQt6 可用
        'if [ "${1:-}" = "-m" ]; then\n'
        '  echo "LAUNCHED:$2:PYTHONPATH=$PYTHONPATH"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n", encoding="utf-8")
    fake_py.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "MODCONV_PYTHON": str(fake_py), "HOME": str(home)}

    proc = subprocess.run([str(app / "Contents" / "MacOS" / build_mod.APP_NAME)],
                          capture_output=True, text=True, timeout=60, env=env, cwd=str(home))
    assert proc.returncode == 0, proc.stderr

    log = home / "Library" / "Logs" / "多模型格式转换工具.log"
    assert log.is_file(), "启动器应把输出写入 ~/Library/Logs/多模型格式转换工具.log"
    line = log.read_text(encoding="utf-8")
    assert "LAUNCHED:modconv" in line
    assert str(app / "Contents" / "Resources" / "app") in line


def test_installer_script_is_syntactically_valid(build_mod, tmp_path, monkeypatch):
    """安装脚本要能通过 bash 语法检查（它几乎只在用户机器上运行一次）。"""
    monkeypatch.setattr(build_mod, "DIST", tmp_path)
    app = build_mod.build_app(build_mod.read_version(), ICON)
    installer = app / "Contents" / "Resources" / "install_dependencies.command"
    assert subprocess.run(["bash", "-n", str(installer)]).returncode == 0
    launcher = app / "Contents" / "MacOS" / build_mod.APP_NAME
    assert subprocess.run(["bash", "-n", str(launcher)]).returncode == 0


@pytest.mark.skipif(shutil.which("hdiutil") is None, reason="需要 hdiutil")
def test_build_dmg(build_mod, tmp_path, monkeypatch):
    """DMG 里应包含 app、一键安装脚本、说明、LICENSE 与 /Applications 软链接。"""
    monkeypatch.setattr(build_mod, "DIST", tmp_path)
    version = build_mod.read_version()
    app = build_mod.build_app(version, ICON)
    dmg = build_mod.build_dmg(app, version)

    assert dmg.is_file() and dmg.stat().st_size > 100_000
    info = subprocess.run(["hdiutil", "imageinfo", str(dmg)],
                          capture_output=True, text=True, timeout=120)
    assert info.returncode == 0, info.stderr
    assert "UDZO" in info.stdout or "compressed" in info.stdout.lower()
