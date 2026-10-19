"""应用入口。"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from modconv.main_window import MainWindow


def main() -> int:
    # 显式设置应用名：以 `python -m modconv` 直接启动时，macOS 的 Dock/关于菜单
    # 否则会显示解释器名（"Python"）
    app = QApplication(sys.argv)
    app.setApplicationName("多模型格式转换工具")
    app.setApplicationDisplayName("多模型格式转换工具")
    app.setOrganizationName("多模型格式转换工具")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
