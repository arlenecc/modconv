#!/bin/bash
# 将 ONNXRuntimeTester 打包为可双击运行的 macOS .app
set -euo pipefail

cd "$(dirname "$0")"

echo "🔨 正在编译 Release 版本 …"
swift build -c release

APP="ONNXRuntimeTester.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cp .build/release/ONNXRuntimeTester "$APP/Contents/MacOS/ONNXRuntimeTester"

# onnxruntime 通过 SwiftPM binaryTarget（xcframework）静态链接进可执行文件，无需额外拷贝；
# 但 mlx-swift 的 Metal 资源包（mlx-swift_Cmlx.bundle，含 default.metallib）必须随 .app 分发，
# 否则加载 MLX 模型、首次调用 GPU 时会因找不到 metallib 而崩溃。
mkdir -p "$APP/Contents/Resources"
for bundle in .build/release/*.bundle; do
    [ -e "$bundle" ] || continue
    cp -R "$bundle" "$APP/Contents/Resources/"
done

# 写入 Info.plist
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>ONNXRuntimeTester</string>
    <key>CFBundleDisplayName</key>
    <string>ONNX Runtime 测试工具</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.onnxruntimetester</string>
    <key>CFBundleExecutable</key>
    <string>ONNXRuntimeTester</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

echo "✅ 已生成 ${APP}（双击即可运行）"
