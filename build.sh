#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/output/build"
INSTALL_DIR="$ROOT_DIR/output/install"

if [ "${1:-}" = "clean" ]; then
    echo "==> 清理构建目录..."
    rm -rf "$ROOT_DIR/output"
    echo "    done"
    exit 0
fi

# Step 1: 配置 + 构建库
echo "==> 配置 & 构建库..."
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
cmake --build "$BUILD_DIR" --target install
echo "    库已安装到 $INSTALL_DIR"

# Step 2: 构建 + 运行测试
echo "==> 构建 & 运行测试..."
cmake -S "$ROOT_DIR/test" -B "$BUILD_DIR/test" \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR"
cmake --build "$BUILD_DIR/test"
ctest --test-dir "$BUILD_DIR/test" --output-on-failure

echo "==> 全部完成"
