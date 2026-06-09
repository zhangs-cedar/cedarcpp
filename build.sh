#!/usr/bin/env bash
# ============================================================================
# cedarcpp 一键构建脚本
# ============================================================================
# 用法:
#   ./build.sh         一键构建库 + 运行测试
#   ./build.sh clean   清理所有构建产物和日志
#
# 流程说明:
#   1. 用 CMake 配置并构建 cedarcpp 库（header-only，安装头文件 + CMake 配置）
#   2. 安装到 install/ 目录
#   3. 用已安装的库构建 test/ 下的单元测试
#   4. 运行 ctest 执行全部测试
#
# 目录职责:
#   build/       — CMake 构建中间文件 + 可执行文件（保留，下次增量构建更快）
#   install/     — 最终产物：头文件 + CMake 包配置
#   logs/        — 运行时日志（如 cedar::print 的输出日志）
#
# 注意:
#   - 第二次运行 ./build.sh 时，CMake 会自动跳过未变更的步骤（增量构建）
#   - 如需完全清理，执行 ./build.sh clean
#
# 依赖:
#   - CMake >= 3.5
#   - C++17 编译器（g++ / clang++）
#   - （可选）OpenCV + xtensor — 仅在使用 cedar/image.hpp 时需要
# ============================================================================

# ---- 安全选项 ----
# -e: 任何命令失败立即退出
# -u: 使用未定义变量时报错
# -o pipefail: pipeline 中任一命令失败即算失败
set -euo pipefail

# ---- 路径定义 ----
# 脚本所在目录（项目根目录）
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 构建目录：存放所有编译中间文件和可执行文件
# 生成自 cmake -B + cmake --build
BUILD_DIR="$ROOT_DIR/build"

# 安装目录：最终产物（头文件 + CMake 包配置）
# 生成自 cmake --build --target install
# 结构：install/include/cedar/ + install/lib/cmake/cedar/
INSTALL_DIR="$ROOT_DIR/install"

# 日志目录：运行时日志输出（如 cedar::print 的日志文件）
# 设置 PRINT_LOG_PATH 环境变量可指定日志路径
LOGS_DIR="$ROOT_DIR/logs"

# ============================================================================
# clean 模式
# ============================================================================
# 清理 build（中间文件）、install（最终产物）、logs（运行时日志），
# 以及旧的 output/（旧版产物目录，兼容清理）
if [ "${1:-}" = "clean" ]; then
    echo "==> 清理构建产物..."
    rm -rf "$BUILD_DIR" "$INSTALL_DIR" "$LOGS_DIR" "$ROOT_DIR/output"
    echo "    done"
    exit 0
fi

# ============================================================================
# Step 1: 构建并安装库
# ============================================================================
echo "==> 配置 & 构建库..."

# -S: 源码目录（项目根，包含 CMakeLists.txt）
# -B: 构建目录（所有编译产物，不污染源码树）
# -DCMAKE_INSTALL_PREFIX: 指定安装根目录
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

# --build: 编译并执行 install target
# 对于 header-only 库，实际工作是将头文件拷贝到安装目录，
# 并生成 CMake 包配置文件（cedarConfig.cmake 等）
cmake --build "$BUILD_DIR" --target install

echo "    库已安装到 $INSTALL_DIR"

# ============================================================================
# Step 2: 构建并运行测试
# ============================================================================
echo "==> 构建 & 运行测试..."

# test/CMakeLists.txt 通过 find_package(cedar CONFIG) 找到已安装的库
# -DCMAKE_PREFIX_PATH: 告诉 CMake 在哪里搜索已安装的包
cmake -S "$ROOT_DIR/test" -B "$BUILD_DIR/test" \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR"

# 编译测试可执行文件
cmake --build "$BUILD_DIR/test"

# 将运行时日志定向到 logs/ 目录，不污染 /tmp/
# --test-dir: 指定测试目录
# --output-on-failure: 测试失败时显示详细输出
PRINT_LOG_PATH="$LOGS_DIR/print.log" \
ctest --test-dir "$BUILD_DIR/test" --output-on-failure

# ============================================================================
# 完成
# ============================================================================
echo "==> 全部完成"
echo "    库文件:  $INSTALL_DIR/include/"
echo "    配置:    $INSTALL_DIR/lib/cmake/cedar/"
echo "    日志:    $LOGS_DIR/"
echo "    再次运行 ./build.sh 走增量构建，速度更快"

/home/coder/data/Github/cedarcpp/build/test/test_sprint