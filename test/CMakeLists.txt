cmake_minimum_required(VERSION 3.5)
project(cedar_test LANGUAGES CXX)

# 显式指定 C++ 标准（clangd 需要从 compile_commands.json 读到 -std= 标志）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 通过 find_package 找到已安装的 cedar 库
# 使用时需设置 -DCMAKE_PREFIX_PATH=<install_prefix>
find_package(cedar CONFIG REQUIRED)

# 启用测试
enable_testing()

# ---- s_print 测试 ----
add_executable(test_sprint test_sprint.cpp)
target_link_libraries(test_sprint PRIVATE cedar)
add_test(NAME sprint COMMAND test_sprint)

# ---- image 测试（需要 OpenCV，可选） ----
find_package(OpenCV QUIET COMPONENTS core imgproc highgui imgcodecs)
if(OpenCV_FOUND)
    add_executable(test_image test_image.cpp)
    target_link_libraries(test_image PRIVATE cedar ${OpenCV_LIBS})
    add_test(NAME image COMMAND test_image)
    message(STATUS "OpenCV 已找到: ${OpenCV_DIR}  → 启用 image 测试")
else()
    message(STATUS "OpenCV 未找到 → 跳过 image 测试（不影响 library 编译）")
endif()
