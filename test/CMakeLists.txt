cmake_minimum_required(VERSION 3.5)
project(cedar_test LANGUAGES CXX)

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
# add_executable(test_image test_image.cpp)
# target_link_libraries(test_image PRIVATE cedar)
# add_test(NAME image COMMAND test_image)
