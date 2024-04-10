# 设置最低版本要求
cmake_minimum_required(VERSION 3.5)

# 设置项目名称
project(cedar)

# 设置包含目录路径
set(CEDAR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# build
# 设置头文件列表
set(CEDAR_HEADERS
    ${CEDAR_INCLUDE_DIR}/cedar/image.hpp
)

# 创建库目标
add_library(cedar INTERFACE)

# 将库的头文件目录添加到编译选项中
target_include_directories(cedar INTERFACE
    $<BUILD_INTERFACE:${CEDAR_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# 设置公共头文件属性
set_target_properties(cedar
    PROPERTIES
    PUBLIC_HEADER "${CEDAR_HEADERS}"
)

# 设置编译特性
target_compile_features(cedar INTERFACE cxx_std_14)

# install
# 设置安装规则，包括将库安装到系统目录以及导出 CMake 目标文件
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS cedar
        EXPORT ${PROJECT_NAME}-targets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
