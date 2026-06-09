# 设置最低版本要求
cmake_minimum_required(VERSION 3.5)

# 设置项目名称与版本
project(cedar VERSION 0.1.0)

# 设置包含目录路径
set(CEDAR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# build
# 设置头文件列表
set(CEDAR_HEADERS
    ${CEDAR_INCLUDE_DIR}/cedar/image.hpp
    ${CEDAR_INCLUDE_DIR}/cedar/print.hpp
    ${CEDAR_INCLUDE_DIR}/cedar/print_detail.hpp
)

# install 路径配置（必须在使用 INSTALL_INTERFACE 之前）
include(GNUInstallDirs)

# 创建库目标（header-only）
add_library(cedar INTERFACE)

# 将库的头文件目录添加到编译选项中
target_include_directories(cedar INTERFACE
    $<BUILD_INTERFACE:${CEDAR_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# 设置公共头文件属性
set_target_properties(cedar
    PROPERTIES
    PUBLIC_HEADER "${CEDAR_HEADERS}"
)

# 设置编译特性
target_compile_features(cedar INTERFACE cxx_std_17)

# install 配置
include(CMakePackageConfigHelpers)

# 安装头文件
install(TARGETS cedar
        EXPORT ${PROJECT_NAME}-targets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

# 生成并安装 CMake Config 文件
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cedarConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cedarConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cedar
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cedarConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# 安装导出目标文件
install(EXPORT ${PROJECT_NAME}-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cedar
)

# 安装 Config 和 Version 文件
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cedarConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cedarConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cedar
)
