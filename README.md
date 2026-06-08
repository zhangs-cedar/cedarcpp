# cedarcpp

C++ header-only 库，提供彩色打印（`cedar::print`）和图像处理工具函数。

## 目录结构

```
cedarcpp/
├── include/cedar/    # 头文件（print.hpp, image.hpp）
├── test/             # 单元测试
├── tutorial/         # 教程项目
├── cmake/            # CMake 配置模板
├── docs/             # 设计文档
├── build.sh          # 一键构建脚本
└── output/           # 构建产物（git ignored）
```

## 构建与测试

```bash
# 一键构建库 + 运行测试
./build.sh

# 清理构建产物
./build.sh clean
```

构建产物安装在 `output/install` 下，不污染源码树。

## 手动编译

```bash
cmake -B output/build -DCMAKE_INSTALL_PREFIX=output/install
cmake --build output/build --target install
```