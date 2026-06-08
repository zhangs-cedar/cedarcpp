# cedarcpp 结构整理设计

> 日期: 2026-06-08
> 目标: 让 cedarcpp 目录结构更清晰简单，编译测试形成为对应 sh 脚本

## 1. 目标与范围

### 做
- 重新组织目录结构，职责分层清晰
- 将 `test/no_gpu_cpp_vision_tutorial/` 移到 `tutorial/`
- 重建 `image.hpp`：放入 `namespace cedar`，去掉全局 `using namespace`
- 创建 `build.sh` 一键构建 + 测试脚本（所有产物在 `output/` 下）
- 清理旧的构建产物
- 更新 `.gitignore` 和 `README.md`

### 不做
- 不添加新功能
- 不修改 `print.hpp` 的行为和接口
- 不修改 `image.hpp` 的函数签名和行为
- 不修改 CMake 的 install/export 机制
- 不修改 `tutorial/` 内部内容
- 不添加任何新依赖

## 2. 最终目录结构

```
cedarcpp/
├── include/
│   └── cedar/
│       ├── print.hpp         # 保持现状
│       └── image.hpp         # 重构: namespace cedar, 显式 cv::/std::
├── test/
│   ├── CMakeLists.txt        # 保持现状
│   └── test_sprint.cpp       # 保持现状
├── tutorial/                 # 从 test/no_gpu_cpp_vision_tutorial/ 移入
│   ├── CMakeLists.txt
│   ├── 00_linux_cpp_cmake/
│   ├── 01_opencv_preprocess/
│   ├── 02_roi_measure/
│   ├── 03_onnxruntime_cpu_infer/
│   ├── 04_infer_interface_stub/
│   ├── 05_multithread_pipeline/
│   ├── 06_pybind11_infer/
│   ├── 07_3d_measurement/
│   ├── common/
│   ├── data/
│   ├── docs/
│   ├── external/
│   ├── out/
│   ├── scripts/
│   ├── build/
│   ├── build-ort/
│   ├── build-py/
│   └── README.md
├── cmake/
│   └── cedarConfig.cmake.in  # 保持现状
├── docs/
│   └── specs/                # 设计文档
├── build.sh                  # 一键构建脚本
├── CMakeLists.txt            # 微调（仅 install 路径已由脚本控制）
├── .gitignore                # 补充忽略规则
├── README.md                 # 更新使用说明
└── LICENSE                   # 保持现状
```

## 3. build.sh 脚本设计

位置: 项目根目录 `/home/coder/data/Github/cedarcpp/build.sh`

```
用法:
  ./build.sh        一键构建库 + 运行测试
  ./build.sh clean  清理所有构建产物
```

流程:
1. 配置 + 构建库 → 安装到 `output/install`
2. 构建测试（链接已安装的库）→ 运行 ctest
3. 所有产物在 `output/` 下，`output/` 已在 `.gitignore` 中

## 4. image.hpp 重构规则

| 当前 | 重构后 |
|------|--------|
| `using namespace cv;` | 去掉，使用 `cv::` 前缀 |
| `using namespace std;` | 去掉，使用 `std::` 前缀 |
| 函数在全局作用域 | 包在 `namespace cedar {}` 内 |
| 宏保护 `CEDAR_ANY_HPP` | `CEDAR_IMAGE_HPP` |
| 函数签名 | 保持完全不变 |
| 行为 | 保持完全不变 |

## 5. .gitignore 补充

在现有基础上增加:
```
output/
.vscode/
.idea/
```

## 6. 不修改的部分

- `print.hpp` — 不变
- `test/test_sprint.cpp` — 不变
- `test/CMakeLists.txt` — 不变
- `cmake/cedarConfig.cmake.in` — 不变
- `tutorial/` 内部所有文件 — 不变
- `CMakeLists.txt` 的构建逻辑 — 不变
- `LICENSE` — 不变

## 7. 风险与注意事项

- `image.hpp` 重构后，已有代码若通过 `using namespace cedar;` 或完全限定名调用则不受影响；若原来依赖全局命名空间中的函数名（如直接写 `loadAndCheckImage(...)` 而非 `cedar::loadAndCheckImage(...)`），需添加 `using namespace cedar;`。当前库没有其他消费者，风险可控。
- `test/output/` 中现有构建产物在 `.gitignore` 中已忽略，但文件仍存在磁盘，需清理。
- `tutorial/` 的独立 CMakeLists.txt 保持不变，不与主 CMake 集成，避免相互影响。
