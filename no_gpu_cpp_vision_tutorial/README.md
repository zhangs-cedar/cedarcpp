# no_gpu_cpp_vision_tutorial

无显卡 Linux 版工业视觉 C++ 工程化教程代码。

这套仓库的目标不是跑 TensorRT 性能，而是在没有 NVIDIA GPU 的 Linux 上先完成工业视觉工程闭环：

```text
P0: C++ / CMake / OpenCV C++ / 多线程 / pybind11 / Linux 工程化
P1: ONNXRuntime CPU 替代 TensorRT 做推理闭环
P2: TensorRT 先学接口、工程结构、buffer 思想，等有显卡再真实跑
P3: 3D 基础、OpenCV calib3d、点云处理继续学
```

## 仓库结构

```text
00_linux_cpp_cmake/        纯 C++ / CMake / 文件扫描 / 耗时统计
01_opencv_preprocess/      OpenCV C++ 批量预处理
02_roi_measure/            OpenCV C++ ROI / 缺陷几何测量
03_onnxruntime_cpu_infer/  ONNXRuntime CPU 推理闭环，可选编译
04_infer_interface_stub/   TensorRT 前的接口抽象与 GPU Stub
05_multithread_pipeline/   无 GPU 多线程 Pipeline，可直接跑
06_pybind11_infer/         pybind11 封装 C++ 模块，可选编译
07_3d_measurement/         点云平面拟合 + 可选 OpenCV stereo depth
common/                    公共头文件
scripts/                   生成样例数据、导出 ONNX、检查结果
external/                  第三方依赖放置说明，不提交大依赖
```

## 推荐环境

Ubuntu 22.04 / 24.04。

基础依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build gdb pkg-config libopencv-dev python3 python3-pip
```

可选依赖：

```bash
sudo apt install -y pybind11-dev
# ONNXRuntime C++ 库建议从官方 release 下载，放到 external/onnxruntime-linux-x64-*/
```

## 生成样例数据

```bash
python3 scripts/generate_sample_data.py
```

生成：

```text
data/images/*.png
数据用于 OpenCV 预处理、ROI 测量、Pipeline

data/stereo/left.png, right.png
数据用于 stereo depth demo

data/pointcloud/plane.xyz
数据用于点云平面拟合
```

## 编译基础模块

```bash
cmake -S . -B build -G Ninja \
  -DNG_BUILD_OPENCV=ON \
  -DNG_BUILD_ORT=OFF \
  -DNG_BUILD_PYBIND11=OFF
cmake --build build -j
```

## 运行基础模块

```bash
./build/00_linux_cpp_cmake/00_scan_files --input data/images
./build/01_opencv_preprocess/01_opencv_preprocess --input data/images --output out/01_preprocess
./build/02_roi_measure/02_roi_measure --input data/images --output out/02_roi
./build/04_infer_interface_stub/04_infer_interface_stub --image data/images/part_001.png --backend cpu
./build/05_multithread_pipeline/05_multithread_pipeline --input data/images --output out/05_pipeline --repeat 200
./build/07_3d_measurement/07_plane_fit --input data/pointcloud/plane.xyz --output out/07_plane.json
./build/07_3d_measurement/07_stereo_depth --left data/stereo/left.png --right data/stereo/right.png --output out/07_stereo
```

## 编译 ONNXRuntime CPU 模块

设置 ONNXRuntime 根目录：

```bash
cmake -S . -B build-ort -G Ninja \
  -DNG_BUILD_OPENCV=ON \
  -DNG_BUILD_ORT=ON \
  -DONNXRUNTIME_ROOT=$PWD/external/onnxruntime-linux-x64-1.xx.x
cmake --build build-ort -j
```

运行：

```bash
./build-ort/03_onnxruntime_cpu_infer/03_ort_cpu_infer \
  --model data/models/tiny_classifier.onnx \
  --image data/images/part_001.png
```

如果没有 ONNX 模型，可以先用脚本导出一个最小模型：

```bash
python3 scripts/export_tiny_onnx.py
```

该脚本需要 `torch`。没有 torch 时，可以跳过 ONNXRuntime 案例，不影响其他模块。

## 编译 pybind11 模块

```bash
cmake -S . -B build-py -G Ninja -DNG_BUILD_PYBIND11=ON
cmake --build build-py -j
PYTHONPATH=build-py/06_pybind11_infer python3 scripts/test_pybind11.py
```

## 当前仓库的设计边界

- 不要求 NVIDIA GPU。
- 不直接编译 TensorRT。
- TensorRT 相关只做接口抽象和 Stub，目的是提前训练工程结构。
- 真正有 GPU 后，把 `IInferEngine` 的 CPU 实现替换成 TensorRT 实现即可。
- OpenCV、ONNXRuntime、pybind11 都是可独立学习模块。

## 验收标准

详见：

```text
docs/acceptance.md
```
