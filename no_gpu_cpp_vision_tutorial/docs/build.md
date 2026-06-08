# 构建说明

## 1. 安装基础依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build gdb pkg-config libopencv-dev python3 python3-pip
```

## 2. 生成样例数据

```bash
python3 scripts/generate_sample_data.py
```

## 3. 编译基础模块

```bash
cmake -S . -B build -G Ninja \
  -DNG_BUILD_OPENCV=ON \
  -DNG_BUILD_ORT=OFF \
  -DNG_BUILD_PYBIND11=OFF
cmake --build build -j
```

## 4. 编译 ONNXRuntime CPU 模块

下载 ONNXRuntime C++ Linux 包后放到 `external/`，例如：

```text
external/onnxruntime-linux-x64-1.17.3/
├── include/
└── lib/
```

然后：

```bash
cmake -S . -B build-ort -G Ninja \
  -DNG_BUILD_OPENCV=ON \
  -DNG_BUILD_ORT=ON \
  -DONNXRUNTIME_ROOT=$PWD/external/onnxruntime-linux-x64-1.17.3
cmake --build build-ort -j
```

运行时如找不到库：

```bash
export LD_LIBRARY_PATH=$PWD/external/onnxruntime-linux-x64-1.17.3/lib:$LD_LIBRARY_PATH
```

## 5. 编译 pybind11

```bash
sudo apt install -y pybind11-dev
cmake -S . -B build-py -G Ninja -DNG_BUILD_PYBIND11=ON
cmake --build build-py -j
```
