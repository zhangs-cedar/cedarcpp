# 排障说明

## 找不到 OpenCV

```bash
sudo apt install -y libopencv-dev pkg-config
pkg-config --modversion opencv4
```

重新构建：

```bash
rm -rf build
cmake -S . -B build -G Ninja -DNG_BUILD_OPENCV=ON
```

## ONNXRuntime 找不到头文件或库

确认目录结构：

```bash
ls external/onnxruntime-linux-x64-*/include/onnxruntime_cxx_api.h
ls external/onnxruntime-linux-x64-*/lib/libonnxruntime.so
```

构建时传入：

```bash
-DONNXRUNTIME_ROOT=$PWD/external/onnxruntime-linux-x64-1.xx.x
```

运行时：

```bash
export LD_LIBRARY_PATH=$PWD/external/onnxruntime-linux-x64-1.xx.x/lib:$LD_LIBRARY_PATH
```

## pybind11 找不到

```bash
sudo apt install -y pybind11-dev python3-dev
```

或：

```bash
python3 -m pip install pybind11
cmake -S . -B build-py -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DNG_BUILD_PYBIND11=ON
```

## StereoBM 输出很差

本仓库里的 stereo 图是合成测试图，只用于验证 API 链路。
真实 3D 深度恢复需要：

- 相机标定
- 极线校正
- 合适纹理
- 合理基线
- 视差范围设置

## TensorRT 为什么没有真实代码

因为本仓库目标是无显卡 Linux 学习。
没有 NVIDIA GPU 时，真实 TensorRT 性能、显存、stream、H2D/D2H 都无法验证。
因此这里只保留接口抽象与 Stub，避免制造虚假掌握感。
