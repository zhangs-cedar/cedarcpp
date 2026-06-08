# 构建验证记录

生成时间：2026-06-08

## 当前沙箱验证结果

沙箱环境缺少 OpenCV C++ 开发包，因此 `-DNG_BUILD_OPENCV=ON` 未能在沙箱内完成配置。报错属于环境缺依赖：

```text
Could not find a package configuration file provided by "OpenCV"
```

已完成验证的无 OpenCV / 无 GPU 基础模块：

```bash
cmake -S . -B build-basic -G Ninja \
  -DNG_BUILD_OPENCV=OFF \
  -DNG_BUILD_ORT=OFF \
  -DNG_BUILD_PYBIND11=OFF
cmake --build build-basic -j
```

验证通过模块：

```text
00_linux_cpp_cmake
04_infer_interface_stub
05_multithread_pipeline
07_plane_fit
```

运行验证：

```bash
./build-basic/00_linux_cpp_cmake/00_scan_files --input data/images
./build-basic/04_infer_interface_stub/04_infer_interface_stub --backend cpu
./build-basic/05_multithread_pipeline/05_multithread_pipeline --input data/images --output out/05_pipeline --repeat 50
./build-basic/07_3d_measurement/07_plane_fit --input data/pointcloud/plane.xyz --output out/07_plane.json
```

## 在你的 Linux 上完整验证

安装 OpenCV 后：

```bash
sudo apt update
sudo apt install -y libopencv-dev pkg-config
cmake -S . -B build -G Ninja -DNG_BUILD_OPENCV=ON
cmake --build build -j
bash scripts/run_basic_acceptance.sh
```
