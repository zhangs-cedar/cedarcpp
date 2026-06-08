# 验收标准

## P0 基础验收

```bash
python3 scripts/generate_sample_data.py
cmake -S . -B build -G Ninja -DNG_BUILD_OPENCV=ON
cmake --build build -j
```

必须能运行：

```bash
./build/00_linux_cpp_cmake/00_scan_files --input data/images
./build/01_opencv_preprocess/01_opencv_preprocess --input data/images --output out/01_preprocess
./build/02_roi_measure/02_roi_measure --input data/images --output out/02_roi
./build/04_infer_interface_stub/04_infer_interface_stub --backend cpu
./build/05_multithread_pipeline/05_multithread_pipeline --input data/images --output out/05_pipeline --repeat 1000
./build/07_3d_measurement/07_plane_fit --input data/pointcloud/plane.xyz --output out/07_plane.json
```

## 功能验收

| 模块 | 输出 |
|---|---|
| 01 | `out/01_preprocess/metrics.csv` 和中间图 |
| 02 | `out/02_roi/json/*.json` 和 `vis/*.png` |
| 04 | CPU backend 返回结果；TRT stub 明确返回不可用 |
| 05 | `pipeline_metrics.csv`，吞吐统计 |
| 07 | `out/07_plane.json` |

## 性能验收

- Pipeline 能处理 1000 张以上模拟帧。
- 队列有界，不应随运行时间无限增长。
- 退出后所有线程 join。

## 工程验收

- 所有模块由 CMake 管理。
- ONNXRuntime 和 pybind11 是可选模块。
- 没有 GPU 也能完成基础学习闭环。
