# 无显卡 Linux 学习计划

## 核心判断

没有显卡时，不要把 TensorRT 当第一阶段目标。

正确顺序：

```text
C++ / CMake / Linux
→ OpenCV C++ 工业图像处理
→ ROI 几何测量
→ 多线程 Pipeline
→ ONNXRuntime CPU 推理闭环
→ pybind11 封装
→ 3D 点云 / calib3d
→ 有 NVIDIA GPU 后补 TensorRT
```

## 每阶段验收

| 阶段 | 验收 |
|---|---|
| C++ / CMake | 能独立新建 target，理解 include/link/install |
| OpenCV | 能批量处理图片，输出中间图和 CSV |
| ROI 测量 | 能输出 bbox/area/angle/JSON |
| Pipeline | 连续 10000 帧不死锁，不内存爆 |
| ONNXRuntime CPU | 能用 CPU 跑 ONNX，输出与 Python 对齐 |
| pybind11 | Python 能调用 C++ 粗粒度接口 |
| 3D | 能拟合平面，解释误差 |
| TensorRT | 有 GPU 后再做 engine、buffer、stream、H2D/D2H |
```
