# 01_opencv_preprocess

目标：批量读取工业图，输出灰度、滤波、二值化、形态学和轮廓可视化。

运行：

```bash
./build/01_opencv_preprocess/01_opencv_preprocess --input data/images --output out/01_preprocess
```

核心训练：

- `cv::Mat` 生命周期
- 图像 IO
- ROI / threshold / morphology / contours
- 单张耗时统计
- CSV 输出
