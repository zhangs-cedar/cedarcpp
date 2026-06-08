# 02_roi_measure

目标：从合成工业图中检测暗色缺陷，输出 bbox、面积、角度和可视化图。

运行：

```bash
./build/02_roi_measure/02_roi_measure --input data/images --output out/02_roi --min-area 80
```

验收：

- 每张图输出 JSON
- 每张图输出缺陷可视化
- bbox、area、angle 可解释
