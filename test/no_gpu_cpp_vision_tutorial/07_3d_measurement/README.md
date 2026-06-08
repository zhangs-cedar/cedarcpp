# 07_3d_measurement

目标：无 GPU 也可以学 3D 基础。

## 点云平面拟合

```bash
./build/07_3d_measurement/07_plane_fit --input data/pointcloud/plane.xyz --output out/07_plane.json
```

输入格式：

```text
x y z
```

输出：

- 拟合平面 `z = ax + by + c`
- 点到平面的最大残差
- 平均绝对残差
- 点数

## Stereo Depth，可选 OpenCV

```bash
./build/07_3d_measurement/07_stereo_depth --left data/stereo/left.png --right data/stereo/right.png --output out/07_stereo
```

训练目标：

- 左右图读取
- 灰度化
- StereoBM
- 视差归一化可视化
