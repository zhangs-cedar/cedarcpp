# 03_onnxruntime_cpu_infer

目标：没有显卡时，用 ONNXRuntime CPU 替代 TensorRT，把“前处理 → 推理 → 后处理”闭环跑通。

## 编译

```bash
cmake -S . -B build-ort -G Ninja \
  -DNG_BUILD_OPENCV=ON \
  -DNG_BUILD_ORT=ON \
  -DONNXRUNTIME_ROOT=$PWD/external/onnxruntime-linux-x64-1.xx.x
cmake --build build-ort -j
```

## 运行

```bash
./build-ort/03_onnxruntime_cpu_infer/03_ort_cpu_infer \
  --model data/models/tiny_classifier.onnx \
  --image data/images/part_001.png
```

## 注意

- 该模块需要 ONNXRuntime C++ 库。
- 没有 GPU 也能跑。
- 训练目标是接口、前处理一致性、输出解析和工程闭环。
