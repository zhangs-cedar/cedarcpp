# 04_infer_interface_stub

目标：没有显卡时先学 TensorRT 前的接口抽象、buffer 思想和 backend 解耦。

这个模块不依赖 TensorRT，也不依赖 OpenCV。

运行：

```bash
./build/04_infer_interface_stub/04_infer_interface_stub --backend cpu
./build/04_infer_interface_stub/04_infer_interface_stub --backend trt
```

`--backend trt` 会返回明确的不可用状态，而不是假装推理成功。
