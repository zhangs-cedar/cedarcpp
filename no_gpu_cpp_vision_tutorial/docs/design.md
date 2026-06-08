# 设计说明

## 设计原则

1. 没有 GPU，也必须能学到工业视觉工程主干。
2. 推理后端必须可替换。
3. TensorRT 不可用时，不假装完成 TensorRT。
4. Python 不控制底层细节，只调用 C++ 粗粒度接口。
5. 每个模块都要有可运行命令、输出文件和验收标准。

## 后端抽象

```cpp
class IInferEngine {
public:
    virtual ~IInferEngine() = default;
    virtual InferResult infer(const Tensor& input) = 0;
};
```

当前阶段：

```text
CpuDummyEngine / OrtCpuEngine
```

未来有 GPU 后：

```text
TrtGpuEngine
```

替换点只在 engine 层，不应该污染业务后处理和 Pipeline。

## Pipeline 设计

```text
Reader -> Preprocess -> Infer -> Postprocess -> Writer
```

核心约束：

- 队列有界
- close 机制明确
- 每帧有 id
- 每阶段有耗时
- 区分 throughput 和 latency

## 3D 设计

先做工程可解释的 3D：

- 点云读取
- 平面拟合
- 残差统计
- stereo disparity 可视化

不要一开始卷 SLAM / NeRF / Gaussian Splatting。
