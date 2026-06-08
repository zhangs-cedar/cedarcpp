#pragma once
#include <memory>
#include <string>
#include <vector>

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct InferResult {
    bool ok = false;
    std::string backend;
    std::string message;
    Tensor output;
    double cost_ms = 0.0;
};

class IInferEngine {
public:
    virtual ~IInferEngine() = default;
    virtual InferResult infer(const Tensor& input) = 0;
};

class CpuDummyEngine final : public IInferEngine {
public:
    InferResult infer(const Tensor& input) override;
};

class TrtGpuEngineStub final : public IInferEngine {
public:
    InferResult infer(const Tensor& input) override;
};

std::unique_ptr<IInferEngine> create_engine(const std::string& backend);
Tensor make_dummy_input(int c, int h, int w);
