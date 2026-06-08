#include "infer_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include "ng/common/timer.hpp"

InferResult CpuDummyEngine::infer(const Tensor& input) {
    ng::Timer timer;
    Tensor output;
    output.shape = {1, 2};
    const float sum = std::accumulate(input.data.begin(), input.data.end(), 0.0f);
    const float mean = input.data.empty() ? 0.0f : sum / static_cast<float>(input.data.size());
    output.data = {1.0f - mean, mean};

    InferResult r;
    r.ok = true;
    r.backend = "cpu_dummy";
    r.message = "CPU dummy backend. Replace this with ONNXRuntime CPU or TensorRT later.";
    r.output = std::move(output);
    r.cost_ms = timer.elapsed_ms();
    return r;
}

InferResult TrtGpuEngineStub::infer(const Tensor& input) {
    (void)input;
    InferResult r;
    r.ok = false;
    r.backend = "trt_gpu_stub";
    r.message = "TensorRT backend is intentionally disabled in no-GPU tutorial. Implement this when NVIDIA GPU is available.";
    r.output.shape = {0};
    r.cost_ms = 0.0;
    return r;
}

std::unique_ptr<IInferEngine> create_engine(const std::string& backend) {
    if (backend == "cpu") return std::make_unique<CpuDummyEngine>();
    if (backend == "trt") return std::make_unique<TrtGpuEngineStub>();
    return std::make_unique<CpuDummyEngine>();
}

Tensor make_dummy_input(int c, int h, int w) {
    Tensor t;
    t.shape = {1, c, h, w};
    t.data.resize(static_cast<size_t>(c * h * w));
    for (size_t i = 0; i < t.data.size(); ++i) {
        t.data[i] = static_cast<float>((i % 255) / 255.0);
    }
    return t;
}
