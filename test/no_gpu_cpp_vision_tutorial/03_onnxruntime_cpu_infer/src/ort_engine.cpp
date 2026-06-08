#include "ort_engine.h"
#include <algorithm>
#include <array>
#include <iostream>
#include "ng/common/timer.hpp"

OrtCpuEngine::OrtCpuEngine(const std::string& model_path, int input_w, int input_h)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ng_ort"),
      session_options_(),
      session_(nullptr),
      input_w_(input_w),
      input_h_(input_h) {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

#if ORT_API_VERSION >= 14
    auto input_alloc = session_.GetInputNameAllocated(0, allocator_);
    auto output_alloc = session_.GetOutputNameAllocated(0, allocator_);
    input_name_ = input_alloc.get();
    output_name_ = output_alloc.get();
#else
    input_name_ = session_.GetInputName(0, allocator_);
    output_name_ = session_.GetOutputName(0, allocator_);
#endif
}

std::vector<float> OrtCpuEngine::preprocess(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image;
    }

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(input_w_, input_h_));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> chw(1 * 3 * input_h_ * input_w_);
    const int area = input_h_ * input_w_;
    for (int y = 0; y < input_h_; ++y) {
        for (int x = 0; x < input_w_; ++x) {
            const auto pix = rgb.at<cv::Vec3f>(y, x);
            chw[0 * area + y * input_w_ + x] = pix[0];
            chw[1 * area + y * input_w_ + x] = pix[1];
            chw[2 * area + y * input_w_ + x] = pix[2];
        }
    }
    return chw;
}

OrtResult OrtCpuEngine::infer(const cv::Mat& image) {
    OrtResult result;
    ng::Timer t_pre;
    auto input_data = preprocess(image);
    result.preprocess_ms = t_pre.elapsed_ms();

    std::array<int64_t, 4> input_shape{1, 3, input_h_, input_w_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    ng::Timer t_infer;
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    result.infer_ms = t_infer.elapsed_ms();

    float* out = outputs[0].GetTensorMutableData<float>();
    auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    size_t count = 1;
    for (auto v : shape) count *= static_cast<size_t>(v > 0 ? v : 1);
    result.logits.assign(out, out + count);

    auto it = std::max_element(result.logits.begin(), result.logits.end());
    if (it != result.logits.end()) {
        result.class_id = static_cast<int>(std::distance(result.logits.begin(), it));
        result.score = *it;
    }
    return result;
}
