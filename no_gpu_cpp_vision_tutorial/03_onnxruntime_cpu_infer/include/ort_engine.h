#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct OrtResult {
    int class_id = -1;
    float score = 0.0f;
    std::vector<float> logits;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
};

class OrtCpuEngine {
public:
    explicit OrtCpuEngine(const std::string& model_path, int input_w = 64, int input_h = 64);
    OrtResult infer(const cv::Mat& image);

private:
    std::vector<float> preprocess(const cv::Mat& image);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
    int input_w_;
    int input_h_;
};
