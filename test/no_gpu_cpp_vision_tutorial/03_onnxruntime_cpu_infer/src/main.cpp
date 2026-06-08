#include <iostream>
#include <opencv2/opencv.hpp>
#include "ng/common/cli.hpp"
#include "ort_engine.h"

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const std::string model = args.get("model", "data/models/tiny_classifier.onnx");
    const std::string image_path = args.get("image", "data/images/part_001.png");

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Cannot read image: " << image_path << "\n";
        return 2;
    }

    OrtCpuEngine engine(model);
    const auto result = engine.infer(image);

    std::cout << "class_id=" << result.class_id
              << " score=" << result.score
              << " preprocess_ms=" << result.preprocess_ms
              << " infer_ms=" << result.infer_ms << "\n";
    std::cout << "logits:";
    for (float v : result.logits) std::cout << ' ' << v;
    std::cout << "\n";
    return 0;
}
