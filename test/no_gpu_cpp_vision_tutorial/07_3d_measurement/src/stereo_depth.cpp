#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ng/common/cli.hpp"
#include "ng/common/fs.hpp"
#include "ng/common/timer.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const fs::path left_path = args.get("left", "data/stereo/left.png");
    const fs::path right_path = args.get("right", "data/stereo/right.png");
    const fs::path output_dir = args.get("output", "out/07_stereo");
    ng::ensure_dir(output_dir);

    cv::Mat left = cv::imread(left_path.string(), cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_path.string(), cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty()) {
        std::cerr << "Cannot read stereo images.\n";
        return 2;
    }
    if (left.size() != right.size()) {
        std::cerr << "Left/right image size mismatch.\n";
        return 3;
    }

    ng::Timer timer;
    int num_disparities = 16 * 4;
    int block_size = 15;
    auto stereo = cv::StereoBM::create(num_disparities, block_size);
    cv::Mat disparity16;
    stereo->compute(left, right, disparity16);

    cv::Mat disp8;
    cv::normalize(disparity16, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite((output_dir / "disparity_vis.png").string(), disp8);

    std::cout << "wrote " << (output_dir / "disparity_vis.png") << " cost_ms=" << timer.elapsed_ms() << "\n";
    return 0;
}
