#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ng/common/cli.hpp"
#include "ng/common/fs.hpp"
#include "ng/common/timer.hpp"

namespace fs = std::filesystem;

static cv::Mat to_gray(const cv::Mat& img) {
    if (img.channels() == 1) return img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const fs::path input_dir = args.get("input", "data/images");
    const fs::path output_dir = args.get("output", "out/01_preprocess");
    const int threshold_value = args.get_int("threshold", 100);

    ng::ensure_dir(output_dir / "gray");
    ng::ensure_dir(output_dir / "blur");
    ng::ensure_dir(output_dir / "binary");
    ng::ensure_dir(output_dir / "morphology");
    ng::ensure_dir(output_dir / "contour_vis");

    const auto files = ng::list_files(input_dir, {".png", ".jpg", ".jpeg", ".bmp"});
    if (files.empty()) {
        std::cerr << "No images found in " << input_dir << "\n";
        return 2;
    }

    std::ofstream csv(output_dir / "metrics.csv");
    csv << "image,width,height,contours,cost_ms\n";

    for (const auto& file : files) {
        ng::Timer timer;
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Skip bad image: " << file << "\n";
            continue;
        }

        cv::Mat gray = to_gray(img);
        cv::Mat blur;
        cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);

        cv::Mat binary;
        cv::threshold(blur, binary, threshold_value, 255, cv::THRESH_BINARY_INV);

        cv::Mat morph;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(binary, morph, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat vis = img.clone();
        for (size_t i = 0; i < contours.size(); ++i) {
            const double area = cv::contourArea(contours[i]);
            if (area < 50.0) continue;
            cv::drawContours(vis, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), 2);
            cv::Rect box = cv::boundingRect(contours[i]);
            cv::rectangle(vis, box, cv::Scalar(0, 255, 0), 1);
        }

        const auto name = file.filename().string();
        cv::imwrite((output_dir / "gray" / name).string(), gray);
        cv::imwrite((output_dir / "blur" / name).string(), blur);
        cv::imwrite((output_dir / "binary" / name).string(), binary);
        cv::imwrite((output_dir / "morphology" / name).string(), morph);
        cv::imwrite((output_dir / "contour_vis" / name).string(), vis);

        const double cost = timer.elapsed_ms();
        csv << name << ',' << img.cols << ',' << img.rows << ',' << contours.size() << ',' << cost << '\n';
        std::cout << "processed " << name << ", cost_ms=" << cost << "\n";
    }

    return 0;
}
