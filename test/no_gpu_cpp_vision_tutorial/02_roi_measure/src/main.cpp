#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ng/common/cli.hpp"
#include "ng/common/fs.hpp"
#include "ng/common/timer.hpp"

namespace fs = std::filesystem;

struct Defect {
    int id = 0;
    cv::Rect bbox;
    double area = 0.0;
    double angle = 0.0;
    cv::RotatedRect rotated;
};

static cv::Mat gray_image(const cv::Mat& img) {
    if (img.channels() == 1) return img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

static std::vector<Defect> measure_defects(const cv::Mat& img, int threshold_value, double min_area) {
    cv::Mat gray = gray_image(img);
    cv::Mat blur;
    cv::medianBlur(gray, blur, 3);
    cv::Mat binary;
    cv::threshold(blur, binary, threshold_value, 255, cv::THRESH_BINARY_INV);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Defect> defects;
    int id = 0;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < min_area) continue;
        Defect d;
        d.id = id++;
        d.bbox = cv::boundingRect(c);
        d.area = area;
        if (c.size() >= 5) {
            d.rotated = cv::minAreaRect(c);
            d.angle = d.rotated.angle;
        } else {
            d.rotated = cv::RotatedRect();
            d.angle = 0.0;
        }
        defects.push_back(d);
    }
    return defects;
}

static void write_json(const fs::path& path, const std::string& image_name, const std::vector<Defect>& defects, double cost_ms) {
    std::ofstream os(path);
    os << std::fixed << std::setprecision(3);
    os << "{\n";
    os << "  \"image\": \"" << image_name << "\",\n";
    os << "  \"cost_ms\": " << cost_ms << ",\n";
    os << "  \"defects\": [\n";
    for (size_t i = 0; i < defects.size(); ++i) {
        const auto& d = defects[i];
        os << "    {\"id\": " << d.id
           << ", \"bbox\": [" << d.bbox.x << ", " << d.bbox.y << ", " << d.bbox.width << ", " << d.bbox.height << "]"
           << ", \"area\": " << d.area
           << ", \"angle\": " << d.angle << "}";
        if (i + 1 != defects.size()) os << ",";
        os << "\n";
    }
    os << "  ]\n";
    os << "}\n";
}

static void draw_defects(cv::Mat& vis, const std::vector<Defect>& defects) {
    for (const auto& d : defects) {
        cv::rectangle(vis, d.bbox, cv::Scalar(0, 255, 0), 2);
        cv::putText(vis, "id=" + std::to_string(d.id) + " area=" + std::to_string(static_cast<int>(d.area)),
                    cv::Point(d.bbox.x, std::max(0, d.bbox.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
    }
}

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const fs::path input_dir = args.get("input", "data/images");
    const fs::path output_dir = args.get("output", "out/02_roi");
    const int threshold_value = args.get_int("threshold", 100);
    const double min_area = args.get_double("min-area", 80.0);

    ng::ensure_dir(output_dir / "json");
    ng::ensure_dir(output_dir / "vis");

    const auto files = ng::list_files(input_dir, {".png", ".jpg", ".jpeg", ".bmp"});
    if (files.empty()) {
        std::cerr << "No images found in " << input_dir << "\n";
        return 2;
    }

    for (const auto& file : files) {
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;

        ng::Timer timer;
        auto defects = measure_defects(img, threshold_value, min_area);
        const double cost_ms = timer.elapsed_ms();

        cv::Mat vis = img.clone();
        draw_defects(vis, defects);

        const auto stem = file.stem().string();
        write_json(output_dir / "json" / (stem + ".json"), file.filename().string(), defects, cost_ms);
        cv::imwrite((output_dir / "vis" / file.filename()).string(), vis);

        std::cout << file.filename().string() << " defects=" << defects.size() << " cost_ms=" << cost_ms << "\n";
    }

    return 0;
}
