#ifndef CEDAR_IMAGE_HPP
#define CEDAR_IMAGE_HPP

#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

namespace cedar {

// 加载图像并检查是否成功加载
cv::Mat loadAndCheckImage(const std::string &filename)
{
    cv::Mat image = cv::imread(filename);
    if (image.empty())
    {
        std::cerr << "无法加载图像: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return image;
}

// 交换图像通道
cv::Mat swapChannels(const cv::Mat &image)
{
    cv::Mat swapped_image;
    cv::cvtColor(image, swapped_image, cv::COLOR_BGR2RGB);
    return swapped_image;
}

// 灰度化
cv::Mat BGR2GRAY(const cv::Mat &image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// 二值化
cv::Mat thresholding(const cv::Mat &image, int threshold_value)
{
    cv::Mat binary_image;
    cv::threshold(image, binary_image, threshold_value, 255, cv::THRESH_BINARY);
    return binary_image;
}

// 二值化（Otsu 方法）
cv::Mat thresholdingOtsu(const cv::Mat &image)
{
    cv::Mat binary_image;
    cv::threshold(image, binary_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binary_image;
}

// 使用OpenCV函数实现平均池化
cv::Mat averagePooling(const cv::Mat &img, int pool_size)
{
    int height = img.rows;         // 图像高度
    int width = img.cols;          // 图像宽度
    int channels = img.channels(); // 图像通道数

    // 准备输出图像
    cv::Mat out = cv::Mat::zeros(height / pool_size, width / pool_size, img.type());

    // 对输入图像进行遍历，以池化窗口大小为步长进行池化操作
    for (int y = 0; y < height; y += pool_size)
    {
        for (int x = 0; x < width; x += pool_size)
        {
            for (int c = 0; c < channels; c++)
            {
                // 计算池化窗口内像素值的平均值
                cv::Rect roi(x, y, pool_size, pool_size);
                cv::Scalar mean = cv::mean(img(roi));
                // 将平均值赋值给输出图像中相应位置的像素
                out.at<cv::Vec3b>(y / pool_size, x / pool_size)[c] = mean.val[c];
            }
        }
    }

    return out;
}

// 显示图像
void showImage(const std::string &window_name, const cv::Mat &image)
{
    cv::imshow(window_name, image);
    cv::waitKey(0);
}

// 保存图像
void saveImage(const std::string &filename, const cv::Mat &image)
{
    cv::imwrite(filename, image);
}

}  // namespace cedar

#endif