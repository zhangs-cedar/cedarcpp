#ifndef CEDAR_IMAGE_HPP
#define CEDAR_IMAGE_HPP

// ============================================================
// cedar::image  —  图像加载、处理与保存工具函数
//
// 用法:
//   #include <cedar/image.hpp>
//   auto img = cedar::loadImage("photo.jpg");
//   auto gray = cedar::toGray(img);
//
// 依赖:
//   - OpenCV >= 4.x
//
// 功能:
//   - loadImage:       加载图像，失败抛异常（而非 exit）
//   - bgrToRgb:        BGR → RGB 通道交换
//   - toGray:          彩色 → 灰度
//   - thresholding:    二值化（手动阈值 / Otsu）
//   - averagePooling:  平均池化（向量化，cv::resize INTER_AREA）
//   - showImage:       显示图像窗口
//   - saveImage:       保存图像到文件
//
// 相当于 Python 的:
//   import cv2
//   img = cv2.imread(path)
//   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
//   ...
// ============================================================

#include <string>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace cedar {

// ============================================================
// 内部实现
// ============================================================
namespace detail {

// ---------- average_pooling ----------
// 平均池化：将图像缩小为原来的 1/pool_size
//
// 用 cv::resize(INTER_AREA) 代替手写三重循环。
// INTER_AREA 在缩小图像时做像素区域平均，等价于 average pooling。
//
// Python 版的话相当于:
//   def average_pooling(img, pool_size):
//       h, w = img.shape[:2]
//       return cv2.resize(img, (w // pool_size, h // pool_size),
//                         interpolation=cv2.INTER_AREA)
inline cv::Mat average_pooling(const cv::Mat& img, int pool_size) {
    int h = img.rows;
    int w = img.cols;
    cv::Mat dst;
    cv::resize(img, dst, cv::Size(w / pool_size, h / pool_size),
               0, 0, cv::INTER_AREA);
    return dst;
}

}  // namespace detail


// ============================================================
// 公共 API
// ============================================================

/// 加载图像，失败时抛出 std::runtime_error
///
/// 相当于 Python 的 cv2.imread()，但失败时抛异常而非返回 None:
///   img = cv2.imread(path)
///   if img is None: raise RuntimeError(f"无法加载图像: {path}")
///
/// @param filename  图像文件路径
/// @return          加载成功的 cv::Mat
/// @throws std::runtime_error  图像加载失败
inline cv::Mat loadImage(const std::string& filename) {
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        throw std::runtime_error("无法加载图像: " + filename);
    }
    return image;
}


/// BGR → RGB 通道交换
///
/// OpenCV 默认以 BGR 顺序加载图像，需转为 RGB 才能用 matplotlib 等库显示。
/// 相当于 cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
///
/// @param image  输入图像（BGR 顺序）
/// @return       通道交换后的图像（RGB 顺序）
inline cv::Mat bgrToRgb(const cv::Mat& image) {
    cv::Mat out;
    cv::cvtColor(image, out, cv::COLOR_BGR2RGB);
    return out;
}


/// 彩色 → 灰度图
///
/// 相当于 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
///
/// @param image  输入彩色图像
/// @return       灰度图像
inline cv::Mat toGray(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}


/// 手动阈值二值化
///
/// 相当于 cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
///
/// @param image           输入图像（灰度图）
/// @param threshold_value 阈值（0~255）
/// @return                二值图像
inline cv::Mat thresholding(const cv::Mat& image, int threshold_value) {
    cv::Mat binary;
    cv::threshold(image, binary, threshold_value, 255, cv::THRESH_BINARY);
    return binary;
}


/// Otsu 自动阈值二值化
///
/// 自动计算最佳阈值，适合双峰直方图的图像。
/// 相当于 cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
///
/// @param image  输入图像（灰度图）
/// @return       二值图像
inline cv::Mat thresholdingOtsu(const cv::Mat& image) {
    cv::Mat binary;
    cv::threshold(image, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binary;
}


/// 平均池化（下采样）
///
/// 将图像缩小为原来的 1/pool_size，每个输出像素是 pool_size×pool_size
/// 窗口内像素的平均值。使用 OpenCV 的 INTER_AREA 插值实现（向量化，
/// 比手写三重循环快 10~100 倍）。
///
/// 相当于:
///   cv2.resize(img, (w // pool_size, h // pool_size),
///              interpolation=cv2.INTER_AREA)
///
/// @param img        输入图像
/// @param pool_size  池化窗口大小（>= 1，需能被图像宽高整除）
/// @return           池化后的图像
inline cv::Mat averagePooling(const cv::Mat& img, int pool_size) {
    return detail::average_pooling(img, pool_size);
}


/// 显示图像（阻塞等待按键）
///
/// 相当于 cv2.imshow() + cv2.waitKey(0)
///
/// @param window_name  窗口标题
/// @param image        待显示的图像
inline void showImage(const std::string& window_name, const cv::Mat& image) {
    cv::imshow(window_name, image);
    cv::waitKey(0);
}


/// 保存图像到文件
///
/// 相当于 cv2.imwrite(filename, img)
///
/// @param filename  保存路径（格式由扩展名决定: .jpg, .png 等）
/// @param image     待保存的图像
inline void saveImage(const std::string& filename, const cv::Mat& image) {
    cv::imwrite(filename, image);
}

}  // namespace cedar

#endif  // CEDAR_IMAGE_HPP
