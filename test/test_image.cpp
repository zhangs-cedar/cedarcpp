// ============================================================
// cedar::image 编译与基本功能验证
//
// 验证 cedar::image 提供的各个函数能正常编译和运行。
// 不依赖外部图像文件，用纯色图像测试管道。
// ============================================================

#include <cedar/image.hpp>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

// ---- 测试统计 ----
int tested = 0;
int failed = 0;

void verify(const string& desc, bool ok) {
    ++tested;
    if (ok) {
        cout << "    ✓ " << desc << endl;
    } else {
        cerr << "    ✗ " << desc << "  → 失败" << endl;
        ++failed;
    }
}

void print_separator() {
    cout << "\n----------------------------------------------" << endl;
}

// ============================================================
int main()
{
    cout << "\n";
    cout << "  cedar::image  功能验证" << endl;
    cout << "  ============================================\n" << endl;

    // ============================================================
    // 场景 1: 函数编译与基本调用
    // ============================================================
    cout << "  [场景 1/3]  函数编译与基本调用" << endl;
    cout << "  说明: 验证各函数能正常编译并返回正确类型\n" << endl;

    // 创建一个 100x100 的彩色图像用于测试
    cv::Mat test_img(100, 100, CV_8UC3, cv::Scalar(64, 128, 192));

    // loadImage 测试（不存在的文件 → 应抛异常）
    cout << "  >> loadImage 异常测试:" << endl;
    try {
        cedar::loadImage("/tmp/nonexistent_image_for_test.jpg");
        verify("loadImage 应抛异常", false);
    } catch (const std::exception& e) {
        verify("loadImage 抛异常: " + string(e.what()), true);
    }

    // bgrToRgb
    cout << "  >> bgrToRgb:" << endl;
    cv::Mat rgb = cedar::bgrToRgb(test_img);
    verify("bgrToRgb 返回非空", !rgb.empty());
    verify("bgrToRgb 尺寸不变",
           rgb.rows == test_img.rows && rgb.cols == test_img.cols);

    // toGray
    cout << "  >> toGray:" << endl;
    cv::Mat gray = cedar::toGray(test_img);
    verify("toGray 返回非空", !gray.empty());
    verify("toGray 通道数 = 1", gray.channels() == 1);

    // thresholding
    cout << "  >> thresholding:" << endl;
    cv::Mat bin = cedar::thresholding(gray, 100);
    verify("thresholding 返回非空", !bin.empty());
    verify("thresholding 通道数 = 1", bin.channels() == 1);

    // thresholdingOtsu
    cout << "  >> thresholdingOtsu:" << endl;
    cv::Mat bin_otsu = cedar::thresholdingOtsu(gray);
    verify("thresholdingOtsu 返回非空", !bin_otsu.empty());

    // averagePooling
    cout << "  >> averagePooling:" << endl;
    cv::Mat pooled = cedar::averagePooling(test_img, 10);
    verify("averagePooling 返回非空", !pooled.empty());
    verify("averagePooling 尺寸 = 10x10",
           pooled.rows == 10 && pooled.cols == 10);

    print_separator();

    // ============================================================
    // 场景 2: 保存与加载
    // ============================================================
    cout << "  [场景 2/3]  保存与加载" << endl;
    cout << "  说明: 验证 saveImage + loadImage 的往返一致性\n" << endl;

    const string tmp_img = "/tmp/cedar_test_save.png";

    cout << "  >> saveImage:" << endl;
    cedar::saveImage(tmp_img, test_img);
    ifstream f(tmp_img);
    verify("文件 " + tmp_img + " 已保存", f.good());
    f.close();

    cout << "  >> loadImage:" << endl;
    cv::Mat loaded = cedar::loadImage(tmp_img);
    verify("加载的图像非空", !loaded.empty());
    verify("加载的图像尺寸一致",
           loaded.rows == test_img.rows && loaded.cols == test_img.cols);

    print_separator();

    // ============================================================
    // 场景 3: showImage（只验证编译，不实际显示）
    // ============================================================
    cout << "  [场景 3/3]  showImage（编译验证）" << endl;
    cout << "  说明: showImage 在无显示环境的 CI 中跳过运行，" << endl;
    cout << "       仅验证函数签名能正常编译\n" << endl;

    // 在函数指针层面验证 showImage 可编译（不实际调用）
    using ShowImageFunc = void (*)(const std::string&, const cv::Mat&);
    ShowImageFunc show_ptr = cedar::showImage;
    verify("showImage 编译通过", show_ptr != nullptr);

    // 清理临时文件
    std::remove(tmp_img.c_str());

    print_separator();

    // ============================================================
    // 结论
    // ============================================================
    cout << "\n";
    cout << "  ============================================" << endl;
    if (failed == 0) {
        cout << "  结论: 全部 " << tested << " 项测试通过  ✓" << endl;
    } else {
        cout << "  结论: " << tested << " 项测试中 "
             << failed << " 项失败  ✗" << endl;
    }
    cout << "  ============================================\n" << endl;

    return failed > 0 ? 1 : 0;
}
