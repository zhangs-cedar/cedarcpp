// ============================================================
// cedar::print 功能验证
//
// 测试 cedar::print 能否正确处理各种基本类型:
//   字符串、整数、浮点数、布尔值、字符、空指针
// 并验证日志文件是否正常生成。
// ============================================================

#include <cedar/print.hpp>
#include <cstdlib>
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
    cout << "  cedar::print  功能验证" << endl;
    cout << "  ============================================" << endl;
    cout << "  以下每一行彩色输出都是 cedar::print 的效果，" << endl;
    cout << "  同时相同内容（无颜色码）已写入日志文件。\n" << endl;

    // ============================================================
    // 场景 1: 基本类型
    // ============================================================
    cout << "  [场景 1/3]  基本类型" << endl;
    cout << "  说明: 验证 print 能否打印字符串、数字、布尔值、字符\n" << endl;

    cout << "  >> 纯文本:" << endl;
    cedar::print("Hello World");

    cout << "  >> 混合参数 (字符串 + 整数 + 浮点数 + 布尔):" << endl;
    cedar::print("Hello", 42, 3.14, true);

    cout << "  >> 字符序列:" << endl;
    cedar::print('A', 'B', 'C');

    cout << "  >> 空指针:" << endl;
    cedar::print("nullptr:", nullptr);

    print_separator();

    // ============================================================
    // 场景 2: 多个参数混合
    // ============================================================
    cout << "  [场景 2/3]  多参数混合" << endl;
    cout << "  说明: 验证不同类型的参数混在一起能否正常输出\n" << endl;

    cout << "  >> 数字 + 字符串 + 布尔 + 浮点数:" << endl;
    cedar::print(1, "plus", 2.0, "equals", 3.0, true);

    print_separator();

    // ============================================================
    // 场景 3: 日志文件验证
    // ============================================================
    cout << "  [场景 3/3]  日志文件验证" << endl;
    cout << "  说明: 验证 print 是否同时将内容写入了日志文件\n" << endl;

    // 日志路径: 优先使用环境变量 PRINT_LOG_PATH，否则默认 /tmp/print.log
    const char* env_log = std::getenv("PRINT_LOG_PATH");
    string log = (env_log && env_log[0]) ? string(env_log) : "/tmp/print.log";

    ifstream f(log);
    verify("日志文件 " + log + " 存在", f.good());

    int lines = 0;
    string line;
    while (getline(f, line)) {
        ++lines;
    }
    verify("日志文件行数 > 0", lines > 0);
    cout << "      实际行数: " << lines << endl;

    f.clear();
    f.seekg(0);
    if (getline(f, line)) {
        verify("日志行以时间戳 [ 开头", line.rfind("[", 0) == 0);
        cout << "      示例行: " << line.substr(0, 60) << "..." << endl;
    }

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
