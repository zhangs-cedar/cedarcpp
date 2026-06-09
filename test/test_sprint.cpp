// ============================================================
// cedar::print 功能验证
// 本程序演示并验证 cedar::print 的各项功能。
// 输出彩色文本 → 控制台（供 FAE 直观确认效果）
// 同时写入纯文本 → 日志文件（由 PRINT_LOG_PATH 环境变量指定）
// ============================================================

#include <cedar/print.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <tuple>
#include <optional>

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
    cout << "  [场景 1/6]  基本类型" << endl;
    cout << "  说明: 验证 print 能否正常打印字符串、数字、布尔值、字符\n" << endl;

    cout << "  >> 纯文本:" << endl;
    cedar::print("Hello World");

    cout << "  >> 混合参数 (字符串 + 整数 + 浮点数 + 布尔):" << endl;
    cedar::print("Hello", 42, 3.14, true);

    cout << "  >> 字符序列:" << endl;
    cedar::print('A', 'B', 'C');

    print_separator();

    // ============================================================
    // 场景 2: 容器类型
    // ============================================================
    cout << "  [场景 2/6]  容器类型" << endl;
    cout << "  说明: 验证 print 能否正确打印 vector / list / map / set\n" << endl;

    vector<int> v = {1, 2, 3};
    cout << "  >> vector<int>:" << endl;
    cedar::print("vector:", v);

    list<string> lst = {"a", "b"};
    cout << "  >> list<string>:" << endl;
    cedar::print("list:", lst);

    map<string, int> mp = {{"x", 1}, {"y", 2}};
    cout << "  >> map<string, int>:" << endl;
    cedar::print("map:", mp);

    set<int> s = {10, 20, 30};
    cout << "  >> set<int>:" << endl;
    cedar::print("set:", s);

    print_separator();

    // ============================================================
    // 场景 3: pair / tuple
    // ============================================================
    cout << "  [场景 3/6]  元组与键值对" << endl;
    cout << "  说明: 验证 print 能否正确打印 pair 和 tuple\n" << endl;

    cout << "  >> pair<string, int>:" << endl;
    cedar::print("pair:", make_pair("key", 42));

    cout << "  >> tuple<int, string, double>:" << endl;
    cedar::print("tuple:", make_tuple(1, "two", 3.0));

    print_separator();

    // ============================================================
    // 场景 4: optional / nullptr
    // ============================================================
    cout << "  [场景 4/6]  空值与可选值" << endl;
    cout << "  说明: 验证 print 能否正确处理有值/无值的 optional 和 nullptr\n" << endl;

    optional<int> opt_yes = 100;
    optional<int> opt_no;
    cout << "  >> optional（有值 + 无值）:" << endl;
    cedar::print("optional:", opt_yes, opt_no);

    cout << "  >> nullptr:" << endl;
    cedar::print("nullptr:", nullptr);

    print_separator();

    // ============================================================
    // 场景 5: 自定义分隔符与结尾
    // ============================================================
    cout << "  [场景 5/6]  自定义格式" << endl;
    cout << "  说明: 使用 print_with 自定义分隔符和结尾\n" << endl;

    cout << "  >> 用 \" | \" 分隔，以换行结尾:" << endl;
    cedar::print_with(" | ", "\n", "A", "B", "C");

    print_separator();

    // ============================================================
    // 场景 6: 日志文件验证
    // ============================================================
    cout << "  [场景 6/6]  日志文件验证" << endl;
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
