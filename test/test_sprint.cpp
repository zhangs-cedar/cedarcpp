#include <cedar/s_print.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <deque>
#include <tuple>
#include <optional>

using namespace std;

// 简单的测试辅助：检查条件，失败时打印并退出
int test_count = 0;
int fail_count = 0;

void check(bool cond, const string& msg) {
    ++test_count;
    if (!cond) {
        cerr << "FAIL: " << msg << endl;
        ++fail_count;
    }
}

int main() {
    // ===== 1. 基本类型 =====
    cedar::print("Hello World");
    cedar::print("Hello", 42, 3.14, true);
    cedar::print("int:", 123, "float:", 45.67, "bool:", false);
    cedar::print('A', 'B', 'C');

    // ===== 2. 容器 =====
    vector<int> v = {1, 2, 3};
    cedar::print("vector:", v);

    list<string> lst = {"a", "b"};
    cedar::print("list:", lst);

    deque<double> dq = {1.5, 2.5};
    cedar::print("deque:", dq);

    map<string, int> mp = {{"x", 1}, {"y", 2}};
    cedar::print("map:", mp);

    unordered_map<int, string> ump = {{1, "one"}};
    cedar::print("unordered_map:", ump);

    set<int> s = {10, 20, 30};
    cedar::print("set:", s);

    unordered_set<string> us = {"a", "b"};
    cedar::print("unordered_set:", us);

    // ===== 3. Pair & Tuple =====
    auto p = make_pair("key", 42);
    cedar::print("pair:", p);

    auto t = make_tuple(1, "two", 3.0);
    cedar::print("tuple:", t);

    // ===== 4. Optional & Nullptr =====
    optional<int> opt_yes = 100;
    optional<int> opt_no;
    cedar::print("opt_with_value:", opt_yes, "opt_empty:", opt_no);
    cedar::print("nullptr:", nullptr);

    // ===== 5. 自定义 sep/end =====
    cedar::print_with(" | ", "\n", "A", "B", "C");

    // ===== 6. 验证日志文件 =====
    string log_path = "/tmp/print.log";
    ifstream log_file(log_path);
    check(log_file.good(), "log file should exist at " + log_path);

    if (log_file.good()) {
        // 读取日志文件行数
        int lines = 0;
        string line;
        while (getline(log_file, line)) {
            ++lines;
            // 每行应以时间戳开头
            check(line.rfind("[", 0) == 0,
                  "log line should start with '[': " + line.substr(0, 40));
        }
        check(lines > 0, "log file should have content");
        log_file.close();
    }

    // ===== 结果 =====
    cout << "\n=== " << test_count << " tests, "
         << fail_count << " failures ===" << endl;

    return fail_count > 0 ? 1 : 0;
}
