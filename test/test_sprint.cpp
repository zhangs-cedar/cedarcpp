// cedar::print 功能测试
// 覆盖全部支持的数据类型 + 日志文件验证
// 编译: g++ -std=c++17 -I <install>/include test_sprint.cpp

#include <cedar/s_print.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <deque>
#include <tuple>
#include <optional>

using namespace std;

// 测试统计
int tested = 0;
int failed = 0;

void verify(bool ok, const string& msg) {
    ++tested;
    if (!ok) { cerr << "FAIL: " << msg << endl; ++failed; }
}

int main() {
    // ---- 1. 基本类型 ----
    cedar::print("Hello World");
    cedar::print("Hello", 42, 3.14, true);
    cedar::print('A', 'B', 'C');

    // ---- 2. 容器 ----
    vector<int> v = {1, 2, 3};
    cedar::print("vector:", v);

    list<string> lst = {"a", "b"};
    cedar::print("list:", lst);

    map<string, int> mp = {{"x", 1}, {"y", 2}};
    cedar::print("map:", mp);

    set<int> s = {10, 20, 30};
    cedar::print("set:", s);

    // ---- 3. pair / tuple ----
    cedar::print("pair:", make_pair("key", 42));
    cedar::print("tuple:", make_tuple(1, "two", 3.0));

    // ---- 4. optional / nullptr ----
    optional<int> opt_yes = 100;
    optional<int> opt_no;
    cedar::print("optional:", opt_yes, opt_no);
    cedar::print("nullptr:", nullptr);

    // ---- 5. 自定义 sep/end ----
    cedar::print_with(" | ", "\n", "A", "B", "C");

    // ---- 6. 验证日志文件 ----
    string log = "/tmp/print.log";
    ifstream f(log);
    verify(f.good(), "日志文件 " + log + " 应存在");

    int lines = 0;
    string line;
    while (getline(f, line)) {
        ++lines;
        verify(line.rfind("[", 0) == 0,
               "日志行应以 [ 开头: " + line.substr(0, 40));
    }
    verify(lines > 0, "日志文件应有内容");

    cout << "\n=== " << tested << " 项测试, "
         << failed << " 项失败 ===" << endl;
    return failed > 0 ? 1 : 0;
}
