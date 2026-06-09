#ifndef CEDAR_PRINT_DETAIL_HPP
#define CEDAR_PRINT_DETAIL_HPP

// ============================================================
// print_detail.hpp  —  cedar::print 的 C++ 内部实现细节
//
// 这个文件包含 cedar::print 正常工作所需的 C++ 工具代码。
// 现在只剩下两样东西:
//   1. ANSI 颜色码
//   2. 判断一个类型是不是字符串（为了给字符串上绿色）
//      (其他类型不判断，全部交给 << 运算符处理)
// ============================================================

// ---------- 需要的 C++ 标准库 ----------
#include <chrono>         // 时间处理
#include <cstdlib>        // getenv 读取环境变量
#include <ctime>          // strftime 格式化时间
#include <filesystem>     // 创建目录
#include <fstream>        // 写文件
#include <iostream>       // 控制台输出
#include <sstream>        // 字符串流（类似 io.StringIO）
#include <string>         // 字符串
#include <string_view>    // 字符串视图
#include <type_traits>    // C++ 类型判断（is_arithmetic_v, is_same_v 等）
#include <unistd.h>       // isatty() 判断是否是终端

namespace cedar {
namespace detail {

// ============================================================
// ANSI 颜色码
// ============================================================
// 给控制台文字加颜色: \033[32m文字\033[0m
// 相当于 Python 的: print(f"\033[32m{text}\033[0m")
// 数字: 32=绿 94=蓝 93=黄 37=白

constexpr const char* RESET  = "\033[0m";   // 重置颜色
constexpr const char* C_STR  = "\033[32m";  // 绿   → 字符串
constexpr const char* C_NUM  = "\033[94m";  // 蓝   → 数字
constexpr const char* C_BOOL = "\033[93m";  // 黄   → 布尔值
constexpr const char* C_DEF  = "\033[37m";  // 白   → 其他类型


// ============================================================
// 判断字符串类型
// ============================================================
// 为什么要这个？因为 C++ 里"字符串"有好几种写法:
//   "hello"        → const char[6]（C 风格字面量）
//   std::string    → C++ 标准字符串
//   const char*    → C 风格字符串指针
//   std::string_view → 字符串只读引用
//
// 我们需要把它们都识别为"字符串"，统一显示绿色。

template <typename T>
struct is_string_type : std::false_type {};

template <> struct is_string_type<std::string>       : std::true_type {};
template <> struct is_string_type<std::string_view>   : std::true_type {};
template <> struct is_string_type<const char*>        : std::true_type {};
template <> struct is_string_type<char*>              : std::true_type {};
template <std::size_t N> struct is_string_type<const char[N]> : std::true_type {};
template <std::size_t N> struct is_string_type<char[N]>       : std::true_type {};

template <typename T>
inline constexpr bool is_string_type_v = is_string_type<std::decay_t<T>>::value;


// ============================================================
// 根据类型返回颜色
// ============================================================
// 字符串 → 绿，数字 → 蓝，其他 → 白

template <typename T>
constexpr const char* color_for() {
    if constexpr (is_string_type_v<T>) {
        return C_STR;
    } else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
        return C_NUM;
    } else {
        return C_DEF;
    }
}

}}  // namespace cedar::detail

#endif  // CEDAR_PRINT_DETAIL_HPP
