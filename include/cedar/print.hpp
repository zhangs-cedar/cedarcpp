#ifndef CEDAR_PRINT_HPP
#define CEDAR_PRINT_HPP

/**
 * cedar::print / cedar::print_with
 *
 * 对标 cedarpy 中 cedar/utils/s_print.py 的彩色打印函数。
 * 控制台输出 ANSI 彩色文本，同时将纯文本（无颜色码）写入日志文件。
 *
 * 用法:
 *   cedar::print("hello", 42, true);              // 默认 sep=" " end="\n"
 *   cedar::print_with(" | ", "\n", "a", "b");     // 自定义 sep 和 end
 *
 * 日志: 默认 /tmp/print.log，环境变量 PRINT_LOG_PATH 可覆盖
 */

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cedar {

// ============================================================
// 内部实现
// ============================================================
namespace detail {

// ---------- ANSI 颜色码 ----------
constexpr const char* RESET  = "\033[0m";
constexpr const char* C_STR  = "\033[32m";  // 绿: 字符串
constexpr const char* C_NUM  = "\033[94m";  // 蓝: 数字
constexpr const char* C_BOOL = "\033[93m";  // 黄: 布尔
constexpr const char* C_SEQ  = "\033[95m";  // 紫: 序列(vector/list/tuple/pair)
constexpr const char* C_MAP  = "\033[96m";  // 青: 关联容器(map/set)
constexpr const char* C_NONE = "\033[90m";  // 灰: 空值(nullptr/empty optional)
constexpr const char* C_DEF  = "\033[37m";  // 白: 其他

// ---------- 判断 T 是否是 Primary<...> 的实例 ----------
template <typename T, template <typename...> class Primary>
struct is_specialization : std::false_type {};

template <template <typename...> class Primary, typename... Args>
struct is_specialization<Primary<Args...>, Primary> : std::true_type {};

template <typename T, template <typename...> class Primary>
inline constexpr bool is_specialization_v = is_specialization<T, Primary>::value;

// ---------- 判断 T 是否是字符串类型 ----------
template <typename T>
struct is_string_type : std::false_type {};

template <> struct is_string_type<std::string>     : std::true_type {};
template <> struct is_string_type<std::string_view> : std::true_type {};
template <> struct is_string_type<const char*>      : std::true_type {};
template <> struct is_string_type<char*>            : std::true_type {};
template <std::size_t N> struct is_string_type<const char[N]> : std::true_type {};
template <std::size_t N> struct is_string_type<char[N]>       : std::true_type {};

template <typename T>
inline constexpr bool is_string_type_v = is_string_type<std::decay_t<T>>::value;

// ---------- 获取类型对应的 ANSI 颜色 ----------
template <typename T>
constexpr const char* color_for() {
    using U = std::decay_t<T>;
    if constexpr (is_string_type_v<T>)             return C_STR;
    else if constexpr (std::is_same_v<U, bool>)    return C_BOOL;
    else if constexpr (std::is_arithmetic_v<U>)    return C_NUM;
    else if constexpr (is_specialization_v<U, std::vector>  ||
                       is_specialization_v<U, std::list>    ||
                       is_specialization_v<U, std::deque>   ||
                       is_specialization_v<U, std::tuple>   ||
                       is_specialization_v<U, std::pair>)   return C_SEQ;
    else if constexpr (is_specialization_v<U, std::map>        ||
                       is_specialization_v<U, std::unordered_map> ||
                       is_specialization_v<U, std::set>        ||
                       is_specialization_v<U, std::unordered_set>) return C_MAP;
    else if constexpr (std::is_same_v<U, std::nullptr_t> ||
                       is_specialization_v<U, std::optional>)  return C_NONE;
    else                                                     return C_DEF;
}

// ---------- 获取类型的显示名称 ----------
template <typename T>
std::string type_name() {
    using U = std::decay_t<T>;
    if constexpr (std::is_same_v<U, int>)                return "int";
    else if constexpr (std::is_same_v<U, long>)          return "long";
    else if constexpr (std::is_same_v<U, long long>)     return "long long";
    else if constexpr (std::is_same_v<U, unsigned>)      return "unsigned";
    else if constexpr (std::is_same_v<U, unsigned long>) return "unsigned long";
    else if constexpr (std::is_same_v<U, float>)         return "float";
    else if constexpr (std::is_same_v<U, double>)        return "double";
    else if constexpr (std::is_same_v<U, bool>)          return "bool";
    else if constexpr (std::is_same_v<U, char>)          return "char";
    else if constexpr (std::is_same_v<U, std::string>)   return "str";
    else if constexpr (is_specialization_v<U, std::vector>)    return "vector";
    else if constexpr (is_specialization_v<U, std::list>)      return "list";
    else if constexpr (is_specialization_v<U, std::deque>)     return "deque";
    else if constexpr (is_specialization_v<U, std::map>)       return "map";
    else if constexpr (is_specialization_v<U, std::unordered_map>) return "unordered_map";
    else if constexpr (is_specialization_v<U, std::set>)       return "set";
    else if constexpr (is_specialization_v<U, std::unordered_set>) return "unordered_set";
    else if constexpr (is_specialization_v<U, std::pair>)      return "pair";
    else if constexpr (is_specialization_v<U, std::tuple>)     return "tuple";
    else if constexpr (is_specialization_v<U, std::optional>)  return "optional";
    else if constexpr (std::is_same_v<U, std::nullptr_t>)      return "NoneType";
    else                                                       return "unknown";
}

// ---------- 前向声明（供下面相互递归调用） ----------
template <typename T>
std::string format_arg(const T& arg, bool colored);

// 格式化序列容器: vector, list, deque, set, unordered_set
template <typename Container>
std::string fmt_seq(const Container& c, bool colored) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& e : c) {
        if (!first) oss << ", ";
        oss << format_arg(e, colored);
        first = false;
    }
    oss << "}";
    return oss.str();
}

// 格式化关联容器: map, unordered_map
template <typename Map>
std::string fmt_map(const Map& m, bool colored) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [k, v] : m) {
        if (!first) oss << ", ";
        oss << format_arg(k, colored) << ": " << format_arg(v, colored);
        first = false;
    }
    oss << "}";
    return oss.str();
}

// 格式化 std::pair
template <typename T1, typename T2>
std::string fmt_pair(const std::pair<T1, T2>& p, bool colored) {
    std::ostringstream oss;
    oss << "(" << format_arg(p.first, colored) << ", " << format_arg(p.second, colored) << ")";
    return oss.str();
}

// 格式化 std::tuple（递归展开）
template <typename Tuple, std::size_t... I>
std::string fmt_tuple_impl(const Tuple& t, bool colored, std::index_sequence<I...>) {
    std::ostringstream oss;
    oss << "(";
    ((oss << (I == 0 ? "" : ", ") << format_arg(std::get<I>(t), colored)), ...);
    oss << ")";
    return oss.str();
}

template <typename... Ts>
std::string fmt_tuple(const std::tuple<Ts...>& t, bool colored) {
    return fmt_tuple_impl(t, colored, std::index_sequence_for<Ts...>{});
}

// ---------- 核心: 按类型格式化单个参数 ----------
// colored=true  → 输出带 ANSI 颜色，false → 纯文本（写日志用）
template <typename T>
std::string format_arg(const T& arg, bool colored) {
    using U = std::decay_t<T>;
    std::ostringstream oss;

    const char* color = colored ? color_for<T>() : "";
    const char* reset = colored ? RESET : "";

    if constexpr (is_string_type_v<T>) {
        // 字符串直接输出，但指针类型要检查 nullptr
        if constexpr (std::is_pointer_v<U>) {
            if (arg == nullptr) { oss << color << "None" << reset; return oss.str(); }
        }
        oss << color << arg << reset;
    } else if constexpr (std::is_same_v<U, bool>) {
        oss << color << "(bool) " << (arg ? "true" : "false") << reset;
    } else if constexpr (std::is_same_v<U, char>) {
        oss << color << "(char) '" << arg << "'" << reset;
    } else if constexpr (std::is_arithmetic_v<U>) {
        oss << color << "(" << type_name<T>() << ") " << arg << reset;
    } else if constexpr (std::is_same_v<U, std::nullptr_t>) {
        oss << color << "None" << reset;
    } else if constexpr (is_specialization_v<U, std::optional>) {
        if (arg.has_value()) oss << format_arg(*arg, colored);
        else                 oss << color << "(optional) None" << reset;
    } else if constexpr (is_specialization_v<U, std::vector>  ||
                          is_specialization_v<U, std::list>    ||
                          is_specialization_v<U, std::deque>   ||
                          is_specialization_v<U, std::set>     ||
                          is_specialization_v<U, std::unordered_set>) {
        oss << color << "(" << type_name<T>() << ") " << fmt_seq(arg, colored) << reset;
    } else if constexpr (is_specialization_v<U, std::map> ||
                          is_specialization_v<U, std::unordered_map>) {
        oss << color << "(" << type_name<T>() << ") " << fmt_map(arg, colored) << reset;
    } else if constexpr (is_specialization_v<U, std::pair>) {
        oss << color << "(" << type_name<T>() << ") " << fmt_pair(arg, colored) << reset;
    } else if constexpr (is_specialization_v<U, std::tuple>) {
        oss << color << "(" << type_name<T>() << ") " << fmt_tuple(arg, colored) << reset;
    } else {
        // 未知类型: 尝试 operator<< 兜底
        oss << color << "(" << type_name<T>() << ") " << arg << reset;
    }
    return oss.str();
}

// ---------- 当前时间戳 [YYYY-MM-DD HH:MM:SS] ----------
inline std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "[%Y-%m-%d %H:%M:%S]", &tm);
    return std::string(buf);
}

// ---------- 解析日志路径: 环境变量 > 默认值 ----------
inline std::string log_path() {
    const char* env = std::getenv("PRINT_LOG_PATH");
    return (env && env[0]) ? std::string(env) : "/tmp/print.log";
}

// ---------- 追加写入日志文件（自动创建父目录） ----------
inline void append_log(const std::string& path, const std::string& content) {
    if (path.empty()) return;
    try {
        std::filesystem::create_directories(
            std::filesystem::path(path).parent_path());
        std::ofstream f(path, std::ios::app);
        if (f) { f << content; f.flush(); }
    } catch (...) {
        // 日志写入失败不影响主流程（与 Python 版一致）
    }
}

// ---------- 核心实现: 拼接参数 → 控制台(彩色) + 日志(纯文本) ----------
template <typename... Args>
std::string print_impl(const std::string& sep, const std::string& end, const Args&... args) {
    std::string ts = timestamp();
    std::string prefix = ts + "   ";

    // 构建彩色输出 → 打屏
    std::ostringstream out;
    out << prefix;
    std::size_t i = 0;
    ((out << (i++ ? sep : "") << format_arg(args, true)), ...);
    std::string colored = out.str() + end;
    std::cout << colored << std::flush;

    // 构建纯文本输出 → 写日志
    out.str(""); out.clear();
    out << prefix;
    i = 0;
    ((out << (i++ ? sep : "") << format_arg(args, false)), ...);
    append_log(log_path(), out.str() + "\n");

    return colored;
}

}  // namespace detail

// ============================================================
// 公共 API
// ============================================================

/// 彩色打印, 默认 sep=" " end="\n"
template <typename... Args>
std::string print(Args&&... args) {
    return detail::print_impl(" ", "\n", std::forward<Args>(args)...);
}

/// 彩色打印, 自定义分隔符和结尾
template <typename... Args>
std::string print_with(const std::string& sep, const std::string& end, Args&&... args) {
    return detail::print_impl(sep, end, std::forward<Args>(args)...);
}

}  // namespace cedar

#endif  // CEDAR_PRINT_HPP
