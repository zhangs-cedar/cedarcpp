#ifndef CEDAR_PRINT_HPP
#define CEDAR_PRINT_HPP

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
// Internal helpers
// ============================================================
namespace detail {

// ---------- Color constants ----------
constexpr const char* RESET      = "\033[0m";
constexpr const char* C_STR      = "\033[32m";  // green
constexpr const char* C_NUM      = "\033[94m";  // blue
constexpr const char* C_BOOL     = "\033[93m";  // yellow
constexpr const char* C_SEQ      = "\033[95m";  // purple  (list/tuple/pair)
constexpr const char* C_MAP      = "\033[96m";  // cyan    (map/set)
constexpr const char* C_NONE     = "\033[90m";  // gray    (nullptr)
constexpr const char* C_DEF      = "\033[37m";  // white

// ---------- is_specialization trait ----------
template <typename T, template <typename...> class Primary>
struct is_specialization : std::false_type {};

template <template <typename...> class Primary, typename... Args>
struct is_specialization<Primary<Args...>, Primary> : std::true_type {};

template <typename T, template <typename...> class Primary>
inline constexpr bool is_specialization_v = is_specialization<T, Primary>::value;

// ---------- is_string_type trait ----------
template <typename T>
struct is_string_type : std::false_type {};

template <>
struct is_string_type<std::string> : std::true_type {};

template <>
struct is_string_type<std::string_view> : std::true_type {};

template <>
struct is_string_type<const char*> : std::true_type {};

template <>
struct is_string_type<char*> : std::true_type {};

template <std::size_t N>
struct is_string_type<const char[N]> : std::true_type {};

template <std::size_t N>
struct is_string_type<char[N]> : std::true_type {};

template <typename T>
inline constexpr bool is_string_type_v = is_string_type<std::decay_t<T>>::value;

// ---------- Color resolution ----------
template <typename T>
constexpr const char* color_for() {
    using U = std::decay_t<T>;
    if constexpr (is_string_type_v<T>) {
        return C_STR;
    } else if constexpr (std::is_same_v<U, bool>) {
        return C_BOOL;
    } else if constexpr (std::is_arithmetic_v<U>) {
        return C_NUM;
    } else if constexpr (is_specialization_v<U, std::vector> ||
                         is_specialization_v<U, std::list> ||
                         is_specialization_v<U, std::deque> ||
                         is_specialization_v<U, std::tuple> ||
                         is_specialization_v<U, std::pair>) {
        return C_SEQ;
    } else if constexpr (is_specialization_v<U, std::map> ||
                         is_specialization_v<U, std::unordered_map> ||
                         is_specialization_v<U, std::set> ||
                         is_specialization_v<U, std::unordered_set>) {
        return C_MAP;
    } else if constexpr (std::is_same_v<U, std::nullptr_t>) {
        return C_NONE;
    } else if constexpr (is_specialization_v<U, std::optional>) {
        return C_NONE;
    } else {
        return C_DEF;
    }
}

// ---------- Type name ----------
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

// ---------- Forward declarations ----------
template <typename T>
std::string format_arg(const T& arg, bool colored);

// ---------- Sequence container formatter ----------
template <typename Container>
std::string format_sequence(const Container& c, bool colored) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& elem : c) {
        if (!first) oss << ", ";
        oss << format_arg(elem, colored);
        first = false;
    }
    oss << "}";
    return oss.str();
}

// ---------- Map container formatter ----------
template <typename Map>
std::string format_map(const Map& m, bool colored) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [key, value] : m) {
        if (!first) oss << ", ";
        oss << format_arg(key, colored) << ": " << format_arg(value, colored);
        first = false;
    }
    oss << "}";
    return oss.str();
}

// ---------- Pair formatter ----------
template <typename T1, typename T2>
std::string format_pair(const std::pair<T1, T2>& p, bool colored) {
    std::ostringstream oss;
    oss << "(" << format_arg(p.first, colored) << ", " << format_arg(p.second, colored) << ")";
    return oss.str();
}

// ---------- Tuple formatter ----------
template <typename Tuple, std::size_t... I>
std::string format_tuple_impl(const Tuple& t, bool colored, std::index_sequence<I...>) {
    std::ostringstream oss;
    oss << "(";
    ((oss << (I == 0 ? "" : ", ") << format_arg(std::get<I>(t), colored)), ...);
    oss << ")";
    return oss.str();
}

template <typename... Ts>
std::string format_tuple(const std::tuple<Ts...>& t, bool colored) {
    return format_tuple_impl(t, colored, std::index_sequence_for<Ts...>{});
}

// ---------- Single-argument formatter ----------
template <typename T>
std::string format_arg(const T& arg, bool colored) {
    using U = std::decay_t<T>;
    std::ostringstream oss;

    const char* color = colored ? color_for<T>() : "";
    const char* reset = colored ? RESET : "";

    if constexpr (is_string_type_v<T>) {
        // string_view / const char* / char[] / std::string — bare value
        if constexpr (std::is_pointer_v<U>) {
            // const char* / char* — check for null
            if (arg == nullptr) {
                oss << color << "None" << reset;
                return oss.str();
            }
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
        if (arg.has_value()) {
            oss << format_arg(*arg, colored);
        } else {
            oss << color << "(optional) None" << reset;
        }
    } else if constexpr (is_specialization_v<U, std::vector> ||
                          is_specialization_v<U, std::list> ||
                          is_specialization_v<U, std::deque> ||
                          is_specialization_v<U, std::set> ||
                          is_specialization_v<U, std::unordered_set>) {
        oss << color << "(" << type_name<T>() << ") " << format_sequence(arg, colored) << reset;
    } else if constexpr (is_specialization_v<U, std::map> ||
                          is_specialization_v<U, std::unordered_map>) {
        oss << color << "(" << type_name<T>() << ") " << format_map(arg, colored) << reset;
    } else if constexpr (is_specialization_v<U, std::pair>) {
        oss << color << "(" << type_name<T>() << ") " << format_pair(arg, colored) << reset;
    } else if constexpr (is_specialization_v<U, std::tuple>) {
        oss << color << "(" << type_name<T>() << ") " << format_tuple(arg, colored) << reset;
    } else {
        // Fallback: try operator<<
        oss << color << "(" << type_name<T>() << ") " << arg << reset;
    }

    return oss.str();
}

// ---------- Timestamp ----------
inline std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "[%Y-%m-%d %H:%M:%S]", &tm);
    return std::string(buf);
}

// ---------- Log path ----------
inline std::string resolve_log_path() {
    const char* env = std::getenv("PRINT_LOG_PATH");
    if (env && env[0] != '\0') {
        return std::string(env);
    }
    return "/tmp/print.log";
}

// ---------- Log writer ----------
inline void append_log(const std::string& path, const std::string& content) {
    if (path.empty()) return;
    try {
        auto p = std::filesystem::path(path);
        std::filesystem::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::app);
        if (f) {
            f << content;
            f.flush();
        }
    } catch (...) {
        // silently ignore (same as Python: no crash on log failure)
    }
}

// ---------- Core implementation ----------
// Takes any number of arguments (sep and end are baked into the params)
template <typename... Args>
std::string print_impl(const std::string& sep, const std::string& end, const Args&... args) {
    std::string timestamp = current_timestamp();
    std::string prefix = timestamp + "   ";

    // --- Build colored (console) output ---
    std::ostringstream colored;
    colored << prefix;
    std::size_t idx = 0;
    ((colored << (idx++ ? sep : "") << format_arg(args, true)), ...);
    std::string colored_output = colored.str() + end;

    // Write to console
    std::cout << colored_output << std::flush;

    // --- Build plain (log) output ---
    std::ostringstream plain;
    plain << prefix;
    idx = 0;
    ((plain << (idx++ ? sep : "") << format_arg(args, false)), ...);
    std::string plain_output = plain.str() + "\n";

    // Write to log file
    append_log(resolve_log_path(), plain_output);

    return colored_output;
}

}  // namespace detail

// ============================================================
// Public API
// ============================================================

/// Default: print(arg1, arg2, ...) with sep=" " end="\n"
template <typename... Args>
std::string print(Args&&... args) {
    return detail::print_impl(" ", "\n", std::forward<Args>(args)...);
}

/// Custom sep/end: print_with(" | ", "\n", arg1, arg2, ...)
template <typename... Args>
std::string print_with(const std::string& sep, const std::string& end, Args&&... args) {
    return detail::print_impl(sep, end, std::forward<Args>(args)...);
}

}  // namespace cedar

#endif  // CEDAR_PRINT_HPP
