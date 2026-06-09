#ifndef CEDAR_PRINT_HPP
#define CEDAR_PRINT_HPP

// ============================================================
// cedar::print  —  打印任意值到控制台（带颜色）+ 日志文件
//
// 用法:
//   #include <cedar/print.hpp>
//   cedar::print("hello", 42, true);
//
// 像 Python 的 print()，但:
//   - 字符串→绿色，数字→蓝色，布尔值→黄色，其他→白色
//   - 每行开头有 [时间戳]
//   - 同时写入日志文件（纯文本，无颜色码）
//   - 返回拼接后的字符串（带颜色码）
//
// C++ 的 << 运算符负责把值转成字符串——int、float、string
// 都是自带的。如果你传了一个不支持 << 的类型，编译器会报错。
// ============================================================

#include <cedar/print_detail.hpp>

namespace cedar {

// ============================================================
// 内部实现
// ============================================================
namespace detail {

// ---------- color_enabled ----------
// 判断 stdout 是否连接了终端（是→输出颜色，否→不输出）
inline bool color_enabled() {
    static const bool enabled = isatty(STDOUT_FILENO);
    return enabled;
}

// ---------- format_arg ----------
// 把单个参数格式化成字符串。
//
// Python 版的话，其实就是:
//   def format_arg(arg, colored):
//       if isinstance(arg, bool):
//           return f"{color}true/false{reset}"
//       elif arg is None:
//           return f"{color}None{reset}"
//       else:
//           return f"{color}{arg}{reset}"
//
// C++ 的 << 运算符会自动处理 int、float、string、char 等内置类型。
// 如果你传的是 list/map 等没有 << 的类型，编译会报错。
// 这是 C++ 的设计哲学: 只有明确支持的操作才能用。

template <typename T>
std::string format_arg(const T& arg, bool colored) {
    using U = std::decay_t<T>;  // 去掉引用/const，拿到"干净"的类型

    const char* color = colored ? color_for<T>() : "";
    const char* reset = colored ? RESET : "";

    std::ostringstream oss;  // 类似 io.StringIO()

    if constexpr (std::is_same_v<U, bool>) {
        // bool: 输出 true/false（而不是 C++ 默认的 1/0）
        oss << color << (arg ? "true" : "false") << reset;

    } else if constexpr (std::is_same_v<U, std::nullptr_t>) {
        // nullptr: 输出 None（和 Python 保持一致）
        oss << color << "None" << reset;

    } else {
        // 其他所有类型: 交给 << 处理
        // int, float, double, char, string 等都有内置的 <<
        oss << color << arg << reset;
    }

    return oss.str();
}


// ---------- timestamp ----------
// 返回当前时间: [YYYY-MM-DD HH:MM:SS]
// 相当于 Python 的 datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
inline std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "[%Y-%m-%d %H:%M:%S]", &tm);
    return std::string(buf);
}


// ---------- log_path ----------
// 日志路径: 环境变量 PRINT_LOG_PATH，没有则默认 /tmp/print.log
// 相当于 Python 的 os.environ.get("PRINT_LOG_PATH", "/tmp/print.log")
inline std::string log_path() {
    const char* env = std::getenv("PRINT_LOG_PATH");
    return (env && env[0]) ? std::string(env) : "/tmp/print.log";
}


// ---------- append_log ----------
// 向日志文件追加一行（自动创建目录）
// 相当于 Python 的:
//   os.makedirs(os.path.dirname(path), exist_ok=True)
//   with open(path, "a") as f: f.write(content)
inline void append_log(const std::string& path, const std::string& content) {
    if (path.empty()) return;
    try {
        std::filesystem::create_directories(
            std::filesystem::path(path).parent_path());
        std::ofstream f(path, std::ios::app);
        if (f) { f << content; f.flush(); }
    } catch (...) {
        // 日志写失败不中断程序
    }
}


// ---------- print_impl ----------
// 核心: 拼接参数 → 打屏（彩色）→ 写日志（纯文本）
//
// 相当于 Python 的:
//   def _print_impl(*args):
//       prefix = f"[{timestamp()}]   "
//       # 打屏（带颜色）
//       colored = prefix
//       for i, a in enumerate(args):
//           if i > 0: colored += " "
//           colored += format_arg(a, colored=True)
//       print(colored, flush=True)
//       # 写日志（纯文本）
//       plain = prefix
//       for i, a in enumerate(args):
//           if i > 0: plain += " "
//           plain += format_arg(a, colored=False)
//       append_to_log(plain)
//       return colored

template <typename... Args>
std::string print_impl(const Args&... args) {
    std::string prefix = timestamp() + "   ";

    // ---- 控制台: 彩色输出 ----
    std::ostringstream out;
    out << prefix;
    std::size_t i = 0;
    ((out << (i++ ? " " : "") << format_arg(args, color_enabled())), ...);
    std::string colored = out.str() + "\n";
    std::cout << colored << std::flush;

    // ---- 日志: 纯文本（不带颜色码）----
    out.str("");
    out.clear();
    out << prefix;
    i = 0;
    ((out << (i++ ? " " : "") << format_arg(args, false)), ...);
    append_log(log_path(), out.str() + "\n");

    return colored;
}

}  // namespace detail


// ============================================================
// 公共 API
// ============================================================

/// 打印任意值到控制台（带颜色），同时写入日志
///
/// 参数: 任意数量的值（必须有 << 运算符支持）
/// 返回: 拼接后的彩色字符串
///
/// 示例:
///   cedar::print("hello");           // → [时间戳]   hello
///   cedar::print(42, 3.14, true);    // → [时间戳]   42 3.14 true
///   auto s = cedar::print("ok");     // s 包含带颜色码的字符串
template <typename... Args>
std::string print(Args&&... args) {
    return detail::print_impl(std::forward<Args>(args)...);
}

}  // namespace cedar

#endif  // CEDAR_PRINT_HPP
