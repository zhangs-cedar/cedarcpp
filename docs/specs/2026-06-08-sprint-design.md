# cedar::sprint — C++ 彩色打印函数

## 目标

为 cedarcpp 项目提供 C++ 版本的 `sprint` 函数，对应 cedarpy 中 `cedar/utils/s_print.py` 的功能。

## 功能描述

- 可变参数彩色打印（ANSI 终端颜色），按类型区分颜色
- 可选文件日志（纯文本，无颜色）
- 行为对齐 Python 版：含时间戳、类型标注、分隔符控制

## 非目标

- 不涉及图像处理、OpenCV 等功能
- 不修改已有代码（除 CMakeLists.txt 一行改动外）

## 设计

### 编译标准

C++17（原 CMakeLists.txt 中 `cxx_std_14` 升级至 `cxx_std_17`）

### 文件结构

```
cedarcpp/
├── CMakeLists.txt              # cxx_std_14 → cxx_std_17
├── include/cedar/
│   ├── image.hpp               # 已有，无改动
│   └── s_print.hpp             # 新建：sprint 实现
```

### 命名空间

`cedar::sprint(...)`

### 函数签名

```cpp
namespace cedar {

// 完整版：可指定 sep, end
template <typename... Args>
std::string sprint(const std::string& sep,
                   const std::string& end,
                   Args&&... args);

// 简版：使用默认 sep=" " end="\n"
template <typename... Args>
std::string sprint(Args&&... args);

}  // namespace cedar
```

### 返回

`std::string` — 格式化后的完整输出字符串（带时间戳、无颜色）

### 颜色映射

| C++ 类型 | ANSI 颜色 |
|---|---|
| `std::string`, `const char*` | 绿 `\033[32m` |
| 整数/浮点 (`int/double/float` 等) | 蓝 `\033[94m` |
| `bool` | 黄 `\033[93m` |
| 序列容器 (`vector/deque/list`) | 紫 `\033[95m` |
| 有序/无序关联容器 (`map/set` 等) | 青 `\033[96m` |
| `std::tuple`, `std::pair` | 紫 `\033[95m` |
| `std::optional` 空值 / `std::nullptr_t` | 灰 `\033[90m` |
| 其他 | 白 `\033[37m` |

### 日志

- 默认路径：`/tmp/sprint.log`
- 可被 `SPRINT_LOG_PATH` 环境变量覆盖
- 不支持每次调用单独指定路径（YAGNI——Python 版此参数极少使用）
- 文件日志纯文本，无 ANSI 转义码

### 时间戳格式

`[YYYY-MM-DD HH:MM:SS]`，与 Python 版一致。

### 实现策略

- C++17 `if constexpr` + 类型萃取 (`std::is_same_v`, `std::is_arithmetic_v` 等) 编译期分发
- C++17 fold expressions 拼接输出
- `std::filesystem::create_directories` 自动创建日志目录
- 单头文件（header-only），无额外依赖

### 使用示例

```cpp
#include <cedar/s_print.hpp>
#include <vector>
#include <map>

int main() {
    cedar::sprint("Hello", 42, 3.14, true);

    std::vector<int> v = {1, 2, 3};
    std::map<std::string, int> m = {{"a", 1}};
    cedar::sprint(v, m);

    cedar::sprint(" | ", "\n", "line1", "line2", "line3");
    return 0;
}
```

## 风险

- ANSI 颜色转义码在非终端环境（重定向到文件、管道）会显示为乱码，保持与 Python 版一致的行为（不自动检测 `isatty`）

## 验证

- 编写测试 main 函数编译运行，确认彩色输出及日志文件生成
