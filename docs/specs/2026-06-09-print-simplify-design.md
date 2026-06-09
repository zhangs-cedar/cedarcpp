# cedar::print 简化设计

> 日期: 2026-06-09
> 目标: 让 cedar::print 的代码读起来像 Python，C++ 模板元编程封装到内部

## 设计原则

- **Python 友好**: 一个函数 `print(args...)`，没有重载选择困惑
- **统一用 `<<` 处理**: 不单独处理 vector/map/tuple/optional 等类型，全交给 `<<` 运算符
- **不炫技**: 没有 SFINAE、tag dispatch、enable_if 等
- **封装**: C++ 模板元编程放在 `print_detail.hpp`，`print.hpp` 只放逻辑代码

## 最终代码量

| 文件 | 行数 | 说明 |
|------|------|------|
| `print.hpp` | ~170 行 | 逻辑代码（含详细注释） |
| `print_detail.hpp` | ~80 行 | 颜色码 + is_string_type + color_for |

对比修改前: `print.hpp` 311 行（逻辑 + 模板混在一起）

## API

```cpp
#include <cedar/print.hpp>

cedar::print("hello", 42, true);     // 打印任意值
cedar::print("hello");               // 一个参数也行
```

输出格式: `[2026-06-09 10:34:52]   hello 42 true`

- 字符串 → 绿色，数字 → 蓝色，布尔值 → 黄色，其他 → 白色
- bool 输出 `true`/`false` 而不是 `1`/`0`
- `nullptr` 输出 `None`
- 其他类型全交给 C++ 的 `<<` 运算符

## 不做

- 不自定义 sep/end（C++ 不支持变参包后放 keyword-only 参数）
- 不单独格式化容器/元组/optional（交给 `<<`）
- 不加类型标注前缀（如 `(int)`、`(vector)`）
- 不改颜色、时间戳、日志行为、返回值

## 文件结构

```
include/cedar/
├── print.hpp              ← 公共 API + 逻辑代码
├── print_detail.hpp       ← C++ 内部实现（颜色码、类型判断）
└── image.hpp              ← 不变
```

## 验证

- `test_sprint.cpp` 测试基本类型（字符串/数字/布尔/字符/空指针）+ 日志验证
- 运行 `build.sh` 编译无警告 + ctest 全部通过
