# 06_pybind11_infer

目标：把 C++ 粗粒度接口暴露给 Python。

编译：

```bash
cmake -S . -B build-py -G Ninja -DNG_BUILD_PYBIND11=ON
cmake --build build-py -j
```

测试：

```bash
PYTHONPATH=build-py/06_pybind11_infer python3 scripts/test_pybind11.py
```

设计原则：

- Python 只做调度、报表、可视化。
- C++ 保留核心推理和后处理。
- 不把每个细碎步骤都暴露给 Python。
