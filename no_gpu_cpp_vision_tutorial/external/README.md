# external

放第三方 C++ 预编译依赖。

推荐结构：

```text
external/
└── onnxruntime-linux-x64-1.xx.x/
    ├── include/
    │   └── onnxruntime_cxx_api.h
    └── lib/
        └── libonnxruntime.so
```

不要把大型第三方库提交进自己的代码仓库。
