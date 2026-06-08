# 05_multithread_pipeline

目标：不依赖 GPU，不依赖 OpenCV，先把工业实时系统里的 Pipeline、阻塞队列、退出机制、耗时统计练扎实。

运行：

```bash
./build/05_multithread_pipeline/05_multithread_pipeline --input data/images --output out/05_pipeline --repeat 200
```

模拟链路：

```text
Reader -> Preprocess -> Infer -> Postprocess -> Writer
```

重点验收：

- 队列有界
- 能 close
- 不死锁
- 能连续处理大量帧
- 能输出 latency 和 throughput
