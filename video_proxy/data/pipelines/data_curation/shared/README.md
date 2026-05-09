# 筛选共享工具

这里放候选筛选阶段的共享辅助代码。

## 文件

- `local_screen.py`：本地 VLM 打分时复用的 prompt、解析和判断逻辑。

## 使用方式

通常不直接运行本目录文件，而是通过：

```bash
LOCAL_SCORE=1 bash video_proxy/data/pipelines/data_curation/run.sh
```

或者直接调用：

```bash
python -m video_proxy.data.pipelines.data_curation.curation.local_score --help
```
