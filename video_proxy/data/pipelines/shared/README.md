# Pipeline 共享工具

这里放多个数据流水线共同使用的工具代码，避免在各任务目录里重复实现路径解析、标注读取和帧缓存逻辑。

## 文件

- `seg_source.py`：读取层级标注、解析 clip 路径、计算 L3 clip 时间范围等。
- `frame_cache.py`：构建和复用源视频帧缓存。

## 使用方式

这些模块一般由 `proxy_construction/` 下的脚本 import，不直接作为命令入口。

```python
from video_proxy.data.pipelines.shared.seg_source import load_annotations
from video_proxy.data.pipelines.shared.frame_cache import ensure_source_frame_cache
```

新增共享逻辑时，请确认至少两个 pipeline 会复用；只被单个任务使用的代码应留在对应任务目录。
