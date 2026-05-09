# 筛选配置

这里保存不同原始数据集的默认筛选配置，供 `data_curation/run.sh` 和后续实验对齐参数。

## 文件

- `et_instruct_164k.yaml`：ET-Instruct-164K 候选视频筛选配置。
- `timelens_100k.yaml`：TimeLens-100K 候选视频筛选配置。

## 使用方式

当前主入口仍然是环境变量驱动的 shell 脚本：

```bash
DATASET=et_instruct_164k bash video_proxy/data/pipelines/data_curation/run.sh
DATASET=timelens_100k bash video_proxy/data/pipelines/data_curation/run.sh
```

如果实验需要固定参数，请优先把参数同步到对应 YAML，再在运行脚本里显式覆盖，避免只存在 shell history 里。
