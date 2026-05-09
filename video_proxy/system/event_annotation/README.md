# Event Annotation System

独立的事件数据标注系统，用于查看和编辑视频理解结果（层次事件、caption、因果链等）。

这个系统放在 `video_proxy/system/event_annotation`，不是常用 rollout visualization 的一部分。一般只在需要人工检查或标注事件数据时启动。

## Quick Start (Development)

```bash
# Backend
cd video_proxy/system/event_annotation/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd video_proxy/system/event_annotation/frontend
npm install
npm run dev
```

## Production (Docker)

```bash
docker-compose up -d
# Open http://localhost:8080
```

## Adding Examples

Place example data in `data/examples/<name>/`:
- `video.mp4` — source video
- `analysis.json` — analysis result (see docs for schema)
- `frames/` — key frame images (optional)
