# VideoProxy Visualization System

Web-based visualization of video understanding results (hierarchical segmentation + causal reasoning).

## Quick Start (Development)

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
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
