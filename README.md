# Adaptive Tone Engine

Adaptive music technology that learns how musicians play and helps equipment respond intelligently.

## Vision

Spend more time playing.
Spend less time tweaking.

## Current Status

Day 1 prototype.

## Goals

- Analyse guitar playing
- Detect attack, sustain and dynamics
- Recommend tone changes
- Learn player preferences over time
- Explore future software and hardware versions

## Stack

- Ubuntu
- Python
- FastAPI
- GitHub
- Docker planned

## Developer

Quickstart (local):

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Docker (local):

```bash
docker compose up --build
# app available at http://localhost:8000
```

Run tests:

```bash
pip install -r requirements.txt
pytest
```

API endpoints:

- GET / -> root status
- GET /status -> service status and short summary
- POST /analyze -> analyze tone traits (stub)
- POST /recommend -> return recommendations based on analysis (stub)

Interactive docs: visit http://127.0.0.1:8000/docs when the app is running.

Example requests:

GET status

```bash
curl http://127.0.0.1:8000/status
```

POST analyze

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"instrument":"guitar","style":"metal","signal_level_db":-5,"brightness":0.8,"muddiness":0.2,"harshness":0.7}'
```

POST recommend

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"target_tone":"tight modern rhythm","analysis":{"status":"ok","summary":"Detected metal guitar tone profile.","detected_traits":["bright","harsh"]}}'
```
