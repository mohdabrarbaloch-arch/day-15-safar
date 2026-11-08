# Safar

An autonomous-vehicle simulation platform that shows its work.

Safar drives a car down a track and then tells you exactly how close it came to
disaster - and at which timestep. Time to collision, minimum gap, lane deviation,
acceleration, jerk: every number on the dashboard is traceable back to a line of
physics, not a summary of a summary.

It also trains a tabular Q-learning agent in-process and lets you read the
policy it learned, one row at a time.

## Why this exists

Anyone can render a car moving down a road. The hard part is answering the
question a safety engineer actually asks: how close did that get to a collision,
and why?

So Safar is built around that question. It runs a real kinematic simulation -
not a canned animation - and reports the safety envelope of what happened. If the
ego vehicle crashes, it tells you at what speed and at what closing angle. If it
does not, it tells you how much margin was left.

The second half is behaviour. The Q-learning agent is deliberately small (140
states, 9 actions) so that its learned policy is legible. A neural network would
score better and teach you nothing.

## Features

Simulation:
- Kinematic bicycle model with midpoint integration
- Pure-pursuit path following with a speed-dependent lookahead
- IDM car-following for NPC traffic
- Weather as friction - wet tarmac costs 38% of braking authority
- Deterministic: same seed and scenario produce byte-identical frames

Safety metrics:
- Minimum time to collision, with the timestep where it occurred
- Minimum gap to any object, in metres
- Max and RMS lateral deviation from the centreline
- Max forward acceleration, max braking, max lateral acceleration, max jerk
- A documented composite safety score (0-100)

Reinforcement learning:
- Tabular Q-learning, 140 discretised states x 9 actions
- Seeded training, so the reward curve is reproducible
- The full greedy policy returned as a readable grid and a heatmap

Platform:
- JWT auth with bcrypt password hashing
- Sliding-window rate limiting per client IP
- Pydantic v2 validation on every request body
- SQLite for development, PostgreSQL for production
- Mobile-first vanilla-JS frontend, no build step
- Docker + docker-compose, and a Vercel serverless config

## Quickstart

Docker:

    git clone https://github.com/mohdabrarbaloch-arch/day-15-safar.git
    cd day-15-safar
    cp .env.example .env
    docker compose up --build

Open http://localhost:8000. Demo account: demo@safar.pk / safar-demo-2026.

Local Python:

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env
    uvicorn app.main:app --reload

With no DATABASE_URL set it uses SQLite and creates the file on first boot.

Full instructions: docs/setup.md

## Tech stack

| Layer | Choice |
|---|---|
| Simulation core | Pure Python + math, zero dependencies |
| API | FastAPI 0.115, Pydantic v2 |
| ORM | SQLAlchemy 2.0 (typed), SQLite / PostgreSQL |
| Auth | JWT HS256 + bcrypt |
| Frontend | Vanilla JS + Canvas, no build step |
| Tests | pytest, 114 tests |
| Deploy | Docker, docker-compose, Vercel |

The most important structural decision: sim/ never imports app/. The simulation
is a library; the API is a thin shell over it. The physics can be tested without
a database, and the engine could be lifted into a CLI without touching a line.

Architecture and scaling notes: ARCHITECTURE.md

## API

Interactive docs at /api/docs. Full reference: docs/api.md

## Running the tests

    pytest -q
    ruff check . && black --check .

## What this is not

- Not a photorealistic renderer. 2D top-down canvas, no graphics pipeline.
- Not a vehicle dynamics package. Kinematic bicycle, no tyre slip or suspension.
- Not a safety certification. The composite score is a documented heuristic.
- Not a trained AV stack. Tabular Q-learning, deliberately small and legible.

## License

MIT - see LICENSE.

Built by ABraz Baloch as Day 15 of the Autonomous AI Software Engineer 30-day
challenge.
