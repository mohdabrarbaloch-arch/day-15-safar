# Safar - API reference

Base URL: `http://localhost:8000`. Interactive docs: `/api/docs`.

All endpoints except `/api/health` and `/api/meta` require a bearer token:

```
Authorization: Bearer <access_token>
```

---

## Conventions

**Errors** are always JSON with a `detail` field:

```json
{ "detail": "Scenario not found" }
```

Validation failures return `422` with a structured body:

```json
{
  "detail": {
    "message": "Scenario failed validation",
    "errors": ["duration must be between 1 and 120 seconds"]
  }
}
```

**Status codes used**

| Code | Meaning |
|---|---|
| 200 | OK |
| 201 | Created |
| 204 | Deleted, no body |
| 401 | Missing or invalid token |
| 403 | Authenticated but not allowed (e.g. deleting a built-in) |
| 404 | Not found, or not yours |
| 409 | Conflict (duplicate email or scenario name) |
| 422 | Validation failed |
| 429 | Rate limited |

**Rate limits** are sliding-window, per client IP:

| Bucket | Default | Applies to |
|---|---|---|
| auth | 10/min | `/auth/register`, `/auth/login`, `/auth/token` |
| sim | 30/min | `POST /runs`, `POST /agents/train` |
| default | 120/min | everything else |

Every response carries `X-RateLimit-Remaining`. A `429` carries `Retry-After: 60`.

---

## Meta

### `GET /api/health`

```json
{ "status": "ok", "version": "1.0.0", "environment": "development", "simulation_core": "deterministic" }
```

### `GET /api/meta`

Static description of the platform: available agents, weather modes, limits, and
the disclaimer. Used by the About view.

---

## Auth

### `POST /api/auth/register` -> 201

```json
{ "email": "you@example.com", "password": "hunter2hunter2", "full_name": "Your Name" }
```

Returns the created user. `409` if the email already exists.

### `POST /api/auth/login` -> 200

```json
{ "email": "demo@safar.pk", "password": "safar-demo-2026" }
```

```json
{ "access_token": "eyJ...", "token_type": "bearer", "expires_in": 86400 }
```

### `POST /api/auth/token` -> 200

OAuth2 password-flow variant (`application/x-www-form-urlencoded`), so the
Swagger UI's Authorize button works. Same response shape.

### `GET /api/auth/me` -> 200

Returns the authenticated user.

---

## Scenarios

### `GET /api/scenarios` -> 200

Your scenarios plus every built-in one, built-ins first.

```json
[
  {
    "id": 1,
    "name": "Straight - slow lead vehicle",
    "description": "A lead car doing 6 m/s while the ego closes at 14...",
    "track_points": [[0, 0], [300, 0]],
    "closed_track": false,
    "lane_width": 3.5,
    "ego_lateral": 0.0,
    "ego_speed": 14.0,
    "weather": "dry",
    "difficulty": "normal",
    "duration": 20.0,
    "dt": 0.05,
    "npcs": [{ "s": 55, "lateral": 0, "speed": 6, "desired_speed": 6.5, "label": "lead-car" }],
    "obstacles": [],
    "is_builtin": true,
    "created_at": "2026-11-09T00:00:00Z"
  }
]
```

### `GET /api/scenarios/{id}` -> 200

One scenario. `404` if it isn't yours and isn't built-in.

### `POST /api/scenarios` -> 201

```json
{
  "name": "My wet brake test",
  "description": "Same obstacle, wet road",
  "track_points": [[0, 0], [300, 0]],
  "weather": "wet",
  "ego_speed": 16,
  "duration": 22,
  "npcs": [],
  "obstacles": [{ "s": 90, "lateral": 0, "radius": 1.1, "label": "stalled-car" }]
}
```

Field bounds: `track_points` 2-500 points, `duration` 1-120 s, `dt` 0.01-0.2 s,
`npcs` <= 20, `obstacles` <= 40. `409` on a duplicate name for your account.

### `DELETE /api/scenarios/{id}` -> 204

Deletes one of your own scenarios. `403` for built-ins.

---

## Runs

### `POST /api/runs` -> 201

```json
{ "scenario_id": 2, "agent": "pursuit", "seed": 1 }
```

For the learned policy, add `"agent_id": 7` and set `"agent": "qlearning"`.

Returns the full run: frames for the replay plus the metrics.

```json
{
  "id": 12,
  "scenario_id": 2,
  "agent": "pursuit",
  "seed": 1,
  "duration": 20.0,
  "created_at": "2026-11-09T00:00:00Z",
  "track_points": [[0, 0], [300, 0]],
  "lane_width": 3.5,
  "frames": [
    { "t": 0.0, "step": 0, "ego": { "x": 0, "y": 0, "heading": 0, "speed": 14, "steer": 0, "accel": 0 },
      "npcs": [{ "x": 55, "y": 0, "heading": 0, "speed": 6 }], "s": 0, "lateral": 0 }
  ],
  "metrics": {
    "collided": false,
    "collision": null,
    "completed": true,
    "progress": 1.0,
    "distance_travelled_m": 268.4,
    "min_ttc_s": 3.075,
    "min_ttc_at": { "value": 3.075, "step": 118, "time": 5.9 },
    "min_gap_m": 12.44,
    "min_gap_at": { "value": 12.44, "step": 118, "time": 5.9 },
    "max_lateral_error_m": { "value": 0.02, "step": 40, "time": 2.0 },
    "rms_lateral_error_m": 0.011,
    "max_accel_forward_mps2": { "value": 3.0, "step": 12, "time": 0.6 },
    "max_decel_mps2": { "value": 2.31, "step": 96, "time": 4.8 },
    "max_lateral_accel_mps2": { "value": 0.0, "step": 0, "time": 0.0 },
    "max_jerk_mps3": { "value": 1.2, "step": 13, "time": 0.65 },
    "safety_score": 80.0,
    "observations": 400
  }
}
```

`min_ttc_s` is `null` when nothing ever closed on the ego. `collision` is `null`
unless a collision happened, in which case it carries the step, time, the object
hit, and the speed in both m/s and km/h.

### `GET /api/runs?limit=20` -> 200

Your recent runs, newest first, **without** frames. `limit` is clamped to 1-100.

### `GET /api/runs/{id}` -> 200

One run, with frames.

### `GET /api/runs/{id}/summary` -> 200

Metrics only. Cheap enough to poll.

### `DELETE /api/runs/{id}` -> 204

Deletes one of your runs.

---

## Agents

### `POST /api/agents/train` -> 201

```json
{
  "name": "safar-agent",
  "scenario_id": 1,
  "episodes": 300,
  "steps_per_episode": 120,
  "alpha": 0.22,
  "gamma": 0.92,
  "seed": 7
}
```

Bounds: `episodes` 10-2000, `steps_per_episode` 20-600, `alpha` and `gamma` in
(0, 1].

```json
{
  "id": 3,
  "name": "safar-agent",
  "episodes": 300,
  "mean_reward": 9.46,
  "created_at": "2026-11-09T00:00:00Z",
  "summary": {
    "episodes": 300,
    "states_visited": 61,
    "state_count": 140,
    "action_count": 9,
    "best_reward": 24.1,
    "mean_reward_last_50": 9.46,
    "final_epsilon": 0.05,
    "reward_curve": [ -12.4, -8.1, "..." ]
  },
  "policy": [
    { "state": "3:2:1", "lateral_bucket": 3, "heading_bucket": 2, "speed_bucket": 1,
      "best_action": 4, "accel": 0.0, "steer": 0.0, "value": 3.81 }
  ]
}
```

`policy` is the greedy action for every state the agent visited - the raw data
behind the heatmap.

### `GET /api/agents` -> 200

Your trained agents, newest first, without the Q-table.

### `GET /api/agents/{id}` -> 200

One agent, including its full policy grid.

### `DELETE /api/agents/{id}` -> 204

Deletes one of your agents.

---

## Security headers

Every response includes:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Permissions-Policy: geolocation=(), microphone=()
```

CORS is an explicit allow-list from `CORS_ORIGINS`. `*` is rejected at startup
when `ENVIRONMENT=production`.
