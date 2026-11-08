# Architecture - Safar

> An autonomous-vehicle simulation platform where every number the safety
> dashboard shows can be traced back to a line of physics.

---

## 1. What problem this solves

Anyone can render a car moving down a road. The hard part is answering the
question a safety engineer actually asks: *"how close did that get to a
collision, and why?"*

Safar is built around that question. It runs a real kinematic simulation - not a
canned animation - and then reports the safety envelope of what happened: time
to collision, minimum gap, lane deviation, longitudinal and lateral acceleration,
jerk, and the exact timestep where each peaked. If the ego vehicle crashes, the
simulation tells you at what speed and at what closing angle.

The second half is behaviour. Safar ships a hand-written tabular Q-learning agent
that is trained *inside the process*, deterministically, against the same
simulator. You can watch its reward curve climb and then watch the trained policy
drive the track - or drive it badly, which it also reports honestly.

---

## 2. System diagram

```
+---------------------------------------------------------------------+
|  BROWSER  (static/, no build step)                                  |
|                                                                     |
|   index.html + app.js  --fetch(JSON)-->  render                     |
|     - canvas viewer (60 fps playback)                               |
|     - metrics cards, TTC chart, Q-table heatmap                     |
+---------------------------------------------------------------------+
                              |
                    HTTP / JSON  (JWT bearer)
                              v
+---------------------------------------------------------------------+
|  FASTAPI  (app/)                                                    |
|                                                                     |
|  routers/auth.py       register - login - me        security.py     |
|  routers/scenarios.py  CRUD on scenarios            models.py       |
|  routers/runs.py       POST /runs -> simulate, persist              |
|  routers/agents.py     POST /agents/train -> RL, persist            |
|                                                                     |
|  ratelimit.py   sliding-window per-IP budget                        |
|  schemas.py     Pydantic v2 request/response contracts              |
|  database.py    SQLAlchemy 2.0 - SQLite dev / PostgreSQL prod       |
+---------------------------------------------------------------------+
                              |
                    in-process import, no network
                              v
+---------------------------------------------------------------------+
|  SIMULATION CORE  (sim/)  - pure Python + math, zero deps           |
|                                                                     |
|  geometry.py   angle wrapping, segment projection, circle overlap   |
|  vehicle.py    kinematic bicycle model (state + integrator)         |
|  track.py      centreline arc-length parameterisation, projection   |
|  traffic.py    IDM car-following for NPC vehicles                   |
|  scenario.py   declarative world definition + validation            |
|  engine.py     fixed-step loop -> frames[] + metrics{}              |
|  metrics.py    TTC - gap - deviation - accel - jerk - composite     |
|  agent.py      tabular Q-learning (state discretiser + trainer)     |
+---------------------------------------------------------------------+
```

---

## 3. Tech stack and why

| Layer | Choice | Reason |
|---|---|---|
| Sim core | Pure Python + `math` | Deterministic, testable, no numpy needed for scalars. Runs anywhere. |
| API | FastAPI 0.115 | Pydantic v2 validation is the input-sanitisation layer. |
| ORM | SQLAlchemy 2.0 typed | Same code path on SQLite (dev) and PostgreSQL (prod). |
| Auth | JWT HS256 + bcrypt | Stateless, standard, no session store to lose. |
| Frontend | Vanilla JS + Canvas | No build step means the repo clones and runs. Canvas at 60 fps is plenty for a 2D sim. |
| Tests | pytest | 114 tests, weighted toward physics correctness. |
| Deploy | Docker / Vercel serverless | Both wired; Vercel needs a token this environment lacks. |

The single most important structural decision: **`sim/` never imports `app/`.**
The simulation is a library. The API is a thin shell over it. That means the
physics can be tested without a database, and the engine could be lifted into a
CLI, a notebook, or a batch job without touching a line.

---

## 4. Data flow - one simulation run

```
POST /api/runs {scenario_id, agent: "idm"|"qlearning", seed}
        
         1. Load scenario row (track, ego start, NPCs, obstacles, duration)
        
         2. Build runtime objects
               Track.from_polyline(...)
               Vehicle(pose=ego_start, params=...)
               [TrafficVehicle(...) for each NPC]
        
         3. Fixed-step loop, dt = 0.05 s  (20 Hz)
               for step in range(n_steps):
                   state   = vehicle.state
                   proj    = track.project(state.x, state.y)      lateral error
                   action  = policy(state, proj)                  IDM or Q-table
                   vehicle.step(action.accel, action.steer, dt)
                   for npc in npcs: npc.step(leader, dt)          IDM
                   metrics.observe(state, proj, npcs, obstacles, dt)
                   frames.append(snapshot())
                   if metrics.collided: break
        
         4. metrics.finalise() -> safety dict
        
         5. Persist Run(frames=json, metrics=json) -> return both
```

The loop is **fixed-step and seed-driven**. Two runs with the same seed and the
same scenario produce byte-identical frames. That property is what makes the
safety numbers defensible: you can reproduce a crash exactly.

---

## 5. Physics - the parts that matter

### 5.1 Vehicle model (kinematic bicycle)

State is `(x, y, psi, v, delta)` - position, heading, speed, steering angle.
Wheelbase `L`. Integration uses the midpoint rule so that a turning vehicle
doesn't visibly drift at low step counts:

```
yaw_rate = (v / L) * tan(delta)
psi_mid  = psi + yaw_rate * dt / 2
x'       = x + v * cos(psi_mid) * dt
y'       = y + v * sin(psi_mid) * dt
psi'     = wrap(psi + yaw_rate * dt)
v'       = clamp(v + a * dt, 0, v_max)
```

Acceleration is clamped to `[-b_max, a_max]`; steering to `+/-delta_max`. Those clamps
are not cosmetic - they are the vehicle's physical limits, and `b_max` is
*friction-scaled by weather*, which is why a wet scenario needs a longer braking
distance and sometimes doesn't stop in time. That is a real effect the simulator
produces, not a scripted outcome.

### 5.2 Path following (pure pursuit)

Lookahead `L_d = clamp(k * v, L_min, L_max)`. Target point at arc length
`s + L_d`. Steering command:

```
alpha = wrap(psi_path - psi_vehicle)   # heading error to the lookahead point
delta = atan2(2 * L * sin(alpha), L_d)
```

Arc-length parameterisation (rather than nearest-point) keeps the lookahead
stable through curves, which is the usual failure mode of naive implementations.

### 5.3 Traffic (IDM)

Each NPC follows its leader using the Intelligent Driver Model:

```
s_star = s0 + v * T + (v * dv) / (2 * sqrt(a * b))
a      = a_max * [ 1 - (v / v0)^4 - (s_star / s)^2 ]
```

Free-flowing when the gap is large, hard-braking when it closes. Two NPCs in the
same lane therefore produce realistic congestion without any hand-authored
behaviour.

---

## 6. Safety metrics

Every metric records *where* it peaked, not just the peak value.

| Metric | Definition | Why it's here |
|---|---|---|
| `collided` | circle-vs-circle overlap beyond a margin | the ground truth |
| `min_ttc` | min over time of `gap / closing_speed` | the leading indicator - a crash is TTC approaching 0 |
| `min_gap` | min centre-to-centre distance to any obstacle | metres, human-readable |
| `max_lateral_error` | max \|lateral offset\| from centreline | did it hold its lane |
| `rms_lateral_error` | RMS of the same | average quality, not just worst moment |
| `max_accel_forward` | max positive longitudinal accel | comfort |
| `max_decel` | max negative longitudinal accel | comfort / braking violence |
| `max_lateral_accel` | `v * yaw_rate` along the path | rollover proxy |
| `max_jerk` | max \|d(a)/dt\| | the metric passengers actually feel |
| `safety_score` | 0-100 composite | one number for the dashboard |

The composite is deliberately simple and documented in `metrics.py`: collisions
zero it out; otherwise it subtracts weighted penalties for low TTC, lane
excursions, and jerk. It is a summary, not a certification - the README says so.

---

## 7. The RL agent

Tabular Q-learning, because a neural network would hide the thing that makes
this educational: you can *read* the learned policy as a table.

**State** - 140 cells:

| Dimension | Buckets |
|---|---|
| lateral error | 7 (<= -1.5 m ... > 1.5 m) |
| heading error | 5 (<= -0.15 rad ... > 0.15 rad) |
| speed | 4 (< 3, 3-6, 6-9, > 9 m/s) |

**Actions** - 9: accel in {-2, 0, +2} x steer in {-0.25, 0, +0.25}.

**Reward** - `+progress - w_e*|lateral| - w_s*|steer| - w_j*|accel|`, with a
large terminal penalty on collision and a bonus on completion.

**Update** - `Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])`, epsilon-greedy with decay.

Training is seeded, so the reward curve is reproducible. The API returns the full
learning curve, the final Q-table, and the greedy policy as a readable grid -
which the frontend renders as a heatmap.

---

## 8. Security

- Passwords: bcrypt, cost 12. Never logged, never returned.
- Tokens: HS256, `JWT_SECRET` from env only. `config.py` refuses to boot in
  production with the dev default - a weak key is a startup error, not a warning.
- Input: every request body is a Pydantic model with bounded fields. Scenario
  geometry is validated for finiteness and dimension limits before it reaches the
  engine, so a malicious payload cannot allocate unbounded memory.
- Rate limiting: sliding window per client IP on auth and on the two expensive
  endpoints (`/runs`, `/agents/train`).
- CORS: explicit allow-list from env, not `*`.
- Security headers: `nosniff`, `DENY` framing, no-referrer, and a strict
  `Referrer-Policy` on every response.

Note on the demo: there is a seeded demo account (`demo@safar.pk`) so the app is
usable immediately. It is created only when `SEED_DEMO=true`.

---

## 9. Scaling notes

The current design runs one simulation per request, in-process, in ~0.1-0.4 s for
a 20-second scenario. That is fine for a portfolio-scale product. What changes at
10x and 100x:

| Concern | Now | At scale |
|---|---|---|
| Run execution | in-process, blocking | Celery/RQ worker pool; API returns a job id and the client polls or subscribes over a WebSocket |
| Frame storage | JSON blob in one column | object store (S3) with the row holding a key + metrics only |
| Metrics query | per-run | pre-aggregated rollups per scenario, materialised |
| Postgres | single instance | read replicas for the dashboard; the write path is append-only |
| RL training | 400 episodes, seconds | move to a batch job; the Q-table is already serialisable, so workers can share updates |
| Rate limiting | in-memory per process | Redis sliding window, shared across workers |

The append-only run model is the important one: runs are never mutated, only
inserted. That makes the whole analytics layer a matter of adding rollups, not
migrating data.

---

## 10. What this is not

- **Not a photorealistic renderer.** 2D top-down, canvas, no graphics pipeline.
- **Not a real vehicle model.** Kinematic bicycle, no tyre slip, no suspension,
  no load transfer. It is the model you use *before* you have a tyre model.
- **Not a safety certification.** The composite score is a summary heuristic.
- **Not a trained AV stack.** The agent is tabular Q-learning on a discretised
  state - deliberately small so its policy is legible.

The README repeats all four. Overclaiming in an AI project is the fastest way to
lose the reader who actually knows the domain.
