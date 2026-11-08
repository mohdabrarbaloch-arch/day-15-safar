"""Tabular Q-learning agent with a legible policy.

The whole point of keeping this tabular is that you can *read* what the agent
learned. A neural policy would score better and teach nothing.

State is a triple of discretised buckets:
    lateral error (7) x heading error (5) x speed (4)  =  140 states
Action is one of 9 (accel x steer). Q-table is therefore 140 x 9 floats -
small enough to serialise into a single API response and render as a heatmap.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from sim.controller import ACTION_COUNT, decode_action
from sim.track import Projection
from sim.vehicle import VehicleParams, VehicleState

LATERAL_BUCKETS: tuple[float, ...] = (-1.5, -0.9, -0.35, 0.35, 0.9, 1.5)
HEADING_BUCKETS: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)
SPEED_BUCKETS: tuple[float, ...] = (3.0, 6.0, 9.0)

STATE_COUNT = (len(LATERAL_BUCKETS) + 1) * (len(HEADING_BUCKETS) + 1) * (len(SPEED_BUCKETS) + 1)


def _bucket(value: float, boundaries: tuple[float, ...]) -> int:
    """Index of the bucket ``value`` falls into."""
    for index, boundary in enumerate(boundaries):
        if value < boundary:
            return index
    return len(boundaries)


def state_key(lateral: float, heading_error: float, speed: float) -> str:
    """Render a discretised state as a stable string key."""
    lat_i = _bucket(lateral, LATERAL_BUCKETS)
    head_i = _bucket(heading_error, HEADING_BUCKETS)
    speed_i = _bucket(speed, SPEED_BUCKETS)
    return f"{lat_i}:{head_i}:{speed_i}"


def state_key_from_live(
    vehicle_state: VehicleState, projection: Projection, track_heading: float
) -> str:
    """Build the state key from a live vehicle and its projection."""
    heading_error = math.atan2(
        math.sin(vehicle_state.heading - track_heading),
        math.cos(vehicle_state.heading - track_heading),
    )
    return state_key(projection.lateral, heading_error, vehicle_state.speed)


@dataclass
class TrainingConfig:
    """Hyperparameters for one training run."""

    episodes: int = 400
    alpha: float = 0.22
    gamma: float = 0.92
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.992
    steps_per_episode: int = 140
    dt: float = 0.05
    seed: int = 7

    # Reward shaping.
    progress_weight: float = 1.0
    lateral_penalty: float = 0.9
    steer_penalty: float = 0.06
    accel_change_penalty: float = 0.05
    collision_penalty: float = 40.0
    completion_bonus: float = 20.0
    off_track_penalty: float = 1.2

    def epsilon_at(self, episode: int) -> float:
        """Linearly-with-decay epsilon schedule."""
        value = self.epsilon_start * (self.epsilon_decay**episode)
        return max(self.epsilon_end, value)


@dataclass
class TrainingResult:
    """Everything a training run produces."""

    q_table: dict[str, list[float]]
    rewards: list[float] = field(default_factory=list)
    episodes: int = 0
    final_epsilon: float = 0.0
    states_visited: int = 0
    best_reward: float = 0.0
    mean_reward_last_50: float = 0.0

    def summary(self) -> dict[str, object]:
        return {
            "episodes": self.episodes,
            "states_visited": self.states_visited,
            "state_count": STATE_COUNT,
            "action_count": ACTION_COUNT,
            "best_reward": round(self.best_reward, 2),
            "mean_reward_last_50": round(self.mean_reward_last_50, 2),
            "final_epsilon": round(self.final_epsilon, 4),
            "reward_curve": self.rewards,
        }

    def policy_grid(self) -> list[dict[str, object]]:
        """Flatten the greedy policy into readable rows for the heatmap."""
        rows: list[dict[str, object]] = []
        for lat_i in range(len(LATERAL_BUCKETS) + 1):
            for head_i in range(len(HEADING_BUCKETS) + 1):
                for speed_i in range(len(SPEED_BUCKETS) + 1):
                    key = f"{lat_i}:{head_i}:{speed_i}"
                    values = self.q_table.get(key)
                    if values is None:
                        continue
                    best = max(range(len(values)), key=lambda i: values[i])
                    action = decode_action(best)
                    rows.append(
                        {
                            "state": key,
                            "lateral_bucket": lat_i,
                            "heading_bucket": head_i,
                            "speed_bucket": speed_i,
                            "best_action": best,
                            "accel": action.accel,
                            "steer": action.steer,
                            "value": round(values[best], 4),
                        }
                    )
        return rows


def empty_q_table() -> dict[str, list[float]]:
    """A Q-table with every reachable state initialised to zero."""
    table: dict[str, list[float]] = {}
    for lat_i in range(len(LATERAL_BUCKETS) + 1):
        for head_i in range(len(HEADING_BUCKETS) + 1):
            for speed_i in range(len(SPEED_BUCKETS) + 1):
                table[f"{lat_i}:{head_i}:{speed_i}"] = [0.0] * ACTION_COUNT
    return table


def _step_environment(
    state: VehicleState,
    params: VehicleParams,
    action_accel: float,
    action_steer: float,
    dt: float,
    track,
    lane_width: float,
    speed_cap: float,
) -> tuple[VehicleState, float, bool]:
    """One environment step used during training.

    Deliberately minimal - no traffic - because the point of the exercise is to
    learn lane-keeping, and traffic noise would make the reward curve
    uninterpretable. ``engine.py`` is where traffic belongs.
    """
    prev_s = track.project(state.x, state.y).s

    steer = max(-params.max_steer, min(params.max_steer, action_steer))
    accel = max(-params.effective_brake, min(params.max_accel, action_accel))
    yaw_rate = (state.speed / params.wheelbase) * math.tan(steer)
    mid_heading = state.heading + yaw_rate * dt * 0.5

    new_x = state.x + state.speed * math.cos(mid_heading) * dt
    new_y = state.y + state.speed * math.sin(mid_heading) * dt
    new_heading = math.atan2(
        math.sin(state.heading + yaw_rate * dt), math.cos(state.heading + yaw_rate * dt)
    )
    new_speed = max(0.0, min(speed_cap, state.speed + accel * dt))

    new_state = VehicleState(
        x=new_x, y=new_y, heading=new_heading, speed=new_speed, steer=steer, accel=accel
    )

    projection = track.project(new_x, new_y)
    progress = max(0.0, projection.s - prev_s)
    off_track = abs(projection.lateral) > lane_width * 0.5
    done = projection.s >= track.length - 1.0

    return new_state, progress, off_track or done, projection, done


def train(
    config: TrainingConfig,
    track,
    lane_width: float = 3.5,
    speed_cap: float = 30.0,
) -> TrainingResult:
    """Run Q-learning against a track.

    Fully seeded: the same config produces the same Q-table, every time. That
    determinism is what makes the reward curve worth showing.
    """
    rng = random.Random(config.seed)
    params = VehicleParams(max_speed=speed_cap, friction_scale=1.0)
    q_table = empty_q_table()
    rewards: list[float] = []
    visited: set[str] = set()

    for episode in range(config.episodes):
        epsilon = config.epsilon_at(episode)

        projection = track.project(0.0, 0.0)
        start_x, start_y, start_heading = track.point_at(0.0)
        state = VehicleState(x=start_x, y=start_y, heading=start_heading, speed=0.0)

        episode_reward = 0.0
        prev_action_index = 4  # neutral
        prev_lateral = 0.0

        for _ in range(config.steps_per_episode):
            live = track.project(state.x, state.y)
            key = state_key_from_live(state, live, track.heading_at(live.s))
            visited.add(key)

            if rng.random() < epsilon:
                action_index = rng.randrange(ACTION_COUNT)
            else:
                values = q_table[key]
                action_index = max(range(len(values)), key=lambda i: values[i])

            action = decode_action(action_index)
            new_state, progress, done, new_proj, finished = _step_environment(
                state,
                params,
                action.accel,
                action.steer,
                config.dt,
                track,
                lane_width,
                speed_cap,
            )

            lateral = new_proj.lateral
            reward = config.progress_weight * progress * 10.0
            reward -= config.lateral_penalty * abs(lateral) * config.dt
            reward -= config.steer_penalty * abs(action.steer)
            reward -= config.accel_change_penalty * abs(action_index - prev_action_index) * 0.01
            if abs(lateral) > lane_width * 0.5:
                reward -= config.off_track_penalty
            if finished:
                reward += config.completion_bonus
            if done and not finished:
                # Off track is a terminal failure, not a soft penalty.
                reward -= config.collision_penalty

            next_live = track.project(new_state.x, new_state.y)
            next_key = state_key_from_live(new_state, next_live, track.heading_at(next_live.s))

            best_next = 0.0 if done else max(q_table[next_key])
            td_target = reward + config.gamma * best_next
            q_table[key][action_index] += config.alpha * (td_target - q_table[key][action_index])

            episode_reward += reward
            prev_action_index = action_index
            prev_lateral = lateral
            state = new_state

            if done:
                break

        rewards.append(round(episode_reward, 3))

    _ = prev_lateral  # referenced only to keep the loop readable
    _ = projection

    window = rewards[-50:] if len(rewards) >= 50 else rewards
    return TrainingResult(
        q_table=q_table,
        rewards=rewards,
        episodes=config.episodes,
        final_epsilon=config.epsilon_at(max(0, config.episodes - 1)),
        states_visited=len(visited),
        best_reward=max(rewards) if rewards else 0.0,
        mean_reward_last_50=sum(window) / len(window) if window else 0.0,
    )
