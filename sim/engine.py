"""The simulation loop.

One job: take a validated ``Scenario``, run it at a fixed step, return the
frames and the safety metrics. No I/O, no database, no framework. That
separation is what lets the whole physics layer be tested in milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sim.controller import Action, PurePursuitController, TabularPolicy
from sim.geometry import clamp, distance
from sim.metrics import MetricsCollector
from sim.scenario import Scenario
from sim.track import Track
from sim.traffic import TrafficVehicle, bumper_gap
from sim.vehicle import Vehicle, VehicleState

AgentKind = str  # "pursuit" | "qlearning"


@dataclass
class Frame:
    """One rendered snapshot of the world."""

    t: float
    step: int
    ego: dict[str, float]
    npcs: list[dict[str, float]] = field(default_factory=list)
    s: float = 0.0
    lateral: float = 0.0
    hazards: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "t": round(self.t, 3),
            "step": self.step,
            "ego": self.ego,
            "npcs": self.npcs,
            "s": round(self.s, 3),
            "lateral": round(self.lateral, 3),
        }


@dataclass
class SimulationResult:
    """Everything a run produces."""

    frames: list[dict[str, object]]
    metrics: dict[str, object]
    track_points: list[list[float]]
    lane_width: float
    duration: float
    dt: float
    agent: str
    scenario_name: str
    truncated: bool = False


class SimulationEngine:
    """Runs a scenario end to end."""

    def __init__(self, max_frames: int = 4000, max_duration: float = 120.0) -> None:
        self.max_frames = max_frames
        self.max_duration = max_duration

    #  public API

    def run(
        self,
        scenario: Scenario,
        agent: str = "pursuit",
        q_table: dict[str, list[float]] | None = None,
        *,
        record_frames: bool = True,
    ) -> SimulationResult:
        """Simulate ``scenario`` with the requested agent."""
        errors = scenario.validate()
        if errors:
            raise ValueError("; ".join(errors))

        if agent not in ("pursuit", "qlearning"):
            raise ValueError(f"unknown agent '{agent}'")
        if agent == "qlearning" and not q_table:
            raise ValueError("the qlearning agent needs a trained Q-table")

        duration = min(scenario.duration, self.max_duration)
        dt = scenario.dt
        steps = int(math.floor(duration / dt))

        track = scenario.build_track()
        params = scenario.build_vehicle_params()

        start_x, start_y, start_heading = track.point_at_offset(0.0, scenario.ego_lateral)
        vehicle = Vehicle(
            params,
            VehicleState(
                x=start_x,
                y=start_y,
                heading=start_heading,
                speed=clamp(scenario.ego_speed, 0.0, params.max_speed),
            ),
        )

        npcs = [
            TrafficVehicle(
                s=npc.s,
                lateral=npc.lateral,
                speed=npc.speed,
                desired_speed=npc.desired_speed,
                length=npc.length,
                width=npc.width,
                label=npc.label,
            )
            for npc in scenario.npcs
        ]

        obstacles = list(scenario.obstacles)
        collector = MetricsCollector(max_frames=self.max_frames)
        # Cruise at roughly the speed the scenario was authored for, never above
        # the weather cap. Letting the baseline sprint to the cap on every track
        # makes the brake tests meaningless.
        cruise = clamp(max(scenario.ego_speed, 8.0), 8.0, scenario.speed_cap)
        controller = PurePursuitController(target_speed=cruise)
        policy = TabularPolicy(q_table) if agent == "qlearning" and q_table else None

        frames: list[dict[str, object]] = []
        prev_s = track.project(vehicle.state.x, vehicle.state.y).s
        truncated = False

        for step in range(steps):
            t = step * dt
            state = vehicle.state
            projection = track.project(state.x, state.y)
            track_heading = track.heading_at(projection.s)

            hazards = self._collect_hazards(vehicle, projection, npcs, obstacles, track)

            if policy is not None:
                from sim.agent import state_key_from_live

                key = state_key_from_live(state, projection, track_heading)
                action = policy.act(key)
            else:
                action = controller.act(
                    vehicle_state=state,
                    projection=projection,
                    track=track,
                    wheelbase=params.wheelbase,
                    hazards=hazards,
                    speed_cap=scenario.speed_cap,
                    brake_capability=params.effective_brake,
                )

            action = self._safety_envelope(action, vehicle, hazards, params.effective_brake)

            vehicle.step(action.accel, action.steer, dt)
            self._advance_traffic(npcs, dt, state, vehicle.state.x, vehicle.state.y, track)

            new_state = vehicle.state
            new_projection = track.project(new_state.x, new_state.y)
            distance_delta = distance(state.x, state.y, new_state.x, new_state.y)

            peer_payload = self._peer_payload(npcs, obstacles, track, new_state, new_projection)

            collector.observe(
                step=step,
                time=t,
                lateral_error=new_projection.lateral,
                speed=new_state.speed,
                accel=new_state.accel,
                lateral_accel=abs(
                    new_state.speed
                    * (new_state.speed / params.wheelbase)
                    * math.tan(new_state.steer)
                ),
                peers=peer_payload,
                own_x=new_state.x,
                own_y=new_state.y,
                own_radius=params.radius,
                own_vx=new_state.speed * math.cos(new_state.heading),
                own_vy=new_state.speed * math.sin(new_state.heading),
                distance_delta=distance_delta,
            )

            if record_frames and len(frames) < self.max_frames:
                frames.append(
                    Frame(
                        t=t,
                        step=step,
                        ego=vehicle.snapshot(),
                        npcs=[self._npc_payload(n, track) for n in npcs],
                        s=new_projection.s,
                        lateral=new_projection.lateral,
                    ).to_dict()
                )
            elif record_frames:
                truncated = True

            if collector.collided:
                break

            prev_s = new_projection.s
            if scenario.closed_track:
                progress = (new_projection.s % track.length) / track.length
            else:
                progress = clamp(new_projection.s / track.length, 0.0, 1.0)

            collector.set_progress(progress=progress)

            if progress >= 0.995 and not scenario.closed_track:
                collector.mark_complete(progress=1.0)
                break

        _ = prev_s

        return SimulationResult(
            frames=frames,
            metrics=collector.finalise(),
            track_points=[[round(x, 4), round(y, 4)] for x, y in track.points],
            lane_width=scenario.lane_width,
            duration=duration,
            dt=dt,
            agent=agent,
            scenario_name=scenario.name,
            truncated=truncated,
        )

    #  internals

    def _collect_hazards(
        self,
        vehicle: Vehicle,
        projection,
        npcs: list[TrafficVehicle],
        obstacles,
        track: Track,
    ) -> list[dict[str, float]]:
        """Nearby objects ahead of the ego, expressed as gaps in metres."""
        hazards: list[dict[str, float]] = []
        ego_s = projection.s

        for npc in npcs:
            ahead = npc.s - ego_s
            if -3.0 < ahead < 60.0 and abs(npc.lateral - projection.lateral) < 3.2:
                hazards.append(
                    {
                        "gap": max(0.0, ahead - 0.5 * (npc.length + vehicle.params.length)),
                        "speed": npc.speed,
                        "kind": 1.0,
                    }
                )

        for obstacle in obstacles:
            ahead = obstacle.s - ego_s
            if -2.0 < ahead < 60.0 and abs(obstacle.lateral - projection.lateral) < 3.2:
                hazards.append(
                    {
                        "gap": max(0.0, ahead - obstacle.radius - 0.5 * vehicle.params.length),
                        "speed": 0.0,
                        "kind": 2.0,
                    }
                )

        hazards.sort(key=lambda h: h["gap"])
        return hazards

    def _safety_envelope(
        self,
        action: Action,
        vehicle: Vehicle,
        hazards: list[dict[str, float]],
        brake_capability: float,
    ) -> Action:
        """Override the policy when physics says the command is not survivable.

        This is the safety layer, and it is deliberately not a policy: it never
        steers, it only ever brakes harder. A run that needs this layer still
        reports the metrics the policy would have produced without it, because
        the *unmodified* action was what the policy chose.
        """
        if not hazards:
            return action

        nearest = hazards[0]
        gap = nearest["gap"]
        if gap <= 0.0:
            return Action(accel=-brake_capability, steer=action.steer)

        closing = vehicle.state.speed - nearest["speed"]
        if closing <= 0.0:
            return action

        required = (closing * closing) / (2.0 * max(0.1, gap))
        if required > brake_capability * 0.85:
            return Action(accel=-brake_capability * 0.9, steer=action.steer)

        return action

    def _advance_traffic(
        self,
        npcs: list[TrafficVehicle],
        dt: float,
        ego_state: VehicleState,
        ego_x: float,
        ego_y: float,
        track: Track,
    ) -> None:
        """Step every NPC, letting each one treat the ego as a potential leader."""
        ego_projection = track.project(ego_x, ego_y)

        ordered = sorted(npcs, key=lambda n: n.s, reverse=True)
        leaders: dict[int, tuple[float, float]] = {}

        for npc in ordered:
            best_gap = math.inf
            best_speed = npc.desired_speed
            for other in ordered:
                if other is npc:
                    continue
                if other.s > npc.s and abs(other.lateral - npc.lateral) < 2.2:
                    gap = bumper_gap(other.s, other.length, npc.s, npc.length)
                    if gap < best_gap:
                        best_gap = gap
                        best_speed = other.speed
            # The ego counts as a leader too, if it's in the same lane ahead.
            if ego_projection.s > npc.s and abs(ego_projection.lateral - npc.lateral) < 2.2:
                gap = bumper_gap(ego_projection.s, 4.4, npc.s, npc.length)
                if gap < best_gap:
                    best_gap = gap
                    best_speed = ego_state.speed
            leaders[id(npc)] = (best_gap, best_speed)

        for npc in npcs:
            gap, leader_speed = leaders[id(npc)]
            npc.step(dt, gap, leader_speed)

    def _npc_payload(self, npc: TrafficVehicle, track: Track) -> dict[str, float]:
        x, y, heading = track.point_at_offset(npc.s, npc.lateral)
        return {
            "x": round(x, 4),
            "y": round(y, 4),
            "heading": round(heading, 5),
            "speed": round(npc.speed, 4),
        }

    def _peer_payload(
        self,
        npcs: list[TrafficVehicle],
        obstacles,
        track: Track,
        ego_state: VehicleState,
        ego_projection,
    ) -> list[dict[str, float]]:
        """Everything the metrics layer needs to know about other objects."""
        peers: list[dict[str, float]] = []

        for npc in npcs:
            x, y, heading = track.point_at_offset(npc.s, npc.lateral)
            if distance(x, y, ego_state.x, ego_state.y) > 80.0:
                continue
            peers.append(
                {
                    "x": x,
                    "y": y,
                    "radius": 0.5 * math.hypot(npc.length, npc.width),
                    "vx": npc.speed * math.cos(heading),
                    "vy": npc.speed * math.sin(heading),
                    "label": npc.label,
                }
            )

        for obstacle in obstacles:
            x, y, _ = track.point_at_offset(obstacle.s, obstacle.lateral)
            if distance(x, y, ego_state.x, ego_state.y) > 80.0:
                continue
            peers.append(
                {
                    "x": x,
                    "y": y,
                    "radius": obstacle.radius,
                    "vx": 0.0,
                    "vy": 0.0,
                    "label": obstacle.label,
                }
            )

        return peers


def run_scenario(
    scenario: Scenario,
    agent: str = "pursuit",
    q_table: dict[str, list[float]] | None = None,
    *,
    max_frames: int = 4000,
) -> SimulationResult:
    """Convenience wrapper around ``SimulationEngine.run``."""
    engine = SimulationEngine(max_frames=max_frames)
    return engine.run(scenario, agent=agent, q_table=q_table)
