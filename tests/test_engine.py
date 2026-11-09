"""The simulation engine, end to end."""

from __future__ import annotations

import pytest

from sim.engine import SimulationEngine, run_scenario
from sim.scenario import (
    NPCConfig,
    ObstacleConfig,
    Scenario,
    builtin_scenarios,
    straight_road,
)


def test_every_builtin_scenario_is_valid():
    for scenario in builtin_scenarios():
        assert scenario.validate() == [], scenario.name


def test_run_produces_frames_and_observations():
    scenario = Scenario(name="t", track_points=straight_road(140.0), ego_speed=8.0, duration=10.0)
    result = run_scenario(scenario)
    assert len(result.frames) > 0
    assert result.metrics["observations"] > 0


def test_runs_are_deterministic():
    scenario = Scenario(name="t", track_points=straight_road(140.0), ego_speed=8.0, duration=8.0)
    first = run_scenario(scenario)
    second = run_scenario(scenario)
    assert first.frames == second.frames
    assert first.metrics == second.metrics


def test_open_road_completes_without_collision():
    scenario = Scenario(name="t", track_points=straight_road(140.0), ego_speed=8.0, duration=18.0)
    metrics = run_scenario(scenario).metrics
    assert metrics["collided"] is False
    assert metrics["completed"] is True


def test_an_impossible_stop_ends_in_a_collision():
    """30 m/s with 20 m to a wall is not survivable, and the engine says so."""
    scenario = Scenario(
        name="t",
        track_points=straight_road(300.0),
        ego_speed=30.0,
        duration=12.0,
        obstacles=[ObstacleConfig(s=20.0, radius=1.1, label="wall")],
    )
    metrics = run_scenario(scenario).metrics
    assert metrics["collided"] is True
    assert metrics["safety_score"] == 0.0
    assert metrics["collision"]["with"] == "wall"


def test_wet_weather_reduces_grip_and_speed_cap():
    dry = Scenario(name="d", track_points=straight_road(), weather="dry")
    wet = Scenario(name="w", track_points=straight_road(), weather="wet")
    assert wet.friction_scale < dry.friction_scale
    assert wet.speed_cap < dry.speed_cap


def test_invalid_scenario_raises():
    scenario = Scenario(name="", track_points=[(0.0, 0.0)])
    with pytest.raises(ValueError):
        run_scenario(scenario)


def test_unknown_agent_raises():
    scenario = Scenario(name="t", track_points=straight_road(140.0), duration=5.0)
    with pytest.raises(ValueError):
        run_scenario(scenario, agent="nope")


def test_qlearning_without_a_table_raises():
    scenario = Scenario(name="t", track_points=straight_road(140.0), duration=5.0)
    with pytest.raises(ValueError):
        run_scenario(scenario, agent="qlearning")


def test_traffic_appears_in_the_frames():
    scenario = Scenario(
        name="t",
        track_points=straight_road(300.0),
        ego_speed=14.0,
        duration=10.0,
        npcs=[NPCConfig(s=55.0, speed=6.0, desired_speed=6.5, label="lead")],
    )
    result = run_scenario(scenario)
    assert len(result.frames[0]["npcs"]) == 1


def test_frame_cap_truncates_and_flags_it():
    scenario = Scenario(name="t", track_points=straight_road(300.0), ego_speed=10.0, duration=30.0)
    result = SimulationEngine(max_frames=50).run(scenario)
    assert len(result.frames) == 50
    assert result.truncated is True


def test_track_points_are_returned_for_rendering():
    scenario = Scenario(name="t", track_points=straight_road(140.0), duration=5.0)
    result = run_scenario(scenario)
    assert len(result.track_points) == 2
    assert result.lane_width == pytest.approx(3.5)
