"""Safety metric collection and scoring."""

from __future__ import annotations

import math

import pytest

from sim.metrics import MetricsCollector, Peak


def test_peak_tracks_maximum():
    peak = Peak(value=0.0)
    peak.observe(5.0, 1, 0.1, maximum=True)
    peak.observe(3.0, 2, 0.2, maximum=True)
    assert peak.value == 5.0
    assert peak.step == 1


def test_peak_tracks_minimum():
    peak = Peak(value=math.inf)
    peak.observe(5.0, 1, 0.1, maximum=False)
    peak.observe(3.0, 2, 0.2, maximum=False)
    assert peak.value == 3.0
    assert peak.step == 2


def test_peak_to_dict():
    payload = Peak(value=1.23456, step=3, time=0.15).to_dict()
    assert payload["value"] == pytest.approx(1.2346, abs=1e-4)
    assert payload["step"] == 3


def observe(collector: MetricsCollector, **overrides) -> None:
    """One observation with sensible defaults, overridable per test."""
    payload = {
        "step": 0,
        "time": 0.0,
        "lateral_error": 0.0,
        "speed": 10.0,
        "accel": 0.0,
        "lateral_accel": 0.0,
        "peers": [],
        "own_x": 0.0,
        "own_y": 0.0,
        "own_radius": 1.0,
        "own_vx": 10.0,
        "own_vy": 0.0,
        "distance_delta": 0.5,
    }
    payload.update(overrides)
    collector.observe(**payload)


def test_clean_run_scores_full_marks():
    collector = MetricsCollector()
    observe(collector)
    assert collector.safety_score() == 100.0


def test_collision_scores_zero():
    collector = MetricsCollector()
    observe(
        collector,
        peers=[{"x": 0.5, "y": 0.0, "radius": 1.0, "vx": 0.0, "vy": 0.0, "label": "car"}],
    )
    assert collector.collided is True
    assert collector.safety_score() == 0.0


def test_collision_records_label_and_speed():
    collector = MetricsCollector()
    observe(
        collector,
        speed=12.0,
        peers=[{"x": 0.5, "y": 0.0, "radius": 1.0, "vx": 0.0, "vy": 0.0, "label": "stalled-car"}],
    )
    metrics = collector.finalise()
    assert metrics["collision"]["with"] == "stalled-car"
    assert metrics["collision"]["speed_mps"] == pytest.approx(12.0)


def test_rms_lateral_error():
    collector = MetricsCollector()
    observe(collector, lateral_error=1.0)
    observe(collector, lateral_error=-1.0)
    assert collector.rms_lateral_error == pytest.approx(1.0)


def test_min_ttc_is_recorded_for_a_closing_peer():
    collector = MetricsCollector()
    observe(
        collector,
        peers=[{"x": 20.0, "y": 0.0, "radius": 1.0, "vx": 0.0, "vy": 0.0, "label": "car"}],
    )
    # gap 18 m, closing at 10 m/s -> 1.8 s
    assert collector.min_ttc.value == pytest.approx(1.8)


def test_finalise_has_the_documented_keys():
    collector = MetricsCollector()
    observe(collector)
    metrics = collector.finalise()
    for key in (
        "collided",
        "completed",
        "progress",
        "safety_score",
        "min_ttc_s",
        "min_gap_m",
        "max_lateral_error_m",
        "rms_lateral_error_m",
        "max_jerk_mps3",
    ):
        assert key in metrics


def test_progress_setter_records_partial_runs():
    collector = MetricsCollector()
    collector.set_progress(progress=0.42)
    assert collector.finalise()["progress"] == pytest.approx(0.42)


def test_mark_complete():
    collector = MetricsCollector()
    collector.mark_complete(progress=1.0)
    metrics = collector.finalise()
    assert metrics["completed"] is True
    assert metrics["progress"] == 1.0


def test_jerk_penalises_the_score():
    collector = MetricsCollector()
    observe(collector, accel=0.0)
    observe(collector, accel=8.0)
    assert collector.max_jerk.value > 0.0
    assert collector.safety_score() < 100.0


def test_lane_excursion_penalises_the_score():
    collector = MetricsCollector()
    observe(collector, lateral_error=2.5)
    assert collector.safety_score() < 100.0
