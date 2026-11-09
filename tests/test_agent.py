"""Tabular Q-learning agent."""

from __future__ import annotations

import pytest

from sim.agent import (
    STATE_COUNT,
    TrainingConfig,
    empty_q_table,
    state_key,
    train,
)
from sim.controller import (
    ACCEL_CHOICES,
    ACTION_COUNT,
    STEER_CHOICES,
    decode_action,
    encode_action,
)
from sim.scenario import straight_road
from sim.track import Track


def test_action_count():
    assert ACTION_COUNT == 9


def test_state_count():
    assert STATE_COUNT == 140


def test_decode_action_corners():
    first = decode_action(0)
    assert (first.accel, first.steer) == (ACCEL_CHOICES[0], STEER_CHOICES[0])
    last = decode_action(8)
    assert (last.accel, last.steer) == (ACCEL_CHOICES[2], STEER_CHOICES[2])


def test_decode_action_neutral():
    neutral = decode_action(4)
    assert (neutral.accel, neutral.steer) == (0.0, 0.0)


def test_decode_action_rejects_out_of_range():
    with pytest.raises(ValueError):
        decode_action(9)


def test_encode_round_trips_every_action():
    for index in range(ACTION_COUNT):
        action = decode_action(index)
        assert encode_action(action.accel, action.steer) == index


def test_empty_q_table_covers_every_state():
    assert len(empty_q_table()) == STATE_COUNT


def test_state_key_buckets():
    assert state_key(0.0, 0.0, 0.0) == "3:2:0"
    assert state_key(5.0, 5.0, 50.0) == "6:4:3"


def test_epsilon_decays_toward_the_floor():
    config = TrainingConfig(episodes=100)
    assert config.epsilon_at(0) > config.epsilon_at(50)
    assert config.epsilon_at(500) == pytest.approx(config.epsilon_end)


def test_training_is_deterministic():
    track = Track(straight_road(120.0))
    config = TrainingConfig(episodes=25, steps_per_episode=40, seed=3)
    first = train(config, track)
    second = train(config, track)
    assert first.q_table == second.q_table
    assert first.rewards == second.rewards


def test_training_produces_a_reward_curve():
    track = Track(straight_road(120.0))
    result = train(TrainingConfig(episodes=20, steps_per_episode=30), track)
    assert len(result.rewards) == 20
    assert result.states_visited > 0


def test_policy_grid_is_complete():
    track = Track(straight_road(120.0))
    result = train(TrainingConfig(episodes=10, steps_per_episode=20), track)
    rows = result.policy_grid()
    assert len(rows) == STATE_COUNT
    assert {"state", "best_action", "accel", "steer", "value"} <= set(rows[0])


def test_summary_reports_the_expected_fields():
    track = Track(straight_road(120.0))
    summary = train(TrainingConfig(episodes=10, steps_per_episode=20), track).summary()
    for key in ("episodes", "states_visited", "state_count", "reward_curve"):
        assert key in summary
