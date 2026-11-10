"""V3-B forward-reward and trajectory-grace regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "rl_train/train/train_configs/"
    "myolegs22_motor_arms_62obs_128_contact_smooth_V3b.json"
)


def load_v3b_config():
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig,
        json.loads(CONFIG.read_text(encoding="utf-8")),
    )
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    return config


@pytest.fixture(scope="module")
def v3b_env():
    config = load_v3b_config()
    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=False, is_evaluate_mode=False
    )
    try:
        yield config, env
    finally:
        env.close()


@pytest.mark.parametrize(
    ("velocity", "expected"),
    [
        (0.0, 0.0009948243843538613),
        (0.5, 0.012286071058743077),
        (1.25, 0.032),
        (1.5, 0.030317429069029917),
    ],
)
def test_v3b_forward_reward_exact_values(v3b_env, velocity, expected):
    config, env = v3b_env
    raw = env.unwrapped
    raw._target_velocity = 1.25
    raw.sim.data.joint("pelvis_tx").qvel[0] = velocity

    component = raw._calculate_forward_reward()

    assert raw.dt == pytest.approx(0.02, abs=1e-12)
    assert component == pytest.approx(expected, rel=0.0, abs=1e-14)
    # The component already contains w_p, w_t and dt, so the dense-reward
    # aggregator must include it once with unit outer weight.
    assert config.env_params.reward_keys_and_weights.forward_reward == 1.0
    assert raw.rwd_keys_wt["forward_reward"] == 1.0
    assert raw.rwd_keys_wt["forward_reward"] * component == pytest.approx(
        expected, rel=0.0, abs=1e-14
    )


def test_v3b_interface_is_unchanged(v3b_env):
    _, env = v3b_env
    observation, _ = env.reset(seed=1234)
    assert observation.shape == (62,)
    assert env.observation_space.shape == (62,)
    assert env.action_space.shape == (22,)
    assert env.unwrapped.sim.model.nu == 26


def test_trajectory_grace_terminates_on_tenth_consecutive_step(
    v3b_env, monkeypatch
):
    _, env = v3b_env
    raw = env.unwrapped
    env.reset(seed=11)
    monkeypatch.setattr(raw, "_get_done", lambda: False)
    monkeypatch.setattr(raw, "_get_qpos_diff", lambda: {"pelvis_tilt": 0.61})

    action = np.zeros(22, dtype=np.float32)
    for expected_count in range(1, 10):
        _obs, _reward, terminated, truncated, info = env.step(action)
        assert terminated is False
        assert truncated is False
        assert info["trajectory_error_consecutive_count"] == expected_count

    _obs, _reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert truncated is False
    assert info["trajectory_error_consecutive_count"] == 10
    assert info["termination_reason"] == "joint_out_of_trajectory:pelvis_tilt"


def test_trajectory_grace_clears_after_recovery(v3b_env, monkeypatch):
    _, env = v3b_env
    raw = env.unwrapped
    env.reset(seed=12)
    monkeypatch.setattr(raw, "_get_done", lambda: False)
    error = [0.61]
    monkeypatch.setattr(
        raw, "_get_qpos_diff", lambda: {"pelvis_tilt": error[0]}
    )
    action = np.zeros(22, dtype=np.float32)

    for _ in range(5):
        _obs, _reward, terminated, truncated, _info = env.step(action)
        assert not terminated and not truncated
    assert raw._trajectory_error_consecutive_count == 5

    error[0] = 0.0
    _obs, _reward, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert info["trajectory_error_consecutive_count"] == 0
    assert raw._trajectory_error_consecutive_count == 0

    error[0] = 0.61
    for expected_count in range(1, 10):
        _obs, _reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert info["trajectory_error_consecutive_count"] == expected_count


def test_reset_clears_trajectory_grace_counter(v3b_env):
    _, env = v3b_env
    raw = env.unwrapped
    raw._trajectory_error_consecutive_count = 9
    observation, _ = env.reset(seed=13)
    assert observation.shape == (62,)
    assert raw._trajectory_error_consecutive_count == 0
