"""V3 regression tests: 62 observations, 22 learned actions, fixed-XML Reset."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from torch import nn

from rl_train.envs.action_adapters import ContralateralFixed20MotorArmsPDWrapper
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "rl_train/train/train_configs/myolegs22_motor_arms_62obs_128_contact_smooth_V3.json"
)


def load_v3_config():
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig,
        json.loads(CONFIG.read_text(encoding="utf-8")),
    )
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    return config


@pytest.fixture(scope="module")
def v3_env():
    config = load_v3_config()
    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=False, is_evaluate_mode=False
    )
    try:
        yield config, env
    finally:
        env.close()


def test_v3_interface_and_observation_block_order(v3_env):
    config, env = v3_env
    raw = env.unwrapped
    observation, info = env.reset(seed=1234)

    assert isinstance(env, ContralateralFixed20MotorArmsPDWrapper)
    assert info == {}
    assert env.observation_space.shape == (62,)
    assert env.action_space.shape == (22,)
    assert raw.sim.model.nu == 26
    assert observation.shape == (62,)
    assert np.all(np.isfinite(observation))
    assert raw._configured_obs_keys == [
        "qpos",
        "qvel",
        "act",
        "sensor",
        "target_velocity",
        "arm_qpos",
        "arm_qvel",
        "future_reference_error",
        "gait_phase",
    ]
    assert [np.asarray(raw.obs_dict[key]).size for key in raw._configured_obs_keys] == [
        8,
        9,
        22,
        4,
        1,
        4,
        4,
        8,
        2,
    ]
    assert config.policy_params.custom_policy_params.net_arch["human_actor"] == [128, 128]
    assert config.policy_params.custom_policy_params.net_arch["common_critic"] == [64, 64]


def test_v3_reset_is_complete_and_equality_consistent(v3_env):
    _, env = v3_env
    raw = env.unwrapped
    observation, _ = env.reset(seed=4321)
    assert observation.shape == (62,)

    for name in (
        "motor_shoulder_r",
        "motor_elbow_r",
        "motor_shoulder_l",
        "motor_elbow_l",
    ):
        assert raw.sim.data.joint(name).qpos[0] == pytest.approx(0.0, abs=1e-12)
        assert raw.sim.data.joint(name).qvel[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(raw.sim.data.ctrl, np.zeros(raw.sim.model.nu))
    np.testing.assert_array_equal(raw.sim.data.act, np.zeros(raw.sim.model.na))

    equality_rows = np.asarray(raw.sim.data.efc_type) == int(
        mujoco.mjtConstraint.mjCNSTR_EQUALITY
    )
    assert np.max(np.abs(np.asarray(raw.sim.data.efc_pos)[equality_rows])) < 1e-12


def test_future_reference_error_and_phase_are_synchronised(v3_env):
    config, env = v3_env
    raw = env.unwrapped
    env.reset(seed=2026)
    future_phase = raw._next_control_reference_phase()
    expected = np.asarray(
        [
            raw._reference_value(f"q_{name}", phase=future_phase)
            - float(raw.sim.data.joint(name).qpos[0])
            for name in config.env_params.future_reference_error_keys
        ]
    )
    np.testing.assert_allclose(
        raw.obs_dict["future_reference_error"], expected, rtol=0.0, atol=1e-12
    )
    assert np.linalg.norm(raw.obs_dict["gait_phase"]) == pytest.approx(1.0, abs=1e-12)


def test_trajectory_threshold_is_diagnostic_not_terminal(v3_env):
    _, env = v3_env
    raw = env.unwrapped
    env.reset(seed=123)
    original_threshold = raw._out_of_trajectory_threshold
    raw._out_of_trajectory_threshold = 0.0
    try:
        observation, _reward, terminated, truncated, info = env.step(
            np.zeros(22, dtype=np.float32)
        )
    finally:
        raw._out_of_trajectory_threshold = original_threshold
    assert observation.shape == (62,)
    assert info["trajectory_error_exceeded"] is True
    assert terminated is False
    assert truncated is False


def test_nonfinite_state_has_explicit_termination_reason(v3_env):
    _, env = v3_env
    raw = env.unwrapped
    env.reset(seed=321)
    raw.sim.data.qpos[0] = np.nan
    _observation, _reward, terminated, _truncated, info = env.step(
        np.zeros(22, dtype=np.float32)
    )
    assert terminated is True
    assert info["termination_reason"] == "numerical_instability"


def test_v3_actor_has_two_128_unit_hidden_layers(v3_env):
    config, env = v3_env
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)
    actor_linears = [
        layer
        for layer in model.policy.policy_network.policy_net
        if isinstance(layer, nn.Linear)
    ]
    critic_linears = [
        layer
        for layer in model.policy.policy_network.value_net
        if isinstance(layer, nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in actor_linears] == [
        (62, 128),
        (128, 128),
        (128, 22),
    ]
    assert [(layer.in_features, layer.out_features) for layer in critic_linears] == [
        (62, 64),
        (64, 64),
        (64, 1),
    ]
