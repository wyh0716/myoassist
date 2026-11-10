"""Regression tests for V2 continuation on the final MotorArms model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from rl_train.envs.action_adapters import (
    MOTOR_ARM_ACTUATORS,
    MOTOR_ARM_CONTROL_CONTRALATERAL_FIXED20_PD,
    ContralateralFixed20MotorArmsPDWrapper,
    ZeroMotorArmsActionWrapper,
)
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "rl_train/train/train_configs/human_only_myolegs22_imitation_50hz_V2.json"
)
MODEL_PATH = REPO_ROOT / "artifacts/myolegs22_planar_motor_arms.xml"
V2_CHECKPOINT = (
    REPO_ROOT
    / "rl_train/results/train_session_20261022-225758/trained_models/final_model.zip"
)


def _load_config(
    motor_arms_control_mode: str = "zero",
    smooth_start_seconds: float = 0.0,
):
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig,
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
    )
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    config.env_params.msk_key = None
    config.env_params.device_key = None
    config.env_params.model_path = str(MODEL_PATH)
    config.env_params.motor_arms_control_mode = motor_arms_control_mode
    config.env_params.motor_arms_pd_smooth_start_seconds = smooth_start_seconds
    return config


@pytest.fixture(scope="module")
def motor_arms_training_env():
    assert MODEL_PATH.is_file()
    config = _load_config()
    env = EnvironmentHandler.create_environment(
        config,
        is_rendering_on=False,
        is_evaluate_mode=True,
    )
    try:
        yield config, env
    finally:
        env.close()


@pytest.fixture(scope="module")
def motor_arms_pd_training_env():
    config = _load_config(MOTOR_ARM_CONTROL_CONTRALATERAL_FIXED20_PD)
    env = EnvironmentHandler.create_environment(
        config,
        is_rendering_on=False,
        is_evaluate_mode=True,
    )
    try:
        yield config, env
    finally:
        env.close()


@pytest.fixture(scope="module")
def motor_arms_smooth_pd_training_env():
    config = _load_config(
        MOTOR_ARM_CONTROL_CONTRALATERAL_FIXED20_PD,
        smooth_start_seconds=0.5,
    )
    env = EnvironmentHandler.create_environment(
        config,
        is_rendering_on=False,
        is_evaluate_mode=True,
    )
    try:
        yield config, env
    finally:
        env.close()


def test_v2_policy_interface_maps_22_actions_to_26_physical_controls(
    motor_arms_training_env,
):
    _, env = motor_arms_training_env
    raw = env.unwrapped

    assert isinstance(env, ZeroMotorArmsActionWrapper)
    assert env.action_space.shape == (22,)
    assert env.physical_action_space.shape == (26,)
    assert raw.sim.model.nu == 26
    assert tuple(raw.sim.model.actuator(index).name for index in range(22, 26)) == (
        MOTOR_ARM_ACTUATORS
    )

    muscle_action = np.linspace(-0.75, 0.75, 22, dtype=np.float32)
    physical_action = env.action(muscle_action)
    np.testing.assert_array_equal(physical_action[:22], muscle_action)
    np.testing.assert_array_equal(physical_action[22:], np.zeros(4))

    env.reset(seed=1234)
    env.step(muscle_action)
    np.testing.assert_array_equal(raw.sim.data.ctrl[22:], np.zeros(4))

    with pytest.raises(ValueError, match="exactly 22"):
        env.action(np.zeros(26, dtype=np.float32))


def test_every_episode_restores_arm_keyframe_and_zero_motor_control(
    motor_arms_training_env,
):
    _, env = motor_arms_training_env
    raw = env.unwrapped
    model = raw.sim.model
    data = raw.sim.data

    for episode in range(8):
        observation, info = env.reset(seed=6000 + episode)
        assert observation.shape == (44,)
        assert info == {}
        for name in MOTOR_ARM_ACTUATORS:
            joint = model.joint(name)
            qadr = int(model.jnt_qposadr[joint.id])
            dadr = int(model.jnt_dofadr[joint.id])
            assert data.qpos[qadr] == pytest.approx(model.key_qpos[0, qadr], abs=1e-12)
            assert data.qvel[dadr] == pytest.approx(model.key_qvel[0, dadr], abs=1e-12)
        np.testing.assert_array_equal(data.ctrl[22:], np.zeros(4))

        # Deliberately contaminate every unreferenced arm state.  The next
        # public reset must not inherit any of it.
        for name in MOTOR_ARM_ACTUATORS:
            data.joint(name).qpos[0] = 0.35
            data.joint(name).qvel[0] = 2.5
        data.ctrl[22:] = 1.0


def test_fixed20_pd_keeps_22_policy_actions_and_appends_four_arm_commands(
    motor_arms_pd_training_env,
):
    config, env = motor_arms_pd_training_env
    raw = env.unwrapped
    assert isinstance(env, ContralateralFixed20MotorArmsPDWrapper)
    assert env.action_space.shape == (22,)
    assert env.physical_action_space.shape == (26,)

    observation, info = env.reset(seed=1234)
    assert observation.shape == (44,)
    assert info == {}
    np.testing.assert_array_equal(raw.sim.data.ctrl[22:], np.zeros(4))

    muscle_action = np.linspace(-0.5, 0.5, 22, dtype=np.float32)
    physical_action = env.action(muscle_action)
    np.testing.assert_array_equal(physical_action[:22], muscle_action)
    np.testing.assert_array_equal(physical_action[22:], env.last_arm_controls)
    assert np.any(np.abs(physical_action[22:]) > 0.0)

    amplitude = np.deg2rad(config.env_params.motor_arms_shoulder_amplitude_deg)
    np.testing.assert_array_less(np.abs(env.last_arm_targets_rad[[0, 2]]), amplitude + 1e-12)
    np.testing.assert_allclose(
        env.last_arm_targets_rad[[1, 3]],
        np.deg2rad(config.env_params.motor_arms_elbow_target_deg),
        rtol=0.0,
        atol=1e-12,
    )

    for shoulder_side, opposite_hip_side, slot in (("r", "l", 0), ("l", "r", 2)):
        hip_angle = float(raw.sim.data.joint(f"hip_flexion_{opposite_hip_side}").qpos[0])
        expected = amplitude * np.clip(
            (hip_angle - env.hip_center_rad[opposite_hip_side])
            / env.hip_scale_rad[opposite_hip_side],
            -1.0,
            1.0,
        )
        assert env.last_arm_targets_rad[slot] == pytest.approx(expected)

    env.step(muscle_action)
    np.testing.assert_allclose(raw.sim.data.ctrl[22:], env.last_arm_controls)


def test_fixed20_pd_saturation_has_zero_target_velocity_and_reset_clears_control(
    motor_arms_pd_training_env,
):
    _, env = motor_arms_pd_training_env
    raw = env.unwrapped
    env.reset(seed=777)
    raw.sim.data.joint("hip_flexion_l").qpos[0] = env.hip_center_rad["l"] + 10.0
    raw.sim.data.joint("hip_flexion_l").qvel[0] = 100.0
    env.action(np.zeros(22, dtype=np.float32))
    assert env.last_arm_targets_rad[0] == pytest.approx(env.shoulder_amplitude_rad)
    assert env.last_arm_target_velocities_rad_s[0] == pytest.approx(0.0)

    observation, _ = env.reset(seed=778)
    assert observation.shape == (44,)
    np.testing.assert_array_equal(raw.sim.data.ctrl[22:], np.zeros(4))
    np.testing.assert_array_equal(env.last_arm_controls, np.zeros(4))


def test_fixed20_pd_quintic_smooth_start_begins_at_reset_pose_and_velocity(
    motor_arms_smooth_pd_training_env,
):
    config, env = motor_arms_smooth_pd_training_env
    raw = env.unwrapped
    env.reset(seed=1234)
    reset_qpos = np.asarray(
        [float(raw.sim.data.joint(name).qpos[0]) for name in MOTOR_ARM_ACTUATORS]
    )
    physical_action = env.action(np.zeros(22, dtype=np.float32))

    np.testing.assert_allclose(env.last_arm_targets_rad, reset_qpos, atol=1e-12)
    np.testing.assert_allclose(env.last_arm_target_velocities_rad_s, 0.0, atol=1e-12)
    np.testing.assert_allclose(env.last_arm_requested_torques_nm, 0.0, atol=1e-12)
    np.testing.assert_allclose(physical_action[22:], 0.0, atol=1e-12)
    assert env.smooth_start_elapsed_seconds == pytest.approx(raw.dt)

    # At the midpoint, both target position and target velocity must use the
    # same quintic blend.  For a fixed elbow target, qdot_ref is entirely the
    # blend-rate term.
    env.smooth_start_elapsed_seconds = 0.25
    env.action(np.zeros(22, dtype=np.float32))
    blend, derivative_unit = env._quintic_blend(0.5)
    elbow_target = np.deg2rad(config.env_params.motor_arms_elbow_target_deg)
    expected_q = reset_qpos[1] + blend * (elbow_target - reset_qpos[1])
    expected_qvel = derivative_unit / 0.5 * (elbow_target - reset_qpos[1])
    assert env.last_arm_targets_rad[1] == pytest.approx(expected_q)
    assert env.last_arm_target_velocities_rad_s[1] == pytest.approx(expected_qvel)

    env.smooth_start_elapsed_seconds = 0.5
    env.action(np.zeros(22, dtype=np.float32))
    np.testing.assert_allclose(
        env.last_arm_targets_rad,
        env.last_arm_unsmoothed_targets_rad,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        env.last_arm_target_velocities_rad_s,
        env.last_arm_unsmoothed_target_velocities_rad_s,
        atol=1e-12,
    )


def test_fixed20_pd_distinguishes_requested_commanded_and_actual_motor_torque(
    motor_arms_smooth_pd_training_env,
):
    _, env = motor_arms_smooth_pd_training_env
    raw = env.unwrapped
    env.reset(seed=4321)
    env.smooth_start_elapsed_seconds = 0.5
    action = np.zeros(22, dtype=np.float32)
    env.step(action)

    expected_commanded = env.last_arm_controls * np.asarray(
        [
            float(
                raw.sim.model.actuator_gear[
                    raw.sim.model.actuator(name).id, 0
                ]
            )
            for name in MOTOR_ARM_ACTUATORS
        ]
    )
    np.testing.assert_allclose(env.last_arm_commanded_torques_nm, expected_commanded)
    actual = np.asarray(
        [
            float(raw.sim.data.actuator(name).force[0])
            * float(raw.sim.model.actuator_gear[raw.sim.model.actuator(name).id, 0])
            for name in MOTOR_ARM_ACTUATORS
        ]
    )
    np.testing.assert_allclose(actual, env.last_arm_commanded_torques_nm, atol=1e-6)
    # Requested torque is the unclipped PD result, while commanded/actual are
    # after conversion through gear and ctrlrange.
    assert np.all(np.isfinite(env.last_arm_requested_torques_nm))


@pytest.mark.skipif(not V2_CHECKPOINT.is_file(), reason="local V2 checkpoint is unavailable")
def test_v2_checkpoint_loads_against_motor_arms_training_interface(
    motor_arms_training_env,
):
    _, env = motor_arms_training_env
    model = PPO.load(str(V2_CHECKPOINT), env=env)
    assert model.observation_space.shape == (44,)
    assert model.action_space.shape == (22,)


def test_vector_training_workers_keep_the_22_action_contract(monkeypatch):
    class InlineWorkerProbe:
        """Construct worker factories without requiring sandbox IPC sockets."""

        def __init__(self, factories):
            self.envs = [factory() for factory in factories]
            self.num_envs = len(self.envs)
            self.action_space = self.envs[0].action_space
            self.observation_space = self.envs[0].observation_space

        def close(self):
            for worker in self.envs:
                worker.close()

    monkeypatch.setattr(
        "rl_train.envs.environment_handler.SubprocVecEnv",
        InlineWorkerProbe,
    )
    config = _load_config()
    config.env_params.num_envs = 2
    env = EnvironmentHandler.create_environment(
        config,
        is_rendering_on=False,
        is_evaluate_mode=False,
    )
    try:
        assert env.num_envs == 2
        assert env.action_space.shape == (22,)
        assert env.observation_space.shape == (44,)
        assert all(isinstance(worker, ZeroMotorArmsActionWrapper) for worker in env.envs)
        for worker in env.envs:
            observation, _ = worker.reset()
            assert observation.shape == (44,)
            worker.step(np.zeros(22, dtype=np.float32))
            np.testing.assert_array_equal(
                worker.unwrapped.sim.data.ctrl[22:],
                np.zeros(4),
            )
    finally:
        env.close()


def test_vector_training_workers_select_fixed20_pd_from_config(monkeypatch):
    class InlineWorkerProbe:
        def __init__(self, factories):
            self.envs = [factory() for factory in factories]
            self.num_envs = len(self.envs)
            self.action_space = self.envs[0].action_space
            self.observation_space = self.envs[0].observation_space

        def close(self):
            for worker in self.envs:
                worker.close()

    monkeypatch.setattr(
        "rl_train.envs.environment_handler.SubprocVecEnv",
        InlineWorkerProbe,
    )
    config = _load_config(MOTOR_ARM_CONTROL_CONTRALATERAL_FIXED20_PD)
    config.env_params.num_envs = 8
    env = EnvironmentHandler.create_environment(
        config,
        is_rendering_on=False,
        is_evaluate_mode=False,
    )
    try:
        assert env.num_envs == 8
        assert env.action_space.shape == (22,)
        assert env.observation_space.shape == (44,)
        assert all(
            isinstance(worker, ContralateralFixed20MotorArmsPDWrapper)
            for worker in env.envs
        )
        for rank, worker in enumerate(env.envs):
            observation, _ = worker.reset(seed=1234 + rank)
            assert observation.shape == (44,)
            physical_action = worker.action(np.zeros(22, dtype=np.float32))
            assert physical_action.shape == (26,)
            assert np.any(np.abs(physical_action[22:]) > 0.0)
    finally:
        env.close()
