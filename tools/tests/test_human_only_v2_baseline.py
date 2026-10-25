"""Regression tests for the corrected second-round human-only gait baseline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.evaluate_checkpoint_multistart import (
    _maximum_training_start,
    _select_start_indices,
    _set_reference_start,
)
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.learning_callback import BaseCustomLearningCallback


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "rl_train/train/train_configs/human_only_myolegs22_imitation_v2_50hz.json"
REFERENCE_JOINTS = (
    "ankle_angle_l",
    "ankle_angle_r",
    "hip_flexion_l",
    "hip_flexion_r",
    "knee_angle_l",
    "knee_angle_r",
    "pelvis_tilt",
    "pelvis_tx",
    "pelvis_ty",
)


def _load_config():
    return DictionableDataclass.create(ImitationTrainSessionConfig, json.loads(CONFIG_PATH.read_text()))


@pytest.fixture(scope="module")
def human_env():
    config = _load_config()
    config.env_params.num_envs = 1
    config.ppo_params.n_steps = 32
    config.ppo_params.batch_size = 32
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    try:
        yield config, env.unwrapped
    finally:
        env.close()


def test_v2_config_changes_only_requested_training_contract(human_env):
    config, env = human_env
    weights = config.env_params.reward_keys_and_weights

    assert weights.forward_reward == pytest.approx(0.8)
    assert weights.muscle_activation_penalty == pytest.approx(0.05)
    assert weights.muscle_activation_diff_penalty == pytest.approx(0.01)
    assert weights.joint_constraint_force_penalty == pytest.approx(0.0)
    assert weights.foot_force_penalty == pytest.approx(0.01)
    assert env.rwd_keys_wt["qpos_imitation_rewards"] == pytest.approx(1.0)
    assert env.rwd_keys_wt["qvel_imitation_rewards"] == pytest.approx(1.0)

    obs, _ = env.reset(seed=config.env_params.seed)
    assert obs.shape == (44,)
    assert env.action_space.shape == (22,)
    assert config.policy_params.custom_policy_params.net_arch == {
        "human_actor": [64, 64],
        "common_critic": [64, 64],
    }


def test_50_hz_control_timing_and_reference_timeline(human_env):
    config, env = human_env
    metadata = env._reference_data["metadata"]
    series = env._reference_data["series_data"]

    assert env.sim.model.opt.timestep == pytest.approx(1 / 2000)
    assert env.frame_skip == 40
    assert env.dt == pytest.approx(1 / 50)
    assert config.env_params.custom_max_episode_steps * env.dt == pytest.approx(33.34)

    assert metadata["original_sample_rate"] == pytest.approx(500)
    assert metadata["original_data_length"] == 44_900
    assert metadata["original_duration_seconds"] == pytest.approx(89.8)
    assert metadata["source_declared_data_length"] == 449_000
    assert metadata["source_data_length_matches"] is False
    assert metadata["data_length"] == 44_900
    assert metadata["resampled_sample_rate"] == 50
    assert metadata["resampled_data_length"] == 4_490
    assert metadata["resampled_duration_seconds"] == pytest.approx(89.8)
    assert metadata["resampled_last_timestamp_seconds"] == pytest.approx((4_490 - 1) / 50)
    assert {len(values) for values in series.values()} == {4_490}

    # q and dq are source channels interpolated to exactly the same timestamps.  A numerical
    # derivative of q should remain close to the supplied dq; the ankle channels are the noisiest
    # and stay below 10% normalised RMSE at 50 Hz.
    for joint in REFERENCE_JOINTS:
        qpos = np.asarray(series[f"q_{joint}"])
        qvel = np.asarray(series[f"dq_{joint}"])
        numerical_qvel = np.gradient(qpos, env.dt)
        rmse = np.sqrt(np.mean(np.square(numerical_qvel - qvel)))
        qvel_rms = np.sqrt(np.mean(np.square(qvel)))
        assert rmse / qvel_rms < 0.10, joint


def test_200_resets_keep_target_and_current_reference_velocity(human_env):
    config, env = human_env
    for reset_index in range(200):
        env.reset(seed=config.env_params.seed if reset_index == 0 else None)
        assert env._target_velocity == pytest.approx(1.25)
        assert (
            env._imitation_phase
            + config.env_params.custom_max_episode_steps * env._reference_speed_scale()
            <= env._reference_data_length - 1
        )

    speed_ratio = env._reference_speed_scale()
    for joint in REFERENCE_JOINTS:
        expected = env._reference_value(f"dq_{joint}") * speed_ratio
        if joint == "pelvis_tx":
            expected = env._target_velocity
        assert env.sim.data.joint(joint).qvel[0] == pytest.approx(expected, abs=1e-10), joint


def test_runtime_reference_uses_one_continuous_time_warp(human_env):
    _, env = human_env
    env.reset(seed=2468)
    speed_scale = env._reference_speed_scale()
    start_phase = env._imitation_phase
    phases = start_phase + np.arange(1000, dtype=float) * speed_scale

    # q(phase(t)) and dq(phase(t))*dphase/dt must describe the same motion.  This catches
    # the old per-frame division by instantaneous pelvis velocity, which changed every joint's
    # velocity without advancing the position reference on the corresponding time axis.
    for joint in REFERENCE_JOINTS:
        qpos = np.array([env._reference_value(f"q_{joint}", phase) for phase in phases])
        qvel = np.array([env._reference_value(f"dq_{joint}", phase) * speed_scale for phase in phases])
        numerical_qvel = np.gradient(qpos, env.dt)
        rmse = np.sqrt(np.mean(np.square(numerical_qvel - qvel)))
        qvel_rms = np.sqrt(np.mean(np.square(qvel)))
        assert rmse / qvel_rms < 0.10, joint


def test_reset_bookkeeping_is_initialised_after_sim_time_reset(human_env):
    _, env = human_env
    env.reset(seed=8642)
    for _ in range(7):
        env.step(np.zeros(env.action_space.shape, dtype=float))
    assert env.sim.data.time > 0

    env.reset()
    assert env.sim.data.time == pytest.approx(0.0)
    assert env._prev_step_time == pytest.approx(0.0)
    assert env._prev_step_changed_time == pytest.approx(0.0)


def test_seed_reproduces_reference_start_sequence(human_env):
    _, env = human_env

    def sequence(seed: int):
        result = []
        for index in range(20):
            env.reset(seed=seed if index == 0 else None)
            result.append(int(env._imitation_index))
        return result

    first = sequence(20261022)
    second = sequence(20261022)
    assert first == second
    assert len(set(first)) > 1


def test_multistart_random_selection_is_reproducible_and_unique():
    first = _select_start_indices(3_591, 20, "random", 20261023)
    second = _select_start_indices(3_591, 20, "random", 20261023)
    different_seed = _select_start_indices(3_591, 20, "random", 20261024)

    assert first == second
    assert first != different_seed
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 20
    assert min(first) >= 0
    assert max(first) <= 3_591


def test_multistart_uses_the_training_reset_range(human_env):
    config, env = human_env
    maximum = _maximum_training_start(
        env._reference_data_length,
        config.env_params.custom_max_episode_steps,
        config.env_params.max_target_velocity,
        env._reference_nominal_velocity,
    )

    assert maximum == env._random_reference_start_upper_bound() - 1


def test_multistart_sets_the_continuous_reference_phase(human_env):
    _, env = human_env
    env.reset(seed=31415)
    obs = _set_reference_start(env, 1_234)

    assert env._imitation_phase == pytest.approx(1_234.0)
    assert env._imitation_index == 1_234
    assert np.array_equal(obs, env.get_obs())


def test_first_step_history_and_numeric_dense_reward(human_env):
    _, env = human_env
    env.reset(seed=4321)
    assert env._prev_muscle_activations_for_reward is None

    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=float))
    reward_terms = info["rwd_dict"]
    assert reward_terms["muscle_activation_diff_penalty"] == pytest.approx(0.0)

    qpos_expected = sum(
        env.dt * np.exp(-8 * diff**2) * env._reward_keys_and_weights.qpos_imitation_rewards[joint]
        for joint, diff in env._get_qpos_diff().items()
    )
    qvel_expected = sum(
        env.dt * np.exp(-8 * diff**2) * env._reward_keys_and_weights.qvel_imitation_rewards[joint]
        for joint, diff in env._get_qvel_diff().items()
    )
    assert reward_terms["qpos_imitation_rewards"] == pytest.approx(qpos_expected)
    assert reward_terms["qvel_imitation_rewards"] == pytest.approx(qvel_expected)

    dense_expected = sum(env.rwd_keys_wt[key] * reward_terms[key] for key in env.rwd_keys_wt)
    assert reward_terms["dense"] == pytest.approx(dense_expected)
    assert reward == pytest.approx(dense_expected)

    # A second reset must clear the just-recorded activation history again.
    env.reset()
    assert env._prev_muscle_activations_for_reward is None
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=float))
    assert info["rwd_dict"]["muscle_activation_diff_penalty"] == pytest.approx(0.0)


def test_callback_clears_per_episode_reward_accumulator():
    callback = BaseCustomLearningCallback(
        log_rollout_freq=1,
        evaluate_freq=0,
        log_handler=None,
        verbose=0,
    )
    callback.rewards_sum = np.zeros(1)
    callback.current_episode_rewards = np.zeros(1)
    callback.episode_counts = np.zeros(1)
    callback.episode_length_counts = np.zeros(1)
    callback.current_episode_length_counts = np.zeros(1)
    callback.current_reward_dict_sum = [{}]
    callback.episode_reward_dict_sum = [{}]

    callback.locals = {
        "rewards": np.array([1.0]),
        "dones": np.array([True]),
        "infos": [{"rwd_dict": {"dense": 1.0}}],
    }
    assert callback._on_step()
    assert callback.episode_reward_dict_sum[0]["dense"] == pytest.approx(1.0)
    assert callback.current_reward_dict_sum[0] == {}

    callback.locals = {
        "rewards": np.array([2.0]),
        "dones": np.array([True]),
        "infos": [{"rwd_dict": {"dense": 2.0}}],
    }
    assert callback._on_step()
    assert callback.episode_reward_dict_sum[0]["dense"] == pytest.approx(3.0)
    assert callback.current_reward_dict_sum[0] == {}


def test_interrupted_training_still_saves_final_model(monkeypatch, tmp_path):
    import rl_train.run_train as run_train

    config = _load_config()
    saved_paths = []

    class FakeEnv:
        closed = False

        def close(self):
            self.closed = True

    class FakeModel:
        def learn(self, **kwargs):
            raise KeyboardInterrupt("test interruption")

        def save(self, path):
            saved_paths.append(path)

    fake_env = FakeEnv()
    fake_model = FakeModel()
    fake_log_handler = SimpleNamespace(get_path2save_final_model=lambda: str(tmp_path / "final_model"))
    monkeypatch.setattr(run_train, "log_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(run_train.EnvironmentHandler, "create_environment", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(run_train.EnvironmentHandler, "get_stable_baselines3_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(run_train.EnvironmentHandler, "updateconfig_from_model_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_train.EnvironmentHandler, "get_callback", lambda *args, **kwargs: object())

    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        run_train.ppo_train_with_parameters(
            config,
            train_time_step=64,
            is_rendering_on=False,
            train_log_handler=fake_log_handler,
        )

    assert saved_paths == [str(tmp_path / "final_model")]
    assert fake_env.closed is True
