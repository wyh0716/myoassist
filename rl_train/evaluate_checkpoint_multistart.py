"""Validate selected imitation checkpoints from multiple reference start indices.

This is a robustness companion to ``evaluate_checkpoint_sweep.py``.  It keeps the
target velocity fixed, disables only the imitation-error termination for
measurement, and reports when the original threshold would have been crossed.
Reference starts are deterministic and either evenly cover, or are sampled
without replacement from, the same valid range used by the training
environment's random-reset policy.  That range is capped both at 80 percent of
the trajectory and at the latest start that leaves one complete training
episode.  Random sampling is seeded so every checkpoint is evaluated from
exactly the same starts.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.envs.action_adapters import MOTOR_ARM_ACTUATORS
from rl_train.evaluate_checkpoint_sweep import (
    _checkpoint_steps,
    _joint_metrics,
    _phase_metrics,
)
from rl_train.evaluate_joint_rmse_diagnostics import CHANNELS
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint filenames or integer timesteps")
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=300)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--reference-period-steps", type=int, default=34)
    parser.add_argument(
        "--start-selection",
        choices=("even", "random"),
        default="even",
        help="How to choose starts from the reference range used by training resets.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=None,
        help="Seed for random start selection (defaults to env_params.seed).",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve_checkpoints(model_dir: Path, requested: list[str]) -> list[Path]:
    available = {_checkpoint_steps(path): path for path in model_dir.glob("model_*.zip")}
    resolved = []
    for value in requested:
        candidate = model_dir / value
        if candidate.exists():
            resolved.append(candidate)
            continue
        try:
            step = int(value)
        except ValueError as exc:
            raise FileNotFoundError(f"Checkpoint not found: {value}") from exc
        if step not in available:
            raise FileNotFoundError(f"No checkpoint saved at timestep {step}")
        resolved.append(available[step])
    return resolved


def _set_reference_start(unwrapped, index: int) -> np.ndarray:
    """Put both MuJoCo simulations at an exact reference state after a normal reset."""
    # The environment advances a continuous phase and keeps _imitation_index only as a
    # compatibility view for analyzers. Setting the integer alone leaves reward, termination,
    # and the next phase at the reset value (usually zero), so an apparent multi-start sweep
    # would actually replay one start repeatedly.
    unwrapped._set_reference_phase(float(index))
    unwrapped._follow_reference_motion(False)
    unwrapped.sim.data.ctrl[:] = 0.0
    if unwrapped.sim.model.na:
        unwrapped.sim.data.act[:] = 0.0
    if hasattr(unwrapped, "_reset_motor_arm_state"):
        unwrapped._reset_motor_arm_state()
    if hasattr(unwrapped, "_synchronise_polynomial_joint_equalities"):
        unwrapped._synchronise_polynomial_joint_equalities()
    unwrapped.sim.forward()
    unwrapped.robot.sync_sims(unwrapped.sim, unwrapped.sim_obsd)
    return unwrapped.get_obs()


def _select_start_indices(
    maximum_training_start: int,
    starts: int,
    selection: str,
    seed: int,
) -> list[int]:
    """Choose reproducible reference starts shared by every checkpoint."""
    if starts < 1:
        raise ValueError("--starts must be at least 1")
    available = maximum_training_start + 1
    if available < 1:
        raise ValueError("Reference trajectory has no valid training starts")
    if selection == "even":
        return np.linspace(0, maximum_training_start, starts, dtype=int).tolist()
    if starts > available:
        raise ValueError(
            f"Cannot sample {starts} unique starts from {available} valid indices"
        )
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(available, size=starts, replace=False).tolist())


def _maximum_training_start(
    reference_length: int,
    episode_steps: int,
    max_target_velocity: float,
    reference_nominal_velocity: float,
) -> int:
    """Reproduce the training environment's exclusive reset bound as an index."""
    eighty_percent_bound = int(reference_length * 0.8)
    max_speed_scale = max_target_velocity / reference_nominal_velocity
    latest_full_episode_start = int(
        np.floor((reference_length - 1) - episode_steps * max_speed_scale)
    )
    upper_bound = min(eighty_percent_bound, latest_full_episode_start + 1)
    if upper_bound <= 0:
        raise ValueError("Reference trajectory is too short for one training episode")
    return upper_bound - 1


def _evaluate_checkpoint(
    checkpoint: Path,
    base_config: ImitationTrainSessionConfig,
    starts: int,
    horizon_steps: int,
    target_velocity: float,
    period_steps: int,
    start_selection: str,
    start_seed: int,
) -> dict:
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig,
        DictionableDataclass.to_dict(base_config),
    )
    original_threshold = float(config.env_params.out_of_trajectory_threshold)
    original_episode_steps = int(config.env_params.custom_max_episode_steps)
    max_training_target_velocity = float(config.env_params.max_target_velocity)
    safe_height = float(config.env_params.safe_height)
    fps = int(config.env_params.control_framerate)
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    config.env_params.terminate_on_trajectory_error = False
    config.env_params.custom_max_episode_steps = max(1_000_000, horizon_steps + 1)

    seed = int(config.env_params.seed)
    np.random.seed(seed)
    random.seed(seed)
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    unwrapped = env.unwrapped
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(
            config, env, trained_model_path=str(checkpoint)
        )
        unwrapped.set_target_velocity_mode_manually(
            MyoAssistLegBase.VelocityMode.UNIFORM,
            0.0,
            target_velocity,
            target_velocity,
            target_velocity,
            target_velocity_period=3.0,
        )
        maximum_training_start = _maximum_training_start(
            unwrapped._reference_data_length,
            original_episode_steps,
            max_training_target_velocity,
            unwrapped._reference_nominal_velocity,
        )
        start_indices = _select_start_indices(
            maximum_training_start,
            starts,
            start_selection,
            start_seed,
        )
        trials = []

        for start_index in start_indices:
            env.reset()
            obs = _set_reference_start(unwrapped, start_index)
            initial_x = float(unwrapped.sim.data.joint("pelvis_tx").qpos[0])
            strict_step = None
            strict_reason = None
            simulated = []
            reference = []
            velocity_errors = []
            heights = []
            rewards = []
            arm_angles = []
            arm_velocities = []
            arm_torques = []
            future_reference_errors = []
            termination_reason = None
            terminated = False
            truncated = False

            for step in range(1, horizon_steps + 1):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                reference_index = int(unwrapped._imitation_index)
                simulated.append(
                    [
                        float(np.rad2deg(unwrapped.sim.data.joint(name).qpos[0]))
                        for _joint, _side, name in CHANNELS
                    ]
                )
                reference.append(
                    [
                        float(np.rad2deg(unwrapped._reference_data["series_data"][f"q_{name}"][reference_index]))
                        for _joint, _side, name in CHANNELS
                    ]
                )
                height = float(unwrapped.sim.data.joint("pelvis_ty").qpos[0])
                heights.append(height)
                velocity_errors.append(
                    abs(float(unwrapped.sim.data.joint("pelvis_tx").qvel[0]) - target_velocity)
                )
                rewards.append(float(reward))
                if unwrapped.sim.model.nu == 26:
                    arm_angles.append(
                        [
                            float(unwrapped.sim.data.joint(name).qpos[0])
                            for name in MOTOR_ARM_ACTUATORS
                        ]
                    )
                    arm_velocities.append(
                        [
                            float(unwrapped.sim.data.joint(name).qvel[0])
                            for name in MOTOR_ARM_ACTUATORS
                        ]
                    )
                    arm_torques.append(
                        [
                            float(unwrapped.sim.data.actuator(name).force[0])
                            * float(unwrapped.sim.model.actuator_gear[unwrapped.sim.model.actuator(name).id, 0])
                            for name in MOTOR_ARM_ACTUATORS
                        ]
                    )
                if "future_reference_error" in unwrapped.obs_dict:
                    future_reference_errors.append(
                        np.asarray(
                            unwrapped.obs_dict["future_reference_error"], dtype=float
                        ).copy()
                    )

                qdiff = {
                    name: abs(float(np.asarray(value).squeeze()))
                    for name, value in unwrapped._get_qpos_diff().items()
                }
                if strict_step is None and (height < safe_height or max(qdiff.values()) > original_threshold):
                    strict_step = step
                    strict_reason = (
                        "pelvis_below_safe_height"
                        if height < safe_height
                        else f"joint_out_of_trajectory:{max(qdiff, key=qdiff.get)}"
                    )
                if terminated or truncated:
                    termination_reason = info.get("termination_reason")
                    break

            sim = np.asarray(simulated, dtype=np.float64)
            ref = np.asarray(reference, dtype=np.float64)
            fixed_joints, fixed_overall = _joint_metrics(sim - ref)
            cycle_steps = (len(sim) // period_steps) * period_steps
            phase = None
            if cycle_steps >= period_steps:
                phase = _phase_metrics(sim[:cycle_steps], ref[:cycle_steps], period_steps)
            final_x = float(unwrapped.sim.data.joint("pelvis_tx").qpos[0])
            arm_angles_array = np.asarray(arm_angles, dtype=np.float64)
            arm_velocities_array = np.asarray(arm_velocities, dtype=np.float64)
            arm_torques_array = np.asarray(arm_torques, dtype=np.float64)
            future_error_array = np.asarray(future_reference_errors, dtype=np.float64)
            trials.append(
                {
                    "reference_start_index": start_index,
                    "completed_steps": len(sim),
                    "duration_seconds": len(sim) / fps,
                    "fell": bool(terminated),
                    "truncated": bool(truncated),
                    "environment_termination_reason": termination_reason,
                    "strict_threshold_crossed": strict_step is not None,
                    "strict_tracking_steps": strict_step if strict_step is not None else horizon_steps,
                    "strict_tracking_seconds": (strict_step if strict_step is not None else horizon_steps) / fps,
                    "strict_tracking_censored_at_horizon": strict_step is None,
                    "strict_termination_reason": strict_reason,
                    "fixed_rmse_all_deg": fixed_overall,
                    "fixed_rmse_joint_deg": fixed_joints,
                    "phase_rmse_all_deg": None if phase is None else phase["all_six_joints_rmse_deg"],
                    "phase_shift_drift_steps_per_cycle": None
                    if phase is None
                    else phase["unwrapped_shift_drift_steps_per_cycle"],
                    "forward_velocity_mae_m_per_s": float(np.mean(velocity_errors)),
                    "forward_displacement_m": final_x - initial_x,
                    "minimum_pelvis_height_m": float(np.min(heights)),
                    "mean_reward_per_step": float(np.mean(rewards)),
                    "arm_angle_range_deg": None
                    if arm_angles_array.size == 0
                    else {
                        name: [
                            float(np.rad2deg(np.min(arm_angles_array[:, slot]))),
                            float(np.rad2deg(np.max(arm_angles_array[:, slot]))),
                        ]
                        for slot, name in enumerate(MOTOR_ARM_ACTUATORS)
                    },
                    "arm_velocity_max_abs_rad_per_s": None
                    if arm_velocities_array.size == 0
                    else {
                        name: float(np.max(np.abs(arm_velocities_array[:, slot])))
                        for slot, name in enumerate(MOTOR_ARM_ACTUATORS)
                    },
                    "arm_actual_torque_max_abs_nm": None
                    if arm_torques_array.size == 0
                    else {
                        name: float(np.max(np.abs(arm_torques_array[:, slot])))
                        for slot, name in enumerate(MOTOR_ARM_ACTUATORS)
                    },
                    "future_reference_error_rmse": None
                    if future_error_array.size == 0
                    else float(np.sqrt(np.mean(np.square(future_error_array)))),
                }
            )

        strict_seconds = np.asarray([item["strict_tracking_seconds"] for item in trials])
        phase_values = [item["phase_rmse_all_deg"] for item in trials if item["phase_rmse_all_deg"] is not None]
        aggregate = {
            "trial_count": len(trials),
            "horizon_seconds": horizon_steps / fps,
            "strict_full_horizon_rate": float(np.mean([item["strict_tracking_censored_at_horizon"] for item in trials])),
            "strict_tracking_seconds_mean": float(np.mean(strict_seconds)),
            "strict_tracking_seconds_median": float(np.median(strict_seconds)),
            "strict_tracking_seconds_p10": float(np.percentile(strict_seconds, 10)),
            "fall_rate": float(np.mean([item["fell"] for item in trials])),
            "fixed_rmse_all_deg_mean": float(np.mean([item["fixed_rmse_all_deg"] for item in trials])),
            "fixed_rmse_all_deg_median": float(np.median([item["fixed_rmse_all_deg"] for item in trials])),
            "phase_rmse_all_deg_mean": float(np.mean(phase_values)) if phase_values else None,
            "forward_velocity_mae_m_per_s_mean": float(
                np.mean([item["forward_velocity_mae_m_per_s"] for item in trials])
            ),
            "minimum_pelvis_height_m_min": float(np.min([item["minimum_pelvis_height_m"] for item in trials])),
            "arm_actual_torque_max_abs_nm": None
            if not any(item["arm_actual_torque_max_abs_nm"] for item in trials)
            else max(
                max(item["arm_actual_torque_max_abs_nm"].values())
                for item in trials
                if item["arm_actual_torque_max_abs_nm"] is not None
            ),
        }
        return {
            "checkpoint": checkpoint.name,
            "timesteps": _checkpoint_steps(checkpoint),
            "interface": {
                "observation_dimension": int(env.observation_space.shape[0]),
                "policy_action_dimension": int(env.action_space.shape[0]),
                "physical_actuator_dimension": int(unwrapped.sim.model.nu),
            },
            "reference_start_indices": start_indices,
            "aggregate": aggregate,
            "trials": trials,
        }
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    with (args.log_dir / "session_config.json").open(encoding="utf-8") as file:
        config = DictionableDataclass.create(ImitationTrainSessionConfig, json.load(file))
    checkpoints = _resolve_checkpoints(args.log_dir / "trained_models", args.checkpoints)
    output_path = args.output or args.log_dir / "offline_evaluation_multistart.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_seed = int(config.env_params.seed) if args.start_seed is None else args.start_seed

    results = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"[{index}/{len(checkpoints)}] {checkpoint.name}", flush=True)
        result = _evaluate_checkpoint(
            checkpoint,
            config,
            args.starts,
            args.horizon_steps,
            args.target_velocity,
            args.reference_period_steps,
            args.start_selection,
            start_seed,
        )
        results.append(result)
        print(json.dumps(result["aggregate"], ensure_ascii=False), flush=True)

    output = {
        "method": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "session": args.log_dir.name,
            "target_velocity_m_per_s": args.target_velocity,
            "deterministic_policy": True,
            "reference_start_selection": (
                "Evenly spaced over the valid range used by training random resets"
                if args.start_selection == "even"
                else "Seeded random sample without replacement from the valid range used by training random resets"
            ),
            "reference_start_seed": start_seed,
            "start_count": args.starts,
            "horizon_steps": args.horizon_steps,
            "control_framerate_hz": int(config.env_params.control_framerate),
            "reference_period_steps": args.reference_period_steps,
            "original_out_of_trajectory_threshold_rad": float(config.env_params.out_of_trajectory_threshold),
            "note": "Imitation-error termination is disabled through the environment flag; its first crossing is reconstructed. Fall and time/reference termination remain enabled.",
        },
        "checkpoints": results,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(json.dumps({"json": str(output_path), "checkpoints": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
