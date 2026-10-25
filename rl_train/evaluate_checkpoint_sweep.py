"""Offline checkpoint sweep for the human-only myolegs22 imitation experiment.

The sweep evaluates checkpoints without changing their weights.  One deterministic
rollout per checkpoint is run from reference index zero at a fixed 1.25 m/s target.
The environment's imitation-error termination is disabled for the rollout so that
fall-only stability and long-horizon kinematics can be measured, while the first
step that *would* have triggered the original termination is recorded separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.evaluate_joint_rmse_diagnostics import (
    CHANNELS,
    _estimate_reference_period,
    _joint_metrics,
    _phase_align_by_cycle,
)
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--horizon-steps", type=int, default=900)
    parser.add_argument("--rmse-steps", type=int, default=884)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def _checkpoint_steps(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _select_checkpoints(model_dir: Path, interval: int) -> list[Path]:
    checkpoints = sorted(model_dir.glob("model_*.zip"), key=_checkpoint_steps)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.zip checkpoints found in {model_dir}")

    by_step = {_checkpoint_steps(path): path for path in checkpoints}
    steps = np.asarray(sorted(by_step), dtype=np.int64)
    targets = [int(steps[0])]
    targets.extend(range(interval, int(steps[-1]), interval))
    targets.append(int(steps[-1]))

    selected_steps = []
    for target in targets:
        nearest = int(steps[np.argmin(np.abs(steps - target))])
        if nearest not in selected_steps:
            selected_steps.append(nearest)
    return [by_step[step] for step in selected_steps]


def _flatten_sensor(env, name: str) -> float:
    return float(np.asarray(env.unwrapped.sim.data.sensor(name).data).squeeze())


def _heel_strike_metrics(right_force: list[float], left_force: list[float], fps: int) -> dict:
    strikes: list[tuple[int, str]] = []
    last_side = ""
    for index in range(2, len(right_force)):
        if min(right_force[index - 2 : index + 1]) > 0.1 and last_side != "right":
            last_side = "right"
            strikes.append((index, last_side))
        elif min(left_force[index - 2 : index + 1]) > 0.1 and last_side != "left":
            last_side = "left"
            strikes.append((index, last_side))

    if len(strikes) > 1 and strikes[-1][0] > strikes[0][0]:
        elapsed = (strikes[-1][0] - strikes[0][0]) / fps
        cadence = (len(strikes) - 1) * 60 / elapsed
        intervals = np.diff([item[0] for item in strikes]) / fps
        mean_interval = float(np.mean(intervals))
    else:
        cadence = None
        mean_interval = None
    return {
        "heel_strike_count": len(strikes),
        "cadence_steps_per_minute": cadence,
        "mean_step_interval_seconds": mean_interval,
    }


def _phase_metrics(simulated: np.ndarray, reference: np.ndarray, period_steps: int) -> dict:
    aligned_errors, cycles = _phase_align_by_cycle(simulated, reference, period_steps)
    joint_metrics, overall = _joint_metrics(aligned_errors)
    shifts = np.asarray([item["shift_steps"] for item in cycles], dtype=np.float64)
    if len(shifts) >= 2:
        unwrapped = np.unwrap(shifts * 2 * np.pi / period_steps) * period_steps / (2 * np.pi)
        drift = float(np.polyfit(np.arange(len(unwrapped), dtype=float), unwrapped, deg=1)[0])
    else:
        drift = None
    return {
        "joint_rmse_deg": joint_metrics,
        "all_six_joints_rmse_deg": overall,
        "mean_absolute_shift_steps": float(np.mean(np.abs(shifts))),
        "max_absolute_shift_steps": float(np.max(np.abs(shifts))),
        "unwrapped_shift_drift_steps_per_cycle": drift,
        "cycle_count": len(cycles),
    }


def _segments(errors: np.ndarray, fps: int) -> dict[str, float]:
    bounds = (
        ("0-10s", 0, min(10 * fps, len(errors))),
        ("10-20s", min(10 * fps, len(errors)), min(20 * fps, len(errors))),
        (f"20-{len(errors) / fps:.2f}s", min(20 * fps, len(errors)), len(errors)),
    )
    result = {}
    for label, start, end in bounds:
        if end > start:
            _joint, overall = _joint_metrics(errors[start:end])
            result[label] = overall
    return result


def _evaluate_checkpoint(
    checkpoint: Path,
    base_config: ImitationTrainSessionConfig,
    target_velocity: float,
    horizon_steps: int,
    rmse_steps: int,
) -> tuple[dict, np.ndarray]:
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig,
        DictionableDataclass.to_dict(base_config),
    )
    original_threshold = float(config.env_params.out_of_trajectory_threshold)
    original_episode_limit = int(config.env_params.custom_max_episode_steps)
    safe_height = float(config.env_params.safe_height)
    fps = int(config.env_params.control_framerate)

    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    config.env_params.out_of_trajectory_threshold = 1_000_000.0
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
        obs, _info = env.reset()
        if int(unwrapped._imitation_index) != 0:
            raise RuntimeError(f"Reference did not reset to index zero for {checkpoint.name}")

        initial_x = float(unwrapped.sim.data.joint("pelvis_tx").qpos[0])
        strict_step = None
        strict_reasons: list[str] = []
        strict_worst_joint = None
        strict_worst_error = None
        strict_pelvis_height = None
        strict_distance = None

        simulated_rows = []
        reference_rows = []
        rewards = []
        pelvis_x = []
        pelvis_vx = []
        pelvis_height = []
        right_force = []
        left_force = []
        controls = []
        actual_terminated = False
        actual_truncated = False

        for step in range(1, horizon_steps + 1):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, actual_terminated, actual_truncated, _info = env.step(action)

            reference_index = int(unwrapped._imitation_index)
            simulated_rows.append(
                [float(np.rad2deg(unwrapped.sim.data.joint(name).qpos[0])) for _joint, _side, name in CHANNELS]
            )
            reference_rows.append(
                [
                    float(np.rad2deg(unwrapped._reference_data["series_data"][f"q_{name}"][reference_index]))
                    for _joint, _side, name in CHANNELS
                ]
            )
            rewards.append(float(reward))
            current_x = float(unwrapped.sim.data.joint("pelvis_tx").qpos[0])
            current_height = float(unwrapped.sim.data.joint("pelvis_ty").qpos[0])
            pelvis_x.append(current_x)
            pelvis_vx.append(float(unwrapped.sim.data.joint("pelvis_tx").qvel[0]))
            pelvis_height.append(current_height)
            right_force.append(_flatten_sensor(env, "r_foot") + _flatten_sensor(env, "r_toes"))
            left_force.append(_flatten_sensor(env, "l_foot") + _flatten_sensor(env, "l_toes"))
            controls.extend(np.asarray(unwrapped.sim.data.ctrl, dtype=np.float64).tolist())

            qdiff = {
                name: abs(float(np.asarray(value).squeeze()))
                for name, value in unwrapped._get_qpos_diff().items()
            }
            worst_joint = max(qdiff, key=qdiff.get)
            worst_error = qdiff[worst_joint]
            if strict_step is None and (current_height < safe_height or worst_error > original_threshold):
                strict_step = step
                if current_height < safe_height:
                    strict_reasons.append("pelvis_below_safe_height")
                if worst_error > original_threshold:
                    strict_reasons.append("joint_out_of_trajectory")
                strict_worst_joint = worst_joint
                strict_worst_error = worst_error
                strict_pelvis_height = current_height
                strict_distance = current_x - initial_x

            if actual_terminated or actual_truncated:
                break

        completed = len(simulated_rows)
        if strict_step is None and completed >= original_episode_limit:
            strict_step = original_episode_limit
            strict_reasons = ["episode_step_limit"]

        sim = np.asarray(simulated_rows, dtype=np.float64)
        ref = np.asarray(reference_rows, dtype=np.float64)
        fixed_metrics = None
        phase_metrics = None
        fixed_segments = None
        if completed >= rmse_steps:
            sim_rmse = sim[:rmse_steps]
            ref_rmse = ref[:rmse_steps]
            fixed_errors = sim_rmse - ref_rmse
            fixed_joint, fixed_overall = _joint_metrics(fixed_errors)
            fixed_metrics = {
                "joint_rmse_deg": fixed_joint,
                "all_six_joints_rmse_deg": fixed_overall,
            }
            fixed_segments = _segments(fixed_errors, fps)
        else:
            sim_rmse = sim
            ref_rmse = ref

        model_mass = float(np.sum(unwrapped.sim.model.body_mass))
        body_weight = model_mass * 9.81
        right_contact = np.asarray(right_force) > 0.1
        left_contact = np.asarray(left_force) > 0.1
        controls_array = np.asarray(controls, dtype=np.float64)
        strike_metrics = _heel_strike_metrics(right_force, left_force, fps)

        result = {
            "checkpoint": checkpoint.name,
            "timesteps": _checkpoint_steps(checkpoint),
            "completed_control_steps": completed,
            "fall_only_duration_seconds": completed / fps,
            "fell_before_horizon": bool(actual_terminated),
            "truncated_before_horizon": bool(actual_truncated),
            "strict_termination_reached": strict_step is not None,
            "strict_termination_step": strict_step,
            "strict_termination_seconds": strict_step / fps if strict_step is not None else None,
            "strict_termination_reasons": strict_reasons,
            "strict_worst_joint": strict_worst_joint,
            "strict_worst_error_rad": strict_worst_error,
            "strict_worst_error_deg": float(np.rad2deg(strict_worst_error)) if strict_worst_error is not None else None,
            "strict_pelvis_height_m": strict_pelvis_height,
            "strict_forward_displacement_m": strict_distance,
            "forward_displacement_m": float(pelvis_x[-1] - initial_x),
            "mean_forward_velocity_m_per_s": float(np.mean(pelvis_vx)),
            "forward_velocity_mae_m_per_s": float(np.mean(np.abs(np.asarray(pelvis_vx) - target_velocity))),
            "minimum_pelvis_height_m": float(np.min(pelvis_height)),
            "mean_pelvis_height_m": float(np.mean(pelvis_height)),
            "mean_reward_per_step": float(np.mean(rewards)),
            "double_support_fraction": float(np.mean(right_contact & left_contact)),
            "flight_fraction": float(np.mean(~right_contact & ~left_contact)),
            "peak_total_foot_force_body_weights": float(
                np.max(np.asarray(right_force) + np.asarray(left_force)) / body_weight
            ),
            "muscle_control_rms": float(np.sqrt(np.mean(np.square(controls_array)))),
            "muscle_control_ge_0p95_fraction": float(np.mean(controls_array >= 0.95)),
            **strike_metrics,
            "fixed_time": fixed_metrics,
            "fixed_time_segments": fixed_segments,
            "phase_aligned": phase_metrics,
        }
        return result, np.column_stack((sim_rmse, ref_rmse))
    finally:
        env.close()


def _csv_row(result: dict) -> dict:
    fixed = result["fixed_time"] or {}
    phase = result["phase_aligned"] or {}
    joints = fixed.get("joint_rmse_deg", {})
    phase_joints = phase.get("joint_rmse_deg", {})
    segments = result["fixed_time_segments"] or {}
    return {
        "checkpoint": result["checkpoint"],
        "timesteps": result["timesteps"],
        "strict_termination_seconds": result["strict_termination_seconds"],
        "strict_termination_reasons": "+".join(result["strict_termination_reasons"]),
        "strict_worst_joint": result["strict_worst_joint"],
        "strict_worst_error_deg": result["strict_worst_error_deg"],
        "strict_pelvis_height_m": result["strict_pelvis_height_m"],
        "strict_forward_displacement_m": result["strict_forward_displacement_m"],
        "fall_only_duration_seconds": result["fall_only_duration_seconds"],
        "fell_before_horizon": result["fell_before_horizon"],
        "forward_displacement_m": result["forward_displacement_m"],
        "mean_forward_velocity_m_per_s": result["mean_forward_velocity_m_per_s"],
        "forward_velocity_mae_m_per_s": result["forward_velocity_mae_m_per_s"],
        "minimum_pelvis_height_m": result["minimum_pelvis_height_m"],
        "mean_reward_per_step": result["mean_reward_per_step"],
        "cadence_steps_per_minute": result["cadence_steps_per_minute"],
        "double_support_fraction": result["double_support_fraction"],
        "flight_fraction": result["flight_fraction"],
        "peak_total_foot_force_body_weights": result["peak_total_foot_force_body_weights"],
        "muscle_control_rms": result["muscle_control_rms"],
        "muscle_control_ge_0p95_fraction": result["muscle_control_ge_0p95_fraction"],
        "fixed_rmse_all_deg": fixed.get("all_six_joints_rmse_deg"),
        "fixed_rmse_hip_deg": joints.get("hip", {}).get("bilateral_rmse_deg"),
        "fixed_rmse_knee_deg": joints.get("knee", {}).get("bilateral_rmse_deg"),
        "fixed_rmse_ankle_deg": joints.get("ankle", {}).get("bilateral_rmse_deg"),
        "fixed_rmse_0_10s_deg": segments.get("0-10s"),
        "fixed_rmse_10_20s_deg": segments.get("10-20s"),
        "fixed_rmse_20_end_deg": next((v for k, v in segments.items() if k.startswith("20-")), None),
        "phase_rmse_all_deg": phase.get("all_six_joints_rmse_deg"),
        "phase_rmse_hip_deg": phase_joints.get("hip", {}).get("bilateral_rmse_deg"),
        "phase_rmse_knee_deg": phase_joints.get("knee", {}).get("bilateral_rmse_deg"),
        "phase_rmse_ankle_deg": phase_joints.get("ankle", {}).get("bilateral_rmse_deg"),
        "phase_mean_absolute_shift_steps": phase.get("mean_absolute_shift_steps"),
        "phase_drift_steps_per_cycle": phase.get("unwrapped_shift_drift_steps_per_cycle"),
    }


def _rankings(results: list[dict]) -> dict:
    completed = [item for item in results if item["fixed_time"] is not None]

    def top(metric_path, reverse=False, count=5):
        def value(item):
            current = item
            for key in metric_path:
                current = current[key]
            return current

        available = [item for item in completed if value(item) is not None]
        ranked = sorted(available, key=value, reverse=reverse)[:count]
        return [
            {"checkpoint": item["checkpoint"], "timesteps": item["timesteps"], "value": value(item)}
            for item in ranked
        ]

    return {
        "longest_strict_tracking": top(("strict_termination_seconds",), reverse=True),
        "lowest_fixed_time_rmse": top(("fixed_time", "all_six_joints_rmse_deg")),
        "lowest_phase_aligned_rmse": top(("phase_aligned", "all_six_joints_rmse_deg")),
        "lowest_speed_mae": top(("forward_velocity_mae_m_per_s",)),
        "lowest_muscle_control_rms": top(("muscle_control_rms",)),
    }


def main() -> None:
    args = _parse_args()
    if args.rmse_steps > args.horizon_steps:
        raise ValueError("--rmse-steps cannot exceed --horizon-steps")

    with (args.log_dir / "session_config.json").open(encoding="utf-8") as file:
        config = DictionableDataclass.create(ImitationTrainSessionConfig, json.load(file))
    checkpoints = _select_checkpoints(args.log_dir / "trained_models", args.checkpoint_interval)
    prefix = args.output_prefix or args.log_dir / "offline_evaluation_1m_sweep"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(checkpoints)} checkpoints")
    results = []
    reference_for_period = None
    period_steps = None
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"[{index}/{len(checkpoints)}] {checkpoint.name}", flush=True)
        result, combined = _evaluate_checkpoint(
            checkpoint,
            config,
            args.target_velocity,
            args.horizon_steps,
            args.rmse_steps,
        )
        if len(combined) >= args.rmse_steps:
            simulated = combined[: args.rmse_steps, : len(CHANNELS)]
            reference = combined[: args.rmse_steps, len(CHANNELS) :]
            if reference_for_period is None:
                reference_for_period = reference
                period_steps, _candidates = _estimate_reference_period(reference, int(config.env_params.control_framerate), 0.7, 1.5)
            phase = _phase_metrics(simulated, reference, period_steps)
            result["phase_aligned"] = phase
        results.append(result)
        print(
            f"  strict={result['strict_termination_seconds']}s "
            f"fall={result['fell_before_horizon']} "
            f"fixed_rmse={(result['fixed_time'] or {}).get('all_six_joints_rmse_deg')} "
            f"phase_rmse={(result['phase_aligned'] or {}).get('all_six_joints_rmse_deg')}",
            flush=True,
        )

    output = {
        "method": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "session": args.log_dir.name,
            "checkpoint_selection": "Nearest saved checkpoint to 0M, each integer million from 1M through 29M, and the final saved checkpoint",
            "checkpoint_interval": args.checkpoint_interval,
            "target_velocity_m_per_s": args.target_velocity,
            "reference_start_index": 0,
            "deterministic_policy": True,
            "control_framerate_hz": int(config.env_params.control_framerate),
            "fall_only_horizon_steps": args.horizon_steps,
            "fall_only_horizon_seconds": args.horizon_steps / config.env_params.control_framerate,
            "rmse_steps": args.rmse_steps,
            "rmse_duration_seconds": args.rmse_steps / config.env_params.control_framerate,
            "estimated_reference_period_steps": period_steps,
            "estimated_reference_period_seconds": period_steps / config.env_params.control_framerate,
            "estimated_reference_cadence_steps_per_minute": 2 * 60 * config.env_params.control_framerate / period_steps,
            "original_safe_height_m": float(config.env_params.safe_height),
            "original_out_of_trajectory_threshold_rad": float(config.env_params.out_of_trajectory_threshold),
            "original_custom_max_episode_steps": int(config.env_params.custom_max_episode_steps),
            "note": "The rollout disables imitation-error termination only for measurement. The first original-threshold crossing is reconstructed from the same trajectory.",
        },
        "rankings": _rankings(results),
        "checkpoints": results,
    }

    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    rows = [_csv_row(item) for item in results]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "checkpoints": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
