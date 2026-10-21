"""Diagnose fixed-time and gait-phase joint tracking for PPO checkpoints.

All checkpoints start from reference index zero and are evaluated with deterministic
actions.  The script exports the raw per-step joint trajectories, fixed-time RMSE by
time segment, and RMSE after one common circular phase shift per gait cycle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


JOINTS = {
    "hip": {"right": "hip_flexion_r", "left": "hip_flexion_l"},
    "knee": {"right": "knee_angle_r", "left": "knee_angle_l"},
    "ankle": {"right": "ankle_angle_r", "left": "ankle_angle_l"},
}
CHANNELS = [
    (joint, side, joint_name)
    for joint, side_names in JOINTS.items()
    for side, joint_name in side_names.items()
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=884)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--period-min-seconds", type=float, default=0.7)
    parser.add_argument("--period-max-seconds", type=float, default=1.5)
    return parser.parse_args()


def _checkpoint_steps(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def _joint_metrics(errors_deg: np.ndarray) -> tuple[dict, float]:
    """Return R/L/bilateral metrics from an array shaped [time, six channels]."""
    result = {}
    all_joint_errors = []
    for joint in JOINTS:
        channel_indices = [i for i, channel in enumerate(CHANNELS) if channel[0] == joint]
        right = errors_deg[:, channel_indices[0]]
        left = errors_deg[:, channel_indices[1]]
        bilateral = np.concatenate((right, left))
        result[joint] = {
            "right_rmse_deg": _rmse(right),
            "left_rmse_deg": _rmse(left),
            "bilateral_rmse_deg": _rmse(bilateral),
        }
        all_joint_errors.append(bilateral)
    return result, _rmse(np.concatenate(all_joint_errors))


def _estimate_reference_period(
    reference_deg: np.ndarray, control_framerate: int, minimum_seconds: float, maximum_seconds: float
) -> tuple[int, list[dict]]:
    """Estimate the shared gait period from normalized six-joint autocorrelation."""
    centered = reference_deg - np.mean(reference_deg, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    normalized = centered / np.where(scale > 1e-12, scale, 1.0)
    minimum_lag = int(round(minimum_seconds * control_framerate))
    maximum_lag = int(round(maximum_seconds * control_framerate))
    candidates = []
    for lag in range(minimum_lag, maximum_lag + 1):
        difference = normalized[lag:] - normalized[:-lag]
        candidates.append(
            {
                "lag_steps": lag,
                "lag_seconds": lag / control_framerate,
                "normalized_mse": float(np.mean(np.square(difference))),
            }
        )
    best = min(candidates, key=lambda item: item["normalized_mse"])
    return int(best["lag_steps"]), candidates


def _phase_align_by_cycle(
    simulated_deg: np.ndarray, reference_deg: np.ndarray, period_steps: int
) -> tuple[np.ndarray, list[dict]]:
    """Align each full cycle using one common circular shift for all six joints."""
    cycle_count = len(simulated_deg) // period_steps
    usable_steps = cycle_count * period_steps
    aligned_errors = np.empty((usable_steps, simulated_deg.shape[1]), dtype=np.float64)
    cycle_results = []
    shifts = range(-(period_steps // 2), period_steps - period_steps // 2)

    for cycle in range(cycle_count):
        start = cycle * period_steps
        end = start + period_steps
        simulated_cycle = simulated_deg[start:end]
        reference_cycle = reference_deg[start:end]
        candidates = []
        for shift in shifts:
            aligned_reference = np.roll(reference_cycle, shift=shift, axis=0)
            error = simulated_cycle - aligned_reference
            candidates.append((_rmse(error.reshape(-1)), shift, error))
        best_rmse, best_shift, best_error = min(candidates, key=lambda item: item[0])
        aligned_errors[start:end] = best_error
        cycle_results.append(
            {
                "cycle": cycle,
                "start_step": start + 1,
                "end_step": end,
                "shift_steps": int(best_shift),
                "cycle_overall_rmse_deg": float(best_rmse),
            }
        )
    return aligned_errors, cycle_results


def main() -> None:
    args = _parse_args()
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    with (args.log_dir / "session_config.json").open(encoding="utf-8") as file:
        config = DictionableDataclass.create(ImitationTrainSessionConfig, json.load(file))

    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    # A fixed 884-step diagnostic horizon is required for a like-for-like comparison.
    # The training termination thresholds can end different checkpoints at different
    # times (the 10M checkpoint stops after 46 steps), so disable those two wrappers
    # for this trajectory-only analysis without changing the simulation dynamics.
    config.env_params.custom_max_episode_steps = 1_000_000_000
    config.env_params.out_of_trajectory_threshold = 1_000_000
    np.random.seed(config.env_params.seed)
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    unwrapped = env.unwrapped
    control_framerate = int(config.env_params.control_framerate)

    trajectories = []
    try:
        for checkpoint in args.checkpoints:
            model = EnvironmentHandler.get_stable_baselines3_model(
                config, env, trained_model_path=str(checkpoint)
            )
            unwrapped.set_target_velocity_mode_manually(
                MyoAssistLegBase.VelocityMode.UNIFORM,
                0,
                args.target_velocity,
                args.target_velocity,
                args.target_velocity,
                target_velocity_period=3,
            )
            obs, _info = env.reset()
            if unwrapped._imitation_index != 0:
                raise RuntimeError(
                    f"Reference did not reset to index 0: {unwrapped._imitation_index}"
                )

            simulated_rows = []
            reference_rows = []
            reference_indices = []
            alignment_disagreements_deg = []
            terminated = truncated = False
            terminal_step = None

            for step in range(1, args.steps + 1):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(action)
                reference_index = int(unwrapped._imitation_index)
                environment_qpos_diff = unwrapped._get_qpos_diff()
                simulated_row = []
                reference_row = []

                for _joint, _side, joint_name in CHANNELS:
                    simulated = float(unwrapped.sim.data.joint(joint_name).qpos[0])
                    reference = float(
                        np.asarray(
                            unwrapped._reference_data["series_data"][f"q_{joint_name}"][reference_index]
                        ).squeeze()
                    )
                    manual_error_deg = float(np.rad2deg(simulated - reference))
                    environment_error_deg = float(
                        np.rad2deg(np.asarray(environment_qpos_diff[joint_name]).squeeze())
                    )
                    alignment_disagreements_deg.append(
                        abs(manual_error_deg - environment_error_deg)
                    )
                    simulated_row.append(np.rad2deg(simulated))
                    reference_row.append(np.rad2deg(reference))

                simulated_rows.append(simulated_row)
                reference_rows.append(reference_row)
                reference_indices.append(reference_index)

                if terminated or truncated:
                    terminal_step = step
                    break

            if len(simulated_rows) != args.steps:
                raise RuntimeError(
                    f"{checkpoint.name} stopped at {len(simulated_rows)}/{args.steps} steps "
                    f"(terminated={terminated}, truncated={truncated})"
                )

            trajectories.append(
                {
                    "checkpoint": checkpoint.name,
                    "timesteps": _checkpoint_steps(checkpoint),
                    "simulated_deg": np.asarray(simulated_rows, dtype=np.float64),
                    "reference_deg": np.asarray(reference_rows, dtype=np.float64),
                    "reference_indices": reference_indices,
                    "terminated_at_or_before_horizon": bool(terminated),
                    "truncated_at_or_before_horizon": bool(truncated),
                    "terminal_step": terminal_step,
                    "max_alignment_check_error_deg": float(max(alignment_disagreements_deg)),
                }
            )
    finally:
        env.close()

    canonical_reference = trajectories[0]["reference_deg"]
    canonical_indices = trajectories[0]["reference_indices"]
    for trajectory in trajectories[1:]:
        if trajectory["reference_indices"] != canonical_indices:
            raise RuntimeError("Reference indices differ between checkpoints")
        if not np.array_equal(trajectory["reference_deg"], canonical_reference):
            raise RuntimeError("Reference joint trajectories differ between checkpoints")

    period_steps, period_candidates = _estimate_reference_period(
        canonical_reference,
        control_framerate,
        args.period_min_seconds,
        args.period_max_seconds,
    )
    full_cycle_steps = (args.steps // period_steps) * period_steps
    if full_cycle_steps != args.steps:
        raise RuntimeError(
            f"The requested horizon ({args.steps}) is not an integer number of the estimated "
            f"{period_steps}-step gait cycles; use {full_cycle_steps} steps for strict comparison"
        )

    segment_bounds = [
        ("0-10s", 0, min(10 * control_framerate, args.steps)),
        ("10-20s", min(10 * control_framerate, args.steps), min(20 * control_framerate, args.steps)),
        (
            f"20-{args.steps / control_framerate:.2f}s",
            min(20 * control_framerate, args.steps),
            args.steps,
        ),
    ]
    segment_bounds = [segment for segment in segment_bounds if segment[2] > segment[1]]

    results = []
    raw_rows = []
    for trajectory in trajectories:
        simulated = trajectory["simulated_deg"]
        reference = trajectory["reference_deg"]
        fixed_errors = simulated - reference
        fixed_joint_metrics, fixed_overall = _joint_metrics(fixed_errors)

        segmented = []
        for label, start, end in segment_bounds:
            metrics, overall = _joint_metrics(fixed_errors[start:end])
            segmented.append(
                {
                    "segment": label,
                    "start_step_inclusive": start + 1,
                    "end_step_inclusive": end,
                    "sample_count_per_side": end - start,
                    "joint_rmse_deg": metrics,
                    "all_six_joints_rmse_deg": overall,
                }
            )

        aligned_errors, cycle_results = _phase_align_by_cycle(
            simulated, reference, period_steps
        )
        phase_joint_metrics, phase_overall = _joint_metrics(aligned_errors)
        shifts = np.asarray([item["shift_steps"] for item in cycle_results], dtype=float)
        unwrapped_shifts = np.unwrap(shifts * 2 * np.pi / period_steps) * period_steps / (2 * np.pi)
        cycle_numbers = np.arange(len(unwrapped_shifts), dtype=float)
        drift_per_cycle = float(np.polyfit(cycle_numbers, unwrapped_shifts, deg=1)[0])
        phase_segmented = []
        for label, start, end in segment_bounds:
            metrics, overall = _joint_metrics(aligned_errors[start:end])
            phase_segmented.append(
                {
                    "segment": label,
                    "joint_rmse_deg": metrics,
                    "all_six_joints_rmse_deg": overall,
                }
            )

        results.append(
            {
                "checkpoint": trajectory["checkpoint"],
                "timesteps": trajectory["timesteps"],
                "completed_control_steps": args.steps,
                "terminated_at_horizon": trajectory["terminated_at_or_before_horizon"],
                "truncated_at_horizon": trajectory["truncated_at_or_before_horizon"],
                "terminal_step": trajectory["terminal_step"],
                "max_alignment_check_error_deg": trajectory[
                    "max_alignment_check_error_deg"
                ],
                "fixed_time": {
                    "joint_rmse_deg": fixed_joint_metrics,
                    "all_six_joints_rmse_deg": fixed_overall,
                    "segments": segmented,
                },
                "phase_aligned": {
                    "joint_rmse_deg": phase_joint_metrics,
                    "all_six_joints_rmse_deg": phase_overall,
                    "segments": phase_segmented,
                    "mean_signed_shift_steps": float(np.mean(shifts)),
                    "median_signed_shift_steps": float(np.median(shifts)),
                    "mean_absolute_shift_steps": float(np.mean(np.abs(shifts))),
                    "max_absolute_shift_steps": float(np.max(np.abs(shifts))),
                    "unwrapped_shift_drift_steps_per_cycle": drift_per_cycle,
                    "cycles": cycle_results,
                },
            }
        )

        for sample_index in range(args.steps):
            for channel_index, (joint, side, joint_name) in enumerate(CHANNELS):
                raw_rows.append(
                    {
                        "checkpoint": trajectory["checkpoint"],
                        "timesteps": trajectory["timesteps"],
                        "control_step": sample_index + 1,
                        "time_seconds": (sample_index + 1) / control_framerate,
                        "reference_index": trajectory["reference_indices"][sample_index],
                        "joint": joint,
                        "side": side,
                        "joint_name": joint_name,
                        "simulated_angle_deg": simulated[sample_index, channel_index],
                        "reference_angle_deg": reference[sample_index, channel_index],
                        "fixed_error_deg": fixed_errors[sample_index, channel_index],
                        "phase_aligned_error_deg": aligned_errors[sample_index, channel_index],
                    }
                )

    output_prefix = args.output_prefix or args.log_dir / "joint_rmse_diagnostics_884"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    segment_csv_path = output_prefix.with_name(output_prefix.name + "_segments.csv")
    phase_csv_path = output_prefix.with_name(output_prefix.name + "_phase_aligned.csv")
    raw_csv_path = output_prefix.with_name(output_prefix.name + "_trajectories.csv")

    payload = {
        "method": {
            "reference_start_index": 0,
            "first_error_reference_index": canonical_indices[0],
            "control_framerate_hz": control_framerate,
            "common_control_steps": args.steps,
            "duration_seconds": args.steps / control_framerate,
            "target_velocity_m_per_s": args.target_velocity,
            "deterministic_policy": True,
            "fixed_horizon_evaluation": True,
            "evaluation_only_overrides": {
                "custom_max_episode_steps": 1000000000,
                "out_of_trajectory_threshold": 1000000,
            },
            "fixed_time_formula": "sqrt(mean((simulated_angle_deg - same_step_reference_angle_deg)^2))",
            "phase_period_search_seconds": [
                args.period_min_seconds,
                args.period_max_seconds,
            ],
            "estimated_gait_period_steps": period_steps,
            "estimated_gait_period_seconds": period_steps / control_framerate,
            "full_gait_cycles": args.steps // period_steps,
            "phase_alignment": (
                "For every reference-defined gait cycle, apply one shared integer circular "
                "shift to all six reference joint channels that minimizes pooled six-joint RMSE"
            ),
            "positive_shift_definition": (
                "Positive shift means the simulated cycle lags the reference by that many control steps"
            ),
            "period_candidates": period_candidates,
        },
        "checkpoints": results,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    metric_fields = (
        "checkpoint",
        "timesteps",
        "metric",
        "segment",
        "joint",
        "right_rmse_deg",
        "left_rmse_deg",
        "bilateral_rmse_deg",
        "all_six_joints_rmse_deg",
    )
    with segment_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        for result in results:
            for segment in result["fixed_time"]["segments"]:
                for joint, metrics in segment["joint_rmse_deg"].items():
                    writer.writerow(
                        {
                            "checkpoint": result["checkpoint"],
                            "timesteps": result["timesteps"],
                            "metric": "fixed_time_segment",
                            "segment": segment["segment"],
                            "joint": joint,
                            **metrics,
                            "all_six_joints_rmse_deg": segment[
                                "all_six_joints_rmse_deg"
                            ],
                        }
                    )

    with phase_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        for result in results:
            for joint, metrics in result["phase_aligned"]["joint_rmse_deg"].items():
                writer.writerow(
                    {
                        "checkpoint": result["checkpoint"],
                        "timesteps": result["timesteps"],
                        "metric": "phase_aligned_full_horizon",
                        "segment": "0-29.47s",
                        "joint": joint,
                        **metrics,
                        "all_six_joints_rmse_deg": result["phase_aligned"][
                            "all_six_joints_rmse_deg"
                        ],
                    }
                )

    with raw_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"JSON:       {json_path}")
    print(f"Segments:   {segment_csv_path}")
    print(f"Phase:      {phase_csv_path}")
    print(f"Trajectory: {raw_csv_path}")


if __name__ == "__main__":
    main()
