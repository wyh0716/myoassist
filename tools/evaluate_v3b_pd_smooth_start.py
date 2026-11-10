#!/usr/bin/env python3
"""Paired 12-start evaluation of immediate versus quintic MotorArms PD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_train.envs.action_adapters import MOTOR_ARM_ACTUATORS
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.evaluate_checkpoint_multistart import (
    _maximum_training_start,
    _select_start_indices,
    _set_reference_start,
)
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


LOWER_JOINTS = (
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
)


@dataclass(frozen=True)
class Variant:
    checkpoint_label: str
    checkpoint_path: Path
    smooth_seconds: float

    @property
    def controller_label(self) -> str:
        return "smooth_0p5s" if self.smooth_seconds > 0.0 else "original_pd"

    @property
    def key(self) -> str:
        return f"{self.checkpoint_label}_{self.controller_label}"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    session = root / "rl_train/results/train_session_20261106-123840"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=session)
    parser.add_argument(
        "--checkpoint-100m",
        type=Path,
        default=session / "trained_models/model_100016128.zip",
    )
    parser.add_argument(
        "--checkpoint-200m",
        type=Path,
        default=session / "trained_models/final_model.zip",
    )
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=session / "pd_smooth_start_12start_comparison",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_config(session_dir: Path) -> ImitationTrainSessionConfig:
    return DictionableDataclass.create(
        ImitationTrainSessionConfig,
        json.loads((session_dir / "session_config.json").read_text(encoding="utf-8")),
    )


def clone_config(
    base: ImitationTrainSessionConfig, smooth_seconds: float
) -> ImitationTrainSessionConfig:
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig, DictionableDataclass.to_dict(base)
    )
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    config.env_params.motor_arms_pd_smooth_start_seconds = smooth_seconds
    return config


def resolve_start_indices(
    base: ImitationTrainSessionConfig,
    starts: int,
    seed: int,
) -> list[int]:
    config = clone_config(base, 0.0)
    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=False, is_evaluate_mode=True
    )
    raw = env.unwrapped
    try:
        maximum = _maximum_training_start(
            raw._reference_data_length,
            int(base.env_params.custom_max_episode_steps),
            float(base.env_params.max_target_velocity),
            float(raw._reference_nominal_velocity),
        )
        return _select_start_indices(maximum, starts, "even", seed)
    finally:
        env.close()


def _arm_actual_output_torque(raw) -> np.ndarray:
    model = raw.sim.model
    data = raw.sim.data
    return np.asarray(
        [
            float(data.actuator(name).force[0])
            * float(model.actuator_gear[model.actuator(name).id, 0])
            for name in MOTOR_ARM_ACTUATORS
        ],
        dtype=float,
    )


def _arm_generalized_torque(raw) -> np.ndarray:
    model = raw.sim.model
    data = raw.sim.data
    return np.asarray(
        [
            float(data.qfrc_actuator[model.jnt_dofadr[model.joint(name).id]])
            for name in MOTOR_ARM_ACTUATORS
        ],
        dtype=float,
    )


def evaluate_variant(
    *,
    variant: Variant,
    base_config: ImitationTrainSessionConfig,
    start_indices: list[int],
    seconds: float,
    target_velocity: float,
    seed: int,
) -> dict:
    config = clone_config(base_config, variant.smooth_seconds)
    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=False, is_evaluate_mode=True
    )
    raw = env.unwrapped
    try:
        policy = EnvironmentHandler.get_stable_baselines3_model(
            config, env, trained_model_path=str(variant.checkpoint_path.resolve())
        )
        raw.set_target_velocity_mode_manually(
            MyoAssistLegBase.VelocityMode.UNIFORM,
            0.0,
            target_velocity,
            target_velocity,
            target_velocity,
            target_velocity_period=3.0,
        )
        control_hz = int(config.env_params.control_framerate)
        horizon_steps = int(round(seconds * control_hz))
        startup_steps = int(round(0.5 * control_hz))
        trials = []

        for trial_index, start_index in enumerate(start_indices):
            env.reset(seed=seed + trial_index)
            observation = _set_reference_start(raw, start_index)
            simulated = []
            reference = []
            velocity_errors = []
            heights = []
            rewards = []
            arm_qvel = []
            arm_requested = []
            arm_commanded = []
            arm_actual = []
            arm_generalized = []
            arm_target_velocity = []
            termination_reason = None
            terminated = False
            truncated = False

            previous_arm_qvel = np.asarray(
                [float(raw.sim.data.joint(name).qvel[0]) for name in MOTOR_ARM_ACTUATORS]
            )
            startup_acceleration = []

            for step in range(1, horizon_steps + 1):
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                current_arm_qvel = np.asarray(
                    [
                        float(raw.sim.data.joint(name).qvel[0])
                        for name in MOTOR_ARM_ACTUATORS
                    ],
                    dtype=float,
                )
                simulated.append(
                    [float(raw.sim.data.joint(name).qpos[0]) for name in LOWER_JOINTS]
                )
                reference.append(
                    [float(raw._reference_value(f"q_{name}")) for name in LOWER_JOINTS]
                )
                velocity_errors.append(
                    abs(float(raw.sim.data.joint("pelvis_tx").qvel[0]) - target_velocity)
                )
                heights.append(float(raw.sim.data.joint("pelvis_ty").qpos[0]))
                rewards.append(float(reward))

                if step <= startup_steps:
                    arm_qvel.append(current_arm_qvel.copy())
                    startup_acceleration.append(
                        (current_arm_qvel - previous_arm_qvel) * control_hz
                    )
                    arm_requested.append(env.last_arm_requested_torques_nm.copy())
                    arm_commanded.append(env.last_arm_commanded_torques_nm.copy())
                    arm_actual.append(_arm_actual_output_torque(raw))
                    arm_generalized.append(_arm_generalized_torque(raw))
                    arm_target_velocity.append(
                        env.last_arm_target_velocities_rad_s.copy()
                    )
                previous_arm_qvel = current_arm_qvel
                if terminated or truncated:
                    termination_reason = str(
                        info.get(
                            "termination_reason",
                            "terminated" if terminated else "truncated",
                        )
                    )
                    break

            simulated_array = np.asarray(simulated, dtype=float)
            reference_array = np.asarray(reference, dtype=float)
            error_deg = np.rad2deg(simulated_array - reference_array)
            arm_qvel_array = np.asarray(arm_qvel, dtype=float)
            arm_acc_array = np.asarray(startup_acceleration, dtype=float)
            requested_array = np.asarray(arm_requested, dtype=float)
            commanded_array = np.asarray(arm_commanded, dtype=float)
            actual_array = np.asarray(arm_actual, dtype=float)
            generalized_array = np.asarray(arm_generalized, dtype=float)
            target_velocity_array = np.asarray(arm_target_velocity, dtype=float)
            completed_steps = len(simulated_array)

            trials.append(
                {
                    "reference_start_index": start_index,
                    "completed_steps": completed_steps,
                    "completed_10s": completed_steps == horizon_steps,
                    "termination_reason": termination_reason,
                    "lower_six_rmse_deg": float(
                        np.sqrt(np.mean(np.square(error_deg)))
                    ),
                    "velocity_mae_m_s": float(np.mean(velocity_errors)),
                    "minimum_pelvis_height_m": float(np.min(heights)),
                    "mean_reward_per_step": float(np.mean(rewards)),
                    "startup_0p5s": {
                        "maximum_arm_speed_rad_s": float(
                            np.max(np.abs(arm_qvel_array))
                        ),
                        "maximum_arm_acceleration_rad_s2": float(
                            np.max(np.abs(arm_acc_array))
                        ),
                        "maximum_target_speed_rad_s": float(
                            np.max(np.abs(target_velocity_array))
                        ),
                        "maximum_requested_pd_torque_nm": float(
                            np.max(np.abs(requested_array))
                        ),
                        "maximum_commanded_torque_nm": float(
                            np.max(np.abs(commanded_array))
                        ),
                        "maximum_actual_actuator_output_torque_nm": float(
                            np.max(np.abs(actual_array))
                        ),
                        "maximum_actual_generalized_torque_nm": float(
                            np.max(np.abs(generalized_array))
                        ),
                        "maximum_requested_minus_actual_nm": float(
                            np.max(np.abs(requested_array - actual_array))
                        ),
                        "maximum_commanded_minus_actual_nm": float(
                            np.max(np.abs(commanded_array - actual_array))
                        ),
                        "maximum_actual_minus_generalized_nm": float(
                            np.max(np.abs(actual_array - generalized_array))
                        ),
                        "first_step_requested_pd_torque_nm": requested_array[0].tolist(),
                        "first_step_actual_output_torque_nm": actual_array[0].tolist(),
                    },
                }
            )

        completed = [item for item in trials if item["completed_10s"]]
        startup_keys = tuple(trials[0]["startup_0p5s"])[0:10]
        aggregate = {
            "trial_count": len(trials),
            "full_10s_success_rate": float(
                np.mean([item["completed_10s"] for item in trials])
            ),
            "lower_six_rmse_deg_mean": float(
                np.mean([item["lower_six_rmse_deg"] for item in trials])
            ),
            "velocity_mae_m_s_mean": float(
                np.mean([item["velocity_mae_m_s"] for item in trials])
            ),
            "minimum_pelvis_height_m_min": float(
                np.min([item["minimum_pelvis_height_m"] for item in trials])
            ),
            "mean_reward_per_step": float(
                np.mean([item["mean_reward_per_step"] for item in trials])
            ),
            "completed_trial_count": len(completed),
            "startup_0p5s": {
                key: {
                    "mean_across_starts": float(
                        np.mean([item["startup_0p5s"][key] for item in trials])
                    ),
                    "maximum_across_starts": float(
                        np.max([item["startup_0p5s"][key] for item in trials])
                    ),
                }
                for key in startup_keys
            },
        }
        return {
            "variant": variant.key,
            "checkpoint_label": variant.checkpoint_label,
            "checkpoint_path": str(variant.checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(variant.checkpoint_path),
            "smooth_start_seconds": variant.smooth_seconds,
            "interface": {
                "observation_dimension": int(env.observation_space.shape[0]),
                "policy_action_dimension": int(env.action_space.shape[0]),
                "physical_actuator_dimension": int(raw.sim.model.nu),
            },
            "aggregate": aggregate,
            "trials": trials,
        }
    finally:
        env.close()


def paired_comparison(original: dict, smooth: dict) -> dict:
    original_aggregate = original["aggregate"]
    smooth_aggregate = smooth["aggregate"]
    fields = (
        "full_10s_success_rate",
        "lower_six_rmse_deg_mean",
        "velocity_mae_m_s_mean",
        "minimum_pelvis_height_m_min",
        "mean_reward_per_step",
    )
    deltas = {
        field: float(smooth_aggregate[field] - original_aggregate[field])
        for field in fields
    }
    startup_fields = (
        "maximum_arm_speed_rad_s",
        "maximum_arm_acceleration_rad_s2",
        "maximum_requested_pd_torque_nm",
        "maximum_actual_actuator_output_torque_nm",
    )
    startup_deltas = {}
    for field in startup_fields:
        original_value = original_aggregate["startup_0p5s"][field][
            "mean_across_starts"
        ]
        smooth_value = smooth_aggregate["startup_0p5s"][field][
            "mean_across_starts"
        ]
        startup_deltas[field] = {
            "original": original_value,
            "smooth": smooth_value,
            "absolute_delta": smooth_value - original_value,
            "relative_delta_percent": (
                100.0 * (smooth_value - original_value) / original_value
                if original_value != 0.0
                else None
            ),
        }

    success_degraded = deltas["full_10s_success_rate"] < -1e-12
    rmse_degraded = (
        smooth_aggregate["lower_six_rmse_deg_mean"]
        > 1.05 * original_aggregate["lower_six_rmse_deg_mean"]
    )
    velocity_degraded = (
        smooth_aggregate["velocity_mae_m_s_mean"]
        > 1.05 * original_aggregate["velocity_mae_m_s_mean"]
    )
    return {
        "checkpoint_label": original["checkpoint_label"],
        "smooth_minus_original": deltas,
        "startup_0p5s_mean_comparison": startup_deltas,
        "degradation_flags": {
            "success_rate_lower": success_degraded,
            "lower_six_rmse_more_than_5_percent_worse": rmse_degraded,
            "velocity_mae_more_than_5_percent_worse": velocity_degraded,
        },
        "requires_shorter_ramp_test": bool(
            success_degraded or rmse_degraded or velocity_degraded
        ),
    }


def write_trial_csv(path: Path, variants: list[dict]) -> None:
    fields = (
        "variant",
        "checkpoint_label",
        "smooth_start_seconds",
        "reference_start_index",
        "completed_steps",
        "completed_10s",
        "termination_reason",
        "lower_six_rmse_deg",
        "velocity_mae_m_s",
        "minimum_pelvis_height_m",
        "mean_reward_per_step",
        "maximum_arm_speed_rad_s",
        "maximum_arm_acceleration_rad_s2",
        "maximum_requested_pd_torque_nm",
        "maximum_commanded_torque_nm",
        "maximum_actual_actuator_output_torque_nm",
        "maximum_requested_minus_actual_nm",
        "maximum_commanded_minus_actual_nm",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            for trial in variant["trials"]:
                startup = trial["startup_0p5s"]
                writer.writerow(
                    {
                        "variant": variant["variant"],
                        "checkpoint_label": variant["checkpoint_label"],
                        "smooth_start_seconds": variant["smooth_start_seconds"],
                        **{
                            key: trial[key]
                            for key in fields
                            if key in trial
                        },
                        **{
                            key: startup[key]
                            for key in fields
                            if key in startup
                        },
                    }
                )


def plot_summary(path: Path, variants: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    checkpoints = ("100M", "200M")
    colors = {"original_pd": "#64748b", "smooth_0p5s": "#1f5a8a"}
    labels = {"original_pd": "Original PD", "smooth_0p5s": "0.5 s quintic"}
    metrics = (
        ("full_10s_success_rate", "10 s success rate", "%", 100.0),
        ("lower_six_rmse_deg_mean", "Lower-six RMSE", "deg", 1.0),
        ("velocity_mae_m_s_mean", "Velocity MAE", "m/s", 1.0),
        (
            "maximum_actual_actuator_output_torque_nm",
            "First 0.5 s peak actual arm torque",
            "N·m",
            1.0,
        ),
    )
    lookup = {item["variant"]: item for item in variants}
    x = np.arange(len(checkpoints), dtype=float)
    width = 0.34
    for axis, (metric, title, unit, scale) in zip(axes.flat, metrics):
        for offset, controller in ((-width / 2, "original_pd"), (width / 2, "smooth_0p5s")):
            values = []
            for checkpoint in checkpoints:
                aggregate = lookup[f"{checkpoint}_{controller}"]["aggregate"]
                if metric.startswith("maximum_"):
                    value = aggregate["startup_0p5s"][metric]["mean_across_starts"]
                else:
                    value = aggregate[metric]
                values.append(value * scale)
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=colors[controller],
                label=labels[controller],
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        axis.set_xticks(x, checkpoints)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", color="#cbd5e1", alpha=0.5, linewidth=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "Original vs 0.5 s quintic MotorArms PD — paired 12-start evaluation",
        fontsize=14,
    )
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    for required in (
        args.session_dir / "session_config.json",
        args.checkpoint_100m,
        args.checkpoint_200m,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_base_config(args.session_dir)
    starts = resolve_start_indices(base_config, args.starts, args.seed)
    variants = [
        Variant("100M", args.checkpoint_100m, 0.0),
        Variant("100M", args.checkpoint_100m, 0.5),
        Variant("200M", args.checkpoint_200m, 0.0),
        Variant("200M", args.checkpoint_200m, 0.5),
    ]
    results = []
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] {variant.key}", flush=True)
        result = evaluate_variant(
            variant=variant,
            base_config=base_config,
            start_indices=starts,
            seconds=args.seconds,
            target_velocity=args.target_velocity,
            seed=args.seed,
        )
        results.append(result)
        print(json.dumps(result["aggregate"], ensure_ascii=False), flush=True)

    comparisons = []
    for checkpoint in ("100M", "200M"):
        original = next(
            item for item in results if item["variant"] == f"{checkpoint}_original_pd"
        )
        smooth = next(
            item for item in results if item["variant"] == f"{checkpoint}_smooth_0p5s"
        )
        comparisons.append(paired_comparison(original, smooth))

    output = {
        "method": {
            "checkpoint_weights_updated": False,
            "paired_reference_start_indices": starts,
            "start_count": args.starts,
            "seconds": args.seconds,
            "control_rate_hz": int(base_config.env_params.control_framerate),
            "target_velocity_m_s": args.target_velocity,
            "deterministic_policy": True,
            "trajectory_error_termination_enabled": bool(
                base_config.env_params.terminate_on_trajectory_error
            ),
            "trajectory_error_threshold_rad": float(
                base_config.env_params.out_of_trajectory_threshold
            ),
            "trajectory_error_consecutive_steps": int(
                base_config.env_params.trajectory_error_consecutive_steps
            ),
        },
        "variants": results,
        "paired_comparisons": comparisons,
    }
    output_path = args.output_dir / "pd_smooth_start_12start_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trial_csv(args.output_dir / "pd_smooth_start_trials.csv", results)
    plot_summary(args.output_dir / "pd_smooth_start_comparison.png", results)
    print(json.dumps({"json": str(output_path), "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
