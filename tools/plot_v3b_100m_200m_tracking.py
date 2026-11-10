#!/usr/bin/env python3
"""Compare deterministic 100M and 200M V3-B tracking against one reference.

The two checkpoints are evaluated with the saved V3-B session configuration,
the same seed, reference start index 0, a 1.25 m/s target, and the original
strict termination conditions.  The script writes two figures, an inspectable
CSV time series, and a JSON metric summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


TRACKED = (
    "pelvis_ty",
    "pelvis_tilt",
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
)
LOWER_SIX = TRACKED[2:]
DISPLAY = {
    "pelvis_ty": "Pelvis height",
    "pelvis_tilt": "Pelvis tilt",
    "hip_flexion_r": "Right hip",
    "hip_flexion_l": "Left hip",
    "knee_angle_r": "Right knee",
    "knee_angle_l": "Left knee",
    "ankle_angle_r": "Right ankle",
    "ankle_angle_l": "Left ankle",
}
MODEL_COLORS = {"100M": "#1f5a8a", "200M": "#d97706"}
SIDE_COLORS = {"Right": "#1f5a8a", "Left": "#d97706"}


@dataclass
class Rollout:
    label: str
    checkpoint: Path
    checkpoint_steps: int
    time_s: np.ndarray
    actual: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    velocity: np.ndarray
    target_velocity: np.ndarray
    reward: np.ndarray
    termination_reason: str | None
    info: dict


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
    parser.add_argument("--checkpoint-200m-step", type=int, default=200_007_680)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=session / "trajectory_comparison_100m_200m",
    )
    return parser.parse_args()


def load_config(session_dir: Path) -> ImitationTrainSessionConfig:
    config_dict = json.loads(
        (session_dir / "session_config.json").read_text(encoding="utf-8")
    )
    config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    return config


def rollout_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    checkpoint_steps: int,
    session_dir: Path,
    seconds: float,
    target_velocity: float,
    seed: int,
) -> Rollout:
    config = load_config(session_dir)
    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=False, is_evaluate_mode=True
    )
    raw = env.unwrapped
    try:
        policy = EnvironmentHandler.get_stable_baselines3_model(
            config, env, trained_model_path=str(checkpoint.resolve())
        )
        raw.set_target_velocity_mode_manually(
            MyoAssistLegBase.VelocityMode.UNIFORM,
            0.0,
            target_velocity,
            target_velocity,
            target_velocity,
            target_velocity_period=3.0,
        )
        observation, _ = env.reset(seed=seed)
        if observation.shape != (62,) or policy.action_space.shape != (22,):
            raise RuntimeError(
                f"Unexpected interface for {label}: obs={observation.shape}, "
                f"action={policy.action_space.shape}"
            )

        control_hz = int(config.env_params.control_framerate)
        maximum_steps = int(round(seconds * control_hz))
        actual = {key: [] for key in TRACKED}
        reference = {key: [] for key in TRACKED}
        velocity = []
        targets = []
        rewards = []
        times = []
        termination_reason = None
        last_info: dict = {}

        for step in range(1, maximum_steps + 1):
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(observation)) or not np.isfinite(reward):
                raise FloatingPointError(f"Non-finite rollout for {label} at step {step}")

            for key in TRACKED:
                q_actual = float(raw.sim.data.joint(key).qpos[0])
                q_reference = float(raw._reference_value(f"q_{key}"))
                if key != "pelvis_ty":
                    q_actual = float(np.rad2deg(q_actual))
                    q_reference = float(np.rad2deg(q_reference))
                actual[key].append(q_actual)
                reference[key].append(q_reference)

            times.append(step / control_hz)
            velocity.append(float(raw.sim.data.joint("pelvis_tx").qvel[0]))
            targets.append(target_velocity)
            rewards.append(float(reward))
            last_info = dict(info)
            if terminated or truncated:
                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "terminated" if terminated else "truncated",
                    )
                )
                break

        return Rollout(
            label=label,
            checkpoint=checkpoint.resolve(),
            checkpoint_steps=checkpoint_steps,
            time_s=np.asarray(times, dtype=float),
            actual={key: np.asarray(values, dtype=float) for key, values in actual.items()},
            reference={
                key: np.asarray(values, dtype=float) for key, values in reference.items()
            },
            velocity=np.asarray(velocity, dtype=float),
            target_velocity=np.asarray(targets, dtype=float),
            reward=np.asarray(rewards, dtype=float),
            termination_reason=termination_reason,
            info=last_info,
        )
    finally:
        env.close()


def _style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#cbd5e1", alpha=0.5, linewidth=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_actual_reference(rollouts: list[Rollout], path: Path) -> None:
    rows = (
        ("Pelvis height", ("pelvis_ty",), "Height (m)"),
        ("Pelvis tilt", ("pelvis_tilt",), "Angle (deg)"),
        ("Hip flexion", ("hip_flexion_r", "hip_flexion_l"), "Angle (deg)"),
        ("Knee angle", ("knee_angle_r", "knee_angle_l"), "Angle (deg)"),
        ("Ankle angle", ("ankle_angle_r", "ankle_angle_l"), "Angle (deg)"),
        ("Forward velocity", ("velocity",), "Velocity (m/s)"),
    )
    figure, axes = plt.subplots(
        len(rows), 2, figsize=(15.5, 17.5), sharex="col", constrained_layout=True
    )
    for column, rollout in enumerate(rollouts):
        for row, (title, keys, ylabel) in enumerate(rows):
            axis = axes[row, column]
            if keys == ("velocity",):
                axis.plot(
                    rollout.time_s,
                    rollout.target_velocity,
                    color="#475569",
                    linestyle="--",
                    linewidth=1.35,
                    label="Reference",
                )
                axis.plot(
                    rollout.time_s,
                    rollout.velocity,
                    color=MODEL_COLORS[rollout.label],
                    linewidth=1.15,
                    label="Actual",
                )
            else:
                for key in keys:
                    if key.endswith("_r"):
                        side = "Right"
                    elif key.endswith("_l"):
                        side = "Left"
                    else:
                        side = "Actual"
                    color = SIDE_COLORS.get(side, MODEL_COLORS[rollout.label])
                    actual_label = side if len(keys) > 1 else "Actual"
                    ref_label = f"{side} reference" if len(keys) > 1 else "Reference"
                    axis.plot(
                        rollout.time_s,
                        rollout.actual[key],
                        color=color,
                        linewidth=1.15,
                        label=actual_label,
                    )
                    axis.plot(
                        rollout.time_s,
                        rollout.reference[key],
                        color=color,
                        linestyle="--",
                        linewidth=1.15,
                        alpha=0.9,
                        label=ref_label,
                    )
            axis.set_title(title if row else f"{rollout.label} checkpoint — {title}")
            _style_axis(axis, ylabel)
            if row in (0, 2, 5):
                axis.legend(loc="best", fontsize=8, frameon=False, ncol=2)
        axes[-1, column].set_xlabel("Time (s)")
    figure.suptitle(
        "V3-B deterministic tracking: actual trajectories vs reference\n"
        "Same reference start (index 0), seed 1234, target speed 1.25 m/s",
        fontsize=16,
    )
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)


def plot_errors(rollouts: list[Rollout], path: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(14.5, 13), sharex=True, constrained_layout=True)
    for axis, key in zip(axes.flat, TRACKED, strict=True):
        for rollout in rollouts:
            error = rollout.actual[key] - rollout.reference[key]
            rmse = float(np.sqrt(np.mean(np.square(error))))
            unit = "m" if key == "pelvis_ty" else "deg"
            axis.plot(
                rollout.time_s,
                error,
                color=MODEL_COLORS[rollout.label],
                linewidth=1.2,
                label=f"{rollout.label} (RMSE {rmse:.3f} {unit})",
            )
        axis.axhline(0.0, color="#0f172a", linewidth=0.8)
        axis.set_title(DISPLAY[key])
        _style_axis(axis, "Error (m)" if key == "pelvis_ty" else "Error (deg)")
        axis.legend(loc="best", fontsize=8, frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        "V3-B tracking error (actual − reference)\n"
        "Focused per-joint y-scales; RMSE values are directly comparable between models",
        fontsize=16,
    )
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)


def summarize(rollout: Rollout) -> dict:
    per_quantity = {}
    for key in TRACKED:
        error = rollout.actual[key] - rollout.reference[key]
        unit = "m" if key == "pelvis_ty" else "deg"
        per_quantity[key] = {
            "unit": unit,
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error))),
            "max_abs_error": float(np.max(np.abs(error))),
        }
    lower_errors = np.concatenate(
        [rollout.actual[key] - rollout.reference[key] for key in LOWER_SIX]
    )
    return {
        "checkpoint": str(rollout.checkpoint),
        "checkpoint_steps": rollout.checkpoint_steps,
        "duration_s": float(rollout.time_s[-1]),
        "completed_control_steps": int(len(rollout.time_s)),
        "termination_reason": rollout.termination_reason,
        "mean_forward_velocity_m_per_s": float(np.mean(rollout.velocity)),
        "forward_velocity_mae_m_per_s": float(
            np.mean(np.abs(rollout.velocity - rollout.target_velocity))
        ),
        "mean_reward_per_step": float(np.mean(rollout.reward)),
        "lower_limb_six_joint_rmse_deg": float(
            np.sqrt(np.mean(np.square(lower_errors)))
        ),
        "per_quantity": per_quantity,
    }


def write_csv(rollouts: list[Rollout], path: Path) -> None:
    fields = ["model", "checkpoint_steps", "time_s", "forward_velocity_actual_m_s", "forward_velocity_reference_m_s", "reward"]
    for key in TRACKED:
        suffix = "m" if key == "pelvis_ty" else "deg"
        fields.extend(
            [
                f"{key}_actual_{suffix}",
                f"{key}_reference_{suffix}",
                f"{key}_error_{suffix}",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rollout in rollouts:
            for index, time_s in enumerate(rollout.time_s):
                row = {
                    "model": rollout.label,
                    "checkpoint_steps": rollout.checkpoint_steps,
                    "time_s": float(time_s),
                    "forward_velocity_actual_m_s": float(rollout.velocity[index]),
                    "forward_velocity_reference_m_s": float(rollout.target_velocity[index]),
                    "reward": float(rollout.reward[index]),
                }
                for key in TRACKED:
                    suffix = "m" if key == "pelvis_ty" else "deg"
                    actual = float(rollout.actual[key][index])
                    reference = float(rollout.reference[key][index])
                    row[f"{key}_actual_{suffix}"] = actual
                    row[f"{key}_reference_{suffix}"] = reference
                    row[f"{key}_error_{suffix}"] = actual - reference
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    session_config = args.session_dir / "session_config.json"
    for required in (session_config, args.checkpoint_100m, args.checkpoint_200m):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = [
        rollout_checkpoint(
            label="100M",
            checkpoint=args.checkpoint_100m,
            checkpoint_steps=100_016_128,
            session_dir=args.session_dir,
            seconds=args.seconds,
            target_velocity=args.target_velocity,
            seed=args.seed,
        ),
        rollout_checkpoint(
            label="200M",
            checkpoint=args.checkpoint_200m,
            checkpoint_steps=args.checkpoint_200m_step,
            session_dir=args.session_dir,
            seconds=args.seconds,
            target_velocity=args.target_velocity,
            seed=args.seed,
        ),
    ]

    actual_reference_path = args.output_dir / "actual_vs_reference_100m_200m.png"
    errors_path = args.output_dir / "tracking_error_100m_200m.png"
    csv_path = args.output_dir / "tracking_timeseries_100m_200m.csv"
    summary_path = args.output_dir / "tracking_summary_100m_200m.json"
    plot_actual_reference(rollouts, actual_reference_path)
    plot_errors(rollouts, errors_path)
    write_csv(rollouts, csv_path)

    summary = {
        "evaluation": {
            "deterministic": True,
            "seed": args.seed,
            "reference_start_index": 0,
            "target_velocity_m_per_s": args.target_velocity,
            "requested_duration_s": args.seconds,
            "strict_v3b_termination_preserved": True,
        },
        "models": {rollout.label: summarize(rollout) for rollout in rollouts},
        "outputs": {
            "actual_vs_reference_plot": str(actual_reference_path.resolve()),
            "tracking_error_plot": str(errors_path.resolve()),
            "timeseries_csv": str(csv_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
