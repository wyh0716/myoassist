#!/usr/bin/env python3
"""Validate gait-cycle kinematics, toe-off timing, and bilateral symmetry.

The 100M and 200M V3-B checkpoints are evaluated deterministically with the
same reference start, target speed, and the validated 0.5 s quintic MotorArms
PD start ramp.  Actual foot touch sensors define ipsilateral heel-strike to
heel-strike gait cycles.  Results are reported per complete cycle after the
startup transient rather than over an arbitrary 10 s window.
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


SIDES = ("r", "l")
JOINT_BASES = ("hip_flexion", "knee_angle", "ankle_angle")
JOINTS = tuple(f"{base}_{side}" for side in SIDES for base in JOINT_BASES)
DISPLAY = {
    "hip_flexion": "Hip flexion",
    "knee_angle": "Knee angle",
    "ankle_angle": "Ankle angle",
}
COLORS = {
    "right": "#1f5a8a",
    "left": "#d97706",
    "reference": "#64748b",
    "100M": "#1f5a8a",
    "200M": "#d97706",
}


@dataclass
class Rollout:
    label: str
    checkpoint: Path
    checkpoint_steps: int
    time_s: np.ndarray
    phase: np.ndarray
    actual_deg: dict[str, np.ndarray]
    reference_deg: dict[str, np.ndarray]
    touch_n: dict[str, np.ndarray]
    body_weight_n: float
    joint_limits_deg: dict[str, tuple[float, float]]
    termination_reason: str | None


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
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smooth-start-seconds", type=float, default=0.5)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=session / "smooth_gait_cycle_validation_100m_200m",
    )
    return parser.parse_args()


def load_config(session_dir: Path, smooth_seconds: float) -> ImitationTrainSessionConfig:
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig,
        json.loads((session_dir / "session_config.json").read_text(encoding="utf-8")),
    )
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    config.env_params.motor_arms_pd_smooth_start_seconds = float(smooth_seconds)
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
    smooth_seconds: float,
) -> Rollout:
    config = load_config(session_dir, smooth_seconds)
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
        actual = {name: [] for name in JOINTS}
        reference = {name: [] for name in JOINTS}
        touch = {name: [] for name in ("r_foot", "r_toes", "l_foot", "l_toes")}
        times: list[float] = []
        phases: list[float] = []
        termination_reason = None

        for step in range(1, maximum_steps + 1):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(observation)):
                raise FloatingPointError(f"Non-finite rollout for {label} at step {step}")
            for name in JOINTS:
                actual[name].append(
                    float(np.rad2deg(raw.sim.data.joint(name).qpos[0]))
                )
                reference[name].append(
                    float(np.rad2deg(raw._reference_value(f"q_{name}")))
                )
            for name in touch:
                touch[name].append(float(raw.sim.data.sensor(name).data[0]))
            times.append(step / control_hz)
            phases.append(float(raw._imitation_phase))
            if terminated or truncated:
                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "terminated" if terminated else "truncated",
                    )
                )
                break

        model = raw.sim.model
        limits = {}
        for name in JOINTS:
            joint = model.joint(name)
            limits[name] = tuple(float(v) for v in np.rad2deg(model.jnt_range[joint.id]))
        return Rollout(
            label=label,
            checkpoint=checkpoint.resolve(),
            checkpoint_steps=checkpoint_steps,
            time_s=np.asarray(times, dtype=float),
            phase=np.asarray(phases, dtype=float),
            actual_deg={k: np.asarray(v, dtype=float) for k, v in actual.items()},
            reference_deg={k: np.asarray(v, dtype=float) for k, v in reference.items()},
            touch_n={k: np.asarray(v, dtype=float) for k, v in touch.items()},
            body_weight_n=float(np.sum(model.body_mass) * abs(model.opt.gravity[2])),
            joint_limits_deg=limits,
            termination_reason=termination_reason,
        )
    finally:
        env.close()


def _median_filter_3(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.median(
        np.column_stack((padded[:-2], padded[1:-1], padded[2:])), axis=1
    )


def hysteresis_contact(
    force_n: np.ndarray,
    *,
    on_threshold_n: float,
    off_threshold_n: float,
    confirmation_samples: int = 2,
) -> np.ndarray:
    """Convert force to a debounced contact mask with separate on/off thresholds."""
    force_n = _median_filter_3(np.asarray(force_n, dtype=float))
    state = False
    candidate_count = 0
    result = np.zeros(len(force_n), dtype=bool)
    for index, value in enumerate(force_n):
        requested = state
        if state and value <= off_threshold_n:
            requested = False
        elif not state and value >= on_threshold_n:
            requested = True
        if requested != state:
            candidate_count += 1
            if candidate_count >= confirmation_samples:
                state = requested
                candidate_count = 0
        else:
            candidate_count = 0
        result[index] = state
    return result


def transition_indices(mask: np.ndarray, rising: bool) -> np.ndarray:
    before = mask[:-1]
    after = mask[1:]
    transition = (~before & after) if rising else (before & ~after)
    return np.flatnonzero(transition) + 1


def enforce_minimum_separation(indices: np.ndarray, samples: int) -> np.ndarray:
    kept: list[int] = []
    for value in indices:
        if not kept or value - kept[-1] >= samples:
            kept.append(int(value))
    return np.asarray(kept, dtype=int)


def contact_events(
    rollout: Rollout,
    *,
    control_hz: int,
    on_fraction_bw: float = 0.02,
    off_fraction_bw: float = 0.01,
) -> dict:
    # 2% BW to establish contact and 1% BW to release it.  This suppresses
    # numerical grazing contacts while retaining physiologically meaningful load.
    on = on_fraction_bw * rollout.body_weight_n
    off = off_fraction_bw * rollout.body_weight_n
    result = {"threshold_on_n": on, "threshold_off_n": off, "sides": {}}
    for side in SIDES:
        heel_force = rollout.touch_n[f"{side}_foot"]
        toe_force = rollout.touch_n[f"{side}_toes"]
        heel = hysteresis_contact(
            heel_force, on_threshold_n=on, off_threshold_n=off
        )
        toe = hysteresis_contact(toe_force, on_threshold_n=on, off_threshold_n=off)
        combined = hysteresis_contact(
            heel_force + toe_force, on_threshold_n=on, off_threshold_n=off
        )
        # An ipsilateral gait cycle is first contact to the next first contact.
        strikes = enforce_minimum_separation(
            transition_indices(combined, rising=True), int(round(0.45 * control_hz))
        )
        offs = enforce_minimum_separation(
            transition_indices(combined, rising=False), int(round(0.30 * control_hz))
        )
        result["sides"][side] = {
            "heel_contact": heel,
            "toe_contact": toe,
            "foot_contact": combined,
            "heel_strikes": strikes,
            "toe_offs": offs,
        }
    return result


def complete_cycles(
    events: dict,
    *,
    side: str,
    time_s: np.ndarray,
    settle_seconds: float,
) -> list[tuple[int, int, int]]:
    side_events = events["sides"][side]
    strikes = side_events["heel_strikes"]
    offs = side_events["toe_offs"]
    cycles = []
    for start, end in zip(strikes[:-1], strikes[1:], strict=True):
        if time_s[start] < settle_seconds:
            continue
        candidates = offs[(offs > start) & (offs < end)]
        if len(candidates) == 0:
            continue
        cycles.append((int(start), int(end), int(candidates[0])))
    return cycles


def resample_cycle(values: np.ndarray, start: int, end: int, points: int = 101) -> np.ndarray:
    source_phase = np.linspace(0.0, 100.0, end - start + 1)
    target_phase = np.linspace(0.0, 100.0, points)
    return np.interp(target_phase, source_phase, values[start : end + 1])


def safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) < 1e-9 or np.std(second) < 1e-9:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def summarize_rollout(
    rollout: Rollout,
    events: dict,
    *,
    control_hz: int,
    settle_seconds: float,
) -> tuple[dict, dict]:
    cycles = {
        side: complete_cycles(
            events, side=side, time_s=rollout.time_s, settle_seconds=settle_seconds
        )
        for side in SIDES
    }
    if min(len(cycles["r"]), len(cycles["l"])) < 2:
        raise RuntimeError(
            f"{rollout.label}: insufficient complete cycles after {settle_seconds}s; "
            f"right={len(cycles['r'])}, left={len(cycles['l'])}"
        )

    toe_off = {}
    duration = {}
    for side in SIDES:
        duration_values = [rollout.time_s[end] - rollout.time_s[start] for start, end, _ in cycles[side]]
        toe_values = [
            100.0
            * (rollout.time_s[off] - rollout.time_s[start])
            / (rollout.time_s[end] - rollout.time_s[start])
            for start, end, off in cycles[side]
        ]
        duration[side] = np.asarray(duration_values, dtype=float)
        toe_off[side] = np.asarray(toe_values, dtype=float)

    # Correlations use each right-stride interval.  Zero-lag describes the
    # expected anti-phase motion; shifting the left waveform by half a cycle
    # tests bilateral shape symmetry.
    correlations: dict[str, dict[str, list[float]]] = {
        base: {
            "zero_lag": [],
            "half_cycle_shifted": [],
            "reference_half_cycle_shifted": [],
            "independent_cycle_normalized": [],
            "reference_independent_cycle_normalized": [],
        }
        for base in JOINT_BASES
    }
    representative_curves = {}
    right_cycles = cycles["r"]
    durations = np.asarray(
        [rollout.time_s[end] - rollout.time_s[start] for start, end, _ in right_cycles]
    )
    representative_index = int(np.argmin(np.abs(durations - np.median(durations))))
    representative = right_cycles[representative_index]
    phase_grid = np.linspace(0.0, 100.0, 101)
    for base in JOINT_BASES:
        right_name = f"{base}_r"
        left_name = f"{base}_l"
        for start, end, _off in right_cycles:
            right = resample_cycle(rollout.actual_deg[right_name], start, end)
            left = resample_cycle(rollout.actual_deg[left_name], start, end)
            ref_right = resample_cycle(rollout.reference_deg[right_name], start, end)
            ref_left = resample_cycle(rollout.reference_deg[left_name], start, end)
            correlations[base]["zero_lag"].append(safe_corr(right, left))
            correlations[base]["half_cycle_shifted"].append(
                safe_corr(right[:-1], np.roll(left[:-1], -50))
            )
            correlations[base]["reference_half_cycle_shifted"].append(
                safe_corr(ref_right[:-1], np.roll(ref_left[:-1], -50))
            )
        paired_cycle_count = min(len(cycles["r"]), len(cycles["l"]))
        for cycle_index in range(paired_cycle_count):
            right_start, right_end, _ = cycles["r"][cycle_index]
            left_start, left_end, _ = cycles["l"][cycle_index]
            right_independent = resample_cycle(
                rollout.actual_deg[right_name], right_start, right_end
            )
            left_independent = resample_cycle(
                rollout.actual_deg[left_name], left_start, left_end
            )
            ref_right_independent = resample_cycle(
                rollout.reference_deg[right_name], right_start, right_end
            )
            ref_left_independent = resample_cycle(
                rollout.reference_deg[left_name], left_start, left_end
            )
            correlations[base]["independent_cycle_normalized"].append(
                safe_corr(right_independent, left_independent)
            )
            correlations[base]["reference_independent_cycle_normalized"].append(
                safe_corr(ref_right_independent, ref_left_independent)
            )
        start, end, _off = representative
        representative_curves[base] = {
            "phase_percent": phase_grid,
            "right_actual_deg": resample_cycle(rollout.actual_deg[right_name], start, end),
            "left_actual_deg": resample_cycle(rollout.actual_deg[left_name], start, end),
            "right_reference_deg": resample_cycle(rollout.reference_deg[right_name], start, end),
            "left_reference_deg": resample_cycle(rollout.reference_deg[left_name], start, end),
        }

    kinematics = {}
    for name in JOINTS:
        side = name.rsplit("_", 1)[-1]
        stable_indices = np.unique(
            np.concatenate(
                [np.arange(start, end + 1, dtype=int) for start, end, _ in cycles[side]]
            )
        )
        actual_values = rollout.actual_deg[name][stable_indices]
        reference_values = rollout.reference_deg[name][stable_indices]
        errors = actual_values - reference_values
        lower, upper = rollout.joint_limits_deg[name]
        actual_min = float(np.min(actual_values))
        actual_max = float(np.max(actual_values))
        kinematics[name] = {
            "actual_min_deg": actual_min,
            "actual_max_deg": actual_max,
            "actual_rom_deg": actual_max - actual_min,
            "reference_min_deg": float(np.min(reference_values)),
            "reference_max_deg": float(np.max(reference_values)),
            "rmse_deg": float(np.sqrt(np.mean(np.square(errors)))),
            "joint_limit_deg": [lower, upper],
            "minimum_limit_margin_deg": float(min(actual_min - lower, upper - actual_max)),
        }

    summary = {
        "checkpoint": str(rollout.checkpoint),
        "checkpoint_steps": rollout.checkpoint_steps,
        "duration_s": float(rollout.time_s[-1]),
        "termination_reason": rollout.termination_reason,
        "body_weight_n": rollout.body_weight_n,
        "contact_thresholds_n": {
            "on": events["threshold_on_n"],
            "off": events["threshold_off_n"],
        },
        "complete_cycle_count": {side: len(cycles[side]) for side in SIDES},
        "cycle_duration_s": {
            side: {
                "mean": float(np.mean(duration[side])),
                "std": float(np.std(duration[side])),
            }
            for side in SIDES
        },
        "toe_off_percent": {
            side: {
                "mean": float(np.mean(toe_off[side])),
                "std": float(np.std(toe_off[side])),
                "minimum": float(np.min(toe_off[side])),
                "maximum": float(np.max(toe_off[side])),
            }
            for side in SIDES
        },
        "kinematics": kinematics,
        "bilateral_pearson": {
            base: {
                metric: {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }
                for metric, values in metrics.items()
            }
            for base, metrics in correlations.items()
        },
        "representative_right_cycle": {
            "start_time_s": float(rollout.time_s[representative[0]]),
            "end_time_s": float(rollout.time_s[representative[1]]),
            "toe_off_percent": float(
                100.0
                * (rollout.time_s[representative[2]] - rollout.time_s[representative[0]])
                / (rollout.time_s[representative[1]] - rollout.time_s[representative[0]])
            ),
        },
    }
    plot_data = {
        "cycles": cycles,
        "duration": duration,
        "toe_off": toe_off,
        "correlations": correlations,
        "representative_curves": representative_curves,
    }
    return summary, plot_data


def contact_threshold_sensitivity(
    rollout: Rollout,
    *,
    control_hz: int,
    settle_seconds: float,
) -> dict:
    result = {}
    for on_fraction, off_fraction in ((0.01, 0.005), (0.02, 0.01), (0.05, 0.025)):
        events = contact_events(
            rollout,
            control_hz=control_hz,
            on_fraction_bw=on_fraction,
            off_fraction_bw=off_fraction,
        )
        key = f"on_{100 * on_fraction:g}pct_off_{100 * off_fraction:g}pct_bw"
        result[key] = {}
        for side in SIDES:
            cycles = complete_cycles(
                events,
                side=side,
                time_s=rollout.time_s,
                settle_seconds=settle_seconds,
            )
            toe_percent = np.asarray(
                [
                    100.0
                    * (rollout.time_s[off] - rollout.time_s[start])
                    / (rollout.time_s[end] - rollout.time_s[start])
                    for start, end, off in cycles
                ],
                dtype=float,
            )
            result[key][side] = {
                "complete_cycle_count": len(cycles),
                "toe_off_percent_mean": float(np.mean(toe_percent)),
                "toe_off_percent_std": float(np.std(toe_percent)),
            }
    return result


def plot_joint_angles(plot_data: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), sharex=True, constrained_layout=True)
    for row, label in enumerate(("100M", "200M")):
        for column, base in enumerate(JOINT_BASES):
            axis = axes[row, column]
            curve = plot_data[label]["representative_curves"][base]
            phase = curve["phase_percent"]
            axis.plot(phase, curve["right_actual_deg"], color=COLORS["right"], label="Right actual")
            axis.plot(phase, curve["left_actual_deg"], color=COLORS["left"], label="Left actual")
            axis.plot(phase, curve["right_reference_deg"], color=COLORS["right"], linestyle="--", alpha=0.65, label="Right reference")
            axis.plot(phase, curve["left_reference_deg"], color=COLORS["left"], linestyle="--", alpha=0.65, label="Left reference")
            axis.set_title(f"{label} — {DISPLAY[base]}")
            axis.set_ylabel("Angle (deg)")
            axis.grid(True, alpha=0.25)
            if row == 1:
                axis.set_xlabel("Right gait cycle (%)")
            if row == 0 and column == 0:
                axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Representative complete gait cycle: actual and reference joint angles")
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def plot_toe_off(plot_data: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    for axis, label in zip(axes, ("100M", "200M"), strict=True):
        values = [plot_data[label]["toe_off"][side] for side in SIDES]
        axis.boxplot(values, tick_labels=["Right", "Left"], widths=0.55, showmeans=True)
        axis.axhspan(
            60.0,
            62.6,
            color="#94a3b8",
            alpha=0.18,
            label="Healthy young adults at 1.2 m/s (mean +/- SD)",
        )
        axis.set_title(label)
        axis.set_xlabel("Leg")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Toe-off timing (% ipsilateral gait cycle)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle("Toe-off timing from debounced foot-contact sensors")
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def plot_correlations(summaries: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(len(JOINT_BASES), dtype=float)
    width = 0.34
    for axis, label in zip(axes, ("100M", "200M"), strict=True):
        zero = [summaries[label]["bilateral_pearson"][base]["zero_lag"]["mean"] for base in JOINT_BASES]
        shifted = [summaries[label]["bilateral_pearson"][base]["half_cycle_shifted"]["mean"] for base in JOINT_BASES]
        axis.bar(x - width / 2, zero, width, label="Same-time r", color="#64748b")
        axis.bar(x + width / 2, shifted, width, label="Half-cycle-shifted r", color=COLORS[label])
        axis.axhline(0.0, color="#0f172a", linewidth=0.8)
        axis.set_xticks(x, ["Hip", "Knee", "Ankle"])
        axis.set_ylim(-1.05, 1.05)
        axis.set_title(label)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Pearson correlation coefficient r")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Bilateral joint-angle correlation within complete right gait cycles")
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def plot_independent_cycle_correlations(summaries: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(len(JOINT_BASES), dtype=float)
    width = 0.34
    for axis, label in zip(axes, ("100M", "200M"), strict=True):
        actual = [
            summaries[label]["bilateral_pearson"][base]["independent_cycle_normalized"]["mean"]
            for base in JOINT_BASES
        ]
        actual_std = [
            summaries[label]["bilateral_pearson"][base]["independent_cycle_normalized"]["std"]
            for base in JOINT_BASES
        ]
        reference = [
            summaries[label]["bilateral_pearson"][base]["reference_independent_cycle_normalized"]["mean"]
            for base in JOINT_BASES
        ]
        reference_std = [
            summaries[label]["bilateral_pearson"][base]["reference_independent_cycle_normalized"]["std"]
            for base in JOINT_BASES
        ]
        axis.bar(
            x - width / 2,
            actual,
            width,
            yerr=actual_std,
            capsize=3,
            label="Actual",
            color=COLORS[label],
        )
        axis.bar(
            x + width / 2,
            reference,
            width,
            yerr=reference_std,
            capsize=3,
            label="Reference",
            color="#64748b",
            alpha=0.82,
        )
        axis.set_xticks(x, ["Hip", "Knee", "Ankle"])
        axis.set_ylim(0.0, 1.05)
        axis.set_title(label)
        axis.grid(True, axis="y", alpha=0.25)
        for position, value in zip(x - width / 2, actual, strict=True):
            axis.text(position, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Pearson correlation coefficient r")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "Bilateral Pearson correlation after independent ipsilateral gait-cycle normalization"
    )
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def write_independent_cycle_correlations(plot_data: dict[str, dict], output: Path) -> None:
    fields = ["model", "joint", "paired_cycle_index", "actual_pearson_r", "reference_pearson_r"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in ("100M", "200M"):
            correlations = plot_data[label]["correlations"]
            for base in JOINT_BASES:
                actual = correlations[base]["independent_cycle_normalized"]
                reference = correlations[base]["reference_independent_cycle_normalized"]
                for cycle_index, (actual_r, reference_r) in enumerate(
                    zip(actual, reference, strict=True), start=1
                ):
                    writer.writerow(
                        {
                            "model": label,
                            "joint": base,
                            "paired_cycle_index": cycle_index,
                            "actual_pearson_r": float(actual_r),
                            "reference_pearson_r": float(reference_r),
                        }
                    )


def write_representative_csv(plot_data: dict[str, dict], output: Path) -> None:
    fields = ["model", "gait_cycle_percent"]
    for base in JOINT_BASES:
        fields.extend(
            [
                f"{base}_right_actual_deg",
                f"{base}_left_actual_deg",
                f"{base}_right_reference_deg",
                f"{base}_left_reference_deg",
            ]
        )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in ("100M", "200M"):
            curves = plot_data[label]["representative_curves"]
            for index in range(101):
                row = {
                    "model": label,
                    "gait_cycle_percent": float(curves[JOINT_BASES[0]]["phase_percent"][index]),
                }
                for base in JOINT_BASES:
                    for key in (
                        "right_actual_deg",
                        "left_actual_deg",
                        "right_reference_deg",
                        "left_reference_deg",
                    ):
                        row[f"{base}_{key}"] = float(curves[base][key][index])
                writer.writerow(row)


def main() -> None:
    args = parse_args()
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
            smooth_seconds=args.smooth_start_seconds,
        ),
        rollout_checkpoint(
            label="200M",
            checkpoint=args.checkpoint_200m,
            checkpoint_steps=200_007_680,
            session_dir=args.session_dir,
            seconds=args.seconds,
            target_velocity=args.target_velocity,
            seed=args.seed,
            smooth_seconds=args.smooth_start_seconds,
        ),
    ]
    summaries = {}
    plot_data = {}
    for rollout in rollouts:
        events = contact_events(rollout, control_hz=50)
        summaries[rollout.label], plot_data[rollout.label] = summarize_rollout(
            rollout,
            events,
            control_hz=50,
            settle_seconds=args.settle_seconds,
        )
        summaries[rollout.label]["contact_threshold_sensitivity"] = (
            contact_threshold_sensitivity(
                rollout,
                control_hz=50,
                settle_seconds=args.settle_seconds,
            )
        )

    result = {
        "method": {
            "target_velocity_m_per_s": args.target_velocity,
            "control_rate_hz": 50,
            "physics_rate_hz": 2000,
            "deterministic_policy": True,
            "reference_start_index": 0,
            "seed": args.seed,
            "motor_arms_pd_smooth_start_seconds": args.smooth_start_seconds,
            "startup_excluded_before_s": args.settle_seconds,
            "gait_cycle_definition": "ipsilateral first foot contact to next ipsilateral first foot contact",
            "toe_off_definition": "debounced transition from ipsilateral foot contact to no contact",
            "contact_threshold_definition": "on at 2% body weight; off at 1% body weight; 2-sample confirmation",
            "pearson_definition": {
                "zero_lag": "left and right joint angles at the same right-gait-cycle phase",
                "half_cycle_shifted": "left waveform circularly advanced by 50% before correlation",
                "independent_cycle_normalized": (
                    "right and left waveforms each normalized from their own ipsilateral "
                    "initial contact to the next ipsilateral initial contact; no phase shift"
                ),
            },
        },
        "models": summaries,
    }
    summary_path = args.output_dir / "gait_cycle_validation_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_representative_csv(
        plot_data, args.output_dir / "representative_gait_cycles.csv"
    )
    plot_joint_angles(plot_data, args.output_dir / "joint_angles_one_cycle.png")
    plot_toe_off(plot_data, args.output_dir / "toe_off_timing.png")
    plot_correlations(summaries, args.output_dir / "bilateral_pearson.png")
    plot_independent_cycle_correlations(
        summaries, args.output_dir / "bilateral_pearson_independent_cycles.png"
    )
    write_independent_cycle_correlations(
        plot_data, args.output_dir / "bilateral_pearson_independent_cycles.csv"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
