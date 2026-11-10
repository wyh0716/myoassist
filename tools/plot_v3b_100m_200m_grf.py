#!/usr/bin/env python3
"""Plot simulated and reference-estimated total vertical GRF for V3-B.

The actual curves sum all terrain contact forces at the end of each 50 Hz
control interval.  The reference curve is an estimate from the XML model's
whole-body COM acceleration: Fz = m * (zddot - gravity_z).  It is not measured
force-plate data and does not attempt to split load between the two feet.
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
import mujoco
import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


COLORS = {"100M": "#1f5a8a", "200M": "#d97706", "Reference estimate": "#4b5563"}


@dataclass
class GrfRollout:
    label: str
    checkpoint: Path
    checkpoint_steps: int
    time_s: np.ndarray
    phase: np.ndarray
    raw_fz_n: np.ndarray
    filtered_fz_n: np.ndarray
    body_weight_n: float
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
    parser.add_argument("--lowpass-hz", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=session / "grf_comparison_100m_200m",
    )
    return parser.parse_args()


def load_config(session_dir: Path) -> ImitationTrainSessionConfig:
    config = DictionableDataclass.create(
        ImitationTrainSessionConfig,
        json.loads((session_dir / "session_config.json").read_text(encoding="utf-8")),
    )
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    return config


def lowpass(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    if not 0.0 < cutoff_hz < sample_rate_hz / 2.0:
        raise ValueError("Low-pass cutoff must lie between zero and Nyquist")
    sos = butter(4, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    return np.asarray(sosfiltfilt(sos, values), dtype=float)


def total_vertical_grf(raw) -> float:
    """Sum terrain contact forces on the articulated model in world z."""
    model = raw.sim.model
    data = raw.sim.data
    terrain_id = int(model.geom("terrain").id)
    total_world = np.zeros(3, dtype=float)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if terrain_id not in (geom1, geom2):
            continue
        contact_wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(
            model.ptr, data.ptr, contact_id, contact_wrench
        )
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_world_on_geom2 = frame.T @ contact_wrench[:3]
        force_world_on_body = (
            -force_world_on_geom2 if geom2 == terrain_id else force_world_on_geom2
        )
        total_world += force_world_on_body
    return float(total_world[2])


def rollout_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    checkpoint_steps: int,
    session_dir: Path,
    seconds: float,
    target_velocity: float,
    seed: int,
    cutoff_hz: float,
) -> tuple[GrfRollout, object, ImitationTrainSessionConfig]:
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
        times, phases, forces = [], [], []
        termination_reason = None
        for step in range(1, maximum_steps + 1):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(observation)):
                raise FloatingPointError(f"Non-finite rollout for {label} at step {step}")
            times.append(step / control_hz)
            phases.append(float(raw._imitation_phase))
            forces.append(total_vertical_grf(raw))
            if terminated or truncated:
                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "terminated" if terminated else "truncated",
                    )
                )
                break

        raw_fz = np.asarray(forces, dtype=float)
        filtered_fz = lowpass(raw_fz, control_hz, cutoff_hz)
        mass_kg = float(np.sum(np.asarray(raw.sim.model.body_mass, dtype=float)))
        gravity_z = float(raw.sim.model.opt.gravity[2])
        body_weight_n = mass_kg * abs(gravity_z)
        rollout = GrfRollout(
            label=label,
            checkpoint=checkpoint.resolve(),
            checkpoint_steps=checkpoint_steps,
            time_s=np.asarray(times, dtype=float),
            phase=np.asarray(phases, dtype=float),
            raw_fz_n=raw_fz,
            filtered_fz_n=filtered_fz,
            body_weight_n=body_weight_n,
            termination_reason=termination_reason,
        )

        # Retain only immutable data needed by the reference estimator; the
        # environment itself is closed below.
        reference_context = {
            "native_model": raw.sim.model.ptr,
            "init_qpos": np.asarray(raw.init_qpos, dtype=float).copy(),
            "reference_data_keys": tuple(raw.reference_data_keys),
            "series_data": {
                key: np.asarray(value, dtype=float).copy()
                for key, value in raw._reference_data["series_data"].items()
            },
        }
        return rollout, reference_context, config
    finally:
        env.close()


def interpolate_reference(series: np.ndarray, phase: float) -> float:
    lower = int(np.floor(phase))
    upper = min(lower + 1, len(series) - 1)
    fraction = float(phase - lower)
    return float((1.0 - fraction) * series[lower] + fraction * series[upper])


def synchronize_joint_equalities(model, data) -> None:
    for equality_id in range(model.neq):
        if int(model.eq_type[equality_id]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        slave_id = int(model.eq_obj1id[equality_id])
        master_id = int(model.eq_obj2id[equality_id])
        if slave_id < 0 or master_id < 0:
            continue
        slave_qadr = int(model.jnt_qposadr[slave_id])
        master_qadr = int(model.jnt_qposadr[master_id])
        coefficients = np.asarray(model.eq_data[equality_id, :5], dtype=float)
        master_delta = float(data.qpos[master_qadr] - model.qpos0[master_qadr])
        data.qpos[slave_qadr] = model.qpos0[slave_qadr] + np.polynomial.polynomial.polyval(
            master_delta, coefficients
        )


def estimate_reference_grf(
    phases: np.ndarray,
    time_s: np.ndarray,
    reference_context: dict,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    model = reference_context["native_model"]
    data = mujoco.MjData(model)
    init_qpos = reference_context["init_qpos"]
    series = reference_context["series_data"]
    z_com = []
    for phase in phases:
        data.qpos[:] = init_qpos
        data.qvel[:] = 0.0
        for joint_name in reference_context["reference_data_keys"]:
            joint_id = int(model.joint(joint_name).id)
            qpos_address = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_address] = interpolate_reference(
                series[f"q_{joint_name}"], phase
            )
        synchronize_joint_equalities(model, data)
        mujoco.mj_forward(model, data)
        body_masses = np.asarray(model.body_mass[1:], dtype=float)
        body_coms = np.asarray(data.xipos[1:], dtype=float)
        whole_body_com = np.sum(body_coms * body_masses[:, None], axis=0) / np.sum(
            body_masses
        )
        z_com.append(float(whole_body_com[2]))

    z_com_array = np.asarray(z_com, dtype=float)
    dt = float(np.median(np.diff(time_s)))
    control_hz = 1.0 / dt
    window = min(11, len(z_com_array) if len(z_com_array) % 2 else len(z_com_array) - 1)
    if window < 7:
        raise ValueError("Reference rollout is too short for stable COM differentiation")
    z_acceleration = savgol_filter(
        z_com_array,
        window_length=window,
        polyorder=3,
        deriv=2,
        delta=dt,
        mode="interp",
    )
    total_mass_kg = float(np.sum(np.asarray(model.body_mass, dtype=float)))
    gravity_z = float(model.opt.gravity[2])
    estimated_fz_n = total_mass_kg * (z_acceleration - gravity_z)
    estimated_fz_filtered = lowpass(estimated_fz_n, control_hz, cutoff_hz)
    return estimated_fz_n, estimated_fz_filtered, total_mass_kg, window // 2


def style_axis(axis: plt.Axes, body_weight_n: float) -> None:
    axis.axhline(
        body_weight_n,
        color="#64748b",
        linestyle=":",
        linewidth=1.0,
        label="1 body weight",
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Total vertical GRF (N)")
    axis.grid(True, color="#cbd5e1", alpha=0.5, linewidth=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    secondary = axis.secondary_yaxis(
        "right",
        functions=(lambda value: value / body_weight_n, lambda value: value * body_weight_n),
    )
    secondary.set_ylabel("Body weight (BW)")


def plot_actual(rollout: GrfRollout, path: Path, cutoff_hz: float) -> None:
    color = COLORS[rollout.label]
    figure, axis = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
    axis.plot(
        rollout.time_s,
        rollout.raw_fz_n,
        color=color,
        alpha=0.20,
        linewidth=0.8,
        label="Control-rate contact samples",
    )
    axis.plot(
        rollout.time_s,
        rollout.filtered_fz_n,
        color=color,
        linewidth=1.5,
        label=f"{cutoff_hz:g} Hz low-pass",
    )
    style_axis(axis, rollout.body_weight_n)
    axis.set_title(
        f"{rollout.label} checkpoint: simulated total vertical GRF\n"
        "Sum of all terrain contact forces; 50 Hz control-rate sampling"
    )
    axis.legend(loc="upper right", frameon=False, ncol=3)
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def plot_reference(
    time_s: np.ndarray,
    raw_reference_n: np.ndarray,
    filtered_reference_n: np.ndarray,
    body_weight_n: float,
    path: Path,
    cutoff_hz: float,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
    color = COLORS["Reference estimate"]
    axis.plot(
        time_s,
        raw_reference_n,
        color=color,
        alpha=0.22,
        linewidth=0.8,
        label="COM-acceleration estimate",
    )
    axis.plot(
        time_s,
        filtered_reference_n,
        color=color,
        linewidth=1.5,
        label=f"{cutoff_hz:g} Hz low-pass",
    )
    style_axis(axis, body_weight_n)
    axis.set_title(
        "Reference trajectory: model-estimated total vertical GRF\n"
        "Fz = mass × (vertical COM acceleration + gravity); not force-plate data"
    )
    axis.legend(loc="upper right", frameon=False, ncol=3)
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def plot_overlay(
    rollouts: list[GrfRollout],
    reference_n: np.ndarray,
    path: Path,
) -> None:
    body_weight_n = rollouts[0].body_weight_n
    figure, axis = plt.subplots(figsize=(14, 5.6), constrained_layout=True)
    for rollout in rollouts:
        axis.plot(
            rollout.time_s,
            rollout.filtered_fz_n,
            color=COLORS[rollout.label],
            linewidth=1.35,
            label=f"{rollout.label} simulated",
        )
    axis.plot(
        rollouts[0].time_s,
        reference_n,
        color=COLORS["Reference estimate"],
        linestyle="--",
        linewidth=1.45,
        label="Reference estimate",
    )
    style_axis(axis, body_weight_n)
    axis.set_title(
        "Total vertical GRF comparison: 100M vs 200M vs reference estimate\n"
        "All curves use the same start index, seed and 1.25 m/s target"
    )
    axis.legend(loc="upper right", frameon=False, ncol=4)
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

    rollout_100m, reference_context, config = rollout_checkpoint(
        label="100M",
        checkpoint=args.checkpoint_100m,
        checkpoint_steps=100_016_128,
        session_dir=args.session_dir,
        seconds=args.seconds,
        target_velocity=args.target_velocity,
        seed=args.seed,
        cutoff_hz=args.lowpass_hz,
    )
    rollout_200m, _unused_context, _unused_config = rollout_checkpoint(
        label="200M",
        checkpoint=args.checkpoint_200m,
        checkpoint_steps=200_007_680,
        session_dir=args.session_dir,
        seconds=args.seconds,
        target_velocity=args.target_velocity,
        seed=args.seed,
        cutoff_hz=args.lowpass_hz,
    )
    rollouts = [rollout_100m, rollout_200m]
    if not np.allclose(rollout_100m.time_s, rollout_200m.time_s):
        raise RuntimeError("The two checkpoints completed different evaluation horizons")
    if not np.allclose(rollout_100m.phase, rollout_200m.phase):
        raise RuntimeError("The two checkpoints did not use the same reference phases")
    if not np.isclose(rollout_100m.body_weight_n, rollout_200m.body_weight_n):
        raise RuntimeError("The two checkpoints used different model body weights")

    reference_raw, reference_filtered, mass_kg, edge_samples = estimate_reference_grf(
        rollout_100m.phase,
        rollout_100m.time_s,
        reference_context,
        args.lowpass_hz,
    )
    body_weight_n = rollout_100m.body_weight_n
    if not np.isclose(body_weight_n, mass_kg * 9.81, rtol=0.01):
        raise RuntimeError("Reference estimator and rollout model masses disagree")
    # Savitzky-Golay differentiation is not supported by measurements outside
    # the finite record.  Keep the calculated edge values in the CSV for full
    # reproducibility, but omit them from figures and error metrics rather than
    # displaying extrapolation artifacts as physical GRF.
    reference_raw_for_plot = reference_raw.copy()
    reference_filtered_for_plot = reference_filtered.copy()
    reference_raw_for_plot[:edge_samples] = np.nan
    reference_raw_for_plot[-edge_samples:] = np.nan
    reference_filtered_for_plot[:edge_samples] = np.nan
    reference_filtered_for_plot[-edge_samples:] = np.nan

    paths = {
        "100m": args.output_dir / "grf_100m.png",
        "200m": args.output_dir / "grf_200m.png",
        "reference": args.output_dir / "grf_reference_estimate.png",
        "overlay": args.output_dir / "grf_overlay_100m_200m_reference.png",
        "csv": args.output_dir / "grf_timeseries_100m_200m_reference.csv",
        "summary": args.output_dir / "grf_summary_100m_200m_reference.json",
    }
    plot_actual(rollout_100m, paths["100m"], args.lowpass_hz)
    plot_actual(rollout_200m, paths["200m"], args.lowpass_hz)
    plot_reference(
        rollout_100m.time_s,
        reference_raw_for_plot,
        reference_filtered_for_plot,
        body_weight_n,
        paths["reference"],
        args.lowpass_hz,
    )
    plot_overlay(rollouts, reference_filtered_for_plot, paths["overlay"])

    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "time_s",
            "reference_phase",
            "grf_100m_raw_n",
            "grf_100m_filtered_n",
            "grf_200m_raw_n",
            "grf_200m_filtered_n",
            "grf_reference_raw_n",
            "grf_reference_filtered_n",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, time_s in enumerate(rollout_100m.time_s):
            writer.writerow(
                {
                    "time_s": float(time_s),
                    "reference_phase": float(rollout_100m.phase[index]),
                    "grf_100m_raw_n": float(rollout_100m.raw_fz_n[index]),
                    "grf_100m_filtered_n": float(rollout_100m.filtered_fz_n[index]),
                    "grf_200m_raw_n": float(rollout_200m.raw_fz_n[index]),
                    "grf_200m_filtered_n": float(rollout_200m.filtered_fz_n[index]),
                    "grf_reference_raw_n": float(reference_raw[index]),
                    "grf_reference_filtered_n": float(reference_filtered[index]),
                }
            )

    valid = slice(edge_samples, -edge_samples)
    summary_models = {}
    for rollout in rollouts:
        error = rollout.filtered_fz_n[valid] - reference_filtered[valid]
        summary_models[rollout.label] = {
            "checkpoint": str(rollout.checkpoint),
            "checkpoint_steps": rollout.checkpoint_steps,
            "duration_s": float(rollout.time_s[-1]),
            "termination_reason": rollout.termination_reason,
            "mean_filtered_grf_n": float(np.mean(rollout.filtered_fz_n[valid])),
            "mean_filtered_grf_bw": float(
                np.mean(rollout.filtered_fz_n[valid]) / body_weight_n
            ),
            "peak_raw_grf_n": float(np.max(rollout.raw_fz_n[valid])),
            "peak_raw_grf_bw": float(np.max(rollout.raw_fz_n[valid]) / body_weight_n),
            "peak_filtered_grf_n": float(np.max(rollout.filtered_fz_n[valid])),
            "peak_filtered_grf_bw": float(
                np.max(rollout.filtered_fz_n[valid]) / body_weight_n
            ),
            "rmse_vs_reference_estimate_n": float(np.sqrt(np.mean(np.square(error)))),
            "rmse_vs_reference_estimate_bw": float(
                np.sqrt(np.mean(np.square(error))) / body_weight_n
            ),
            "mae_vs_reference_estimate_n": float(np.mean(np.abs(error))),
        }
    summary = {
        "method": {
            "deterministic": True,
            "seed": args.seed,
            "reference_start_index": 0,
            "target_velocity_m_per_s": args.target_velocity,
            "control_sample_rate_hz": int(config.env_params.control_framerate),
            "physics_rate_hz": int(config.env_params.physics_sim_framerate),
            "lowpass_cutoff_hz": args.lowpass_hz,
            "model_mass_kg": mass_kg,
            "body_weight_n": body_weight_n,
            "reference_method": "whole-body COM vertical acceleration",
            "reference_is_estimated_not_measured": True,
            "rmse_edge_samples_excluded_each_side": edge_samples,
        },
        "models": summary_models,
        "reference": {
            "mean_filtered_grf_n": float(np.mean(reference_filtered[valid])),
            "mean_filtered_grf_bw": float(np.mean(reference_filtered[valid]) / body_weight_n),
            "peak_filtered_grf_n": float(np.max(reference_filtered[valid])),
            "peak_filtered_grf_bw": float(np.max(reference_filtered[valid]) / body_weight_n),
        },
        "outputs": {key: str(value.resolve()) for key, value in paths.items() if key != "summary"},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
