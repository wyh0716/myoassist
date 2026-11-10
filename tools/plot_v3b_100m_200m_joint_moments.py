#!/usr/bin/env python3
"""Plot V3-B lower-limb joint moments for 100M, 200M and the reference.

Simulation figures use MuJoCo generalized forces at each lower-limb hinge:

* active muscle-driven moment = qfrc_actuator
* net internal joint moment = qfrc_actuator + qfrc_passive

The reference figure is an inverse-dynamics estimate, not experimental ground
truth.  The NPZ contains kinematics but no force-plate data.  Total ground
reaction force is therefore inferred from the floating-base inverse dynamics,
and distributed over low foot/toe contact sites while matching the required
pelvis-pitch moment when the available support points permit it.
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
from scipy.optimize import minimize
from scipy.signal import butter, savgol_filter, sosfiltfilt

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


JOINTS = (
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
)
DISPLAY = {
    "hip_flexion_r": "Right hip flexion",
    "hip_flexion_l": "Left hip flexion",
    "knee_angle_r": "Right knee",
    "knee_angle_l": "Left knee",
    "ankle_angle_r": "Right ankle",
    "ankle_angle_l": "Left ankle",
}
REFERENCE_COLORS = {
    "hip_flexion_r": "#1f5a8a",
    "hip_flexion_l": "#d97706",
    "knee_angle_r": "#1f5a8a",
    "knee_angle_l": "#d97706",
    "ankle_angle_r": "#1f5a8a",
    "ankle_angle_l": "#d97706",
}
SITE_NAMES = ("r_foot_touch", "r_toes_touch", "l_foot_touch", "l_toes_touch")
SITE_BODIES = ("calcn_r", "toes_r", "calcn_l", "toes_l")


@dataclass
class MomentRollout:
    label: str
    checkpoint: Path
    checkpoint_steps: int
    time_s: np.ndarray
    phase: np.ndarray
    active_nm: dict[str, np.ndarray]
    passive_nm: dict[str, np.ndarray]
    net_nm: dict[str, np.ndarray]
    filtered_active_nm: dict[str, np.ndarray]
    filtered_passive_nm: dict[str, np.ndarray]
    filtered_net_nm: dict[str, np.ndarray]
    termination_reason: str | None


@dataclass
class ReferenceEstimate:
    time_s: np.ndarray
    phase: np.ndarray
    net_nm: dict[str, np.ndarray]
    filtered_net_nm: dict[str, np.ndarray]
    edge_samples: int
    cop_moment_clipped_fraction: float
    maximum_pelvis_residual_nm: float


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
        default=session / "joint_moments_100m_200m_reference",
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
    sos = butter(4, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    return np.asarray(sosfiltfilt(sos, values), dtype=float)


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
) -> tuple[MomentRollout, dict]:
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
        time_s: list[float] = []
        phase: list[float] = []
        active = {name: [] for name in JOINTS}
        passive = {name: [] for name in JOINTS}
        termination_reason = None

        for step in range(1, maximum_steps + 1):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(observation)):
                raise FloatingPointError(f"Non-finite rollout for {label} at step {step}")

            time_s.append(step / control_hz)
            phase.append(float(raw._imitation_phase))
            for name in JOINTS:
                joint_id = int(raw.sim.model.joint(name).id)
                dof_address = int(raw.sim.model.jnt_dofadr[joint_id])
                active[name].append(float(raw.sim.data.qfrc_actuator[dof_address]))
                passive[name].append(float(raw.sim.data.qfrc_passive[dof_address]))

            if terminated or truncated:
                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "terminated" if terminated else "truncated",
                    )
                )
                break

        active_arrays = {name: np.asarray(values) for name, values in active.items()}
        passive_arrays = {name: np.asarray(values) for name, values in passive.items()}
        net_arrays = {
            name: active_arrays[name] + passive_arrays[name] for name in JOINTS
        }
        context = {
            "init_qpos": np.asarray(raw.init_qpos, dtype=float).copy(),
            "reference_data_keys": tuple(raw.reference_data_keys),
            "series_data": {
                key: np.asarray(value, dtype=float).copy()
                for key, value in raw._reference_data["series_data"].items()
            },
            "model_path": Path(config.env_params.model_path).resolve(),
        }
        return (
            MomentRollout(
                label=label,
                checkpoint=checkpoint.resolve(),
                checkpoint_steps=checkpoint_steps,
                time_s=np.asarray(time_s),
                phase=np.asarray(phase),
                active_nm=active_arrays,
                passive_nm=passive_arrays,
                net_nm=net_arrays,
                filtered_active_nm={
                    name: lowpass(active_arrays[name], control_hz, cutoff_hz)
                    for name in JOINTS
                },
                filtered_passive_nm={
                    name: lowpass(passive_arrays[name], control_hz, cutoff_hz)
                    for name in JOINTS
                },
                filtered_net_nm={
                    name: lowpass(net_arrays[name], control_hz, cutoff_hz)
                    for name in JOINTS
                },
                termination_reason=termination_reason,
            ),
            context,
        )
    finally:
        env.close()


def _contact_weights(
    moment_coefficients: np.ndarray,
    target_moment: float,
    site_heights: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Find nonnegative force shares with support-site and pitch constraints."""
    active = site_heights < 0.045
    if not np.any(active):
        active[np.argmin(site_heights)] = True

    height_scores = 1.0 / (
        1.0 + np.exp(np.clip((site_heights - 0.025) / 0.004, -60.0, 60.0))
    )
    height_scores[~active] = 0.0
    if float(np.sum(height_scores)) <= 1e-12:
        height_scores[np.argmin(site_heights)] = 1.0
    base = height_scores / np.sum(height_scores)

    active_moments = moment_coefficients[active]
    clipped_target = float(
        np.clip(target_moment, np.min(active_moments), np.max(active_moments))
    )
    was_clipped = not np.isclose(clipped_target, target_moment, atol=1e-7)
    constraints = (
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {
            "type": "eq",
            "fun": lambda weights: float(
                np.dot(moment_coefficients, weights) - clipped_target
            ),
        },
    )
    bounds = [(0.0, 1.0) if enabled else (0.0, 0.0) for enabled in active]
    result = minimize(
        lambda weights: float(np.sum((weights - base) ** 2)),
        base,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 100},
    )
    if result.success:
        weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        weights /= np.sum(weights)
        return weights, was_clipped

    # A stable fallback for the degenerate single-support / equal-moment case.
    closest = np.flatnonzero(active)[
        int(np.argmin(np.abs(active_moments - clipped_target)))
    ]
    weights = np.zeros_like(base)
    weights[closest] = 1.0
    return weights, True


def estimate_reference_moments(
    *,
    time_s: np.ndarray,
    phase: np.ndarray,
    context: dict,
    cutoff_hz: float,
) -> ReferenceEstimate:
    model = mujoco.MjModel.from_xml_path(str(context["model_path"]))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    series = context["series_data"]
    init_qpos = context["init_qpos"]

    hip_center = {}
    hip_scale = {}
    for side in ("r", "l"):
        hip = np.asarray(series[f"q_hip_flexion_{side}"], dtype=float)
        hip_center[side] = float(np.mean(hip))
        hip_scale[side] = float(np.quantile(np.abs(hip - hip_center[side]), 0.99))

    qpos_rows = []
    site_rows = []
    for current_phase in phase:
        data = mujoco.MjData(model)
        data.qpos[:] = init_qpos
        for name in context["reference_data_keys"]:
            joint_id = int(model.joint(name).id)
            qpos_address = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_address] = interpolate_reference(
                series[f"q_{name}"], current_phase
            )

        for shoulder_side, opposite_hip_side in (("r", "l"), ("l", "r")):
            hip = interpolate_reference(
                series[f"q_hip_flexion_{opposite_hip_side}"], current_phase
            )
            normalized = np.clip(
                (hip - hip_center[opposite_hip_side]) / hip_scale[opposite_hip_side],
                -1.0,
                1.0,
            )
            shoulder_id = int(model.joint(f"motor_shoulder_{shoulder_side}").id)
            elbow_id = int(model.joint(f"motor_elbow_{shoulder_side}").id)
            data.qpos[model.jnt_qposadr[shoulder_id]] = np.deg2rad(10.0) * normalized
            data.qpos[model.jnt_qposadr[elbow_id]] = np.deg2rad(20.0)

        synchronize_joint_equalities(model, data)
        mujoco.mj_forward(model, data)
        qpos_rows.append(np.asarray(data.qpos, dtype=float).copy())
        site_rows.append(
            [
                np.asarray(data.site_xpos[model.site(name).id], dtype=float).copy()
                for name in SITE_NAMES
            ]
        )

    qpos = np.asarray(qpos_rows)
    site_positions = np.asarray(site_rows)
    dt = float(np.median(np.diff(time_s)))
    control_hz = 1.0 / dt
    window = min(11, len(qpos) if len(qpos) % 2 else len(qpos) - 1)
    if window < 7:
        raise ValueError("Reference sequence is too short for inverse dynamics")
    qvel = savgol_filter(
        qpos, window, 3, deriv=1, delta=dt, axis=0, mode="interp"
    )
    qacc = savgol_filter(
        qpos, window, 3, deriv=2, delta=dt, axis=0, mode="interp"
    )

    net = {name: [] for name in JOINTS}
    clipped_count = 0
    maximum_pelvis_residual = 0.0
    for row in range(len(time_s)):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[row]
        data.qvel[:] = qvel[row]
        data.qacc[:] = qacc[row]
        mujoco.mj_inverse(model, data)
        required = np.asarray(data.qfrc_inverse, dtype=float).copy()
        passive = np.asarray(data.qfrc_passive, dtype=float).copy()

        # Floating-base inverse dynamics directly supplies the total planar
        # external force and pitch moment required by the reference motion.
        total_force = np.array([required[0], 0.0, required[1]], dtype=float)
        generalized_per_site = []
        moment_coefficients = []
        for site_index, body_name in enumerate(SITE_BODIES):
            generalized = np.zeros(model.nv, dtype=float)
            mujoco.mj_applyFT(
                model,
                data,
                total_force,
                np.zeros(3),
                site_positions[row, site_index],
                int(model.body(body_name).id),
                generalized,
            )
            generalized_per_site.append(generalized)
            moment_coefficients.append(float(generalized[2]))

        weights, was_clipped = _contact_weights(
            np.asarray(moment_coefficients),
            float(required[2]),
            site_positions[row, :, 2],
        )
        clipped_count += int(was_clipped)
        external = np.sum(
            np.asarray(generalized_per_site) * weights[:, None], axis=0
        )
        internal_net = required - external + passive
        maximum_pelvis_residual = max(
            maximum_pelvis_residual,
            float(np.max(np.abs((required - external)[:3]))),
        )
        for name in JOINTS:
            joint_id = int(model.joint(name).id)
            dof_address = int(model.jnt_dofadr[joint_id])
            net[name].append(float(internal_net[dof_address]))

    net_arrays = {name: np.asarray(values) for name, values in net.items()}
    return ReferenceEstimate(
        time_s=time_s.copy(),
        phase=phase.copy(),
        net_nm=net_arrays,
        filtered_net_nm={
            name: lowpass(net_arrays[name], control_hz, cutoff_hz) for name in JOINTS
        },
        edge_samples=window // 2,
        cop_moment_clipped_fraction=clipped_count / len(time_s),
        maximum_pelvis_residual_nm=maximum_pelvis_residual,
    )


def style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.axhline(0.0, color="#64748b", linewidth=0.8)
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#cbd5e1", alpha=0.5, linewidth=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_simulation_joint(
    rollout: MomentRollout,
    joint: str,
    path: Path,
    cutoff_hz: float,
    y_limits: tuple[float, float],
) -> None:
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.0)},
        constrained_layout=True,
    )
    top.plot(
        rollout.time_s,
        rollout.filtered_active_nm[joint],
        color="#1f5a8a",
        linewidth=1.4,
        label="Active muscle-driven moment (qfrc_actuator)",
    )
    top.plot(
        rollout.time_s,
        rollout.filtered_net_nm[joint],
        color="#d97706",
        linestyle="--",
        linewidth=1.25,
        label="Net internal joint moment (active + passive)",
    )
    top.fill_between(
        rollout.time_s,
        rollout.filtered_active_nm[joint],
        rollout.filtered_net_nm[joint],
        color="#d97706",
        alpha=0.18,
        label="Passive contribution",
    )
    top.set_ylim(*y_limits)
    style_axis(top, "Moment (N·m)")
    top.legend(loc="upper right", frameon=False)
    top.set_title(
        f"{rollout.label} checkpoint — {DISPLAY[joint]} moment\n"
        f"Deterministic 10 s rollout; {cutoff_hz:g} Hz low-pass; "
        "positive direction follows the XML joint axis"
    )

    bottom.plot(
        rollout.time_s,
        rollout.filtered_passive_nm[joint],
        color="#7c3aed",
        linewidth=1.2,
    )
    style_axis(bottom, "Passive (N·m)")
    bottom.set_xlabel("Time (s)")
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def plot_reference(reference: ReferenceEstimate, path: Path, cutoff_hz: float) -> None:
    figure, axes = plt.subplots(
        3, 2, figsize=(15.5, 11.5), sharex=True, constrained_layout=True
    )
    edge = reference.edge_samples
    valid = slice(edge, len(reference.time_s) - edge)
    for axis, joint in zip(axes.flat, JOINTS):
        side = "Right" if joint.endswith("_r") else "Left"
        axis.plot(
            reference.time_s[valid],
            reference.filtered_net_nm[joint][valid],
            color=REFERENCE_COLORS[joint],
            linewidth=1.35,
            label=f"{side} estimated net moment",
        )
        style_axis(axis, "Moment (N·m)")
        axis.set_title(DISPLAY[joint])
        axis.legend(loc="upper right", frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        "Reference trajectory: estimated inverse-dynamics net joint moments\n"
        f"Kinematics + model-inferred GRF/contact distribution; {cutoff_hz:g} Hz low-pass; "
        "not measured force-plate ground truth",
        fontsize=14,
    )
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def write_csv(
    path: Path,
    rollouts: list[MomentRollout],
    reference: ReferenceEstimate,
) -> None:
    count = min(len(reference.time_s), *(len(item.time_s) for item in rollouts))
    fieldnames = ["time_s", "reference_phase"]
    for joint in JOINTS:
        fieldnames.extend(
            [
                f"reference_estimated_net_{joint}_nm",
                f"reference_estimated_net_{joint}_filtered_nm",
            ]
        )
        for rollout in rollouts:
            stem = rollout.label.lower()
            fieldnames.extend(
                [
                    f"{stem}_active_{joint}_nm",
                    f"{stem}_passive_{joint}_nm",
                    f"{stem}_net_{joint}_nm",
                    f"{stem}_active_{joint}_filtered_nm",
                    f"{stem}_net_{joint}_filtered_nm",
                ]
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(count):
            record: dict[str, float] = {
                "time_s": float(reference.time_s[row]),
                "reference_phase": float(reference.phase[row]),
            }
            for joint in JOINTS:
                record[f"reference_estimated_net_{joint}_nm"] = float(
                    reference.net_nm[joint][row]
                )
                record[f"reference_estimated_net_{joint}_filtered_nm"] = float(
                    reference.filtered_net_nm[joint][row]
                )
                for rollout in rollouts:
                    stem = rollout.label.lower()
                    record[f"{stem}_active_{joint}_nm"] = float(
                        rollout.active_nm[joint][row]
                    )
                    record[f"{stem}_passive_{joint}_nm"] = float(
                        rollout.passive_nm[joint][row]
                    )
                    record[f"{stem}_net_{joint}_nm"] = float(
                        rollout.net_nm[joint][row]
                    )
                    record[f"{stem}_active_{joint}_filtered_nm"] = float(
                        rollout.filtered_active_nm[joint][row]
                    )
                    record[f"{stem}_net_{joint}_filtered_nm"] = float(
                        rollout.filtered_net_nm[joint][row]
                    )
            writer.writerow(record)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


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

    rollout_100m, context = rollout_checkpoint(
        label="100M",
        checkpoint=args.checkpoint_100m,
        checkpoint_steps=100_016_128,
        session_dir=args.session_dir,
        seconds=args.seconds,
        target_velocity=args.target_velocity,
        seed=args.seed,
        cutoff_hz=args.lowpass_hz,
    )
    rollout_200m, _unused_context = rollout_checkpoint(
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
        raise RuntimeError("100M and 200M rollouts do not share the same time grid")
    if not np.allclose(rollout_100m.phase, rollout_200m.phase):
        raise RuntimeError("100M and 200M rollouts do not share the same reference phase")

    reference = estimate_reference_moments(
        time_s=rollout_100m.time_s,
        phase=rollout_100m.phase,
        context=context,
        cutoff_hz=args.lowpass_hz,
    )
    plot_reference(
        reference,
        args.output_dir / "reference_estimated_inverse_dynamics_net_joint_moments.png",
        args.lowpass_hz,
    )

    # Keep the corresponding 100M and 200M joint figures on the same vertical
    # scale so visual magnitude comparisons are not distorted.
    for joint in JOINTS:
        all_values = np.concatenate(
            [
                rollout.filtered_active_nm[joint]
                for rollout in rollouts
            ]
            + [rollout.filtered_net_nm[joint] for rollout in rollouts]
        )
        margin = max(5.0, 0.08 * float(np.ptp(all_values)))
        limits = (float(np.min(all_values) - margin), float(np.max(all_values) + margin))
        for rollout in rollouts:
            plot_simulation_joint(
                rollout,
                joint,
                args.output_dir / f"{rollout.label.lower()}_{joint}_moments.png",
                args.lowpass_hz,
                limits,
            )

    write_csv(
        args.output_dir / "joint_moment_timeseries_100m_200m_reference.csv",
        rollouts,
        reference,
    )
    edge = reference.edge_samples
    valid = slice(edge, len(reference.time_s) - edge)
    summary = {
        "method": {
            "simulation_active": "qfrc_actuator at each lower-limb joint DOF",
            "simulation_net": "qfrc_actuator + qfrc_passive",
            "reference": (
                "MuJoCo inverse dynamics from reference kinematics; floating-base "
                "force/moment distributed across inferred foot/toe support sites"
            ),
            "reference_is_measured_ground_truth": False,
            "lowpass_hz": args.lowpass_hz,
            "control_rate_hz": 50.0,
        },
        "conditions": {
            "seconds": args.seconds,
            "seed": args.seed,
            "target_velocity_m_s": args.target_velocity,
            "reference_start_index": 0,
            "xml": str(context["model_path"]),
        },
        "reference_quality": {
            "differentiation_edge_samples_excluded_from_figure_and_metrics": edge,
            "cop_pitch_moment_clipped_fraction": reference.cop_moment_clipped_fraction,
            "maximum_floating_pelvis_residual_nm": reference.maximum_pelvis_residual_nm,
        },
        "reference_estimated_net_moment_rms_nm": {
            joint: rms(reference.filtered_net_nm[joint][valid]) for joint in JOINTS
        },
        "checkpoints": {},
    }
    for rollout in rollouts:
        summary["checkpoints"][rollout.label] = {
            "path": str(rollout.checkpoint),
            "steps": rollout.checkpoint_steps,
            "termination_reason": rollout.termination_reason,
            "active_moment_rms_nm": {
                joint: rms(rollout.filtered_active_nm[joint]) for joint in JOINTS
            },
            "net_moment_rms_nm": {
                joint: rms(rollout.filtered_net_nm[joint]) for joint in JOINTS
            },
            "passive_moment_rms_nm": {
                joint: rms(rollout.filtered_passive_nm[joint]) for joint in JOINTS
            },
        }
    (args.output_dir / "joint_moment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
