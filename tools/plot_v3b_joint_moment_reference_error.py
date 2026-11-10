#!/usr/bin/env python3
"""Overlay 100M/200M muscle-actuator moments with reference ID estimates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
MODEL_COLORS = {"100M": "#1f5a8a", "200M": "#d97706"}
REFERENCE_COLOR = "#4b5563"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    output_dir = (
        root
        / "rl_train/results/train_session_20261106-123840"
        / "joint_moments_100m_200m_reference"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=output_dir / "joint_moment_timeseries_100m_200m_reference.csv",
    )
    parser.add_argument(
        "--input-summary",
        type=Path,
        default=output_dir / "joint_moment_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    return parser.parse_args()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = actual - reference
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    correlation = float(np.corrcoef(actual, reference)[0, 1])
    return {
        "rmse_nm": rmse,
        "mae_nm": mae,
        "reference_rms_nm": reference_rms,
        "normalized_rmse_percent_of_reference_rms": (
            100.0 * rmse / reference_rms if reference_rms > 0.0 else float("nan")
        ),
        "correlation": correlation,
    }


def style_axis(axis: plt.Axes) -> None:
    axis.axhline(0.0, color="#64748b", linewidth=0.8)
    axis.grid(True, color="#cbd5e1", alpha=0.5, linewidth=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylabel("Moment (N·m)")


def plot_model(
    *,
    label: str,
    data: dict[str, np.ndarray],
    valid: slice,
    limits: dict[str, tuple[float, float]],
    metrics_by_joint: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    time_s = data["time_s"][valid]
    stem = label.lower()
    figure, axes = plt.subplots(
        3, 2, figsize=(15.5, 11.5), sharex=True, constrained_layout=True
    )
    for axis, joint in zip(axes.flat, JOINTS):
        simulated = data[f"{stem}_active_{joint}_filtered_nm"][valid]
        reference = data[f"reference_estimated_net_{joint}_filtered_nm"][valid]
        axis.plot(
            time_s,
            reference,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.25,
            label="Reference ID estimate",
        )
        axis.plot(
            time_s,
            simulated,
            color=MODEL_COLORS[label],
            linewidth=1.25,
            label=f"{label} muscle-actuator moment",
        )
        axis.set_ylim(*limits[joint])
        style_axis(axis)
        axis.set_title(DISPLAY[joint])
        item = metrics_by_joint[joint]
        axis.text(
            0.015,
            0.035,
            f"RMSE {item['rmse_nm']:.1f} N·m  |  MAE {item['mae_nm']:.1f} N·m\n"
            f"NRMSE {item['normalized_rmse_percent_of_reference_rms']:.0f}%  |  "
            f"r {item['correlation']:.2f}",
            transform=axis.transAxes,
            fontsize=9.2,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "alpha": 0.88,
            },
        )
        axis.legend(loc="upper right", frameon=False, fontsize=9)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        f"{label} checkpoint vs reference inverse-dynamics estimate\n"
        "Same phase/time grid; 10 Hz low-pass; reference GRF/COP inferred from kinematics",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    if not args.input_summary.is_file():
        raise FileNotFoundError(args.input_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_csv(args.input_csv)
    source_summary = json.loads(args.input_summary.read_text(encoding="utf-8"))
    edge = int(
        source_summary["reference_quality"]
        ["differentiation_edge_samples_excluded_from_figure_and_metrics"]
    )
    valid = slice(edge, len(data["time_s"]) - edge)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for label in ("100M", "200M"):
        stem = label.lower()
        results[label] = {}
        for joint in JOINTS:
            results[label][joint] = metrics(
                data[f"{stem}_active_{joint}_filtered_nm"][valid],
                data[f"reference_estimated_net_{joint}_filtered_nm"][valid],
            )

    limits = {}
    for joint in JOINTS:
        values = np.concatenate(
            [
                data[f"reference_estimated_net_{joint}_filtered_nm"][valid],
                data[f"100m_active_{joint}_filtered_nm"][valid],
                data[f"200m_active_{joint}_filtered_nm"][valid],
            ]
        )
        margin = max(5.0, 0.07 * float(np.ptp(values)))
        limits[joint] = (
            float(np.min(values) - margin),
            float(np.max(values) + margin),
        )

    for label in ("100M", "200M"):
        plot_model(
            label=label,
            data=data,
            valid=valid,
            limits=limits,
            metrics_by_joint=results[label],
            output_path=args.output_dir / f"{label.lower()}_vs_reference_joint_moments.png",
        )

    comparison = {
        "comparison": (
            "filtered simulation qfrc_actuator versus filtered reference "
            "inverse-dynamics estimated net joint moment"
        ),
        "edge_samples_excluded_each_end": edge,
        "reference_is_measured_ground_truth": False,
        "reference_cop_pitch_moment_clipped_fraction": source_summary[
            "reference_quality"
        ]["cop_pitch_moment_clipped_fraction"],
        "models": results,
    }
    (args.output_dir / "joint_moment_reference_error_metrics.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
