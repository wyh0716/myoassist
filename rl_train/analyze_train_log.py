"""Create a validated, compact analysis of a MyoAssist PPO train_log.json.

The callback's episode-level totals and lengths are valid, but its per-component
``average_reward_dict_per_episode`` accumulator is not reset when an episode ends.
This script therefore reconstructs interval reward components by differencing the
monotonic ``reward_accumulate`` counters instead of using that biased field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S.%f"
ROLLING_WINDOW = 31
COMPONENT_KEYS = (
    "forward_reward",
    "muscle_activation_penalty",
    "muscle_activation_diff_penalty",
    "foot_force_penalty",
    "joint_constraint_force_penalty",
    "qpos_imitation_rewards",
    "qvel_imitation_rewards",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def _centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window % 2 != 1:
        raise ValueError("Rolling window must be odd")
    result = np.full(len(values), np.nan, dtype=np.float64)
    half = window // 2
    for index in range(half, len(values) - half):
        result[index] = float(np.mean(values[index - half : index + half + 1]))
    return result


def _nearest_index(steps: np.ndarray, target: int) -> int:
    return int(np.argmin(np.abs(steps - target)))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> None:
    args = _parse_args()
    log_path = args.log_dir / "train_log.json"
    config_path = args.log_dir / "session_config.json"
    with log_path.open(encoding="utf-8") as file:
        records = json.load(file)["log_datas"]
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    if not records:
        raise RuntimeError("train_log.json contains no records")

    steps = np.asarray([record["num_timesteps"] for record in records], dtype=np.int64)
    if len(np.unique(steps)) != len(steps) or np.any(np.diff(steps) <= 0):
        raise RuntimeError("num_timesteps must be unique and strictly increasing")
    step_intervals = np.diff(steps)
    episode_length = np.asarray(
        [record["average_num_timestep"] for record in records], dtype=np.float64
    )
    episode_reward = np.asarray(
        [record["average_reward_per_episode"] for record in records], dtype=np.float64
    )
    if np.any(episode_length <= 0):
        raise RuntimeError("Every logged interval must contain at least one completed episode")
    dense_reward_per_step = episode_reward / episode_length

    reward_weights = config["env_params"]["reward_keys_and_weights"]
    flattened_weights = {
        key: float(sum(value.values())) if isinstance(value, dict) else float(value)
        for key, value in reward_weights.items()
    }
    accumulated_keys = tuple(records[0]["reward_accumulate"])
    previous_values = {key: 0.0 for key in accumulated_keys}
    previous_step = 0
    component_per_step = {key: [] for key in accumulated_keys}
    reconstructed_dense = []
    for record in records:
        interval_steps = record["num_timesteps"] - previous_step
        interval_components = {}
        for key in accumulated_keys:
            current = float(record["reward_accumulate"][key])
            interval_value = (current - previous_values[key]) / interval_steps
            component_per_step[key].append(interval_value)
            interval_components[key] = interval_value
            previous_values[key] = current
        reconstructed_dense.append(
            sum(
                flattened_weights[key] * value
                for key, value in interval_components.items()
                if key in flattened_weights
            )
        )
        previous_step = int(record["num_timesteps"])
    component_per_step = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in component_per_step.items()
    }
    reconstructed_dense = np.asarray(reconstructed_dense, dtype=np.float64)
    reconstruction_difference = dense_reward_per_step - reconstructed_dense

    rolling = {
        "episode_length": _centered_rolling_mean(episode_length, ROLLING_WINDOW),
        "episode_reward": _centered_rolling_mean(episode_reward, ROLLING_WINDOW),
        "dense_reward_per_step": _centered_rolling_mean(
            dense_reward_per_step, ROLLING_WINDOW
        ),
    }
    for key in COMPONENT_KEYS:
        rolling[key] = _centered_rolling_mean(
            component_per_step[key], ROLLING_WINDOW
        )

    valid_ppo_records = [record for record in records if record["approx_kl"] >= 0]
    approx_kl = np.asarray(
        [record["approx_kl"] for record in valid_ppo_records], dtype=np.float64
    )
    explained_variance = np.asarray(
        [record["explained_variance"] for record in valid_ppo_records],
        dtype=np.float64,
    )
    clip_fraction = np.asarray(
        [record["clip_fraction"] for record in valid_ppo_records], dtype=np.float64
    )
    policy_std = np.asarray(
        [record["std"] for record in valid_ppo_records], dtype=np.float64
    )

    raw_length_peak_index = int(np.argmax(episode_length))
    raw_reward_peak_index = int(np.argmax(episode_reward))
    rolling_length_peak_index = int(np.nanargmax(rolling["episode_length"]))
    rolling_reward_peak_index = int(np.nanargmax(rolling["episode_reward"]))
    rolling_density_peak_index = int(np.nanargmax(rolling["dense_reward_per_step"]))

    milestone_targets = [
        1_000_000,
        5_000_000,
        10_000_000,
        15_000_000,
        20_000_000,
        25_000_000,
        28_000_000,
        int(steps[-1]),
    ]
    milestones = []
    for target in milestone_targets:
        index = _nearest_index(steps, target)
        milestones.append(
            {
                "requested_timestep": target,
                "actual_timestep": int(steps[index]),
                "average_episode_length_steps": float(episode_length[index]),
                "average_episode_duration_seconds": float(episode_length[index] / 30),
                "average_episode_reward": float(episode_reward[index]),
                "average_dense_reward_per_step": float(dense_reward_per_step[index]),
                "rolling_episode_length_steps": _finite_or_none(
                    rolling["episode_length"][index]
                ),
                "rolling_episode_reward": _finite_or_none(
                    rolling["episode_reward"][index]
                ),
            }
        )

    phase_summaries = []
    for lower, upper in ((0, 10_000_000), (10_000_000, 20_000_000), (20_000_000, 30_000_000)):
        mask = (steps > lower) & (steps <= upper)
        phase_ppo = [
            record["explained_variance"]
            for record in records
            if lower < record["num_timesteps"] <= upper
            and record["explained_variance"] >= 0
        ]
        phase_summaries.append(
            {
                "range": f"{lower // 1_000_000}-{upper // 1_000_000}M",
                "record_count": int(np.sum(mask)),
                "mean_episode_length_steps": float(np.mean(episode_length[mask])),
                "median_episode_length_steps": float(np.median(episode_length[mask])),
                "mean_episode_reward": float(np.mean(episode_reward[mask])),
                "mean_dense_reward_per_step": float(
                    np.mean(dense_reward_per_step[mask])
                ),
                "mean_explained_variance": float(np.mean(phase_ppo)),
            }
        )

    first_time = datetime.strptime(records[0]["time"], TIMESTAMP_FORMAT)
    last_time = datetime.strptime(records[-1]["time"], TIMESTAMP_FORMAT)
    elapsed_seconds = (last_time - first_time).total_seconds()

    summary = {
        "source": {
            "train_log": str(log_path),
            "session_config": str(config_path),
            "record_count": len(records),
            "first_timestep": int(steps[0]),
            "last_timestep": int(steps[-1]),
            "step_interval_values": sorted({int(value) for value in step_intervals}),
            "first_timestamp": records[0]["time"],
            "last_timestamp": records[-1]["time"],
            "logged_wall_time_seconds": elapsed_seconds,
        },
        "data_quality": {
            "unique_strictly_increasing_timesteps": True,
            "finite_primary_metrics": bool(
                np.all(np.isfinite(episode_length))
                and np.all(np.isfinite(episode_reward))
            ),
            "ppo_record_count": len(valid_ppo_records),
            "component_method": (
                "Difference adjacent reward_accumulate counters and divide by the "
                "corresponding environment-timestep interval"
            ),
            "excluded_field": "average_reward_dict_per_episode",
            "excluded_field_reason": (
                "current_reward_dict_sum is not reset when an episode ends, causing "
                "the per-episode component accumulator to double-count earlier episodes"
            ),
            "dense_reward_reconstruction_mean_absolute_error": float(
                np.mean(np.abs(reconstruction_difference))
            ),
            "dense_reward_reconstruction_max_absolute_error": float(
                np.max(np.abs(reconstruction_difference))
            ),
        },
        "peaks": {
            "raw_episode_length": {
                "timestep": int(steps[raw_length_peak_index]),
                "value_steps": float(episode_length[raw_length_peak_index]),
            },
            "raw_episode_reward": {
                "timestep": int(steps[raw_reward_peak_index]),
                "value": float(episode_reward[raw_reward_peak_index]),
            },
            "rolling_episode_length": {
                "window_records": ROLLING_WINDOW,
                "approximate_window_timesteps": int(
                    ROLLING_WINDOW * np.median(step_intervals)
                ),
                "timestep": int(steps[rolling_length_peak_index]),
                "value_steps": float(
                    rolling["episode_length"][rolling_length_peak_index]
                ),
            },
            "rolling_episode_reward": {
                "timestep": int(steps[rolling_reward_peak_index]),
                "value": float(rolling["episode_reward"][rolling_reward_peak_index]),
            },
            "rolling_dense_reward_per_step": {
                "timestep": int(steps[rolling_density_peak_index]),
                "value": float(
                    rolling["dense_reward_per_step"][rolling_density_peak_index]
                ),
            },
        },
        "relationships": {
            "episode_length_vs_episode_reward_pearson_r": float(
                np.corrcoef(episode_length, episode_reward)[0, 1]
            ),
            "episode_length_vs_dense_reward_per_step_pearson_r": float(
                np.corrcoef(episode_length, dense_reward_per_step)[0, 1]
            ),
        },
        "ppo_health": {
            "target_kl": float(config["ppo_params"]["target_kl"]),
            "approx_kl_median": float(np.median(approx_kl)),
            "approx_kl_p95": float(np.percentile(approx_kl, 95)),
            "approx_kl_max": float(np.max(approx_kl)),
            "fraction_approx_kl_above_1_5x_target": float(
                np.mean(approx_kl > 1.5 * config["ppo_params"]["target_kl"])
            ),
            "explained_variance_median": float(np.median(explained_variance)),
            "explained_variance_final": float(explained_variance[-1]),
            "clip_fraction_median": float(np.median(clip_fraction)),
            "clip_fraction_final": float(clip_fraction[-1]),
            "policy_std_min": float(np.min(policy_std)),
            "policy_std_min_timestep": int(
                valid_ppo_records[int(np.argmin(policy_std))]["num_timesteps"]
            ),
            "policy_std_final": float(policy_std[-1]),
        },
        "phase_summaries": phase_summaries,
        "milestones": milestones,
    }

    output_prefix = args.output_prefix or args.log_dir / "train_log_analysis"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    chart_path = output_prefix.with_name(output_prefix.name + "_episode_length_chart.json")
    chart_sources_path = output_prefix.with_name(
        output_prefix.name + "_episode_length_sources.json"
    )
    findings_sources_path = output_prefix.with_name(
        output_prefix.name + "_findings_sources.json"
    )

    csv_fields = [
        "record_index",
        "num_timesteps",
        "million_timesteps",
        "timestamp",
        "elapsed_seconds",
        "average_episode_length_steps",
        "average_episode_duration_seconds",
        "average_episode_reward",
        "average_dense_reward_per_step",
        "reconstructed_dense_reward_per_step",
        "dense_reward_reconstruction_difference",
        "rolling_episode_length_steps_31",
        "rolling_episode_reward_31",
        "rolling_dense_reward_per_step_31",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "policy_std",
        "value_loss",
        "policy_gradient_loss",
    ]
    for key in COMPONENT_KEYS:
        csv_fields.extend(
            (
                f"{key}_per_step",
                f"{key}_weighted_contribution_per_step",
                f"{key}_rolling_31",
            )
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for index, record in enumerate(records):
            timestamp = datetime.strptime(record["time"], TIMESTAMP_FORMAT)
            row = {
                "record_index": index,
                "num_timesteps": int(steps[index]),
                "million_timesteps": float(steps[index] / 1_000_000),
                "timestamp": record["time"],
                "elapsed_seconds": float((timestamp - first_time).total_seconds()),
                "average_episode_length_steps": float(episode_length[index]),
                "average_episode_duration_seconds": float(episode_length[index] / 30),
                "average_episode_reward": float(episode_reward[index]),
                "average_dense_reward_per_step": float(dense_reward_per_step[index]),
                "reconstructed_dense_reward_per_step": float(
                    reconstructed_dense[index]
                ),
                "dense_reward_reconstruction_difference": float(
                    reconstruction_difference[index]
                ),
                "rolling_episode_length_steps_31": _finite_or_none(
                    rolling["episode_length"][index]
                ),
                "rolling_episode_reward_31": _finite_or_none(
                    rolling["episode_reward"][index]
                ),
                "rolling_dense_reward_per_step_31": _finite_or_none(
                    rolling["dense_reward_per_step"][index]
                ),
                "approx_kl": record["approx_kl"],
                "clip_fraction": record["clip_fraction"],
                "explained_variance": record["explained_variance"],
                "policy_std": record["std"],
                "value_loss": record["value_loss"],
                "policy_gradient_loss": record["policy_gradient_loss"],
            }
            for key in COMPONENT_KEYS:
                row[f"{key}_per_step"] = float(component_per_step[key][index])
                row[f"{key}_weighted_contribution_per_step"] = float(
                    component_per_step[key][index] * flattened_weights[key]
                )
                row[f"{key}_rolling_31"] = _finite_or_none(rolling[key][index])
            writer.writerow(row)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    chart_rows = []
    chart_indices = list(range(0, len(steps), 2))
    if chart_indices[-1] != len(steps) - 1:
        chart_indices.append(len(steps) - 1)
    for index in chart_indices:
        step = steps[index]
        chart_rows.append(
            {
                "million_timesteps": round(float(step / 1_000_000), 6),
                "series": "日志区间均值",
                "episode_steps": round(float(episode_length[index]), 6),
            }
        )
        if math.isfinite(rolling["episode_length"][index]):
            chart_rows.append(
                {
                    "million_timesteps": round(float(step / 1_000_000), 6),
                    "series": "31 条记录滚动均值",
                    "episode_steps": round(
                        float(rolling["episode_length"][index]), 6
                    ),
                }
            )
    chart_spec = {
        "schemaVersion": 1,
        "id": "myolegs22-training-episode-length",
        "queryId": "myolegs22_train_log_episode_length",
        "title": "3000 万步训练中的 Episode 长度",
        "description": (
            "每条日志区间均值与约 101.6 万环境步的居中滚动均值；控制频率为 30 Hz。"
        ),
        "chart": {
            "type": "line",
            "x": "million_timesteps",
            "y": "episode_steps",
            "series": "series",
            "xLabel": "累计环境步数（百万）",
            "yLabel": "平均 Episode 长度（控制步）",
            "showLegend": True,
            "startAtZero": True,
        },
        "rows": chart_rows,
        "source": {
            "label": "MyoAssist 训练日志",
            "files": [
                {"label": "train_log.json"},
                {"label": "session_config.json"},
                {"label": "train_log_analysis.csv"},
            ],
            "filters": ["训练会话：train_session_20261016-101344"],
            "caveats": [
                "训练环境启用了随机参考轨迹起点，因此该曲线不是固定初始条件的独立评估。",
                "为控制内联图体积，曲线每隔一条日志记录显示一个点；所有汇总统计仍使用完整 916 条记录。",
            ],
        },
        "theme": "codex-classic",
    }
    with chart_path.open("w", encoding="utf-8") as file:
        json.dump(chart_spec, file, ensure_ascii=False, separators=(",", ":"))

    preview_rows = [
        {
            "million_steps": round(item["actual_timestep"] / 1_000_000, 3),
            "episode_steps": round(item["average_episode_length_steps"], 3),
            "episode_seconds": round(item["average_episode_duration_seconds"], 3),
            "reward_per_step": round(item["average_dense_reward_per_step"], 6),
        }
        for item in milestones
    ]
    common_source = {
        "label": "MyoAssist 训练日志与分析输出",
        "files": [
            {"label": "train_log.json"},
            {"label": "session_config.json"},
            {"label": "train_log_analysis.csv"},
        ],
        "filters": [
            "训练会话：train_session_20261016-101344",
            "累计环境步：4096–29986816",
        ],
        "metricDefinitions": [
            {
                "label": "平均 Episode 长度",
                "definition": (
                    "平均 Episode 长度是每个日志区间内已结束 Episode 的平均控制步数；"
                    "控制频率为 30 Hz。"
                ),
            },
            {
                "label": "滚动均值",
                "definition": (
                    "滚动均值采用 31 条日志记录的居中窗口，约覆盖 101.6 万环境步。"
                ),
            },
        ],
        "caveats": [
            "训练环境启用了随机参考轨迹起点，因此训练曲线不能替代固定种子、固定起点的 checkpoint 独立评估。"
        ],
        "evidenceFlow": [
            {
                "kind": "validation",
                "title": "日志完整性检查",
                "detail": (
                    "共 916 条记录；训练步数唯一且严格递增；相邻保存间隔均为 32768 环境步。"
                ),
            }
        ],
    }
    chart_sources = {
        "schemaVersion": 1,
        "items": [
            {
                "id": "episode-length-trend",
                "title": "Episode 持续时间如何随训练推进？",
                "queries": [
                    {
                        "id": "episode-length-from-training-log",
                        "source": common_source,
                        "columns": [
                            "million_steps",
                            "episode_steps",
                            "episode_seconds",
                        ],
                        "rows": preview_rows,
                        "preview": {
                            "kind": "aggregate",
                            "note": "展示接近关键里程碑的日志记录；完整 916 条结果保存在分析 CSV 中。",
                            "totalRows": len(records),
                        },
                    }
                ],
            }
        ],
    }
    with chart_sources_path.open("w", encoding="utf-8") as file:
        json.dump(chart_sources, file, ensure_ascii=False, separators=(",", ":"))

    findings_sources = {
        "schemaVersion": 1,
        "items": [
            {
                "id": "training-progress",
                "title": "训练后期是否继续改善？",
                "queries": [
                    {
                        "id": "phase-and-milestone-summary",
                        "source": {
                            **common_source,
                            "metricDefinitions": [
                                *common_source["metricDefinitions"],
                                {
                                    "label": "单位步奖励",
                                    "definition": (
                                        "单位步奖励用日志区间的平均 Episode 奖励除以平均 Episode 长度计算。"
                                    ),
                                },
                            ],
                        },
                        "columns": [
                            "million_steps",
                            "episode_steps",
                            "episode_seconds",
                            "reward_per_step",
                        ],
                        "rows": preview_rows,
                        "preview": {
                            "kind": "aggregate",
                            "note": "关键训练里程碑的聚合预览。",
                            "totalRows": len(records),
                        },
                    }
                ],
            },
            {
                "id": "ppo-health",
                "title": "PPO 更新过程是否稳定？",
                "queries": [
                    {
                        "id": "ppo-health-summary",
                        "source": {
                            "label": "MyoAssist PPO 训练日志与配置",
                            "files": [
                                {"label": "train_log.json"},
                                {"label": "session_config.json"},
                                {"label": "train_log_analysis_summary.json"},
                            ],
                            "filters": [
                                "训练会话：train_session_20261016-101344",
                                "有效 PPO 统计记录：915 条",
                            ],
                            "metricDefinitions": [
                                {
                                    "label": "Explained variance",
                                    "definition": "Explained variance 衡量 Critic 对回报变化的拟合程度。",
                                },
                                {
                                    "label": "Approximate KL",
                                    "definition": "Approximate KL 衡量相邻 PPO 策略更新幅度，配置目标为 0.01。",
                                },
                            ],
                            "evidenceFlow": [
                                {
                                    "kind": "calculation",
                                    "title": "PPO 汇总",
                                    "detail": (
                                        "Approximate KL 中位数为 0.01219，95 分位为 0.01522；"
                                        "Explained variance 的分阶段均值为 0.706、0.661、0.589。"
                                    ),
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "id": "reward-log-quality",
                "title": "哪些奖励日志字段可以可信使用？",
                "queries": [
                    {
                        "id": "reward-accumulator-validation",
                        "source": {
                            "label": "训练回调源码与奖励累计量",
                            "files": [
                                {"label": "learning_callback.py"},
                                {"label": "train_log.json"},
                                {"label": "train_log_analysis_summary.json"},
                            ],
                            "metricDefinitions": [
                                {
                                    "label": "重建单位步奖励",
                                    "definition": (
                                        "重建单位步奖励由相邻 reward_accumulate 累计值作差，"
                                        "再除以对应环境步数间隔。"
                                    ),
                                }
                            ],
                            "caveats": [
                                "average_reward_dict_per_episode 被排除，因为 current_reward_dict_sum 在 Episode 结束时没有清零，会重复累计更早 Episode 的逐项奖励。"
                            ],
                            "evidenceFlow": [
                                {
                                    "kind": "validation",
                                    "title": "奖励重建交叉检查",
                                    "detail": (
                                        "重建单位步总奖励与 Episode 奖励/长度的平均绝对差为 0.000305，最大绝对差为 0.001893。"
                                    ),
                                }
                            ],
                        },
                    }
                ],
            },
        ],
    }
    with findings_sources_path.open("w", encoding="utf-8") as file:
        json.dump(findings_sources, file, ensure_ascii=False, separators=(",", ":"))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Chart:   {chart_path}")
    print(f"Sources: {chart_sources_path}")
    print(f"Findings sources: {findings_sources_path}")


if __name__ == "__main__":
    main()
