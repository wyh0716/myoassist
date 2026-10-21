"""Compare deterministic PPO checkpoints against the imitation reference.

The comparison deliberately fixes the reference start at index zero and uses the
same environment, target speed, and evaluation horizon for every checkpoint.
Joint errors are sampled after each control step against the environment's current
reference index, which is the same alignment used by the imitation reward.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def _checkpoint_steps(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot read timestep from checkpoint name: {path}") from exc


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def main() -> None:
    args = _parse_args()
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    with (args.log_dir / "session_config.json").open(encoding="utf-8") as file:
        config = DictionableDataclass.create(ImitationTrainSessionConfig, json.load(file))

    # Evaluation-only changes: identical, repeatable starting state and enough room
    # for the requested horizon. These do not modify the saved training config.
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    config.env_params.custom_max_episode_steps = 1_000_000_000
    config.env_params.out_of_trajectory_threshold = 1_000_000

    np.random.seed(config.env_params.seed)
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    unwrapped = env.unwrapped

    results = []
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

            errors_deg = {
                joint: {side: [] for side in side_names}
                for joint, side_names in JOINTS.items()
            }
            alignment_disagreements_deg = []
            terminated = truncated = False

            for _ in range(args.steps):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(action)
                reference_index = unwrapped._imitation_index
                environment_qpos_diff = unwrapped._get_qpos_diff()

                for joint, side_names in JOINTS.items():
                    for side, joint_name in side_names.items():
                        simulated = float(unwrapped.sim.data.joint(joint_name).qpos[0])
                        reference = float(
                            np.asarray(
                                unwrapped._reference_data["series_data"][f"q_{joint_name}"][reference_index]
                            ).squeeze()
                        )
                        error_deg = float(np.rad2deg(simulated - reference))
                        errors_deg[joint][side].append(error_deg)
                        environment_error_deg = float(
                            np.rad2deg(np.asarray(environment_qpos_diff[joint_name]).squeeze())
                        )
                        alignment_disagreements_deg.append(abs(error_deg - environment_error_deg))

                if terminated or truncated:
                    break

            completed_steps = len(errors_deg["hip"]["right"])
            if completed_steps != args.steps:
                raise RuntimeError(
                    f"{checkpoint.name} stopped at {completed_steps}/{args.steps} steps "
                    f"(terminated={terminated}, truncated={truncated}); comparison would not be like-for-like"
                )

            joint_results = {}
            all_errors = []
            for joint, side_errors in errors_deg.items():
                right = np.asarray(side_errors["right"], dtype=np.float64)
                left = np.asarray(side_errors["left"], dtype=np.float64)
                bilateral = np.concatenate((right, left))
                joint_results[joint] = {
                    "right_rmse_deg": _rmse(right),
                    "left_rmse_deg": _rmse(left),
                    "bilateral_rmse_deg": _rmse(bilateral),
                }
                all_errors.append(bilateral)

            results.append(
                {
                    "checkpoint": checkpoint.name,
                    "timesteps": _checkpoint_steps(checkpoint),
                    "completed_control_steps": completed_steps,
                    "duration_seconds": completed_steps / config.env_params.control_framerate,
                    "max_alignment_check_error_deg": float(max(alignment_disagreements_deg)),
                    "joint_rmse_deg": joint_results,
                    "all_six_joints_rmse_deg": _rmse(np.concatenate(all_errors)),
                }
            )
    finally:
        env.close()

    output_prefix = args.output_prefix or args.log_dir / "joint_rmse_fixed_reference_900"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    payload = {
        "definition": "RMSE in degrees between simulated qpos and the same-step imitation reference qpos",
        "formula": "sqrt(mean((simulated_angle_deg - reference_angle_deg)^2))",
        "reference_start_index": 0,
        "target_velocity_m_per_s": args.target_velocity,
        "control_framerate_hz": config.env_params.control_framerate,
        "requested_control_steps": args.steps,
        "deterministic_policy": True,
        "checkpoints": results,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "checkpoint",
                "timesteps",
                "joint",
                "right_rmse_deg",
                "left_rmse_deg",
                "bilateral_rmse_deg",
            ),
        )
        writer.writeheader()
        for result in results:
            for joint, metrics in result["joint_rmse_deg"].items():
                writer.writerow(
                    {
                        "checkpoint": result["checkpoint"],
                        "timesteps": result["timesteps"],
                        "joint": joint,
                        **metrics,
                    }
                )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
