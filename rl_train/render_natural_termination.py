"""Render deterministic checkpoints until the original environment ends the episode.

Unlike the standard long-horizon gait evaluator, this script does not relax the
pelvis-height, imitation-error, reference-end, or episode-length termination
conditions.  Every checkpoint starts from reference index zero at the same target
speed, and recording stops on the first real ``terminated``/``truncated`` signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rl_train.analyzer.gait_data import GaitData
from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.train_log_handler import TrainLogHandler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _termination_reasons(
    env,
    *,
    terminated: bool,
    truncated: bool,
    step_count: int,
    max_steps: int,
    safe_height: float,
    trajectory_threshold: float,
) -> tuple[list[str], float, str | None, float]:
    unwrapped = env.unwrapped
    pelvis_height = float(unwrapped.sim.data.joint("pelvis_ty").qpos[0])
    qdiff = {
        name: float(np.asarray(value).squeeze())
        for name, value in unwrapped._get_qpos_diff().items()
    }
    worst_joint = max(qdiff, key=lambda name: abs(qdiff[name])) if qdiff else None
    worst_error = abs(qdiff[worst_joint]) if worst_joint is not None else 0.0

    reasons = []
    if terminated and pelvis_height < safe_height:
        reasons.append("pelvis_below_safe_height")
    if terminated and worst_error > trajectory_threshold:
        reasons.append("joint_out_of_trajectory")
    if truncated and unwrapped._imitation_index >= unwrapped._reference_data_length - 1:
        reasons.append("reference_trajectory_end")
    if truncated and (
        step_count >= max_steps
        or unwrapped._step_count_per_episode >= unwrapped.CUSTOM_MAX_EPISODE_STEPS
    ):
        reasons.append("episode_step_limit")
    if terminated and not reasons:
        reasons.append("other_environment_termination")
    if truncated and not reasons:
        reasons.append("other_environment_truncation")
    return reasons, pelvis_height, worst_joint, worst_error


def main() -> None:
    args = _parse_args()
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    with (args.log_dir / "session_config.json").open(encoding="utf-8") as file:
        config = DictionableDataclass.create(
            ImitationTrainSessionConfig, json.load(file)
        )

    # Evaluation-only reproducibility settings. All termination thresholds remain
    # exactly as stored in the training session config.
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    safe_height = float(config.env_params.safe_height)
    trajectory_threshold = float(config.env_params.out_of_trajectory_threshold)
    configured_max_steps = int(config.env_params.custom_max_episode_steps)
    max_steps = args.max_steps or configured_max_steps
    if max_steps > configured_max_steps:
        raise ValueError(
            f"--max-steps {max_steps} exceeds the original episode limit "
            f"{configured_max_steps}; that would no longer be a natural evaluation"
        )

    output_dir = args.output_dir or args.log_dir / "natural_termination_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for checkpoint in args.checkpoints:
        # Give every checkpoint a fresh MuJoCo environment. Some internal muscle
        # states are not guaranteed to be identical after a reset of an already-used
        # environment, which would make a multi-checkpoint comparison order-dependent.
        np.random.seed(config.env_params.seed)
        env = EnvironmentHandler.create_environment(
            config, is_rendering_on=False, is_evaluate_mode=True
        )
        unwrapped = env.unwrapped
        evaluator = ImitationGaitEvaluator(
            TrainLogHandler(str(args.log_dir)), config
        )
        evaluator.env = env
        evaluator.free_cam = mujoco.MjvCamera()
        _ = unwrapped.sim.renderer.render_offscreen(
            camera_id=evaluator.free_cam, width=1920, height=1080
        )
        try:
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
            # Re-seed both NumPy and Gymnasium before every checkpoint so the
            # comparison is repeatable even when several models share one process.
            np.random.seed(config.env_params.seed)
            obs, _info = env.reset(seed=config.env_params.seed)
            if unwrapped._imitation_index != 0:
                raise RuntimeError(
                    f"Reference did not reset to index 0: {unwrapped._imitation_index}"
                )

            initial_pelvis_x = float(
                unwrapped.sim.data.joint("pelvis_tx").qpos[0]
            )
            gait_data = GaitData()
            terminated = truncated = False
            rewards = []
            steps_completed = 0

            for step_count in range(1, max_steps + 1):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(action)
                gait_data.add_data(
                    mj_model=unwrapped.sim.model,
                    mj_data=unwrapped.sim.data,
                    target_velocity=unwrapped._target_velocity,
                    printing=step_count == 1,
                )
                rewards.append(float(reward))
                steps_completed = step_count
                if terminated or truncated:
                    break

            reasons, pelvis_height, worst_joint, worst_error = _termination_reasons(
                env,
                terminated=bool(terminated),
                truncated=bool(truncated),
                step_count=steps_completed,
                max_steps=max_steps,
                safe_height=safe_height,
                trajectory_threshold=trajectory_threshold,
            )
            final_pelvis_x = float(
                unwrapped.sim.data.joint("pelvis_tx").qpos[0]
            )
            duration = steps_completed / config.env_params.control_framerate

            stem = f"{checkpoint.stem}_natural_termination"
            gait_path = output_dir / f"{stem}.json"
            video_path = output_dir / f"{stem}.mp4"
            gait_data.save_json_data(str(gait_path))
            evaluator.replay(
                str(gait_path),
                str(video_path),
                cam_distance=2.5,
                use_activation_visualization=False,
                cam_type="follow",
                video_fps=config.env_params.control_framerate,
            )

            result = {
                "checkpoint": checkpoint.name,
                "reference_start_index": 0,
                "target_velocity_m_per_s": args.target_velocity,
                "deterministic_policy": True,
                "control_framerate_hz": config.env_params.control_framerate,
                "completed_control_steps": steps_completed,
                "duration_seconds": duration,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reasons": reasons,
                "final_pelvis_height_m": pelvis_height,
                "worst_joint_error_name": worst_joint,
                "worst_joint_error_rad_abs": worst_error,
                "worst_joint_error_deg_abs": float(np.rad2deg(worst_error)),
                "forward_displacement_m": final_pelvis_x - initial_pelvis_x,
                "mean_reward_per_control_step": float(np.mean(rewards)),
                "video": video_path.name,
                "trajectory": gait_path.name,
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            env.close()

    summary = {
        "definition": (
            "Deterministic rollout from reference index 0, stopped immediately on "
            "the original training environment's first terminated/truncated signal"
        ),
        "original_termination_settings": {
            "safe_height_m": safe_height,
            "out_of_trajectory_threshold_rad": trajectory_threshold,
            "out_of_trajectory_threshold_deg": float(
                np.rad2deg(trajectory_threshold)
            ),
            "custom_max_episode_steps": configured_max_steps,
        },
        "checkpoints": results,
    }
    summary_path = output_dir / "natural_termination_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
