"""Render a deterministic V3-B checkpoint with presentation-ready telemetry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_session = root / "rl_train/results/train_session_20261106-123840"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=default_session)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint zip; omitted selects the newest completely written model_N.zip.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Explicit display step for a non-numbered checkpoint such as final_model.zip.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--target-velocity", type=float, default=1.25)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--pd-smooth-start-seconds",
        type=float,
        default=0.5,
        help="Quintic MotorArms PD target-ramp duration after reset.",
    )
    return parser.parse_args()


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as error:
        raise ValueError(f"Checkpoint must be named model_<steps>.zip: {path}") from error


def _newest_complete_checkpoint(session_dir: Path) -> Path:
    candidates = []
    now = time.time()
    for path in (session_dir / "trained_models").glob("model_*.zip"):
        if path.stat().st_size > 0 and now - path.stat().st_mtime >= 1.0:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed model_N.zip in {session_dir}")
    return max(candidates, key=_checkpoint_step)


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _overlay(
    frame: np.ndarray,
    *,
    checkpoint_step: int,
    time_s: float,
    target_velocity: float,
    velocity: float,
    pelvis_height: float,
    displacement: float,
    reward: float,
    worst_joint: str,
    worst_error: float,
    grace_count: int,
    grace_required: int,
    arm_torque: float,
    status: str,
    pd_smooth_start_seconds: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width = image.width
    draw.rounded_rectangle(
        (18, 16, width - 18, 122), radius=12, fill=(14, 27, 43, 218)
    )
    draw.text(
        (34, 27),
        (
            f"V3-B  |  checkpoint {checkpoint_step:,} steps  |  deterministic  |  "
            f"arm PD ramp {pd_smooth_start_seconds:.1f}s"
        ),
        font=_font(21),
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (34, 61),
        (
            f"t={time_s:5.2f}s   target={target_velocity:.2f} m/s   "
            f"vx={velocity:+.2f} m/s   h={pelvis_height:.3f} m   "
            f"x={displacement:+.2f} m"
        ),
        font=_font(17),
        fill=(220, 235, 247, 255),
    )
    draw.text(
        (34, 91),
        (
            f"reward={reward:+.4f}   max|q-qref|={worst_error:.3f} "
            f"({worst_joint})   grace={grace_count}/{grace_required}   "
            f"max|tau_arm|={arm_torque:.2f} Nm"
        ),
        font=_font(16),
        fill=(220, 235, 247, 255),
    )
    status_color = (68, 214, 137, 235) if status == "RUNNING" else (255, 181, 71, 240)
    draw.rounded_rectangle(
        (width - 164, image.height - 58, width - 24, image.height - 20),
        radius=9,
        fill=(14, 27, 43, 220),
    )
    draw.text(
        (width - 147, image.height - 49),
        status,
        font=_font(16),
        fill=status_color,
    )
    return np.asarray(image)


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint or _newest_complete_checkpoint(args.session_dir)
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_step = (
        int(args.checkpoint_step)
        if args.checkpoint_step is not None
        else _checkpoint_step(checkpoint)
    )
    output = args.output or (
        args.session_dir / f"render_latest_{checkpoint_step}_annotated.mp4"
    )
    output = output.resolve()

    config_dict = json.loads(
        (args.session_dir / "session_config.json").read_text(encoding="utf-8")
    )
    config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)
    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    config.env_params.motor_arms_pd_smooth_start_seconds = float(
        args.pd_smooth_start_seconds
    )

    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=False, is_evaluate_mode=True
    )
    raw = env.unwrapped
    try:
        policy = EnvironmentHandler.get_stable_baselines3_model(
            config, env, trained_model_path=str(checkpoint)
        )
        raw.set_target_velocity_mode_manually(
            MyoAssistLegBase.VelocityMode.UNIFORM,
            0.0,
            args.target_velocity,
            args.target_velocity,
            args.target_velocity,
            target_velocity_period=3.0,
        )
        observation, _ = env.reset(seed=args.seed)
        if observation.shape != (62,) or policy.action_space.shape != (22,):
            raise RuntimeError(
                f"Unexpected V3-B interface: obs={observation.shape}, "
                f"action={policy.action_space.shape}"
            )

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 2.55
        camera.azimuth = 90.0
        camera.elevation = -5.0
        raw.sim.renderer.render_offscreen(camera_id=camera, width=960, height=720)
        option = raw.sim.renderer._scene_option
        option.geomgroup[:] = 0
        option.geomgroup[0] = 1
        option.geomgroup[2] = 1

        control_hz = int(config.env_params.control_framerate)
        render_stride = max(1, int(round(control_hz / args.video_fps)))
        max_steps = int(round(args.seconds * control_hz))
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(output), fps=args.video_fps, codec="libx264", macro_block_size=None
        )

        initial_x = float(raw.sim.data.joint("pelvis_tx").qpos[0])
        velocities = []
        heights = []
        rewards = []
        max_errors = []
        max_arm_torques = []
        termination_reason = None
        steps_completed = 0
        last_frame = None
        try:
            for step in range(1, max_steps + 1):
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                if not np.all(np.isfinite(observation)) or not np.isfinite(reward):
                    raise FloatingPointError(f"Non-finite rollout at step {step}")

                velocity = float(raw.sim.data.joint("pelvis_tx").qvel[0])
                height = float(raw.sim.data.joint("pelvis_ty").qpos[0])
                displacement = float(raw.sim.data.joint("pelvis_tx").qpos[0]) - initial_x
                worst_error = float(info.get("max_reference_position_error", 0.0))
                worst_joint = str(info.get("max_reference_error_joint", "none"))
                grace_count = int(info.get("trajectory_error_consecutive_count", 0))
                grace_required = int(
                    info.get(
                        "trajectory_error_consecutive_steps_required",
                        config.env_params.trajectory_error_consecutive_steps,
                    )
                )
                arm_torque = float(
                    np.max(
                        np.abs(
                            [
                                raw.sim.data.qfrc_actuator[
                                    raw.sim.model.jnt_dofadr[
                                        raw.sim.model.joint(name).id
                                    ]
                                ]
                                for name in (
                                    "motor_shoulder_r",
                                    "motor_elbow_r",
                                    "motor_shoulder_l",
                                    "motor_elbow_l",
                                )
                            ]
                        )
                    )
                )
                termination_reason = info.get("termination_reason") if (terminated or truncated) else None
                status = termination_reason or "RUNNING"

                velocities.append(velocity)
                heights.append(height)
                rewards.append(float(reward))
                max_errors.append(worst_error)
                max_arm_torques.append(arm_torque)
                steps_completed = step

                if (step - 1) % render_stride == 0 or terminated or truncated:
                    camera.lookat[:] = raw.sim.data.body("pelvis").xpos
                    camera.lookat[2] = 0.85
                    frame = raw.sim.renderer.render_offscreen(
                        camera_id=camera, width=960, height=720
                    )
                    last_frame = _overlay(
                        frame,
                        checkpoint_step=checkpoint_step,
                        time_s=step / control_hz,
                        target_velocity=args.target_velocity,
                        velocity=velocity,
                        pelvis_height=height,
                        displacement=displacement,
                        reward=float(reward),
                        worst_joint=worst_joint,
                        worst_error=worst_error,
                        grace_count=grace_count,
                        grace_required=grace_required,
                        arm_torque=arm_torque,
                        status=status,
                        pd_smooth_start_seconds=args.pd_smooth_start_seconds,
                    )
                    writer.append_data(last_frame)

                if terminated or truncated:
                    break

            if termination_reason and last_frame is not None:
                for _ in range(args.video_fps):
                    writer.append_data(last_frame)
        finally:
            writer.close()

        final_x = float(raw.sim.data.joint("pelvis_tx").qpos[0])
        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_steps": checkpoint_step,
            "deterministic": True,
            "reference_start_index": 0,
            "target_velocity_m_per_s": args.target_velocity,
            "control_framerate_hz": control_hz,
            "video_fps": args.video_fps,
            "completed_steps": steps_completed,
            "duration_seconds": steps_completed / control_hz,
            "termination_reason": termination_reason,
            "forward_displacement_m": final_x - initial_x,
            "mean_forward_velocity_m_per_s": float(np.mean(velocities)),
            "minimum_pelvis_height_m": float(np.min(heights)),
            "mean_reward_per_step": float(np.mean(rewards)),
            "maximum_reference_error": float(np.max(max_errors)),
            "maximum_arm_torque_nm": float(np.max(max_arm_torques)),
            "motor_arms_pd_smooth_start_seconds": args.pd_smooth_start_seconds,
            "video": str(output),
        }
        summary_path = output.with_suffix(".json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Video: {output}")
        print(f"Summary: {summary_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
