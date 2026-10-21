"""Render the imitation reference directly on the bare myolegs22 model.

No policy or actuator command is used. Each frame writes the reference qpos/qvel
for the same joints consumed by the training environment, advances the reference
index, calls MuJoCo forward kinematics, and renders the resulting pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import mujoco

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with (args.log_dir / "session_config.json").open(encoding="utf-8") as file:
        config = DictionableDataclass.create(
            ImitationTrainSessionConfig, json.load(file)
        )

    config.env_params.num_envs = 1
    config.env_params.flag_random_ref_index = False
    output = args.output or args.log_dir / "reference_trajectory_myolegs22.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    env = EnvironmentHandler.create_environment(
        config, is_rendering_on=False, is_evaluate_mode=True
    )
    unwrapped = env.unwrapped
    try:
        if len(unwrapped._device_actuator_ids):
            raise RuntimeError(
                "Reference rendering requires a human-only model, but device actuators exist"
            )
        if unwrapped.sim.model.nu != 22:
            raise RuntimeError(
                f"Expected 22 human muscle actuators, found {unwrapped.sim.model.nu}"
            )

        unwrapped.set_target_velocity_mode_manually(
            MyoAssistLegBase.VelocityMode.UNIFORM,
            0,
            1.25,
            1.25,
            1.25,
            target_velocity_period=3,
        )
        env.reset(seed=config.env_params.seed)

        total_frames = int(unwrapped._reference_data_length)
        if not 0 <= args.start_index < total_frames:
            raise ValueError(
                f"--start-index must be within [0, {total_frames - 1}]"
            )
        stop_index = total_frames
        if args.frames is not None:
            if args.frames <= 0:
                raise ValueError("--frames must be positive")
            stop_index = min(total_frames, args.start_index + args.frames)

        camera = mujoco.MjvCamera()
        _ = unwrapped.sim.renderer.render_offscreen(
            camera_id=camera, width=args.width, height=args.height
        )
        for flag_name in ("mjRND_SHADOW", "mjRND_REFLECTION"):
            try:
                flag = getattr(mujoco.mjtRndFlag, flag_name)
                unwrapped.sim.renderer._scene.flags[flag] = 0
            except (AttributeError, IndexError):
                pass

        fps = int(config.env_params.control_framerate)
        with imageio.get_writer(
            output,
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        ) as writer:
            for reference_index in range(args.start_index, stop_index):
                unwrapped.imitation_step(
                    is_x_follow=True, specific_index=reference_index
                )
                camera_target = unwrapped.sim.data.body("pelvis").xpos.copy()
                camera_target[2] = 0.8
                camera.distance = args.camera_distance
                camera.azimuth = 90
                camera.elevation = 0
                camera.lookat = camera_target
                frame = unwrapped.sim.renderer.render_offscreen(
                    camera_id=camera, width=args.width, height=args.height
                )
                writer.append_data(frame)

        rendered_frames = stop_index - args.start_index
        print(
            json.dumps(
                {
                    "output": str(output),
                    "reference_data": config.env_params.reference_data_path,
                    "model": config.env_params.msk_key,
                    "device": config.env_params.device_key,
                    "actuator_count": int(unwrapped.sim.model.nu),
                    "start_index": args.start_index,
                    "stop_index_exclusive": stop_index,
                    "rendered_frames": rendered_frames,
                    "fps": fps,
                    "duration_seconds": rendered_frames / fps,
                    "resolution": [args.width, args.height],
                    "policy_loaded": False,
                    "actions_applied": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
