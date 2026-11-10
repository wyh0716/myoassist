"""Action-space adapters used by production RL environments."""

from __future__ import annotations

import mujoco
import numpy as np
from myosuite.utils import gym
from scipy.signal import find_peaks


MOTOR_ARM_ACTUATORS = (
    "motor_shoulder_r",
    "motor_elbow_r",
    "motor_shoulder_l",
    "motor_elbow_l",
)

MOTOR_ARM_CONTROL_ZERO = "zero"
MOTOR_ARM_CONTROL_CONTRALATERAL_FIXED20_PD = "contralateral_fixed20_pd"


def _motor_arms_layout(env):
    """Validate and return the fixed 22-muscle + 4-motor layout."""
    model = env.unwrapped.sim.model
    names = tuple(model.actuator(index).name for index in range(model.nu))
    is_muscle = model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    muscle_ids = np.flatnonzero(is_muscle)
    motor_ids = np.flatnonzero(~is_muscle)

    if model.nu != 26 or tuple(names[-4:]) != MOTOR_ARM_ACTUATORS:
        raise ValueError(
            "MotorArms action adapters require the final 22-muscle + "
            f"4-motor layout; got nu={model.nu}, trailing actuators={names[-4:]}"
        )
    if not np.array_equal(muscle_ids, np.arange(22)):
        raise ValueError(
            "The lower-limb actor requires the 22 muscles to occupy physical actuator "
            f"indices 0..21; got {muscle_ids.tolist()}"
        )
    if not np.array_equal(motor_ids, np.arange(22, 26)):
        raise ValueError(
            "The four arm motors must occupy physical actuator indices "
            f"22..25; got {motor_ids.tolist()}"
        )
    if not isinstance(env.action_space, gym.spaces.Box) or env.action_space.shape != (26,):
        raise TypeError(
            "The physical MotorArms environment must expose a 26-dimensional Box "
            f"action space; got {env.action_space!r}"
        )
    return muscle_ids, motor_ids


def _supports_motor_arms(env) -> bool:
    model = env.unwrapped.sim.model
    if model.nu != 26:
        return False
    names = tuple(model.actuator(index).name for index in range(model.nu))
    is_muscle = model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    return bool(
        tuple(names[-4:]) == MOTOR_ARM_ACTUATORS
        and np.array_equal(np.flatnonzero(is_muscle), np.arange(22))
        and np.array_equal(np.flatnonzero(~is_muscle), np.arange(22, 26))
    )


class _MotorArmsActionWrapperBase(gym.ActionWrapper):
    """Common 22-policy-action to 26-physical-action contract."""

    def __init__(self, env):
        super().__init__(env)
        self.muscle_actuator_ids, self.arm_motor_actuator_ids = _motor_arms_layout(env)
        self.physical_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=self.physical_action_space.low[self.muscle_actuator_ids].copy(),
            high=self.physical_action_space.high[self.muscle_actuator_ids].copy(),
            dtype=self.physical_action_space.dtype,
        )

    @classmethod
    def supports(cls, env) -> bool:
        return _supports_motor_arms(env)

    def _muscle_action(self, action) -> np.ndarray:
        muscle_action = np.asarray(action, dtype=self.physical_action_space.dtype)
        if muscle_action.shape != (22,):
            raise ValueError(
                "The lower-limb actor must provide exactly 22 muscle controls; "
                f"got shape {muscle_action.shape}"
            )
        return muscle_action

    def reverse_action(self, action):
        physical_action = np.asarray(action, dtype=self.physical_action_space.dtype)
        if physical_action.shape != (26,):
            raise ValueError(f"Expected a 26-value physical action, got {physical_action.shape}")
        return physical_action[self.muscle_actuator_ids].copy()

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            _observation, info = reset_result
        else:
            info = {}
        raw = self.env.unwrapped
        raw.sim.data.ctrl[self.arm_motor_actuator_ids] = 0.0
        raw.sim.forward()
        raw.robot.sync_sims(raw.sim, raw.sim_obsd)
        self._reset_controller_state()
        return raw.get_obs(), info

    def _reset_controller_state(self) -> None:
        pass


class ZeroMotorArmsActionWrapper(_MotorArmsActionWrapperBase):
    """Expose only the 22 muscles and pin the four arm motors to zero.

    The wrapped MuJoCo model still has 26 physical actuators.  PPO sees the
    original V2 22-dimensional action space, while every action delivered to
    MuJoCo is ``[22 muscle controls, 0, 0, 0, 0]``.  The exact actuator names
    and ordering are validated before the wrapper can be constructed.
    """

    def __init__(self, env):
        super().__init__(env)
        self.zero_motor_actuator_ids = self.arm_motor_actuator_ids

    def action(self, action):
        muscle_action = self._muscle_action(action)
        physical_action = np.zeros(26, dtype=self.physical_action_space.dtype)
        physical_action[self.muscle_actuator_ids] = muscle_action
        return physical_action


class ContralateralFixed20MotorArmsPDWrapper(_MotorArmsActionWrapperBase):
    """Append deterministic shoulder/arm PD commands to a 22-muscle action.

    Each shoulder target follows the *actual* opposite hip angle, normalized by
    the reference gait's center and 99th-percentile excursion and clipped to
    ``+/- shoulder_amplitude``.  Both elbow targets remain fixed.  Reset leaves
    all arm controls at zero.  ``motor_arms_pd_smooth_start_seconds`` optionally
    applies a quintic target ramp; its zero default preserves the original
    immediate-PD baseline exactly.
    """

    def __init__(self, env, env_params):
        super().__init__(env)
        self.shoulder_amplitude_rad = np.deg2rad(
            float(env_params.motor_arms_shoulder_amplitude_deg)
        )
        self.shoulder_kp = float(env_params.motor_arms_shoulder_kp_nm_per_rad)
        self.shoulder_kd = float(env_params.motor_arms_shoulder_kd_nm_s_per_rad)
        self.elbow_target_rad = np.deg2rad(float(env_params.motor_arms_elbow_target_deg))
        self.elbow_kp = float(env_params.motor_arms_elbow_kp_nm_per_rad)
        self.elbow_kd = float(env_params.motor_arms_elbow_kd_nm_s_per_rad)
        self.smooth_start_seconds = float(
            getattr(env_params, "motor_arms_pd_smooth_start_seconds", 0.0)
        )
        if self.shoulder_amplitude_rad <= 0.0:
            raise ValueError("motor_arms_shoulder_amplitude_deg must be positive")
        if not 0.0 <= self.elbow_target_rad <= 2.269:
            raise ValueError("motor_arms_elbow_target_deg must lie in the elbow joint range")
        if min(self.shoulder_kp, self.shoulder_kd, self.elbow_kp, self.elbow_kd) < 0.0:
            raise ValueError("MotorArms PD gains must be non-negative")
        if self.smooth_start_seconds < 0.0:
            raise ValueError("motor_arms_pd_smooth_start_seconds must be non-negative")

        raw = env.unwrapped
        series = raw._reference_data["series_data"]
        self.hip_center_rad = {}
        self.hip_scale_rad = {}
        for side in ("r", "l"):
            hip = np.asarray(series[f"q_hip_flexion_{side}"], dtype=float)
            center = float(np.mean(hip))
            scale = float(np.quantile(np.abs(hip - center), 0.99))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Reference hip excursion is invalid for side {side}: {scale}")
            self.hip_center_rad[side] = center
            self.hip_scale_rad[side] = scale

        right_hip = np.asarray(series["q_hip_flexion_r"], dtype=float)
        control_hz = 1.0 / raw.dt
        peaks, _ = find_peaks(
            right_hip,
            distance=max(1, int(round(0.6 * control_hz))),
            prominence=max(1e-8, 0.5 * float(np.std(right_hip))),
        )
        if len(peaks) < 3:
            raise ValueError("Could not estimate a gait period for shoulder target-speed limiting")
        median_period_s = float(np.median(np.diff(peaks) / control_hz))
        self.maximum_shoulder_target_speed_rad_s = (
            1.5 * self.shoulder_amplitude_rad * 2.0 * np.pi / median_period_s
        )
        self.last_arm_targets_rad = np.zeros(4, dtype=float)
        self.last_arm_target_velocities_rad_s = np.zeros(4, dtype=float)
        self.last_arm_unsmoothed_targets_rad = np.zeros(4, dtype=float)
        self.last_arm_unsmoothed_target_velocities_rad_s = np.zeros(4, dtype=float)
        self.last_arm_requested_torques_nm = np.zeros(4, dtype=float)
        self.last_arm_commanded_torques_nm = np.zeros(4, dtype=float)
        self.last_arm_controls = np.zeros(4, dtype=float)
        self.smooth_start_elapsed_seconds = 0.0
        self.smooth_start_initial_qpos_rad = np.zeros(4, dtype=float)

    def _reset_controller_state(self) -> None:
        raw = self.env.unwrapped
        model = raw.sim.model
        data = raw.sim.data
        for slot, name in enumerate(MOTOR_ARM_ACTUATORS):
            joint_id = int(model.joint(name).id)
            self.smooth_start_initial_qpos_rad[slot] = float(
                data.qpos[model.jnt_qposadr[joint_id]]
            )
        self.smooth_start_elapsed_seconds = 0.0
        self.last_arm_targets_rad[:] = self.smooth_start_initial_qpos_rad
        self.last_arm_target_velocities_rad_s[:] = 0.0
        self.last_arm_unsmoothed_targets_rad[:] = 0.0
        self.last_arm_unsmoothed_target_velocities_rad_s[:] = 0.0
        self.last_arm_requested_torques_nm[:] = 0.0
        self.last_arm_commanded_torques_nm[:] = 0.0
        self.last_arm_controls[:] = 0.0

    @staticmethod
    def _quintic_blend(unit_time: float) -> tuple[float, float]:
        """Return smoothstep value and derivative with respect to unit time."""
        unit_time = float(np.clip(unit_time, 0.0, 1.0))
        value = unit_time**3 * (10.0 - 15.0 * unit_time + 6.0 * unit_time**2)
        derivative = 30.0 * unit_time**2 * (1.0 - unit_time) ** 2
        return value, derivative

    def action(self, action):
        muscle_action = self._muscle_action(action)
        raw = self.env.unwrapped
        model = raw.sim.model
        data = raw.sim.data
        targets = np.array(
            [0.0, self.elbow_target_rad, 0.0, self.elbow_target_rad],
            dtype=float,
        )
        target_velocities = np.zeros(4, dtype=float)

        for shoulder_side, opposite_hip_side, slot in (("r", "l", 0), ("l", "r", 2)):
            hip_name = f"hip_flexion_{opposite_hip_side}"
            center = self.hip_center_rad[opposite_hip_side]
            scale = self.hip_scale_rad[opposite_hip_side]
            hip_angle = float(data.joint(hip_name).qpos[0])
            hip_velocity = float(data.joint(hip_name).qvel[0])
            raw_normalized_hip = (hip_angle - center) / scale
            normalized_hip = float(np.clip(raw_normalized_hip, -1.0, 1.0))
            normalized_hip_velocity = 0.0
            if -1.0 < raw_normalized_hip < 1.0:
                normalized_hip_velocity = hip_velocity / scale
            targets[slot] = self.shoulder_amplitude_rad * normalized_hip
            target_velocities[slot] = float(
                np.clip(
                    self.shoulder_amplitude_rad * normalized_hip_velocity,
                    -self.maximum_shoulder_target_speed_rad_s,
                    self.maximum_shoulder_target_speed_rad_s,
                )
            )

        self.last_arm_unsmoothed_targets_rad[:] = targets
        self.last_arm_unsmoothed_target_velocities_rad_s[:] = target_velocities
        if (
            self.smooth_start_seconds > 0.0
            and self.smooth_start_elapsed_seconds < self.smooth_start_seconds
        ):
            unit_time = self.smooth_start_elapsed_seconds / self.smooth_start_seconds
            blend, derivative_unit_time = self._quintic_blend(unit_time)
            blend_rate = derivative_unit_time / self.smooth_start_seconds
            target_delta = targets - self.smooth_start_initial_qpos_rad
            target_velocities = blend_rate * target_delta + blend * target_velocities
            targets = self.smooth_start_initial_qpos_rad + blend * target_delta

        torques = np.zeros(4, dtype=float)
        controls = np.zeros(4, dtype=self.physical_action_space.dtype)
        for slot, name in enumerate(MOTOR_ARM_ACTUATORS):
            joint = model.joint(name)
            qpos = float(data.qpos[model.jnt_qposadr[joint.id]])
            qvel = float(data.qvel[model.jnt_dofadr[joint.id]])
            if "shoulder" in name:
                kp, kd = self.shoulder_kp, self.shoulder_kd
            else:
                kp, kd = self.elbow_kp, self.elbow_kd
            torque = kp * (targets[slot] - qpos) + kd * (
                target_velocities[slot] - qvel
            )
            actuator = model.actuator(name)
            gear = float(model.actuator_gear[actuator.id, 0])
            low, high = model.actuator_ctrlrange[actuator.id]
            controls[slot] = np.clip(torque / gear, low, high)
            torques[slot] = torque

        physical_action = np.zeros(26, dtype=self.physical_action_space.dtype)
        physical_action[self.muscle_actuator_ids] = muscle_action
        physical_action[self.arm_motor_actuator_ids] = controls
        self.last_arm_targets_rad[:] = targets
        self.last_arm_target_velocities_rad_s[:] = target_velocities
        self.last_arm_requested_torques_nm[:] = torques
        self.last_arm_commanded_torques_nm[:] = controls * np.asarray(
            [
                float(model.actuator_gear[model.actuator(name).id, 0])
                for name in MOTOR_ARM_ACTUATORS
            ],
            dtype=float,
        )
        self.last_arm_controls[:] = controls
        self.smooth_start_elapsed_seconds += float(raw.dt)
        return physical_action
