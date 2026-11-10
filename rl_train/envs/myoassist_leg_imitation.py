import collections
import numpy as np
import mujoco
from scipy.signal import find_peaks
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
################################################################


class ImitationCustomLearningCallback(BaseCustomLearningCallback):
    def __init__(
        self,
        *,
        log_rollout_freq: int,
        evaluate_freq: int,
        log_handler: train_log_handler.TrainLogHandler,
        original_reward_weights: ImitationTrainSessionConfig.EnvParams.RewardWeights,
        auto_reward_adjust_params: ImitationTrainSessionConfig.AutoRewardAdjustParams,
        verbose=1,
    ):
        super().__init__(
            log_rollout_freq=log_rollout_freq, evaluate_freq=evaluate_freq, log_handler=log_handler, verbose=verbose
        )
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params

    def _init_callback(self):
        super()._init_callback()

        self.reward_accumulate = DictionableDataclass.create(ImitationTrainSessionConfig.EnvParams.RewardWeights)
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0

    # called after all envs step done
    def _on_step(self) -> bool:
        # print("======================self.locals    ======================")
        # # pprint.pprint(self.locals)
        # print(f"DEBUG:: {len(self.locals['infos'])=}")
        # for idx, info in enumerate(self.locals['infos']):
        #     print(f"DEBUG:: {idx=} {info['rwd_dict']=}")
        # print("======================self.locals    ======================")

        # print(f"DEBUG:: {subprocvec_env=}")
        # print(f"DEBUG:: {subprocvec_env.env_method('subproc_env_test', 'This is param from learning callback')=}")
        for info in self.locals["infos"]:
            for key in self.reward_accumulate.keys():
                self.reward_accumulate[key] += info["rwd_dict"][key]

        super()._on_step()

        return True

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log: bool = True) -> "ImitationTrainCheckpointData":
        log_data_base = super()._on_rollout_end(write_log=False)
        if log_data_base is None:
            return
        log_data = ImitationTrainCheckpointData(
            **log_data_base.__dict__,
            reward_weights=DictionableDataclass.to_dict(self._reward_weights),
            reward_accumulate=self.reward_accumulate.copy(),
        )
        if write_log:
            self.train_log_handler.add_log_data(log_data)
            self.train_log_handler.write_json_file()

        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)

        ## ARA (Disabled)

        # print(f"DEBUG:: {self.reward_accumulate=}")
        # joint_rewards = {}
        # for key in self.reward_accumulate.keys():
        #     # print(f"DEBUG:: {key=} {self.reward_accumulate[key]=}")
        #     if MyoLeg18Imitation.Q_POS_DIFF_REWARD_KEY_PREFIX in key:
        #         joint_rewards[key] = self.reward_accumulate[key]
        #     self.reward_accumulate[key] = 0
        # reward_mean = 0
        # for key in joint_rewards.keys():
        #     reward_mean += joint_rewards[key]
        # reward_mean /= len(joint_rewards)
        # joint_reward_deviations = {key: (joint_rewards[key] - reward_mean)/reward_mean for key in joint_rewards.keys()}

        # for key in joint_reward_deviations.keys():
        #     new_reward_weight = getattr(self._reward_weights, key) - self._auto_reward_adjust_params.learning_rate * joint_reward_deviations[key]
        #     setattr(self._reward_weights, key, new_reward_weight)
        # subprocvec_env:SubprocVecEnv = self.model.get_env()
        # subprocvec_env.env_method('set_reward_weights', self._reward_weights)
        # print(f"DEBUG:: {self._reward_weights=}")


##############################################################################


class MyoAssistLegImitation(MyoAssistLegBase):
    # Standing-height excess above which the reference's pelvis height is corrected for the
    # device's under-foot thickness. Measured: devices that add nothing under the foot sit within
    # 1 cm of the reference, STRIDE_L2's sole pads put it 5.5 cm above it.
    REFERENCE_HEIGHT_CORRECTION_THRESHOLD = 0.02

    # automatically inherit from MyoAssistLegBase
    # DEFAULT_OBS_KEYS = ['qpos',
    #                     'qvel',
    #                     'act',
    #                     'target_velocity',
    #                     ]

    def _setup(
        self,
        *,
        env_params: ImitationTrainSessionConfig.EnvParams,
        reference_data: dict | None = None,
        loop_reference_data: bool = False,
        **kwargs,
    ):
        self._flag_random_ref_index = env_params.flag_random_ref_index
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self._terminate_on_trajectory_error = bool(
            env_params.terminate_on_trajectory_error
        )
        self._trajectory_error_consecutive_steps = int(
            env_params.trajectory_error_consecutive_steps
        )
        if self._trajectory_error_consecutive_steps < 1:
            raise ValueError("trajectory_error_consecutive_steps must be at least one")
        self._trajectory_error_consecutive_count = 0
        self.reference_data_keys = env_params.reference_data_keys
        self._loop_reference_data = loop_reference_data
        self._include_future_reference_error_in_observation = bool(
            env_params.include_future_reference_error_in_observation
        )
        self.future_reference_error_keys = tuple(
            env_params.future_reference_error_keys
        )
        self._include_gait_phase_in_observation = bool(
            env_params.include_gait_phase_in_observation
        )
        self._gait_phase_reference_key = env_params.gait_phase_reference_key
        self._derive_gait_phase_if_missing = bool(
            env_params.derive_gait_phase_if_missing
        )
        self._auto_reference_height_correction = bool(
            env_params.auto_reference_height_correction
        )
        self._reward_keys_and_weights: ImitationTrainSessionConfig.EnvParams.RewardWeights = env_params.reward_keys_and_weights

        self.setup_reference_data(data=reference_data)

        super()._setup(
            env_params=env_params,
            **kwargs,
        )

    def _reference_position_value(self, name: str, phase: float | None = None) -> float:
        """Return a q reference in the V3-B local world frame."""
        value = self._reference_value(f"q_{name}", phase=phase)
        if name == "pelvis_tx":
            return 0.0
        return value

    def set_reward_weights(self, reward_keys_and_weights: TrainSessionConfigBase.EnvParams.RewardWeights):
        self._reward_keys_and_weights = reward_keys_and_weights

    # override from MujocoEnv
    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        if self._include_future_reference_error_in_observation:
            obs_dict["future_reference_error"] = self._future_reference_error()
        if self._include_gait_phase_in_observation:
            phase = self._reference_value(self._gait_phase_reference_key)
            obs_dict["gait_phase"] = np.asarray(
                [np.sin(phase), np.cos(phase)], dtype=float
            )
        return obs_dict

    def _next_control_reference_phase(self) -> float:
        """Reference phase one policy-control interval in the future."""
        if self._imitation_phase is None:
            raise RuntimeError("Reference phase has not been initialised")
        next_phase = self._imitation_phase + self._reference_speed_scale()
        last_index = self._reference_data_length - 1
        if self._loop_reference_data:
            return float(next_phase % self._reference_data_length)
        return float(min(next_phase, last_index))

    def _future_reference_error(self) -> np.ndarray:
        """q_ref(t + one control period) - q_actual(t), in fixed V3 order."""
        future_phase = self._next_control_reference_phase()
        errors = []
        for name in self.future_reference_error_keys:
            reference = self._reference_position_value(name, phase=future_phase)
            actual = float(self.sim.data.joint(name).qpos[0])
            errors.append(reference - actual)
        return np.asarray(errors, dtype=float)

    def _get_qpos_diff(self) -> dict:

        def get_qpos_diff_one(key: str):
            diff = (
                self.sim.data.joint(f"{key}").qpos[0].copy()
                - self._reference_position_value(key)
            )
            return diff

        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict

    def _get_qvel_diff(self):
        speed_ratio_to_target_velocity = self._reference_speed_scale()

        def get_qvel_diff_one(key: str):
            diff = (
                self.sim.data.joint(f"{key}").qvel[0].copy()
                - self._reference_value(f"dq_{key}") * speed_ratio_to_target_velocity
            )
            return diff

        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            # joint_weight = self._reward_keys_and_weights.qvel_imitation_rewards[q_key]
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict

    def _get_qpos_diff_nparray(self):
        return np.array([diff for diff in self._get_qpos_diff().values()])

    def _get_end_effector_diff(self):
        # body_pos = self.sim.data.body('pelvis').xpos.copy()
        # diff_array = []
        # for mapping in self.ANCHOR_SIM_TO_REF.values():
        #     sim_anchor = self.sim.data.joint(mapping.sim_name).xanchor.copy() - body_pos
        #     ref_anchor = self._reference_data[mapping.ref_name][self._imitation_index]
        #     diff = np.linalg.norm(sim_anchor - ref_anchor)
        #     diff_array.append(diff)
        # return diff_array
        return np.array([0])

    def _calculate_imitation_rewards(self, obs_dict):
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # Calculate joint position rewards
        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            q_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            dq_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        # Calculate end effector reward
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        # Calculate joint imitation rewards sum
        qpos_imitation_rewards = np.sum(
            [q_reward_dict[key] * self._reward_keys_and_weights.qpos_imitation_rewards[key] for key in q_reward_dict.keys()]
        )
        qvel_imitation_rewards = np.sum(
            [dq_reward_dict[key] * self._reward_keys_and_weights.qvel_imitation_rewards[key] for key in dq_reward_dict.keys()]
        )

        # Add new key-value pairs to the base_reward dictionary
        base_reward.update(
            {
                "qpos_imitation_rewards": qpos_imitation_rewards,
                "qvel_imitation_rewards": qvel_imitation_rewards,
                "end_effector_imitation_reward": anchor_reward,
            }
        )

        # Use the updated base_reward as imitation_rewards
        imitation_rewards = base_reward
        info = base_info
        return imitation_rewards, info

    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):
        # Calculate common rewards
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        # Construct reward dictionary
        # Automatically add all imitation_rewards items to rwd_dict
        rwd_dict = collections.OrderedDict((key, imitation_rewards[key]) for key in imitation_rewards)

        # Add additional fixed keys
        rwd_dict.update(
            {
                "sparse": 0,
                "solved": False,
                "done": self._get_done(),
            }
        )
        # Calculate final reward
        rwd_dict["dense"] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0)

        return rwd_dict

    def _follow_reference_motion(self, is_x_follow: bool):
        for key in self.reference_data_keys:
            if is_x_follow and key == "pelvis_tx":
                qpos = self._reference_value(f"q_{key}")
            else:
                qpos = self._reference_position_value(key)
            self.sim.data.joint(f"{key}").qpos = qpos
            # if key == 'pelvis_ty':
            #     self.sim.data.joint(f"{key}").qpos += 0.05
        speed_ratio_to_target_velocity = self._reference_speed_scale()
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = (
                self._reference_value(f"dq_{key}") * speed_ratio_to_target_velocity
            )

    def imitation_step(self, is_x_follow: bool, specific_index: int | None = None):
        if specific_index is None:
            self._advance_reference_phase(loop=True)
        else:
            self._set_reference_phase(float(specific_index))
        self._follow_reference_motion(is_x_follow)
        # should call this but I don't know why
        # next_obs, reward, terminated, truncated, info = super().step(np.zeros(self.sim.model.nu))
        # return (next_obs, reward, False, False, info)
        self.forward()
        return self._imitation_index
        # pass

    # override
    def step(self, a, **kwargs):
        # Do not pass an already-corrupt state into MuJoCo.  MyoSuite otherwise catches the
        # simulator warning and performs an internal reset, which makes the failure look like a
        # low-pelvis fall and can return a non-finite terminal observation to PPO.
        if not (
            np.all(np.isfinite(self.sim.data.qpos))
            and np.all(np.isfinite(self.sim.data.qvel))
            and np.all(np.isfinite(self.sim.data.ctrl))
            and np.all(np.isfinite(self.sim.data.act))
        ):
            terminal_observation = np.nan_to_num(
                np.asarray(self.get_obs(), dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            return terminal_observation, 0.0, True, False, {
                "termination_reason": "numerical_instability",
                "trajectory_error_exceeded": False,
            }
        # MyoSuite calls step once from its base _setup before any public reset.  That call is
        # only a construction sanity check: do not advance the real trajectory or terminate a
        # composed device because its authored initial pose differs from reference frame zero.
        is_out_of_index = (
            True
            if not self._reference_phase_initialized
            else self._advance_reference_phase(loop=self._loop_reference_data)
        )

        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        info = dict(info or {})
        numerical_failure = not (
            np.all(np.isfinite(np.asarray(next_obs)))
            and np.isfinite(reward)
            and np.all(np.isfinite(self.sim.data.qpos))
            and np.all(np.isfinite(self.sim.data.qvel))
            and np.all(np.isfinite(self.sim.data.ctrl))
            and np.all(np.isfinite(self.sim.data.act))
        )
        if numerical_failure:
            terminated = True
            info["termination_reason"] = "numerical_instability"
            info["trajectory_error_exceeded"] = False
        elif is_out_of_index:
            reward = 0
            truncated = True
            info["termination_reason"] = "reference_exhausted"
            info["trajectory_error_exceeded"] = False
        else:
            q_diff = self._get_qpos_diff()
            absolute_q_diff = {key: abs(float(value)) for key, value in q_diff.items()}
            maximum_error_joint = max(absolute_q_diff, key=absolute_q_diff.get)
            maximum_error = absolute_q_diff[maximum_error_joint]
            is_out_of_trajectory = maximum_error > self._out_of_trajectory_threshold
            info.update(
                {
                    "trajectory_error_exceeded": bool(is_out_of_trajectory),
                    "max_reference_position_error": float(maximum_error),
                    "max_reference_error_joint": maximum_error_joint,
                    "pelvis_height_m": float(self.sim.data.joint("pelvis_ty").qpos[0]),
                    "reference_index": int(self._imitation_index),
                }
            )
            if is_out_of_trajectory:
                self._trajectory_error_consecutive_count += 1
            else:
                self._trajectory_error_consecutive_count = 0
            info.update(
                {
                    "trajectory_error_consecutive_count": int(
                        self._trajectory_error_consecutive_count
                    ),
                    "trajectory_error_consecutive_steps_required": int(
                        self._trajectory_error_consecutive_steps
                    ),
                }
            )

            # A base-level fall remains immediate and takes precedence over the
            # trajectory grace window, even if both conditions occur together.
            if terminated:
                pelvis_height = float(self.sim.data.joint("pelvis_ty").qpos[0])
                # MyoSuite catches a MuJoCo NaN/Inf step and restores the simulator before
                # returning, so the post-step arrays can already be finite again.  A base-level
                # termination with a safe restored pelvis height is therefore a numerical
                # failure, not a fall.
                reason = (
                    "pelvis_below_safe_height"
                    if np.isfinite(pelvis_height) and pelvis_height < self._safe_height
                    else "numerical_instability"
                )
                info.setdefault("termination_reason", reason)
            elif (
                self._terminate_on_trajectory_error
                and self._trajectory_error_consecutive_count
                >= self._trajectory_error_consecutive_steps
            ):
                terminated = True
                info["termination_reason"] = (
                    f"joint_out_of_trajectory:{maximum_error_joint}"
                )
            elif truncated:
                info.setdefault("termination_reason", "time_limit")

        return (next_obs, reward, terminated, truncated, info)

    def setup_reference_data(self, data: dict | None):
        if data is None:
            raise ValueError("Reference data is not set")
        self._reference_data = (
            self._height_corrected_reference(data)
            if self._auto_reference_height_correction
            else data
        )
        self._reference_data_length = self._reference_data["metadata"]["resampled_data_length"]
        series = self._reference_data["series_data"]
        required_reference_keys = {
            f"q_{name}" for name in self.future_reference_error_keys
        }
        missing_reference_keys = sorted(required_reference_keys - set(series))
        if self._include_future_reference_error_in_observation and missing_reference_keys:
            raise ValueError(
                "Future-reference observation is missing NPZ channels: "
                f"{missing_reference_keys}"
            )
        if self._include_gait_phase_in_observation:
            if self._gait_phase_reference_key not in series:
                if not self._derive_gait_phase_if_missing:
                    raise ValueError(
                        "Gait phase observation requires reference channel "
                        f"{self._gait_phase_reference_key!r}"
                    )
                phase = self._derive_cycle_phase(series)
                copied_reference = dict(self._reference_data)
                copied_reference["series_data"] = dict(series)
                copied_reference["series_data"][self._gait_phase_reference_key] = phase
                # Historical alias used by analyzers should see the same arrays.
                copied_reference["resampled_series_data"] = copied_reference["series_data"]
                self._reference_data = copied_reference
                series = self._reference_data["series_data"]
            phase_values = np.asarray(series[self._gait_phase_reference_key], dtype=float)
            if phase_values.shape != (self._reference_data_length,) or not np.all(
                np.isfinite(phase_values)
            ):
                raise ValueError(
                    f"Invalid {self._gait_phase_reference_key!r} reference channel"
                )
        # MyoSuite performs one validation step inside its base _setup before the first public
        # reset.  Give that step a valid reference target; the real episode phase is selected by
        # reset immediately afterwards.
        self._imitation_index = 0
        self._imitation_phase = 0.0
        self._reference_phase_initialized = False
        pelvis_velocity = np.asarray(self._reference_data["series_data"]["dq_pelvis_tx"], dtype=float)
        self._reference_nominal_velocity = float(np.mean(pelvis_velocity))
        if not np.isfinite(self._reference_nominal_velocity) or self._reference_nominal_velocity <= 0:
            raise ValueError(
                "Reference mean pelvis velocity must be finite and positive; "
                f"got {self._reference_nominal_velocity}"
            )

    def _derive_cycle_phase(self, series: dict) -> np.ndarray:
        """Derive a continuous per-stride phase from repeated right-hip peaks.

        A future NPZ may provide ``gait_phase_unwrapped`` directly.  This
        deterministic fallback keeps the V3 framework testable beforehand and
        uses repeated gait events, never one normalization over the full file.
        """
        hip = np.asarray(series["q_hip_flexion_r"], dtype=float)
        sample_rate = float(self._reference_data["metadata"]["resampled_sample_rate"])
        peaks, _ = find_peaks(
            hip,
            distance=max(1, int(round(0.6 * sample_rate))),
            prominence=max(1e-8, 0.35 * float(np.std(hip))),
        )
        if len(peaks) < 3:
            raise ValueError(
                "Could not derive gait phase from q_hip_flexion_r; provide an "
                f"explicit {self._gait_phase_reference_key!r} NPZ channel"
            )
        median_period = float(np.median(np.diff(peaks)))
        anchors = np.concatenate(
            ([peaks[0] - median_period], peaks, [peaks[-1] + median_period])
        )
        anchor_phase = 2.0 * np.pi * np.arange(-1, len(peaks) + 1, dtype=float)
        return np.interp(np.arange(len(hip), dtype=float), anchors, anchor_phase)

    def _reference_speed_scale(self) -> float:
        """Reference-time dilation needed for the current target walking speed."""
        return float(self._target_velocity / self._reference_nominal_velocity)

    def _set_reference_phase(self, phase: float):
        last_index = self._reference_data_length - 1
        self._imitation_phase = float(np.clip(phase, 0.0, last_index))
        # Retain the integer index for existing analyzers and saved diagnostics.  Reward and
        # reset state use the continuous phase, so no sub-frame timing information is lost.
        self._imitation_index = int(np.floor(self._imitation_phase))

    def _reference_value(self, key: str, phase: float | None = None) -> float:
        if phase is None:
            phase = self._imitation_phase
        if phase is None:
            raise RuntimeError("Reference phase has not been initialised")

        values = self._reference_data["series_data"][key]
        lower = int(np.floor(phase))
        upper = min(lower + 1, self._reference_data_length - 1)
        fraction = float(phase - lower)
        return float((1.0 - fraction) * values[lower] + fraction * values[upper])

    def _advance_reference_phase(self, *, loop: bool) -> bool:
        """Advance by one control step; return whether a finite reference was exhausted."""
        if self._imitation_phase is None:
            return True

        next_phase = self._imitation_phase + self._reference_speed_scale()
        last_index = self._reference_data_length - 1
        if next_phase <= last_index:
            self._set_reference_phase(next_phase)
            return False
        if loop:
            self._set_reference_phase(next_phase % self._reference_data_length)
            return False

        self._set_reference_phase(float(last_index))
        return True

    def _random_reference_start_upper_bound(self) -> int:
        """Exclusive start bound that leaves a complete episode in the reference."""
        old_eighty_percent_bound = int(self._reference_data_length * 0.8)
        # Evaluators control their own rollout horizon and often deliberately replace the
        # configured episode limit with a huge sentinel.  A looping reference also cannot run
        # out.  Neither case should constrain the initial phase by the training horizon.
        if self.is_evaluate_mode or self._loop_reference_data:
            return old_eighty_percent_bound
        max_speed_scale = self._max_target_velocity / self._reference_nominal_velocity
        last_index = self._reference_data_length - 1
        latest_full_episode_start = int(
            np.floor(last_index - self.CUSTOM_MAX_EPISODE_STEPS * max_speed_scale)
        )
        upper_bound = min(old_eighty_percent_bound, latest_full_episode_start + 1)
        if upper_bound <= 0:
            required_frames = self.CUSTOM_MAX_EPISODE_STEPS * max_speed_scale
            raise ValueError(
                "Reference trajectory is too short for one full episode: "
                f"available last index={last_index}, required phase span={required_frames:.3f}"
            )
        return upper_bound

    def _height_corrected_reference(self, data: dict) -> dict:
        """Shift the reference's pelvis height so the composed model's feet reach the ground.

        The reference is motion capture of a bare musculoskeletal model, so its `q_pelvis_ty`
        assumes the human foot is the ground-contacting surface. A device that puts material
        under the foot breaks that assumption: `STRIDE_L2` adds sole pads and stands 5.5 cm
        higher, above the reference's entire range (0.859-0.945 m), so every reset placed it
        below its own soles and the imitation term then demanded a pelvis height its geometry
        cannot reach. It never learned to walk.

        The shift is read off the composed model rather than configured or tabulated, because it
        is a geometric consequence of the device and nobody should have to look it up when adding
        one. The model's keyframe is authored per composition with the feet on the ground, so the
        gap between that standing height and the reference's own mean height is exactly the
        thickness the device introduced.

        Applied to a copy of the reference array, once, so every downstream reader -- the pose
        `_follow_reference_motion` writes, the imitation reward, the out-of-trajectory check --
        sees the corrected target without knowing about the shift.
        """
        series = data["series_data"]
        if "q_pelvis_ty" not in series or self.sim.model.nkey == 0:
            return data

        ty_adr = self.sim.model.jnt_qposadr[self.sim.model.joint("pelvis_ty").id]
        standing_height = float(self.sim.model.key_qpos[0][ty_adr])
        reference_height = float(np.mean(series["q_pelvis_ty"]))
        offset = standing_height - reference_height

        # Devices that add nothing under the foot land within a centimetre of the reference, and
        # the existing configs train correctly at that residual, so leave them exactly as they
        # were rather than perturbing every run for a rounding difference.
        if offset < self.REFERENCE_HEIGHT_CORRECTION_THRESHOLD:
            return data

        corrected = dict(data)
        corrected["series_data"] = dict(series)
        corrected["series_data"]["q_pelvis_ty"] = np.asarray(series["q_pelvis_ty"], dtype=float) + offset
        print(
            f"Reference pelvis height shifted by {offset:+.4f} m: this device stands at "
            f"{standing_height:.3f} m against the reference's {reference_height:.3f} m"
        )
        return corrected

    def reset(self, **kwargs):
        self._trajectory_error_consecutive_count = 0
        # Select the new episode target before reference velocities are initialised.  Previously
        # _follow_reference_motion used the previous episode's target and the base reset changed
        # it afterwards, leaving the initial qvel vector internally inconsistent.
        if kwargs.get("seed") is not None:
            self.seed(kwargs["seed"])
        self._prepare_episode_reset()

        if self._flag_random_ref_index:
            start_index = int(
                self.np_random.integers(0, self._random_reference_start_upper_bound())
            )
        else:
            start_index = 0
        self._set_reference_phase(float(start_index))
        self._reference_phase_initialized = True
        # Start every episode from the complete authored keyframe before the
        # reference overwrites pelvis/lower-limb coordinates.  Reference data
        # has no shoulder/elbow (or device) channels, so using the current
        # simulator arrays here used to leak those unreferenced DOFs from the
        # constructor warm-up or the preceding episode into the next reset.
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel
        self.sim.data.ctrl[:] = 0.0
        if self.sim.model.na:
            self.sim.data.act[:] = 0.0
        # generate random targets
        # new_qpos = self.generate_qpos()# TODO: should set qvel too.
        # self.sim.data.qpos = new_qpos
        self._follow_reference_motion(False)
        self._reset_motor_arm_state()
        self._synchronise_polynomial_joint_equalities()

        obs = super().reset(
            reset_qpos=self.sim.data.qpos.copy(),
            reset_qvel=self.sim.data.qvel.copy(),
            _episode_state_prepared=True,
            **kwargs,
        )
        self.sim.data.ctrl[:] = 0.0
        if self.sim.model.na:
            self.sim.data.act[:] = 0.0
        self._reset_motor_arm_state()
        self._synchronise_polynomial_joint_equalities()
        self.sim.forward()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        reset_observation = self.get_obs()
        if isinstance(obs, tuple):
            obs = (reset_observation, *obs[1:])
        else:
            obs = reset_observation
        return obs

    def _reset_motor_arm_state(self) -> None:
        """Reset the four unreferenced arm joints to the XML-authored zero pose."""
        arm_names = (
            "motor_shoulder_r",
            "motor_elbow_r",
            "motor_shoulder_l",
            "motor_elbow_l",
        )
        available = {
            self.sim.model.joint(index).name for index in range(self.sim.model.njnt)
        }
        for name in arm_names:
            if name in available:
                self.sim.data.joint(name).qpos[0] = 0.0
                self.sim.data.joint(name).qvel[0] = 0.0

    def _synchronise_polynomial_joint_equalities(self) -> None:
        """Make equality slave q/dq consistent with the reset master joints."""
        model, data = self.sim.model, self.sim.data
        for equality_id in range(model.neq):
            if int(model.eq_type[equality_id]) != int(mujoco.mjtEq.mjEQ_JOINT):
                continue
            slave_id = int(model.eq_obj1id[equality_id])
            master_id = int(model.eq_obj2id[equality_id])
            if slave_id < 0 or master_id < 0:
                continue
            slave_qadr = int(model.jnt_qposadr[slave_id])
            master_qadr = int(model.jnt_qposadr[master_id])
            slave_dadr = int(model.jnt_dofadr[slave_id])
            master_dadr = int(model.jnt_dofadr[master_id])
            coefficients = np.asarray(model.eq_data[equality_id, :5], dtype=float)
            master_delta = float(data.qpos[master_qadr] - model.qpos0[master_qadr])
            data.qpos[slave_qadr] = model.qpos0[slave_qadr] + np.polynomial.polynomial.polyval(
                master_delta, coefficients
            )
            derivative = np.polynomial.polynomial.polyval(
                master_delta,
                np.polynomial.polynomial.polyder(coefficients),
            )
            data.qvel[slave_dadr] = derivative * data.qvel[master_dadr]

    # override
    def _initialize_pose(self):
        super()._initialize_pose()
