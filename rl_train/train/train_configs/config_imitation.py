from rl_train.train.train_configs.config import TrainSessionConfigBase
from dataclasses import dataclass, field


@dataclass
class ImitationTrainSessionConfig(TrainSessionConfigBase):
    @dataclass
    class AutoRewardAdjustParams:
        learning_rate: float = 0.001

    auto_reward_adjust_params: AutoRewardAdjustParams = field(default_factory=AutoRewardAdjustParams)

    @dataclass
    class EnvParams(TrainSessionConfigBase.EnvParams):
        @dataclass
        class RewardWeights(TrainSessionConfigBase.EnvParams.RewardWeights):
            qpos_imitation_rewards: dict = field(default_factory=dict)
            qvel_imitation_rewards: dict = field(default_factory=dict)

            end_effector_imitation_reward: float = 0.3

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        flag_random_ref_index: bool = False
        out_of_trajectory_threshold: float = 1
        # Keep measuring/reporting out_of_trajectory_threshold even when this
        # switch is false; it then becomes an evaluation diagnostic instead of
        # an episode termination condition.
        terminate_on_trajectory_error: bool = True
        # Number of consecutive control steps for which any tracked qpos error
        # must exceed the shared threshold before trajectory termination.  The
        # default of one retains the original V2 behaviour.
        trajectory_error_consecutive_steps: int = 1
        reference_data_path: str = ""

        reference_data_keys: list[str] = field(default_factory=list[str])

    env_params: EnvParams = field(default_factory=EnvParams)
