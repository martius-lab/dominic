from solo_legged_gym.envs.base.base_task import BaseTask
from solo_legged_gym.envs.base.base_config import BaseEnvCfg, BaseTrainCfg

from solo_legged_gym.envs.solo12_vanilla.solo12_vanilla import Solo12Vanilla
from solo_legged_gym.envs.solo12_vanilla.solo12_vanilla_config import Solo12VanillaEnvCfg, Solo12VanillaTrainCfg

from solo_legged_gym.envs.solo12_phase.solo12_phase import Solo12Phase
from solo_legged_gym.envs.solo12_phase.solo12_phase_config import Solo12PhaseEnvCfg, Solo12PhaseTrainCfg

from solo_legged_gym.utils.task_registry import task_registry

task_registry.register("solo12_vanilla", Solo12Vanilla, Solo12VanillaEnvCfg(), Solo12VanillaTrainCfg())
task_registry.register("solo12_phase", Solo12Phase, Solo12PhaseEnvCfg(), Solo12PhaseTrainCfg())
