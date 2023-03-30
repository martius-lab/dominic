from solo_legged_gym.envs.base.base_task import BaseTask
from solo_legged_gym.envs.base.base_config import BaseEnvCfg, BaseTrainCfg

from solo_legged_gym.envs.a1_vanilla.a1_vanilla import A1Vanilla
from solo_legged_gym.envs.a1_vanilla.a1_vanilla_config import A1VanillaEnvCfg, A1VanillaTrainCfg

from solo_legged_gym.envs.go1_vanilla.go1_vanilla import Go1Vanilla
from solo_legged_gym.envs.go1_vanilla.go1_vanilla_config import Go1VanillaEnvCfg, Go1VanillaTrainCfg

from solo_legged_gym.utils.task_registry import task_registry

task_registry.register("a1_vanilla", A1Vanilla, A1VanillaEnvCfg(), A1VanillaTrainCfg())
task_registry.register("go1_vanilla", Go1Vanilla, Go1VanillaEnvCfg(), Go1VanillaTrainCfg())
