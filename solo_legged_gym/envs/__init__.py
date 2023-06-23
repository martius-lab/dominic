from solo_legged_gym.envs.base.base_task import BaseTask
from solo_legged_gym.envs.base.base_config import BaseEnvCfg, BaseTrainCfg

from solo_legged_gym.envs.solo12_domino.solo12_domino import Solo12DOMINO
from solo_legged_gym.envs.solo12_domino.solo12_domino_config import Solo12DOMINOEnvCfg, Solo12DOMINOTrainCfg

from solo_legged_gym.envs.solo12_domino_position.solo12_domino_position import Solo12DOMINOPosition
from solo_legged_gym.envs.solo12_domino_position.solo12_domino_position_config import Solo12DOMINOPositionEnvCfg, Solo12DOMINOPositionTrainCfg


from solo_legged_gym.utils.task_registry import task_registry
task_registry.register("solo12_domino", Solo12DOMINO, Solo12DOMINOEnvCfg(), Solo12DOMINOTrainCfg())
task_registry.register("solo12_domino_position", Solo12DOMINOPosition, Solo12DOMINOPositionEnvCfg(), Solo12DOMINOPositionTrainCfg())
