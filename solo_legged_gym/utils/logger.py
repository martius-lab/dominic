import os
from torch.utils.tensorboard import SummaryWriter
from solo_legged_gym.utils import class_to_dict

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg, group=None):
        super().__init__(log_dir, flush_secs)
        self.cfg = cfg
        self.use_wandb = cfg.wandb

        if self.use_wandb:
            try:
                project = cfg.experiment_name
            except KeyError:
                raise KeyError(
                    "Please specify wandb_project in the runner config."
                )

            try:
                entity = os.environ["WANDB_USERNAME"]
            except KeyError:
                raise KeyError(
                    "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
                )

            self.run_id = wandb.util.generate_id()
            wandb.init(project=project, entity=entity, dir=log_dir, group=group, id=self.run_id)
            wandb.run.name = cfg.run_name

        self.log_dict = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self.log_dict[tag] = scalar_value

    def flush_logger(self, step):
        if self.use_wandb:
            wandb.log(self.log_dict, step=step)
        self.log_dict = {}

    def stop(self):
        if self.use_wandb:
            wandb.finish(0)

    def log_config(self, env_cfg, runner_cfg, algorithm_cfg, policy_cfg):
        if self.use_wandb:
            wandb.config.update({"runner_cfg": class_to_dict(runner_cfg)})
            wandb.config.update({"policy_cfg": class_to_dict(policy_cfg)})
            wandb.config.update({"algorithm_cfg": class_to_dict(algorithm_cfg)})
            wandb.config.update({"env_cfg": class_to_dict(env_cfg)})
