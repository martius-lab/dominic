import os

from torch.utils.tensorboard import SummaryWriter
from solo_legged_gym.utils import class_to_dict

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["experiment_name"]
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

        project_log_path = os.path.dirname(os.path.abspath(log_dir))
        run_name = os.path.basename(os.path.abspath(log_dir))
        wandb.init(project=project, entity=entity, dir=project_log_path)
        wandb.run.name = run_name + "-" + wandb.run.name.split("-")[-1]

    def store_config(self, env_cfg, runner_cfg, algorithm_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"algorithm_cfg": algorithm_cfg})
        wandb.config.update({"env_cfg": class_to_dict(env_cfg)})

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, algorithm_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, algorithm_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path)
