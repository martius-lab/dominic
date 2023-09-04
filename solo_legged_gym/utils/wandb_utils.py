import os
import allogger
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from solo_legged_gym.utils import class_to_dict

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg, group=None, resume_id=None):
        super().__init__(log_dir, flush_secs)

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

        if resume_id is not None:
            wandb.init(
                project=project,
                entity=entity,
                dir=log_dir,
                group=group,
                resume="allow",
                id=resume_id,
            )
            self.run_id = resume_id
        else:
            self.run_id = wandb.util.generate_id()
            wandb.init(project=project, entity=entity, dir=log_dir, group=group, id=self.run_id)
            wandb.run.name = os.path.basename(os.path.abspath(log_dir))

        allogger.basic_configure(
            logdir=self.log_dir,
            default_outputs=["hdf"],
            manual_flush=True)
        self.logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO},
                                          default_outputs=['hdf'])

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

        v = scalar_value
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        k = tag.replace('/', '-')
        self.logger.log(v, k, to_hdf=True)

    def flush_logger(self):
        allogger.get_root().flush(children=True)

    def stop(self):
        wandb.finish()
        allogger.get_root().flush(children=True)
        allogger.close()

    def log_config(self, env_cfg, runner_cfg, algorithm_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": class_to_dict(runner_cfg)})
        wandb.config.update({"policy_cfg": class_to_dict(policy_cfg)})
        wandb.config.update({"algorithm_cfg": class_to_dict(algorithm_cfg)})
        wandb.config.update({"env_cfg": class_to_dict(env_cfg)})
