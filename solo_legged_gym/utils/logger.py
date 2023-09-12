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


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg, group=None, resume_id=None, use_wandb=False, use_allogger=False):
        super().__init__(log_dir, flush_secs)

        self.use_allogger = use_allogger
        self.use_wandb = use_wandb

        if use_wandb:
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
                wandb.run.name = cfg.run_name

        if self.use_allogger:
            allogger.basic_configure(
                logdir=self.log_dir,
                default_outputs=["hdf"],
                manual_flush=True)
            self.logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO},
                                              default_outputs=['hdf'])

        self.log_dict = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        # wandb.log({tag: scalar_value}, step=global_step)
        self.log_dict[tag] = scalar_value

        if self.use_allogger:
            v = scalar_value
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach().numpy()
            k = tag.replace('/', '-')
            self.logger.log(v, k, to_hdf=True)

    def flush_logger(self, step):
        if self.use_wandb:
            wandb.log(self.log_dict, step=step)
        if self.use_allogger:
            allogger.get_root().flush(children=True)
        self.log_dict = {}

    def stop(self):
        if self.use_wandb:
            wandb.finish(0)

        if self.use_allogger:
            allogger.get_root().flush(children=True)
            allogger.close()

    def get_allogger_step(self):
        return dict(allogger.get_logger("root").step_per_key)

    def load_allogger_step(self, step):
        allogger.get_logger("root").step_per_key = allogger.get_logger("root").manager.dict(step)

    def log_config(self, env_cfg, runner_cfg, algorithm_cfg, policy_cfg):
        if self.use_wandb:
            wandb.config.update({"runner_cfg": class_to_dict(runner_cfg)})
            wandb.config.update({"policy_cfg": class_to_dict(policy_cfg)})
            wandb.config.update({"algorithm_cfg": class_to_dict(algorithm_cfg)})
            wandb.config.update({"env_cfg": class_to_dict(env_cfg)})
