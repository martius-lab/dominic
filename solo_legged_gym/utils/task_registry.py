# just for fun

import os
import json
from datetime import datetime
from solo_legged_gym import ROOT_DIR
from solo_legged_gym.runners.algorithms import *
from .helpers import class_to_dict, set_seed, parse_sim_params, update_env_cfg_from_args, update_train_cfg_from_args


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_cfgs(self, name):
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg

    def make_env(self, name, args, env_cfg=None):
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)

        env_cfg = update_env_cfg_from_args(env_cfg, args)
        set_seed(env_cfg.seed)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(cfg=env_cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name, args, env_cfg, log_root="default", train_cfg=None):
        create_and_save = False
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name)
            create_and_save = True
        train_cfg = update_train_cfg_from_args(train_cfg, args)

        if log_root == "default":
            log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + train_cfg.runner.run_name)

            if create_and_save:
                os.makedirs(log_dir)
                # update the json file for cluster running only!!!
                env_cfg_dict = class_to_dict(env_cfg)
                train_cfg_dict = class_to_dict(train_cfg)
                train_cfg_dict["runner"]["wandb"] = False
                train_cfg_dict["runner"]["on_cluster"] = True
                env_cfg_dict["viewer"]["enable_viewer"] = False
                cfg = {
                    "solo_legged_gym": {
                        "args": {
                            "dv": True,
                            "w": False,
                            "task": train_cfg.runner.experiment_name
                        },
                        "train_cfg": train_cfg_dict,
                        "env_cfg": env_cfg_dict
                    },
                    "working_dir": "./logs/cluster",
                    "id": 1
                }

                with open(os.path.join(ROOT_DIR, 'envs', train_cfg.runner.experiment_name + '/' + train_cfg.runner.experiment_name + '.json'), 'w') as f:
                    json.dump(cfg, f, indent=2)

                with open(os.path.join(log_dir + '/' + train_cfg.runner.experiment_name + '.json'), 'w') as f:
                    json.dump(cfg, f, indent=2)

        else:
            log_dir = log_root

        algorithm = eval(train_cfg.algorithm_name)
        runner = algorithm(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=args.device)
        return runner


# make global task registry
task_registry = TaskRegistry()
