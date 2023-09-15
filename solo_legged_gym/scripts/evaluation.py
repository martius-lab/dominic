import json
import time

from isaacgym import gymapi
from isaacgym.torch_utils import get_euler_xyz, quat_apply
import torch.nn.functional as func

from solo_legged_gym import ROOT_DIR
from solo_legged_gym.envs import task_registry
from solo_legged_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, get_load_path, \
    update_cfgs_from_dict, get_quat_yaw
import os
import numpy as np
import torch
import allogger
import logging

import threading
import csv


class keyboard_play:

    def __init__(self, args):
        workingd_dir = os.path.dirname(os.path.realpath(__file__))
        allogger.basic_configure(
            logdir=workingd_dir,
            default_outputs=["hdf"],
            manual_flush=True)
        self.logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO},
                                          default_outputs=['hdf'])

        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        eval_number = 100
        ROOT_DIR = os.path.join("/is/rg/al/Data/solo12_data/blm_579_alpha_l0/working_directories")
        load_path = get_load_path(
            ROOT_DIR,
            load_run=str(eval_number),
            checkpoint=-1,
        )
        print(f"Loading model from: {load_path}")

        # load_config_path = os.path.join(os.path.dirname(load_path), f"{train_cfg.runner.experiment_name}.json")
        # if os.path.isfile(load_config_path):
        #     f = open(load_config_path)
        #     load_config = json.load(f)
        #     update_cfgs_from_dict(env_cfg, train_cfg, load_config)

        env_cfg.seed = 2023  # evaluation seed
        env_cfg.env.num_envs = 4096  # evaluation seed
        env_cfg.env.evaluation = True
        env_cfg.env.plot_heights = False
        env_cfg.env.plot_target = False
        env_cfg.env.plot_colors = True
        env_cfg.terrain.train_all_together = 0

        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.env = env
        self.runner = task_registry.make_alg_runner(env=self.env, name=args.task, args=args, env_cfg=env_cfg,
                                                    train_cfg=train_cfg)
        self.obs = self.env.get_observations()
        self.runner.load(load_path)
        self.policy = self.runner.get_inference_policy(device=self.env.device)

        self.feature_history = torch.zeros(int(self.env.max_episode_length),
                                           self.env.num_envs,
                                           self.env.num_features,
                                           device=self.env.device,
                                           dtype=torch.float)
        self.feat_gamma = 0.95
        self.successor = torch.zeros(int(self.env.max_episode_length),
                                     self.env.num_envs,
                                     self.env.num_features,
                                     device=self.env.device,
                                     dtype=torch.float)
        self.mean_init_sf = torch.zeros(8, env.num_features, device=env.device, dtype=torch.float)
        self.mean_every_sf = torch.zeros(8, env.num_features, device=env.device, dtype=torch.float)

        self.nearest_init_dist = torch.zeros(8, device=env.device, dtype=torch.float)
        self.avg_nearest_init_dist = 0
        self.all_init_dist = torch.zeros(8, 8, device=env.device, dtype=torch.float)
        self.avg_init_dist = 0

        self.nearest_every_dist = torch.zeros(8, device=env.device, dtype=torch.float)
        self.avg_nearest_every_dist = 0
        self.all_every_dist = torch.zeros(8, 8, device=env.device, dtype=torch.float)
        self.avg_every_dist = 0

    def play(self):
        avg_nearest_init_dists = []
        avg_init_dists = []
        avg_nearest_every_dists = []
        avg_every_dists = []
        for _ in range(1):
            for i in range(8):
                self.env.eval_skill[:] = i
                self.successor[:] = 0
                self.env.reset()

                for j in range(int(self.env.max_episode_length)):
                    print("skill: ", i, "step: ", j)
                    self.step(j)
                self.calculate_successor(i)
            self.calculate_metrics()
            avg_nearest_init_dists.append(self.avg_nearest_init_dist.detach().cpu().numpy())
            avg_init_dists.append(self.avg_init_dist.detach().cpu().numpy())
            avg_nearest_every_dists.append(self.avg_nearest_every_dist.detach().cpu().numpy())
            avg_every_dists.append(self.avg_every_dist.detach().cpu().numpy())

        avg_nearest_init_dists_array = np.array(avg_nearest_init_dists)
        avg_init_dists_array = np.array(avg_init_dists)
        avg_nearest_every_dists_array = np.array(avg_nearest_every_dists)
        avg_every_dists_array = np.array(avg_every_dists)
        print("mean_avg_nearest_init_dists_array: ", np.mean(avg_nearest_init_dists_array))
        print("mean_avg_init_dists_array: ", np.mean(avg_init_dists_array))
        print("mean_avg_nearest_every_dists_array: ", np.mean(avg_nearest_every_dists_array))
        print("mean_avg_every_dists_array: ", np.mean(avg_every_dists_array))
        print("var_avg_nearest_init_dists_array: ", np.var(avg_nearest_init_dists_array))
        print("var_avg_init_dists_array: ", np.var(avg_init_dists_array))
        print("var_avg_nearest_every_dists_array: ", np.var(avg_nearest_every_dists_array))
        print("var_avg_every_dists_array: ", np.var(avg_every_dists_array))

    def step(self, j):
        obs_skills = (self.obs.detach(), self.encode_skills(self.env.skills))

        self.obs, skills, features, _, done, _ = self.env.step(self.policy(obs_skills).detach())
        self.feature_history[j] = features
        # print(skills.detach().cpu().numpy())
        # print(features.detach().cpu().numpy())
        # print(done.detach().cpu().numpy())

    def calculate_successor(self, skill):
        for i in reversed(range(int(self.env.max_episode_length))):
            if i == int(self.env.max_episode_length) - 1:
                self.successor[i] = self.feature_history[i]
            else:
                self.successor[i] = self.feature_history[i] + self.feat_gamma * self.successor[i + 1]

        self.mean_init_sf[skill] = torch.mean(self.successor[0], dim=0)
        self.mean_every_sf[skill] = torch.mean(torch.mean(self.successor, dim=1), dim=0)
        print("skill: ", skill)
        print(self.mean_init_sf[skill].detach().cpu().numpy())
        print(self.mean_every_sf[skill].detach().cpu().numpy())

    def calculate_metrics(self):
        for i in range(8):
            init_sf = self.mean_init_sf[i]
            every_sf = self.mean_every_sf[i]

            for j in range(8):
                self.all_init_dist[i, j] = torch.dist(init_sf, self.mean_init_sf[j])  # default p=2
                self.all_every_dist[i, j] = torch.dist(every_sf, self.mean_every_sf[j])

                self.nearest_init_dist[i], _ = torch.kthvalue(self.all_init_dist[i, :], k=2)
                self.nearest_every_dist[i], _ = torch.kthvalue(self.all_every_dist[i, :], k=2)

        self.avg_nearest_init_dist = torch.mean(self.nearest_init_dist)
        self.avg_init_dist = torch.mean(self.all_init_dist)
        self.avg_nearest_every_dist = torch.mean(self.nearest_every_dist)
        self.avg_every_dist = torch.mean(self.all_every_dist)

        print("avg_nearest_init_dist: ", self.avg_nearest_init_dist.detach().cpu().numpy())
        print("avg_init_dist: ", self.avg_init_dist.detach().cpu().numpy())
        print("avg_nearest_every_dist: ", self.avg_nearest_every_dist.detach().cpu().numpy())
        print("avg_every_dist: ", self.avg_every_dist.detach().cpu().numpy())

        self.logger.log(self.avg_nearest_init_dist.detach().cpu().numpy(), "avg_nearest_init_dist", to_hdf=True)
        self.logger.log(self.avg_init_dist.detach().cpu().numpy(), "avg_init_dist", to_hdf=True)
        self.logger.log(self.avg_nearest_every_dist.detach().cpu().numpy(), "avg_nearest_every_dist", to_hdf=True)
        self.logger.log(self.avg_every_dist.detach().cpu().numpy(), "avg_every_dist", to_hdf=True)
        allogger.get_root().flush(children=True)
        allogger.close()

    def encode_skills(self, skills):
        return func.one_hot(skills, num_classes=self.env.num_skills).squeeze(1)


if __name__ == "__main__":
    args = get_args()
    kp = keyboard_play(args)
    kp.play()
