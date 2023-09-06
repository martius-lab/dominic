import json
import logging
import time
import os
from collections import deque
import statistics
import numpy as np
import wandb
import cluster

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.tensorboard import SummaryWriter

from solo_legged_gym.utils import class_to_dict
from solo_legged_gym.utils.wandb_utils import WandbSummaryWriter
from solo_legged_gym.runners.modules.masked_policy import MaskedPolicy
from solo_legged_gym.runners.modules.masked_value import MaskedValue
from solo_legged_gym.runners.modules.normalizer import EmpiricalNormalization
from solo_legged_gym.runners.modules.masked_successor_feature import MaskedSuccessorFeature
from solo_legged_gym.runners.algorithms.domino.rollout_buffer import RolloutBuffer

torch.autograd.set_detect_anomaly(True)


class DOMINO:
    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.r_cfg = train_cfg.runner
        self.a_cfg = train_cfg.algorithm
        self.n_cfg = train_cfg.network
        self.device = device
        self.env = env
        self.num_ext_values = self.env.cfg.rewards.num_groups

        # set up the networks
        self.policy = MaskedPolicy(num_obs=self.env.num_obs,
                                   num_skills=self.env.num_skills,
                                   num_actions=self.env.num_actions,
                                   drop_out_rate=self.n_cfg.drop_out_rate,
                                   hidden_dims=self.n_cfg.policy_hidden_dims,
                                   activation=self.n_cfg.policy_activation,
                                   device=self.device,
                                   init_log_std=self.n_cfg.init_log_std).to(self.device)

        self.ext_values = [MaskedValue(num_obs=self.env.num_obs,
                                       num_skills=self.env.num_skills,
                                       drop_out_rate=self.n_cfg.drop_out_rate,
                                       hidden_dims=self.n_cfg.value_hidden_dims,
                                       activation=self.n_cfg.value_activation,
                                       device=self.device)
                           for _ in range(self.num_ext_values)]

        self.int_value = MaskedValue(num_obs=self.env.num_obs,
                                     num_skills=self.env.num_skills,
                                     drop_out_rate=self.n_cfg.drop_out_rate,
                                     hidden_dims=self.n_cfg.value_hidden_dims,
                                     activation=self.n_cfg.value_activation,
                                     device=self.device)

        # set up Lagrangian multipliers
        # There should be num_skills Lagrangian multipliers, but we fixed the first one to be sig(la_0) = 1
        self.lagrange = [nn.Parameter(torch.ones(self.env.num_skills - 1, device=self.device) * 0.0,
                                      requires_grad=True).cuda() for _ in range(self.num_ext_values)]

        if self.a_cfg.use_succ_feat:
            self.succ_feat = MaskedSuccessorFeature(num_obs=self.env.num_obs,
                                                    num_skills=self.env.num_skills,
                                                    num_features=self.env.num_features,
                                                    drop_out_rate=self.n_cfg.drop_out_rate,
                                                    hidden_dims=self.n_cfg.succ_feat_hidden_dims,
                                                    activation=self.n_cfg.succ_feat_activation,
                                                    device=self.device)

            self.succ_feat_lr = self.a_cfg.succ_feat_lr
            self.succ_feat_optimizer = optim.Adam(list(self.succ_feat.parameters()), lr=self.succ_feat_lr)

        else:
            self.avg_features = torch.ones(self.env.num_skills, self.env.num_features, device=self.device,
                                           requires_grad=False) * (1 / self.env.num_features)

        # set up moving averages
        self.avg_ext_values = [torch.zeros(self.env.num_skills, device=self.device, requires_grad=False) for _ in
                               range(self.num_ext_values)]
        if self.a_cfg.pretrain_expert:
            self.avg_expert_ext_values = self.a_cfg.expert_ext_values
        else:
            self.avg_expert_ext_values = torch.zeros(self.num_ext_values, device=self.device, requires_grad=False)

        self.num_steps_per_env = self.r_cfg.num_steps_per_env
        self.save_interval = self.r_cfg.save_interval
        self.restart_interval = self.r_cfg.restart_interval

        self.normalize_observation = self.r_cfg.normalize_observation
        if self.normalize_observation:
            self.obs_normalizer = EmpiricalNormalization(shape=self.env.num_obs,
                                                         until=int(1.0e8)).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization

        self.normalize_features = self.r_cfg.normalize_features
        if self.normalize_features:
            self.feat_normalizer = EmpiricalNormalization(shape=self.env.num_features,
                                                          until=int(1.0e8)).to(self.device)
        else:
            self.feat_normalizer = torch.nn.Identity()  # no normalization

        # set up rollout buffer
        self.rollout_buffer = RolloutBuffer(num_envs=self.env.num_envs,
                                            num_transitions_per_env=self.num_steps_per_env,
                                            obs_shape=[self.env.num_obs],
                                            actions_shape=[self.env.num_actions],
                                            features_shape=[self.env.num_features],
                                            num_ext_values=self.num_ext_values,
                                            use_succ_feat=self.a_cfg.use_succ_feat,
                                            device=self.device)
        self.transition = RolloutBuffer.Transition()

        # set up optimizers
        self.policy_lr = self.a_cfg.policy_lr
        policy_params_list = list(self.policy.parameters())
        self.policy_optimizer = optim.Adam(policy_params_list, lr=self.policy_lr)

        self.value_lr = self.a_cfg.value_lr
        value_params_list = list(self.int_value.parameters())
        for i in range(self.num_ext_values):
            value_params_list += list(self.ext_values[i].parameters())
        self.value_optimizer = optim.Adam(value_params_list, lr=self.value_lr)

        self.lagrange_lr = self.a_cfg.lagrange_lr
        self.lagrange_optimizer = optim.Adam(self.lagrange, lr=self.lagrange_lr)

        self.num_learning_iterations = self.r_cfg.max_iterations
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        self.resume_id = None
        self.writer = None
        self.logger = None
        self.logger_state = None

        # resume
        self.log_dir = log_dir
        self.resume = False

        if not self.env.cfg.env.play:
            models = [file for file in os.listdir(log_dir) if 'model' in file]
            if len(models) == 0:
                print("[Resume] No models in this directory: " + log_dir + ", starting from scratch!")
            else:
                models.sort(key=lambda m: '{0:0>15}'.format(m))
                model = models[-1]
                load_path = os.path.join(log_dir, model)
                print("[Resume] Load model from: " + load_path)

                self.load(path=load_path,
                          load_values=True,
                          load_feat=True,
                          load_optimizer=True,
                          load_feat_normalizer=True,
                          load_curriculum=True,
                          load_init_obs=True)
                self.resume = True

        # Log
        if not self.env.cfg.env.play:
            self.init_writer(resume_id=self.resume_id)  # should be updated if resumed
            if self.resume:
                if self.r_cfg.wandb:
                    self.writer.load_allogger_step(self.logger_state)
            else:
                try:
                    self.resume_id = self.writer.run_id
                except:
                    self.resume_id = None

        if self.a_cfg.burning_expert_steps != 0:
            self.burning_expert = True
        else:
            self.burning_expert = False

        self.env.reset()
        if self.resume:
            self.current_learning_iteration += 1
        self.iter = self.current_learning_iteration
        self.avg_score = 0

    def learn(self):
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )

        new_obs = self.obs_normalizer(self.env.get_observations().to(self.device))
        new_skills = self.env.skills.to(self.device)

        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []

        ext_rew_buffers = [deque(maxlen=100) for _ in range(self.num_ext_values)]
        int_rew_buffer = deque(maxlen=100)
        dist_buffer = deque(maxlen=10000)
        len_buffer = deque(maxlen=100)

        cur_ext_rew_sums = [torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                            for _ in range(self.num_ext_values)]
        cur_int_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_dist_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # filming = False
        # filming_imgs = []
        # filming_iter_counter = 0

        collection_time = 0
        learn_time = 0
        misc_time = 0

        for it in range(self.current_learning_iteration, self.num_learning_iterations):
            # Collect
            start = time.time()

            # # filming
            # if self.r_cfg.record_gif and (it % self.r_cfg.record_gif_interval == 0):
            #     filming = True

            # burning expert
            if it <= self.a_cfg.burning_expert_steps:
                self.burning_expert = True
            else:
                self.burning_expert = False

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    obs = new_obs
                    skills = new_skills

                    # use last obs to get the actions
                    actions, log_prob = \
                        self.policy.act_and_log_prob((obs, self.encode_skills(skills)))

                    new_obs, new_skills, features, group_rew, dones, infos = self.env.step(actions)
                    ext_rews = [group_rew[:, i] for i in range(self.num_ext_values)]
                    # features should be part of the outcome of the actions
                    features = self.feat_normalizer(features)
                    int_rew, dist = self.get_intrinsic_reward(skills, obs, features, self.a_cfg.intrinsic_rew_scale)
                    self.process_env_step(obs, actions, ext_rews, int_rew, skills, features, log_prob, dones, infos,
                                          self.a_cfg.bootstrap_value)
                    # normalize new obs
                    new_obs = self.obs_normalizer(new_obs)

                    # if filming:
                    #     filming_imgs.append(self.env.camera_image)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        for j in range(self.num_ext_values):
                            cur_ext_rew_sums[j] += ext_rews[j]
                            ext_rew_buffers[j].extend(cur_ext_rew_sums[j][new_ids][:, 0].cpu().numpy().tolist())
                            cur_ext_rew_sums[j][new_ids] = 0

                        cur_int_rew_sum += int_rew
                        int_rew_buffer.extend(cur_int_rew_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_int_rew_sum[new_ids] = 0

                        cur_dist_sum += dist
                        avg_dist_per_step = cur_dist_sum / (cur_episode_length + 1)
                        dist_buffer.extend(avg_dist_per_step.cpu().numpy().tolist())
                        cur_dist_sum[new_ids] = 0

                        cur_episode_length += 1
                        len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_episode_length[new_ids] = 0

                obs_skills = (obs, self.encode_skills(skills))
                last_ext_values = []
                for i in range(self.num_ext_values):
                    last_ext_values.append(self.ext_values[i](obs_skills).detach())
                last_int_values = self.int_value(obs_skills).detach()
                self.rollout_buffer.compute_returns(last_ext_values, last_int_values, self.a_cfg.gamma, self.a_cfg.lam)

                if self.a_cfg.use_succ_feat:
                    last_succ_feat = self.succ_feat(obs_skills).detach()
                    self.rollout_buffer.compute_succ_feat_target(last_succ_feat, self.a_cfg.succ_feat_gamma)

            stop = time.time()
            collection_time = stop - start

            # Learning update
            start = stop
            mean_ext_value_loss, mean_int_value_loss, mean_surrogate_loss, mean_succ_feat_loss, \
                mean_lagranges, mean_lagrange_coeffs, mean_constraint_satisfaction = self.update()

            with torch.inference_mode():
                avg_nearest_dist = self.get_avg_dist(self.obs_normalizer(self.env.init_obs_buf))

            stop = time.time()
            learn_time = stop - start

            # if filming:
            #     filming_iter_counter += 1
            #     if filming_iter_counter == self.r_cfg.record_iters:
            #         export_imgs = np.array(filming_imgs)
            #         if self.r_cfg.wandb:
            #             wandb.log({'Video': wandb.Video(export_imgs.transpose(0, 3, 1, 2), fps=50, format="mp4")})
            #         del export_imgs
            #         filming = False
            #         filming_imgs = []
            #         filming_iter_counter = 0

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), it)
            self.iter = it
            ep_infos.clear()

            if (it % self.restart_interval == 0 and
                    self.r_cfg.on_cluster and
                    it != self.current_learning_iteration and
                    it != self.num_learning_iterations - 1):
                print("Triggering cluster restart...")
                wandb.alert(title='Restart', text='Restarting the job at iteration {}'.format(it))
                self.writer.stop()
                cluster.exit_for_resume()

        self.current_learning_iteration = self.iter
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),
                  self.current_learning_iteration)
        self.writer.stop()
        # score not implemented yet
        return 0

    def process_env_step(self, obs, actions, ext_rews, int_rew, skills, features, log_prob, dones, infos,
                         bootstrap=True):
        obs_skills = (obs, self.encode_skills(skills))

        self.transition.observations = obs

        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        self.transition.ext_values = []
        for i in range(self.num_ext_values):
            self.transition.ext_values.append(self.ext_values[i](obs_skills).detach())
        self.transition.int_values = self.int_value(obs_skills).detach()

        self.transition.ext_rews = [ext_rews[i].clone() for i in range(self.num_ext_values)]
        self.transition.int_rew = int_rew.clone()

        if bootstrap:
            for i in range(self.num_ext_values):
                self.transition.ext_rews[i] += self.a_cfg.gamma * torch.squeeze(
                    self.transition.ext_values[i] * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.int_rew += self.a_cfg.gamma * torch.squeeze(
                self.transition.int_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.transition.dones = dones
        self.transition.skills = skills
        self.transition.features = features
        if self.a_cfg.use_succ_feat:
            self.transition.succ_feat = self.succ_feat(obs_skills).detach()

        self.rollout_buffer.add_transitions(self.transition)
        self.transition.clear()

    def get_intrinsic_reward(self, skills, obs, features, intrinsic_rew_scale):
        if self.a_cfg.use_succ_feat:
            n_skills = self.env.num_skills
            n_samples = obs.size(0)
            n_obs = obs.size(1)
            all_encoded_skills = self.encode_skills(torch.arange(n_skills).to(self.device))
            all_sfs = self.succ_feat((obs.unsqueeze(1).repeat(1, n_skills, 1).view(-1, n_obs),
                                      all_encoded_skills.unsqueeze(0).repeat(n_samples, 1, 1).view(-1, n_skills))).view(
                n_samples, n_skills, -1)
            sfs = all_sfs[torch.arange(all_sfs.size(0)), skills, :]
            sfs_dist = torch.norm((sfs.unsqueeze(1).repeat(1, self.env.num_skills, 1) - all_sfs), dim=2, p=2)

            _, nearest_sfs_idx = torch.kthvalue(sfs_dist, k=2, dim=-1)  # num_samples
            nearst_sfs = all_sfs[torch.arange(all_sfs.size(0)), nearest_sfs_idx, :]
            psi_diff = sfs - nearst_sfs
        else:
            afs = self.avg_features[skills]  # num_samples * num_features
            afs_dist = torch.norm((afs.unsqueeze(1).repeat(1, self.env.num_skills, 1) -
                                   self.avg_features.unsqueeze(0).repeat(self.env.num_envs, 1, 1)),
                                  dim=2, p=2)

            _, nearest_afs_idx = torch.kthvalue(afs_dist, k=2, dim=-1)  # num_samples
            nearst_afs = self.avg_features[nearest_afs_idx]  # num_samples * num_features
            psi_diff = afs - nearst_afs

        dist = torch.norm(psi_diff, p=2, dim=-1)
        norm_diff = dist / self.a_cfg.target_dist
        c = (1 - self.a_cfg.attractive_coeff) * torch.pow(norm_diff, self.a_cfg.repulsive_power) - \
            self.a_cfg.attractive_coeff * torch.pow(norm_diff, self.a_cfg.attractive_power)
        int_rew = intrinsic_rew_scale * c * torch.sum(features * psi_diff, dim=-1) / self.env.num_features
        return int_rew, dist

    def get_avg_dist(self, obs):
        n_skills = self.env.num_skills
        n_samples = obs.size(0)
        n_obs = obs.size(1)
        nearest_dists = torch.zeros(n_samples, n_skills, device=self.device)

        if self.a_cfg.use_succ_feat:
            all_encoded_skills = self.encode_skills(torch.arange(n_skills).to(self.device))
            all_sfs = self.succ_feat((obs.unsqueeze(1).repeat(1, n_skills, 1).view(-1, n_obs),
                                      all_encoded_skills.unsqueeze(0).repeat(n_samples, 1, 1).view(-1, n_skills))).view(
                n_samples, n_skills, -1)  # num_samples * num_skills * num_features
            for i in range(n_skills):
                sfs = all_sfs[:, i, :]
                sfs_dist = torch.norm((sfs.unsqueeze(1).repeat(1, self.env.num_skills, 1) - all_sfs), dim=2, p=2)
                _, nearest_sfs_idx = torch.kthvalue(sfs_dist, k=2, dim=-1)  # num_samples
                nearst_sfs = all_sfs[torch.arange(all_sfs.size(0)), nearest_sfs_idx, :]
                psi_diff = sfs - nearst_sfs
                nearest_dists[:, i] = torch.norm(psi_diff, p=2, dim=-1)

        else:
            for i in range(n_skills):
                afs = self.avg_features[i]  # num_samples * num_features
                afs_dist = torch.norm((afs.unsqueeze(1).repeat(1, self.env.num_skills, 1) -
                                       self.avg_features.unsqueeze(0).repeat(self.env.num_envs, 1, 1)),
                                      dim=2, p=2)
                _, nearest_afs_idx = torch.kthvalue(afs_dist, k=2, dim=-1)  # num_samples
                nearst_afs = self.avg_features[nearest_afs_idx]  # num_samples * num_features
                psi_diff = afs - nearst_afs
                nearest_dists[:, i] = torch.norm(psi_diff, p=2, dim=-1)

        return torch.mean(nearest_dists)

    def get_lagrange_coeff(self, skills, burning_expert):
        lagrange_coeff = [torch.zeros_like(skills).to(torch.float32) for _ in range(self.num_ext_values)]
        for i in range(self.num_ext_values):
            lagrange_coeff[i][skills > 0] = self.lagrange[i][skills[skills > 0] - 1]
            lagrange_coeff[i] = torch.sigmoid(lagrange_coeff[i] * self.a_cfg.sigmoid_scale)
            if burning_expert:
                lagrange_coeff[i][:] = 1
            else:
                lagrange_coeff[i][skills == 0] = 1  # with only extrinsic reward
        return lagrange_coeff

    def update_value_moving_avg(self, skills, ext_returns):
        encoded_skills = func.one_hot(skills, num_classes=self.env.num_skills)
        encoded_ext_returns = [encoded_skills * ext_returns[i].repeat(1, self.env.num_skills) for i in
                               range(self.num_ext_values)]
        mean_encoded_ext_returns = [torch.nan_to_num(encoded_ext_returns[i].sum(dim=0) / encoded_skills.sum(dim=0)) for
                                    i in range(self.num_ext_values)]
        self.avg_ext_values = [self.a_cfg.avg_values_decay_factor * self.avg_ext_values[i] + \
                               (1 - self.a_cfg.avg_values_decay_factor) * mean_encoded_ext_returns[i] for i in
                               range(self.num_ext_values)]

    def update_feature_moving_avg(self, skills, features):
        encoded_skills = func.one_hot(skills, num_classes=self.env.num_skills)
        encoded_features = encoded_skills.unsqueeze(-1) * features.unsqueeze(1).repeat(1, self.env.num_skills, 1)
        mean_encoded_features = torch.nan_to_num(encoded_features.sum(dim=0) / encoded_skills.sum(dim=0).unsqueeze(-1),
                                                 nan=(1 / self.env.num_features))

        self.avg_features = self.a_cfg.avg_features_decay_factor * self.avg_features + \
                            (1 - self.a_cfg.avg_features_decay_factor) * mean_encoded_features

    def encode_skills(self, skills):
        return (func.one_hot(skills, num_classes=self.env.num_skills)).squeeze(1)

    def update(self):
        # for logging
        mean_ext_value_loss = [0 for _ in range(self.num_ext_values)]
        mean_int_value_loss = 0
        mean_surrogate_loss = 0
        mean_succ_feat_loss = 0
        mean_lagranges = [np.zeros(self.env.num_skills - 1) for _ in range(self.num_ext_values)]
        mean_lagrange_coeffs = [np.zeros(self.env.num_skills) for _ in range(self.num_ext_values)]
        mean_constraint_satisfaction = [np.zeros(self.env.num_skills - 1) for _ in range(self.num_ext_values)]

        generator = self.rollout_buffer.mini_batch_generator(self.a_cfg.num_mini_batches,
                                                             self.a_cfg.num_learning_epochs)

        for (obs,
             actions,
             target_ext_values,
             target_int_values,
             ext_advantages,
             int_advantages,
             ext_returns,
             int_returns,
             old_actions_log_prob,
             old_mu,
             old_sigma,
             skills,
             features,
             succ_feat_target,
             ) in generator:

            ############################################################################################################
            # PPO step
            # using the current policy to get action log prob
            obs_skills = (obs, self.encode_skills(skills))
            self.policy.update_distribution(obs_skills)
            actions_log_prob_batch = self.policy.distribution.log_prob(actions)

            # get values
            ext_values = []
            for i in range(self.num_ext_values):
                ext_values.append(self.ext_values[i](obs_skills))
            int_value = self.int_value(obs_skills)

            # update moving average
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # combine advantages
            with torch.inference_mode():
                lagrange_coeff = self.get_lagrange_coeff(skills, self.burning_expert)

            # The most important part of the algorithm
            advantages = eval(self.a_cfg.fixed_adv_coeff)[0] * ext_advantages[0] * lagrange_coeff[0]  # task
            advantages += eval(self.a_cfg.fixed_adv_coeff)[1] * ext_advantages[1] * lagrange_coeff[1]  # regular
            advantages += eval(self.a_cfg.fixed_adv_coeff)[2] * ext_advantages[2] * lagrange_coeff[2]  # loose
            advantages += self.a_cfg.intrinsic_adv_coeff * int_advantages * (
                        1.0 - torch.max(torch.stack(lagrange_coeff), dim=0)[0])  # intrinsic

            # Using KL to adaptively changing the learning rate
            if self.a_cfg.desired_kl is not None and self.a_cfg.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma + 1.0e-5)
                        + (torch.square(old_sigma) + torch.square(old_mu - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.a_cfg.desired_kl * 2.0:
                        self.policy_lr = max(1e-5, self.policy_lr / 1.5)
                    elif self.a_cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.policy_lr = min(5e-2, self.policy_lr * 1.5)

                    for param_group in self.policy_optimizer.param_groups:
                        param_group["lr"] = self.policy_lr

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob))
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, 1.0 - self.a_cfg.clip_param, 1.0 + self.a_cfg.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            policy_loss = surrogate_loss - \
                          self.a_cfg.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_params_list = list(self.policy.parameters())
            nn.utils.clip_grad_norm_(policy_params_list, self.a_cfg.max_grad_norm)  # clip grad norm
            self.policy_optimizer.step()

            ############################################################################################################
            # Value function loss
            ext_value_loss = [0 for _ in range(self.num_ext_values)]

            if self.a_cfg.use_clipped_value_loss:
                for i in range(self.num_ext_values):
                    ext_value_clipped = target_ext_values[i] + (ext_values[i] - target_ext_values[i]).clamp(
                        -self.a_cfg.clip_param, self.a_cfg.clip_param)
                    ext_value_loss[i] = torch.max((ext_values[i] - ext_returns[i]).pow(2),
                                                  (ext_value_clipped - ext_returns[i]).pow(2)).mean()

                int_value_clipped = target_int_values + (int_value - target_int_values).clamp(
                    -self.a_cfg.clip_param, self.a_cfg.clip_param
                )

                int_value_loss = torch.max((int_value - int_returns).pow(2),
                                           (int_value_clipped - int_returns).pow(2)).mean()
            else:
                for i in range(self.num_ext_values):
                    ext_value_loss[i] = (ext_returns[i] - ext_values[i]).pow(2).mean()
                int_value_loss = (int_returns - int_value).pow(2).mean()

            value_loss = 0
            for i in range(self.num_ext_values):
                value_loss += ext_value_loss[i]

            if not self.burning_expert:
                value_loss += int_value_loss

            # Gradient step
            self.value_optimizer.zero_grad()
            value_loss.backward()
            value_params_list = list(self.int_value.parameters())
            for i in range(self.num_ext_values):
                value_params_list += list(self.ext_values[i].parameters())
            nn.utils.clip_grad_norm_(value_params_list, self.a_cfg.max_grad_norm)  # clip grad norm
            self.value_optimizer.step()

            ############################################################################################################
            # learn lagrange multipliers
            if not self.burning_expert:
                for _ in range(self.a_cfg.num_lagrange_steps):
                    # Lagrange loss
                    lagrange_loss = 0.0
                    for i in range(self.num_ext_values):
                        lagrange_losses = self.lagrange[i] * (
                                self.avg_ext_values[i][1:] - eval(self.a_cfg.alpha)[i] * self.avg_ext_values[i][0]).squeeze(
                            -1)
                        lagrange_loss += torch.sum(lagrange_losses, dim=-1)
                    lagrange_loss.backward()
                    self.lagrange_optimizer.step()

                # clipping Lagrange multipliers if needed
                if self.a_cfg.clip_lagrange is not None:
                    if 'auto' in self.a_cfg.clip_lagrange:
                        if len(self.a_cfg.clip_lagrange.split('_')) == 2:
                            clip_lagrange_threshold = float(
                                self.a_cfg.clip_lagrange.split('_')[1]) / self.a_cfg.sigmoid_scale
                        else:
                            clip_lagrange_threshold = 5 / self.a_cfg.sigmoid_scale
                    else:
                        clip_lagrange_threshold = self.a_cfg.clip_lagrange
                    for i in range(self.num_ext_values):
                        self.lagrange[i].data = torch.clamp(self.lagrange[i].data,
                                                            min=-clip_lagrange_threshold,
                                                            max=clip_lagrange_threshold)

            ############################################################################################################
            # update features sfs/afs
            if self.a_cfg.use_succ_feat:
                succ_feat = self.succ_feat(obs_skills)
                succ_feat_loss = (succ_feat - succ_feat_target).pow(2).mean()
                self.succ_feat_optimizer.zero_grad()
                succ_feat_loss.backward()
                self.succ_feat_optimizer.step()
            else:
                self.update_feature_moving_avg(skills.squeeze(-1),
                                               features)

            ############################################################################################################
            # update value moving average
            self.update_value_moving_avg(skills.squeeze(-1), ext_returns)

            if self.a_cfg.pretrain_expert:
                for i in range(self.num_ext_values):
                    self.avg_ext_values[i][0] = self.avg_expert_ext_values[i]  # assign expert value to pretrained value
            else:
                for i in range(self.num_ext_values):
                    if self.burning_expert:
                        self.avg_expert_ext_values[i] = self.avg_ext_values[i][0].item()  # copy the value as expert
                    else:
                        self.avg_ext_values[i][0] = self.avg_expert_ext_values[i]  # fix expert to be the end of burning

            ############################################################################################################
            # logging
            for i in range(self.num_ext_values):
                mean_ext_value_loss[i] += ext_value_loss[i].item()
                mean_lagranges[i] += self.lagrange[i].detach().cpu().numpy()
                mean_lagrange_coeffs[i] += (torch.sum(func.one_hot(skills.squeeze(1)) * lagrange_coeff[i], dim=0) /
                                            torch.sum(func.one_hot(skills.squeeze(1)), dim=0)).detach().cpu().numpy()
                mean_constraint_satisfaction[i] += (
                        self.avg_ext_values[i][1:] - eval(self.a_cfg.alpha)[i] * self.avg_ext_values[i][0]).detach().cpu().numpy()
            mean_int_value_loss += int_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if self.a_cfg.use_succ_feat:
                mean_succ_feat_loss += succ_feat_loss.item()

        num_updates = self.a_cfg.num_learning_epochs * self.a_cfg.num_mini_batches
        mean_ext_value_loss = [i / num_updates for i in mean_ext_value_loss]
        mean_int_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if self.a_cfg.use_succ_feat:
            mean_succ_feat_loss /= num_updates
        mean_lagranges = [i / num_updates for i in mean_lagranges]
        mean_lagrange_coeffs = [i / num_updates for i in mean_lagrange_coeffs]
        mean_constraint_satisfaction = [i / num_updates for i in mean_constraint_satisfaction]

        # clear buffer
        self.rollout_buffer.clear()

        return mean_ext_value_loss, mean_int_value_loss, mean_surrogate_loss, mean_succ_feat_loss, \
            mean_lagranges, mean_lagrange_coeffs, mean_constraint_satisfaction

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                info_tensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, ep_info[key].to(self.device)))
                value = torch.mean(info_tensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.policy.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        mean_ext_value_losses = locs['mean_ext_value_loss']
        for i in range(len(mean_ext_value_losses)):
            self.writer.add_scalar(f'Learning/ext_value_function_loss_{i}', mean_ext_value_losses[i], locs['it'])
        self.writer.add_scalar('Learning/int_value_function_loss', locs['mean_int_value_loss'], locs['it'])
        self.writer.add_scalar('Learning/surrogate_loss', locs['mean_surrogate_loss'], locs['it'])
        if self.a_cfg.use_succ_feat:
            self.writer.add_scalar('Learning/success_feature_loss', locs['mean_succ_feat_loss'], locs['it'])

        mean_constraint_satisfaction = locs['mean_constraint_satisfaction']
        mean_lagrange_coeffs = locs['mean_lagrange_coeffs']
        mean_lagranges = locs['mean_lagranges']
        for i in range(self.num_ext_values):
            for j in range(self.env.num_skills - 1):
                self.writer.add_scalar(f'Constraint/constraint_satisfaction_rew{i}_skill{j}',
                                       mean_constraint_satisfaction[i][j], locs['it'])
                self.writer.add_scalar(f'Skill/lagrange_rew{i}_skill{j}', mean_lagranges[i][j], locs['it'])
                self.writer.add_scalar(f'Skill/lagrange_coeff_rew{i}_skill{j}', mean_lagrange_coeffs[i][j + 1],
                                       locs['it'])
            for j in range(self.env.num_skills):
                self.writer.add_scalar(f'Constraint/avg_ext_values_rew{i}_skill{j}', self.avg_ext_values[i][j],
                                       locs['it'])

        self.writer.add_scalar('Learning/policy_lr', self.policy_lr, locs['it'])
        self.writer.add_scalar('Learning/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        self.writer.add_scalar('Train/mean_curriculum_cols', np.mean(self.env.terrain_cols.detach().cpu().numpy()),
                               locs['it'])
        self.writer.add_scalar('Train/mean_curriculum_rows', np.mean(self.env.terrain_rows.detach().cpu().numpy()),
                               locs['it'])
        self.writer.add_scalar('Train/std_curriculum_cols', np.std(self.env.terrain_cols.detach().cpu().numpy()),
                               locs['it'])

        if len(locs['len_buffer']) > 0:
            ext_rew_bufs = locs['ext_rew_buffers']
            for i in range(self.num_ext_values):
                self.writer.add_scalar(f'Train/mean_ext_reward_{i}', statistics.mean(ext_rew_bufs[i]), locs['it'])
            self.writer.add_scalar('Train/mean_intrinsic_reward', statistics.mean(locs['int_rew_buffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['len_buffer']), locs['it'])

        self.writer.add_scalar('Feature/avg_nearest_dist_per_step', statistics.mean(locs['dist_buffer']), locs['it'])
        self.writer.add_scalar('Feature/avg_nearest_dist', locs['avg_nearest_dist'], locs['it'])

        if self.r_cfg.wandb:
            self.writer.flush_logger()

        title = f" \033[1m Learning iteration {locs['it']}/{self.num_learning_iterations} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{title.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                          'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Burning expert:':>{pad}} {self.burning_expert}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               self.num_learning_iterations - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, it, infos=None):
        if self.r_cfg.wandb:
            self.logger_state = self.writer.get_allogger_step()
        saved_dict = {
            "policy_state_dict": self.policy.state_dict(),
            "intrinsic_value_state_dict": self.int_value.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "lagrange_optimizer_state_dict": self.lagrange_optimizer.state_dict(),
            "lagrange": self.lagrange,
            "avg_ext_values": self.avg_ext_values,
            "infos": infos,
            "iteration": it,
            "tot_timesteps": self.tot_timesteps,
            "tot_time": self.tot_time,
            "logger_state": self.logger_state,
            "resume_id": self.resume_id,
            "curriculum": [self.env.terrain_cols, self.env.terrain_rows],
            "init_obs": self.env.init_obs_buf,
        }
        for i in range(self.num_ext_values):
            saved_dict[f"ext_value_{i}_state_dict"] = self.ext_values[i].state_dict()
        if self.normalize_observation:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        if self.normalize_features:
            saved_dict["feat_norm_state_dict"] = self.feat_normalizer.state_dict()
        if self.a_cfg.use_succ_feat:
            saved_dict["succ_feat_state_dict"] = self.succ_feat.state_dict()
            saved_dict["succ_feat_optimizer_state_dict"] = self.succ_feat_optimizer.state_dict()
        else:
            saved_dict["avg_features"] = self.avg_features
        torch.save(saved_dict, path)

    def load(self, path, load_values=False, load_feat=False, load_optimizer=False, load_feat_normalizer=False,
             load_curriculum=False, load_init_obs=False):
        loaded_dict = torch.load(path)
        self.current_learning_iteration = loaded_dict['iteration']
        self.policy.load_state_dict(loaded_dict["policy_state_dict"])
        self.lagrange = loaded_dict["lagrange"]
        self.avg_ext_values = loaded_dict["avg_ext_values"]
        self.resume_id = loaded_dict["resume_id"]
        self.tot_timesteps = loaded_dict["tot_timesteps"]
        self.tot_time = loaded_dict["tot_time"]
        if load_init_obs:
            self.env.init_obs_buf = loaded_dict["init_obs"]
        if load_curriculum:
            self.env.terrain_cols = loaded_dict["curriculum"][0]
            self.env.terrain_rows = loaded_dict["curriculum"][1]
        if self.r_cfg.wandb:
            self.logger_state = loaded_dict["logger_state"]

        if load_values:
            self.int_value.load_state_dict(loaded_dict["intrinsic_value_state_dict"])
            for i in range(self.num_ext_values):
                self.ext_values[i].load_state_dict(loaded_dict[f"ext_value_{i}_state_dict"])
        if load_feat:
            if self.a_cfg.use_succ_feat:
                self.succ_feat.load_state_dict(loaded_dict["succ_feat_state_dict"])
                self.succ_feat_optimizer.load_state_dict(loaded_dict["succ_feat_optimizer_state_dict"])
            else:
                self.avg_features = loaded_dict["avg_features"]
        if self.normalize_observation:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if self.normalize_features and load_feat_normalizer:
            self.feat_normalizer.load_state_dict(loaded_dict["feat_norm_state_dict"])
        if load_optimizer:
            self.policy_optimizer.load_state_dict(loaded_dict["policy_optimizer_state_dict"])
            self.value_optimizer.load_state_dict(loaded_dict["value_optimizer_state_dict"])
            self.lagrange_optimizer.load_state_dict(loaded_dict["lagrange_optimizer_state_dict"])
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.policy.to(device)
            self.obs_normalizer.to(device)

        def inference_policy(input_x):
            x, z = input_x
            obs_skills = (self.obs_normalizer(x), z)
            return self.policy.act_inference(obs_skills)

        return inference_policy

    def train_mode(self):
        self.policy.train()
        self.int_value.train()
        for i in range(self.num_ext_values):
            self.ext_values[i].train()
        if self.normalize_observation:
            self.obs_normalizer.train()
        if self.normalize_features:
            self.feat_normalizer.train()

    def eval_mode(self):
        self.policy.eval()
        self.int_value.eval()
        for i in range(self.num_ext_values):
            self.ext_values[i].eval()
        if self.normalize_observation:
            self.obs_normalizer.eval()
        if self.normalize_features:
            self.feat_normalizer.eval()

    def init_writer(self, resume_id=None):
        # initialize writer
        if self.r_cfg.wandb:
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.r_cfg,
                                             group=self.r_cfg.wandb_group, resume_id=resume_id, use_allogger=True)
            if resume_id is None:
                self.writer.log_config(self.env.cfg, self.r_cfg, self.a_cfg, self.n_cfg)
        else:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # print(json.dumps(class_to_dict(self.env.cfg), indent=2, default=str))
        # print(json.dumps(class_to_dict(self.r_cfg), indent=2, default=str))
        # print(json.dumps(class_to_dict(self.a_cfg), indent=2, default=str))
        # print(json.dumps(class_to_dict(self.n_cfg), indent=2, default=str))
