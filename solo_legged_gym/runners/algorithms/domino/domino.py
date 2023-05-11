import json
import time
import os
from collections import deque
import statistics
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.tensorboard import SummaryWriter

from solo_legged_gym.utils import class_to_dict
from solo_legged_gym.utils.wandb_utils import WandbSummaryWriter
from solo_legged_gym.runners.algorithms.domino.domino_policy import DOMINOPolicy
from solo_legged_gym.runners.modules.value import Value
from solo_legged_gym.runners.modules.normalizer import EmpiricalNormalization
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

        # set up the networks
        self.policy = DOMINOPolicy(num_obs=self.env.num_obs,
                                   num_actions=self.env.num_actions,
                                   hidden_dims=self.n_cfg.policy_hidden_dims,
                                   activation=self.n_cfg.policy_activation,
                                   log_std_init=self.n_cfg.log_std_init).to(self.device)

        self.ext_value = Value(num_obs=self.env.num_obs,
                               hidden_dims=self.n_cfg.value_hidden_dims,
                               activation=self.n_cfg.value_activation).to(self.device)

        self.int_value = Value(num_obs=self.env.num_obs,
                               hidden_dims=self.n_cfg.value_hidden_dims,
                               activation=self.n_cfg.value_activation).to(self.device)

        # set up Lagrangian multipliers
        # There should be num_skills Lagrangian multipliers, but we fixed the first one to be sig(la_0) = 1
        self.lagrange = nn.Parameter(torch.ones(self.env.num_skills - 1,
                                                device=self.device) * self.a_cfg.init_lagrange, requires_grad=True)

        # set up moving averages
        self.avg_ext_values = torch.zeros(self.env.num_skills, device=self.device, requires_grad=False)
        self.avg_features = torch.ones(self.env.num_skills, self.env.num_features, device=self.device,
                                       requires_grad=False) * (1 / self.env.num_features)

        self.num_steps_per_env = self.r_cfg.num_steps_per_env
        self.save_interval = self.r_cfg.save_interval

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
                                            device=self.device)
        self.transition = RolloutBuffer.Transition()

        # set up optimizer
        self.learning_rate = self.a_cfg.learning_rate
        self.optimizer = optim.Adam(list(self.policy.parameters()) +
                                    list(self.ext_value.parameters()) +
                                    list(self.int_value.parameters()),
                                    lr=self.learning_rate)

        self.lagrange_learning_rate = self.a_cfg.lagrange_learning_rate
        self.lagrange_optimizer = optim.Adam([self.lagrange], lr=self.lagrange_learning_rate)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.init_writer(self.env.cfg.env.play)

        # iterations
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.num_learning_iterations = self.r_cfg.max_iterations

        self.env.reset()
        self.avg_score = 0

    def learn(self):
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )

        new_obs = self.obs_normalizer(self.env.get_observations().to(self.device))
        new_skills = self.env.skills.to(self.device)

        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        ext_rew_buffer = deque(maxlen=100)
        int_rew_buffer = deque(maxlen=100)
        len_buffer = deque(maxlen=100)
        cur_ext_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_int_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):

            # Rollout
            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    obs = new_obs
                    skills = new_skills
                    # use last obs to get the actions
                    actions, log_prob = self.policy.act_and_log_prob(obs)
                    new_obs, new_skills, features, ext_rew, group_rew, dones, infos = self.env.step(actions)
                    # features should be part of the outcome of the actions
                    features = self.feat_normalizer(features)
                    int_rew = self.a_cfg.intrinsic_rew_scale * self.get_intrinsic_reward(skills, features)
                    self.process_env_step(obs, actions, ext_rew, int_rew, skills, features, log_prob, dones, infos)
                    # normalize new obs
                    new_obs = self.obs_normalizer(new_obs)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_ext_rew_sum += ext_rew
                        cur_int_rew_sum += int_rew
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        ext_rew_buffer.extend(cur_ext_rew_sum[new_ids][:, 0].cpu().numpy().tolist())
                        int_rew_buffer.extend(cur_int_rew_sum[new_ids][:, 0].cpu().numpy().tolist())
                        len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_ext_rew_sum[new_ids] = 0
                        cur_int_rew_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                last_ext_values = self.ext_value.evaluate(obs).detach()
                last_int_values = self.int_value.evaluate(obs).detach()
                self.rollout_buffer.compute_returns(last_ext_values, last_int_values, self.a_cfg.gamma, self.a_cfg.lam)
            stop = time.time()
            collection_time = stop - start

            # Learning update
            start = stop
            mean_value_loss, mean_ext_value_loss, mean_int_value_loss, mean_surrogate_loss, \
                mean_lagrange_losses, mean_lagrange_coeffs, mean_constraint_satisfaction = self.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += self.num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        # score not implemented yet
        return 0

    def process_env_step(self, obs, actions, ext_rew, int_rew, skills, features, log_prob, dones, infos):
        self.transition.observations = obs

        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        self.transition.ext_values = self.ext_value.evaluate(obs).detach()
        self.transition.int_values = self.int_value.evaluate(obs).detach()

        self.transition.ext_rew = ext_rew.clone()
        self.transition.ext_rew += self.a_cfg.gamma * torch.squeeze(
            self.transition.ext_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)  # bootstrap for timeouts

        self.transition.int_rew = int_rew.clone()
        self.transition.int_rew += self.a_cfg.gamma * torch.squeeze(
            self.transition.int_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)  # bootstrap for timeouts

        self.transition.dones = dones
        self.transition.skills = skills
        self.transition.features = features

        self.rollout_buffer.add_transitions(self.transition)
        self.transition.clear()

    def get_intrinsic_reward(self, skills, features):
        latents = self.avg_features[skills]  # num_samples * num_features
        latents_dist = torch.norm((latents.unsqueeze(1).repeat(1, self.env.num_skills, 1) -
                                   self.avg_features.unsqueeze(0).repeat(self.env.num_envs, 1, 1)),
                                  dim=2, p=2)

        _, nearest_latents_idx = torch.kthvalue(latents_dist, k=2, dim=-1)  # num_samples
        nearst_latents = self.avg_features[nearest_latents_idx]  # num_samples * num_features
        psi_diff = latents - nearst_latents
        norm_diff = torch.norm(psi_diff, p=2, dim=-1) / self.a_cfg.target_d
        c = (1 - self.a_cfg.attractive_coeff) * torch.pow(norm_diff, self.a_cfg.repulsive_power) - \
            self.a_cfg.attractive_coeff * torch.pow(norm_diff, self.a_cfg.attractive_power)
        int_rew = c * torch.sum(features * psi_diff, dim=-1) / self.env.num_features
        return int_rew

    def get_lagrange_coeff(self, skills):
        lagrange_coeff = torch.zeros_like(skills).to(torch.float32)
        lagrange_coeff[skills > 0] = self.lagrange[skills[skills > 0] - 1]
        lagrange_coeff = torch.sigmoid(lagrange_coeff / self.a_cfg.sigmoid_scale)
        lagrange_coeff[skills == 0] = 1  # with only extrinsic reward
        return lagrange_coeff

    def update_moving_avg(self, skills, ext_returns, features):
        encoded_skills = func.one_hot(skills, num_classes=self.env.num_skills)
        encoded_ext_returns = encoded_skills * ext_returns.unsqueeze(-1).repeat(1, self.env.num_skills)
        encoded_features = encoded_skills.unsqueeze(-1) * features.unsqueeze(1).repeat(1, self.env.num_skills, 1)

        mean_encoded_ext_returns = encoded_ext_returns.sum(dim=0) / encoded_skills.sum(dim=0)
        mean_encoded_features = encoded_features.sum(dim=0) / encoded_skills.sum(dim=0).unsqueeze(-1)

        self.avg_ext_values = self.a_cfg.avg_values_decay_factor * self.avg_ext_values + \
                              (1 - self.a_cfg.avg_values_decay_factor) * mean_encoded_ext_returns

        self.avg_features = self.a_cfg.avg_features_decay_factor * self.avg_features + \
                            (1 - self.a_cfg.avg_features_decay_factor) * mean_encoded_features

    def update(self):
        mean_value_loss = 0
        mean_ext_value_loss = 0
        mean_int_value_loss = 0
        mean_surrogate_loss = 0
        mean_lagrange_coeffs = np.zeros(self.env.num_skills)
        mean_constraint_satisfaction = np.zeros(self.env.num_skills - 1)
        mean_lagrange_losses = np.zeros(self.env.num_skills - 1)
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
             ) in generator:

            # using the current policy
            _, _ = self.policy.act_and_log_prob(obs)  # update the distribution
            actions_log_prob_batch = self.policy.distribution.log_prob(actions)
            ext_value = self.ext_value.evaluate(obs)
            int_value = self.int_value.evaluate(obs)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # combine advantages
            with torch.inference_mode():
                lagrange_coeff = self.get_lagrange_coeff(skills)
            advantages = lagrange_coeff * ext_advantages + (1 - lagrange_coeff) * int_advantages
            mean_lagrange_coeffs += (torch.sum(func.one_hot(skills.squeeze(1)) * lagrange_coeff,
                                               dim=0) / torch.sum(func.one_hot(skills.squeeze(1)),
                                                                  dim=0)).cpu().detach().numpy()

            # KL
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
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.a_cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob))
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, 1.0 - self.a_cfg.clip_param, 1.0 + self.a_cfg.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.a_cfg.use_clipped_value_loss:
                ext_value_clipped = target_ext_values + (ext_value - target_ext_values).clamp(
                    -self.a_cfg.clip_param, self.a_cfg.clip_param
                )
                int_value_clipped = target_int_values + (int_value - target_int_values).clamp(
                    -self.a_cfg.clip_param, self.a_cfg.clip_param
                )
                ext_value_loss = torch.max((ext_value - ext_returns).pow(2),
                                           (ext_value_clipped - ext_returns).pow(2)).mean()
                int_value_loss = torch.max((int_value - int_returns).pow(2),
                                           (int_value_clipped - int_returns).pow(2)).mean()
            else:
                ext_value_loss = (ext_returns - ext_value).pow(2).mean()
                int_value_loss = (int_returns - int_value).pow(2).mean()

            value_loss = ext_value_loss + int_value_loss

            weighted_loss = surrogate_loss + \
                            self.a_cfg.value_loss_coef * value_loss - \
                            self.a_cfg.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.parameters()) +
                                     list(self.ext_value.parameters()) +
                                     list(self.int_value.parameters()),
                                     self.a_cfg.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_ext_value_loss += ext_value_loss.item()
            mean_int_value_loss += int_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # Lagrange loss
            lagrange_losses = self.lagrange * (self.avg_ext_values[1:] - self.a_cfg.alpha * self.avg_ext_values[0]).squeeze(-1)
            lagrange_loss = torch.sum(lagrange_losses, dim=-1)
            lagrange_loss.backward()
            self.lagrange_optimizer.step()
            mean_lagrange_losses += lagrange_losses.cpu().detach().numpy()
            mean_constraint_satisfaction += (self.avg_ext_values[1:] - self.a_cfg.alpha * self.avg_ext_values[0]).cpu().detach().numpy()

            # update moving average
            self.update_moving_avg(skills.squeeze(-1), ext_returns.squeeze(-1), features)

        num_updates = self.a_cfg.num_learning_epochs * self.a_cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_ext_value_loss /= num_updates
        mean_int_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_lagrange_losses /= num_updates
        mean_lagrange_coeffs /= num_updates
        mean_constraint_satisfaction /= num_updates
        self.rollout_buffer.clear()

        return mean_value_loss, mean_ext_value_loss, mean_int_value_loss, mean_surrogate_loss, mean_lagrange_losses, mean_lagrange_coeffs, mean_constraint_satisfaction

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

        self.writer.add_scalar('Learning/value_function_loss', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Learning/ext_value_function_loss', locs['mean_ext_value_loss'], locs['it'])
        self.writer.add_scalar('Learning/int_value_function_loss', locs['mean_int_value_loss'], locs['it'])
        self.writer.add_scalar('Learning/surrogate_loss', locs['mean_surrogate_loss'], locs['it'])
        mean_lagrange_losses = locs['mean_lagrange_losses']
        for i in range(len(mean_lagrange_losses)):
            self.writer.add_scalar(f'Skill/lagrange_loss_{i}', mean_lagrange_losses[i], locs['it'])
        mean_constraint_satisfaction = locs['mean_constraint_satisfaction']
        for i in range(len(mean_lagrange_losses)):
            self.writer.add_scalar(f'Skill/constraint_satisfaction_{i}', mean_constraint_satisfaction[i], locs['it'])
        mean_lagrange_coeffs = locs['mean_lagrange_coeffs']
        for i in range(len(mean_lagrange_coeffs)):
            self.writer.add_scalar(f'Skill/lagrange_coeff_{i}', mean_lagrange_coeffs[i], locs['it'])
        for i in range(self.env.num_skills - 1):
            self.writer.add_scalar(f'Skill/lagrange_{i}', self.lagrange[i].item(), locs['it'])
        self.writer.add_scalar('Learning/learning_rate', self.learning_rate, locs['it'])
        self.writer.add_scalar('Learning/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['len_buffer']) > 0:
            self.writer.add_scalar('Train/mean_extrinsic_reward', statistics.mean(locs['ext_rew_buffer']), locs['it'])
            self.writer.add_scalar('Train/mean_intrinsic_reward', statistics.mean(locs['int_rew_buffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['len_buffer']), locs['it'])

        title = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + self.num_learning_iterations} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{title.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                          'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               self.num_learning_iterations - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "policy_state_dict": self.policy.state_dict(),
            "extrinsic_value_state_dict": self.ext_value.state_dict(),
            "intrinsic_value_state_dict": self.int_value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.normalize_observation:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        if self.normalize_features:
            saved_dict["feat_norm_state_dict"] = self.feat_normalizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.policy.load_state_dict(loaded_dict["policy_state_dict"])
        self.ext_value.load_state_dict(loaded_dict["extrinsic_value_state_dict"])
        self.int_value.load_state_dict(loaded_dict["intrinsic_value_state_dict"])
        if self.normalize_observation:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if self.normalize_features:
            self.feat_normalizer.load_state_dict(loaded_dict["feat_norm_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.policy.to(device)
        policy = self.policy.act_inference
        if self.r_cfg.normalize_observation:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.policy.act_inference(self.obs_normalizer(x))
        return policy

    def train_mode(self):
        self.policy.train()
        self.ext_value.train()
        self.int_value.train()
        if self.normalize_observation:
            self.obs_normalizer.train()
        if self.normalize_features:
            self.feat_normalizer.train()

    def eval_mode(self):
        self.policy.eval()
        self.ext_value.eval()
        self.int_value.eval()
        if self.normalize_observation:
            self.obs_normalizer.eval()
        if self.normalize_features:
            self.feat_normalizer.eval()

    def init_writer(self, play):
        if play:
            return
        # initialize writer
        if self.r_cfg.wandb:
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.r_cfg,
                                             group=self.r_cfg.wandb_group)
            self.writer.log_config(self.env.cfg, self.r_cfg, self.a_cfg, self.n_cfg)
        else:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        print(json.dumps(class_to_dict(self.env.cfg), indent=2, default=str))
        print(json.dumps(class_to_dict(self.r_cfg), indent=2, default=str))
        print(json.dumps(class_to_dict(self.a_cfg), indent=2, default=str))
        print(json.dumps(class_to_dict(self.n_cfg), indent=2, default=str))
