import time
import os
from collections import deque
import numpy as np
import statistics

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from solo_legged_gym.utils.wandb_utils import WandbSummaryWriter
from solo_legged_gym.runners.algorithms.ppo.ppo_policy import PPOPolicy
from solo_legged_gym.runners.modules.value import Value
from solo_legged_gym.runners.modules.normalizer import EmpiricalNormalization
from solo_legged_gym.runners.storage.rollout_buffer import RolloutBuffer


class PPO:
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
        self.policy = PPOPolicy(num_obs=self.env.num_obs,
                                num_actions=self.env.num_actions,
                                hidden_dims=self.n_cfg.policy_hidden_dims,
                                activation=self.n_cfg.policy_activation,
                                log_std_init=self.n_cfg.log_std_init).to(self.device)

        self.value = Value(num_obs=self.env.num_obs,
                           hidden_dims=self.n_cfg.value_hidden_dims,
                           activation=self.n_cfg.value_activation).to(self.device)

        self.num_steps_per_env = self.r_cfg.num_steps_per_env
        self.save_interval = self.r_cfg.save_interval
        self.normalize_observation = self.r_cfg.normalize_observation

        if self.normalize_observation:
            self.obs_normalizer = EmpiricalNormalization(shape=self.env.num_obs,
                                                         until=int(1.0e8)).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization

        # set up rollout buffer
        self.rollout_buffer = RolloutBuffer(num_envs=self.env.num_envs,
                                            num_transitions_per_env=self.num_steps_per_env,
                                            obs_shape=[self.env.num_obs],
                                            actions_shape=[self.env.num_actions],
                                            device=self.device)
        self.transition = RolloutBuffer.Transition()

        # set up optimizer
        self.learning_rate = self.a_cfg.learning_rate
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),
                                    lr=self.learning_rate)

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

        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):

            # Rollout
            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    previous_obs = obs
                    actions, log_prob = self.policy.act(previous_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)
                    self.process_env_step(previous_obs, actions, log_prob, rewards, dones, infos)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                last_values = self.value.evaluate(obs).detach()
                self.rollout_buffer.compute_returns(last_values, self.a_cfg.gamma, self.a_cfg.lam)
            stop = time.time()
            collection_time = stop - start

            # Learning update
            start = stop
            mean_value_loss, mean_surrogate_loss = self.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                rew_each_step = np.array(list(rewbuffer)) / np.array(list(lenbuffer))
                len_percent_buffer = np.array(list(lenbuffer)) / self.env.max_episode_length
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            if it >= tot_iter - 1:
                if len(rewbuffer) > 0:
                    self.avg_score = statistics.mean(rewbuffer)
            ep_infos.clear()

        self.current_learning_iteration += self.num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        return self.avg_score

    def process_env_step(self, obs, actions, log_prob, rewards, dones, infos):
        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.values = self.value.evaluate(obs).detach()
        self.transition.observations = obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.rewards += self.a_cfg.gamma * torch.squeeze(
            self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.rollout_buffer.add_transitions(self.transition)
        self.transition.clear()

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.rollout_buffer.mini_batch_generator(self.a_cfg.num_mini_batches,
                                                             self.a_cfg.num_learning_epochs)
        for (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
        ) in generator:

            # using the current policy
            _, _ = self.policy.act(obs_batch)  # update the distribution
            actions_log_prob_batch = self.policy.distribution.log_prob(actions_batch)
            value_batch = self.value.evaluate(obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL
            if self.a_cfg.desired_kl is not None and self.a_cfg.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.a_cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.a_cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.a_cfg.clip_param, 1.0 + self.a_cfg.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.a_cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.a_cfg.clip_param, self.a_cfg.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.a_cfg.value_loss_coef * value_loss - self.a_cfg.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()),
                                     self.a_cfg.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.a_cfg.num_learning_epochs * self.a_cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.rollout_buffer.clear()

        return mean_value_loss, mean_surrogate_loss

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.policy.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Learning/value_function_loss', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Learning/surrogate_loss', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Learning/learning_rate', self.learning_rate, locs['it'])
        self.writer.add_scalar('Learning/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward_each_step', statistics.mean(locs['rew_each_step']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length_percentage', statistics.mean(locs['len_percent_buffer']),
                                   locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + self.num_learning_iterations} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

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
            "value_state_dict": self.value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.normalize_observation:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.r_cfg.wandb: self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.policy.load_state_dict(loaded_dict["policy_state_dict"])
        self.value.load_state_dict(loaded_dict["value_state_dict"])
        if self.normalize_observation:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
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
        self.value.train()
        if self.normalize_observation:
            self.obs_normalizer.train()

    def eval_mode(self):
        self.policy.eval()
        self.value.eval()
        if self.normalize_observation:
            self.obs_normalizer.eval()

    def init_writer(self, play):
        if play:
            return
        # initialize writer
        if self.r_cfg.wandb:
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.r_cfg)
            self.writer.log_config(self.env.cfg, self.r_cfg, self.a_cfg, self.n_cfg)
        else:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
