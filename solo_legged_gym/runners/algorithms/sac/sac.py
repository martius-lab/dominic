import time
import os
from collections import deque
import numpy as np
import statistics

import torch
import torch.optim as optim
from torch.nn import functional as func

from torch.utils.tensorboard import SummaryWriter
from solo_legged_gym.utils.wandb_utils import WandbSummaryWriter
from solo_legged_gym.runners.storage.replay_buffer import ReplayBuffer
from solo_legged_gym.runners.modules.normalizer import EmpiricalNormalization
from solo_legged_gym.runners.algorithms.sac.sac_policy import SACPolicy
from solo_legged_gym.runners.modules.qvalues import QValues
from solo_legged_gym.runners.utils.utils import polyak_update


class SAC:
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
        self.num_steps_per_env = self.r_cfg.num_steps_per_env
        self.save_interval = self.r_cfg.save_interval
        self.normalize_observation = self.r_cfg.normalize_observation

        # set up the networks
        self.policy = SACPolicy(num_obs=self.env.num_obs,
                                num_actions=self.env.num_actions,
                                hidden_dims=self.n_cfg.policy_hidden_dims,
                                activation=self.n_cfg.policy_activation,
                                init_noise_std=self.n_cfg.policy_init_noise_std).to(self.device)

        self.qvalues = QValues(num_obs=self.env.num_obs,
                               num_actions=self.env.num_actions,
                               num_qvalues=self.a_cfg.num_critic,
                               hidden_dims=self.n_cfg.qvalue_hidden_dims,
                               activation=self.n_cfg.qvalue_activation).to(self.device)

        self.qvalues_target = QValues(num_obs=self.env.num_obs,
                                      num_actions=self.env.num_actions,
                                      num_qvalues=self.a_cfg.num_critic,
                                      hidden_dims=self.n_cfg.qvalue_hidden_dims,
                                      activation=self.n_cfg.qvalue_activation).to(self.device)
        # value target should always be in eval mode
        self.qvalues_target.eval()

        # set up normalizer
        if self.normalize_observation:
            self.obs_normalizer = EmpiricalNormalization(shape=self.env.num_obs,
                                                         until=int(1.0e8)).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization

        # set up optimizer
        self.learning_rate = self.a_cfg.learning_rate
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.learning_rate)
        self.qvalues_optimizer = optim.Adam(self.qvalues.parameters(),
                                            lr=self.learning_rate)

        # set up replay buffer
        self.replay_buffer = ReplayBuffer(num_envs=self.env.num_envs,
                                          num_transitions_per_env=int(self.a_cfg.buffer_size // self.env.num_envs),
                                          obs_shape=[self.env.num_obs],
                                          actions_shape=[self.env.num_actions],
                                          device=self.device)
        self.transition = ReplayBuffer.Transition()

        # set up algorithm parameters
        self.target_entropy = self.a_cfg.target_entropy
        if self.target_entropy == "auto":
            self.target_entropy = -self.env.num_actions

        self.ent_coef = self.a_cfg.ent_coef
        self.ent_coef_optimizer = None
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            self.ent_coef_tensor = torch.tensor(self.ent_coef, device=self.device)

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
        self.learn_percentage = 0.0

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
            self.learn_percentage = (it - self.current_learning_iteration) / tot_iter
            # Rollout
            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    previous_obs = obs
                    actions = self.policy.act(previous_obs).detach()
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)
                    self.process_env_step(previous_obs, actions, obs, rewards, dones, infos)

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
            stop = time.time()
            collection_time = stop - start

            # Learning update
            start = stop
            mean_ent_coef_loss, mean_ent_coef, mean_actor_loss, mean_critic_loss = self.update()
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

    def process_env_step(self, obs, actions, next_obs, rewards, dones, infos):
        self.transition.observations = obs
        self.transition.actions = actions
        self.transition.next_observations = next_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.timeouts = infos['time_outs'].unsqueeze(1).to(self.device)
        self.replay_buffer.add_transitions(self.transition)
        self.transition.clear()

    def update(self):
        optimizers = [self.policy_optimizer, self.qvalues_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        if self.a_cfg.schedule == "adaptive":
            self.learning_rate = max(1e-5, (1 - self.learn_percentage) * self.a_cfg.learning_rate)
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        mean_ent_coef_loss = 0
        mean_ent_coef = 0
        mean_actor_loss = 0
        mean_critic_loss = 0
        generator = self.replay_buffer.mini_batch_generator(self.a_cfg.mini_batch_size, self.a_cfg.num_mini_batches,
                                                            self.a_cfg.num_learning_epochs)
        for (obs_batch,
             next_obs_batch,
             actions_batch,
             rewards_batch,
             dones_batch,
             ) in generator:

            # actions by the current policy
            actions_pi = self.policy.act(obs_batch)
            actions_pi_log_prob = self.policy.get_actions_log_prob(actions_pi)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (
                            actions_pi_log_prob.reshape(-1, 1) + self.target_entropy).detach()).mean()
                mean_ent_coef_loss += ent_coef_loss.item()
            else:
                ent_coef = self.ent_coef_tensor

            mean_ent_coef += ent_coef.item()

            # Optimize entropy coefficient
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions_pi = self.policy.act(next_obs_batch)
                next_actions_pi_log_prob = self.policy.get_actions_log_prob(next_actions_pi)
                next_q_values = torch.cat(self.qvalues_target.evaluate(next_obs_batch, next_actions_pi), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_actions_pi_log_prob.reshape(-1, 1)
                target_q_values = rewards_batch + (1 - dones_batch) * self.a_cfg.gamma * next_q_values

            # Compute critic loss
            current_q_values = self.qvalues.evaluate(obs_batch, actions_batch)
            critic_loss = 0.5 * sum(func.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            mean_critic_loss += critic_loss.item()

            # Optimize the critic
            self.qvalues_optimizer.zero_grad()
            critic_loss.backward()
            self.qvalues_optimizer.step()

            # Compute actor loss
            qvalues_pi = torch.cat(self.qvalues.evaluate(obs_batch, actions_pi), dim=1)
            min_qvalues_pi, _ = torch.min(qvalues_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * actions_pi_log_prob - min_qvalues_pi).mean()
            mean_actor_loss += actor_loss.item()

            # Optimize the actor
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            polyak_update(self.qvalues.parameters(), self.qvalues_target.parameters(), self.a_cfg.tau)

        num_updates = self.a_cfg.num_learning_epochs * self.a_cfg.num_mini_batches
        mean_ent_coef_loss /= num_updates
        mean_ent_coef /= num_updates
        mean_actor_loss /= num_updates
        mean_critic_loss /= num_updates

        return mean_ent_coef_loss, mean_ent_coef, mean_actor_loss, mean_critic_loss

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
        mean_std = self.policy.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Learning/mean_ent_coef_loss', locs['mean_ent_coef_loss'], locs['it'])
        self.writer.add_scalar('Learning/mean_ent_coef', locs['mean_ent_coef'], locs['it'])
        self.writer.add_scalar('Learning/mean_actor_loss', locs['mean_actor_loss'], locs['it'])
        self.writer.add_scalar('Learning/mean_critic_loss', locs['mean_critic_loss'], locs['it'])
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
                          f"""{'Entropy coefficient loss:':>{pad}} {locs['mean_ent_coef_loss']:.4f}\n"""
                          f"""{'Entropy coefficient:':>{pad}} {locs['mean_ent_coef']:.4f}\n"""
                          f"""{'Actor loss:':>{pad}} {locs['mean_actor_loss']:.4f}\n"""
                          f"""{'Critic loss:':>{pad}} {locs['mean_critic_loss']:.4f}\n"""
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
                          f"""{'Entropy coefficient loss:':>{pad}} {locs['mean_ent_coef_loss']:.4f}\n"""
                          f"""{'Entropy coefficient:':>{pad}} {locs['mean_ent_coef']:.4f}\n"""
                          f"""{'Actor loss:':>{pad}} {locs['mean_actor_loss']:.4f}\n"""
                          f"""{'Critic loss:':>{pad}} {locs['mean_critic_loss']:.4f}\n"""
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
            "actor_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.qvalues.state_dict(),
            "critic_target_state_dict": self.qvalues_target.state_dict(),
            "actor_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.qvalues_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.ent_coef_optimizer is not None:
            saved_dict["ent_coef_optimizer_state_dict"] = self.ent_coef_optimizer.state_dict()
        if self.normalize_observation:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.r_cfg.wandb: self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.policy.load_state_dict(loaded_dict["actor_state_dict"])
        self.qvalues.load_state_dict(loaded_dict["critic_state_dict"])
        self.qvalues_target.load_state_dict(loaded_dict["critic_target_state_dict"])
        if self.normalize_observation:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if load_optimizer:
            self.policy_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.qvalues_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.load_state_dict(loaded_dict["ent_coef_optimizer_state_dict"])
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
        self.qvalues.train()
        if self.normalize_observation:
            self.obs_normalizer.train()

    def eval_mode(self):
        self.policy.eval()
        self.qvalues.eval()
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
