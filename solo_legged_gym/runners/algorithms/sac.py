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
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    previous_obs = obs
                    actions = self.policy.act(previous_obs).detach()
                    obs, rewards, dones, infos = self.env.step(actions)
                    self.process_env_step(previous_obs, actions, rewards, dones, infos)
                    obs = self.obs_normalizer(obs)

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

                # Learning step
                start = stop

                last_values = self.value.evaluate(obs).detach()
                self.rollout_buffer.compute_returns(last_values, self.a_cfg.gamma, self.a_cfg.lam)

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