import torch
import numpy as np
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import torch_rand_float, quat_rotate_inverse, get_euler_xyz

from solo_legged_gym.envs import BaseTask
from solo_legged_gym.utils import class_to_dict, get_quat_yaw


class Solo12DOMINO(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_commands = self.cfg.commands.num_commands
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        self.num_skills = self.cfg.env.num_skills
        self.skills = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.add_noise = self.cfg.observations.add_noise
        if self.add_noise:
            self.noise_scale_vec = self._get_noise_scale_vec()
        self.push_robots = self.cfg.domain_rand.push_robots
        if self.push_robots:
            self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.change_commands = self.cfg.commands.change_commands
        if self.change_commands:
            self.change_commands_interval = np.ceil(self.cfg.commands.change_commands_interval_s / self.dt)

        self._set_default_dof_pos()
        self._prepare_reward()

    def _init_buffers(self):
        super()._init_buffers()
        self.num_features = self.cfg.env.num_features
        self.num_feature_history_dim = self.cfg.env.num_feature_history_dim
        self.feature_buf = torch.zeros(self.num_envs, self.num_features, device=self.device, dtype=torch.float)

        self.HAA_indices = torch.tensor(
            [i for i in range(self.num_dof) if "HAA" not in self.dof_names[i]],
            device=self.device,
            requires_grad=False,
        )
        self.KFE_indices = torch.tensor(
            [i for i in range(self.num_dof) if "KFE" not in self.dof_names[i]],
            device=self.device,
            requires_grad=False,
        )
        self.HFE_indices = torch.tensor(
            [i for i in range(self.num_dof) if "HFE" in self.dof_names[i]],
            device=self.device,
            requires_grad=False,
        )
        self.torque_limits[:] = self.cfg.control.torque_limits

        # self.dof_vel_limits = torch.zeros_like(self.dof_vel)
        # self.dof_vel_limits[:, self.HAA_indices] = self.cfg.control.dof_vel_limits
        # self.dof_vel_limits[:, self.KFE_indices] = self.cfg.control.dof_vel_limits
        # self.dof_vel_limits[:, self.HFE_indices] = self.cfg.control.dof_vel_limits

        self.joint_targets = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.last_joint_targets = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.joint_targets_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.torques = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.total_power = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.total_torque = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.last_ee_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.ee_local = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                    requires_grad=False)
        self.ee_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False)
        self.ee_vel_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.actuator_lag_buffer = torch.zeros(self.num_envs, self.cfg.domain_rand.actuator_lag_steps + 1, 12,
                                               dtype=torch.float, device=self.device, requires_grad=False)
        self.actuator_lag_index = torch.randint(low=0, high=self.cfg.domain_rand.actuator_lag_steps,
                                                size=[self.num_envs], device=self.device)
        self.feet_contact_force = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)

        feature_history_length = self.cfg.env.feature_history_length
        self.feature_history = torch.zeros(self.num_envs, self.num_feature_history_dim, feature_history_length,
                                           dtype=torch.float,
                                           device=self.device, requires_grad=False)
        feature_freq = torch.fft.fftfreq(feature_history_length).to(self.device)
        feature_focus_freq = self.cfg.env.feature_focus_freq
        feature_focus_freq_len = len(feature_focus_freq)
        self.feature_focus_freq_idx = torch.zeros(feature_focus_freq_len, dtype=torch.long,
                                                  device=self.device, requires_grad=False)
        for i in range(feature_focus_freq_len):
            self.feature_focus_freq_idx[i] = \
                (abs(feature_freq - feature_focus_freq[i]) < 1e-4).nonzero(as_tuple=True)[0]

        self.feature_focus_freq_mag = torch.zeros(self.num_envs, self.num_feature_history_dim, feature_focus_freq_len,
                                                  dtype=torch.float, device=self.device,
                                                  requires_grad=False)
        self.feature_focus_freq_phase = torch.zeros(self.num_envs, self.num_feature_history_dim, feature_focus_freq_len,
                                                    dtype=torch.float, device=self.device,
                                                    requires_grad=False)

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_skills(env_ids)
        self._refresh_quantities()

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.last_ee_global[env_ids] = 0.
        self.total_power[env_ids] = 0.
        self.total_torque[env_ids] = 0.
        self.actuator_lag_buffer[env_ids] = 0.
        self.feature_history[env_ids] = 0

        self.episode_length_buf[env_ids] = 0

        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_term_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_term_sums[key][env_ids]) / self.max_episode_length
            self.episode_term_sums[key][env_ids] = 0.

        for key in self.episode_group_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_group_sums[key][env_ids]) / self.max_episode_length
            self.episode_group_sums[key][env_ids] = 0.

        self.extras["time_outs"] = self.time_out_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        _, _, _, _, _, _ = self.step(torch.zeros(self.num_envs,
                                                 self.num_actions,
                                                 device=self.device,
                                                 requires_grad=False))

    def pre_physics_step(self, actions):
        self.actions = actions

        clip_joint_target = self.cfg.control.clip_joint_target
        scale_joint_target = self.cfg.control.scale_joint_target
        self.joint_targets = torch.clip(actions * scale_joint_target, -clip_joint_target, clip_joint_target).to(
            self.device)

    def step(self, actions, pause=False):
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.joint_targets).view(self.torques.shape)
            power_ = self.torques * self.dof_vel
            total_power_ = torch.sum(power_ * (power_ >= 0), dim=1)
            total_torque_ = torch.sum(torch.square(self.torques), dim=1)
            self.total_power += total_power_
            self.total_torque += total_torque_
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # the skills are separated from the obs because we do not want to normalize it.
        return self.obs_buf, self.skills, self.feature_buf, self.group_rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._refresh_quantities()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._check_termination()
        self.compute_reward()

        if self.change_commands:
            env_ids_change_commands = (self.episode_length_buf % self.change_commands_interval == 0).nonzero(
                as_tuple=False).flatten()
            self._resample_commands(env_ids_change_commands)

        if self.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_robots()

        env_ids_reset = self.reset_buf.nonzero().flatten()
        self.reset_idx(env_ids_reset)

        self.compute_observations()
        self.compute_features()

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_joint_targets[:] = self.joint_targets[:]
        self.last_ee_global[:] = self.ee_global[:]

    def _check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def compute_observations(self):
        self.obs_buf = torch.cat((self.base_lin_vel,  # 3
                                  self.base_ang_vel,  # 3
                                  (self.dof_pos - self.default_dof_pos),  # 12
                                  self.dof_vel,  # 12
                                  self.projected_gravity,  # 3
                                  self.actions,
                                  self.commands,
                                  ), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # clip the observation if needed
        if self.cfg.observations.clip_obs:
            self.obs_buf = torch.clip(self.obs_buf, -self.cfg.observations.clip_limit, self.cfg.observations.clip_limit)

    def compute_features(self):
        # FL, FR, HL, HR
        feet_contact_phase_offsets = torch.concatenate(
            ((self.feature_focus_freq_phase[:, 1, :] - self.feature_focus_freq_phase[:, 0, :]),
             (self.feature_focus_freq_phase[:, 2, :] - self.feature_focus_freq_phase[:, 0, :]),
             (self.feature_focus_freq_phase[:, 3, :] - self.feature_focus_freq_phase[:, 0, :]),
             ), dim=-1)

        feet_contact_phase_offsets[feet_contact_phase_offsets >= 2 * np.pi] -= 2 * np.pi
        feet_contact_phase_offsets[feet_contact_phase_offsets < 0.0] += 2 * np.pi

        # we care about the magnitude of the rest
        focus_freq_mags = self.feature_focus_freq_mag[:, 4:, :].view(self.num_envs, -1)

        self.feature_buf = torch.cat((
            self.root_states[:, 2:3],  # 1
            self.base_lin_vel[:, 2:3],  # 1
            self.base_ang_vel[:, :2],  # 2
            focus_freq_mags,  # num_focus_freq * 4
            # feet_contact_phase_offsets  # num_focus_freq * 3
        ), dim=-1)

        # no noise added, no clipping

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.cfg.observations.noise_scales
        noise_level = self.cfg.observations.noise_level
        noise_vec[:3] = noise_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel
        noise_vec[6:18] = noise_scales.dof_pos
        noise_vec[18:30] = noise_scales.dof_vel
        noise_vec[30:33] = noise_scales.gravity
        return noise_vec * noise_level

    def _refresh_quantities(self):
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.feet_contact_force = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        self.ee_contact = self.feet_contact_force > 0.1
        self.ee_global = self.rigid_body_state[:, self.feet_indices, 0:3]
        self.ee_vel_global = self.rigid_body_state[:, self.feet_indices, 7:10]
        ee_local_ = self.rigid_body_state[:, self.feet_indices, 0:3]
        ee_local_[:, :, 0:2] -= self.root_states[:, 0:2].unsqueeze(1)
        for i in range(len(self.feet_indices)):
            self.ee_local[:, i, :] = quat_rotate_inverse(get_quat_yaw(self.base_quat), ee_local_[:, i, :])
        self.joint_targets_rate = torch.norm(self.last_joint_targets - self.joint_targets, p=2, dim=1)
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt

        # contact history: 1 in contact, -1 in swing, 0 placeholder
        selected_features = torch.concatenate(((self.ee_contact * 2 - 1).to(torch.float),
                                               self.root_states[:, 2:3],
                                               self.base_lin_vel[:, 2:3],
                                               self.base_ang_vel[:, :2],
                                               ), dim=-1)

        self.feature_history = torch.cat((self.feature_history[:, :, 1:], selected_features.unsqueeze(-1)), dim=-1)

        feature_fft = torch.fft.fft(self.feature_history, dim=2)  # num_envs * dim_selected_features * buf_len
        self.feature_focus_freq_mag = feature_fft[:, :, self.feature_focus_freq_idx].abs()
        self.feature_focus_freq_phase = torch.remainder(feature_fft[:, :, self.feature_focus_freq_idx].angle(),
                                                        2 * np.pi)

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                     self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        # clip the small command to zero
        self.commands[env_ids, :] *= torch.any(torch.abs(self.commands[env_ids, :]) >= 0.2, dim=1).unsqueeze(1)
        if self.cfg.env.play:
            self.commands[:] = 0.0
            # self.commands[:, 0] = 1.0

    def _resample_skills(self, env_ids):
        self.skills[env_ids] = torch.randint(low=0, high=self.num_skills, size=(len(env_ids),), device=self.device)

        if self.cfg.env.play:
            self.skills[env_ids] = 0

    def _compute_torques(self, joint_targets):
        # pd controller
        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.actuator_lag:
            self.actuator_lag_buffer = torch.cat((self.actuator_lag_buffer[:, 1:, :],
                                                  joint_targets.unsqueeze(1)), dim=1)
            if self.cfg.domain_rand.randomize_actuator_lag:
                joint_targets_ = self.actuator_lag_buffer[torch.arange(self.num_envs), self.actuator_lag_index]
            else:
                joint_targets_ = self.actuator_lag_buffer[:, 0, :]
        else:
            joint_targets_ = joint_targets

        if control_type == "P":
            torques = self.p_gains * (
                    joint_targets_ + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (joint_targets_ - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = joint_targets_
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        if self.cfg.env.play:
            self.dof_pos[env_ids] = self.default_dof_pos
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
            self.dof_pos[env_ids][:, [1, 2, 4, 5, 7, 8, 10, 11]] += torch_rand_float(-0.25, 0.25, (len(env_ids), 8),
                                                                                     device=self.device)

        self.dof_vel[env_ids] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        if self.cfg.env.play:
            self.root_states[env_ids, 7:13] = 0.0
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                               device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        # base velocity impulse
        max_vel = self.cfg.domain_rand.max_push_vel_xyz
        self.root_states[:, 7:10] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 3),
                                                      device=self.device)  # lin vel x/y/z
        max_avel = self.cfg.domain_rand.max_push_avel_xyz
        self.root_states[:, 10:13] += torch_rand_float(-max_avel, max_avel, (self.num_envs, 3),
                                                       device=self.device)  # ang vel x/y/z
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _set_default_dof_pos(self):
        self.p_gains = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward(self):
        self.reward_terms = class_to_dict(self.cfg.rewards.terms)
        self.reward_powers = self.cfg.rewards.powers
        self.reward_groups = {}
        for i in range(len(self.reward_powers)):
            self.reward_groups[str(int(i))] = []
        for name, info in self.reward_terms.items():
            group = str(int(eval(info)[0]))
            self.reward_groups[group].append(name)

        self.episode_term_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_terms.keys()}
        self.episode_group_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_groups.keys()}

        # allocate
        self.group_rew_buf = torch.ones(self.num_envs, len(self.reward_powers), dtype=torch.float, device=self.device,
                                        requires_grad=False)

    def compute_reward(self):
        self.rew_buf[:] = 1.0
        for group_name, terms in self.reward_groups.items():
            group_idx = int(group_name)
            group_power = self.reward_powers[group_idx]
            self.group_rew_buf[:, group_idx] = 1.0
            for i in range(len(terms)):
                reward_name = terms[i]
                reward_function = getattr(self, '_reward_' + reward_name)
                reward_sigma = eval(self.reward_terms[reward_name])[1]
                term_reward = reward_function(reward_sigma)
                self.episode_term_sums[reward_name] += term_reward
                self.group_rew_buf[:, group_idx] *= term_reward
            self.group_rew_buf[:, group_idx] = torch.pow(self.group_rew_buf[:, group_idx], group_power)
            self.episode_group_sums[group_name] += self.group_rew_buf[:, group_idx]
            self.rew_buf *= self.group_rew_buf[:, group_idx]

    # ------------------------------------------------------------------------------------------------------------------

    def _reward_lin_vel_x(self, sigma):
        lin_vel_x_error = self.commands[:, 0] - self.base_lin_vel[:, 0]
        return torch.exp(-torch.square(lin_vel_x_error / sigma))

    def _reward_lin_vel_y(self, sigma):
        lin_vel_y_error = self.commands[:, 1] - self.base_lin_vel[:, 1]
        return torch.exp(-torch.square(lin_vel_y_error / sigma))

    def _reward_ang_vel_z(self, sigma):
        ang_vel_error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        return torch.exp(-torch.square(ang_vel_error / sigma))

    def _reward_lin_z(self, sigma):
        lin_z_error = self.root_states[:, 2] - self.cfg.rewards.base_height_target
        return torch.exp(-torch.square(lin_z_error / sigma))

    def _reward_lin_vel_z(self, sigma):
        lin_vel_z = self.base_lin_vel[:, 2]
        return torch.exp(-torch.square(lin_vel_z / sigma))

    def _reward_lin_acc_z(self, sigma):
        lin_vel_z = self.base_lin_vel[:, 2]
        last_lin_vel_z = self.last_root_vel[:, 2]
        lin_acc_z = torch.abs(lin_vel_z - last_lin_vel_z)
        return torch.exp(-torch.square(lin_acc_z / sigma))

    def _reward_ang_xy(self, sigma):
        ang_xy = torch.stack(list(get_euler_xyz(self.base_quat)[:2]), dim=1)
        ang_xy = torch.norm(ang_xy, p=2, dim=1)
        return torch.clip(torch.exp(-torch.square(ang_xy / sigma)), min=None, max=0.9) / 0.9

    def _reward_ang_vel_xy(self, sigma):
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], p=2, dim=1)
        return torch.exp(-torch.square(ang_vel_xy / sigma))

    def _reward_ang_acc_xy(self, sigma):
        ang_vel_xy = self.base_ang_vel[:, :2]
        last_ang_vel_xy = self.last_root_vel[:, 3:5]
        ang_acc_xy = torch.norm(ang_vel_xy - last_ang_vel_xy, p=2, dim=1)
        return torch.exp(-torch.square(ang_acc_xy / sigma))

    def _reward_joint_default(self, sigma):
        joint_deviation = torch.norm(self.dof_pos - self.default_dof_pos, p=2, dim=1)
        return torch.clip(torch.exp(-torch.square(joint_deviation / sigma)), min=None, max=0.7) / 0.7

    def _reward_joint_targets_rate(self, sigma):
        return torch.exp(-torch.square(self.joint_targets_rate / sigma))

    def _reward_feet_height(self, sigma):
        feet_height_error = torch.norm(self.ee_global[:, :, 2] - sigma[0], p=2, dim=1) * (
                torch.norm(self.commands, dim=1) < 0.1)
        return torch.exp(-torch.square(feet_height_error / sigma[1]))

    def _reward_feet_slip(self, sigma):
        feet_low = self.ee_global[:, :, 2] < sigma[0]
        feet_move = torch.norm(self.ee_global[:, :, :2] - self.last_ee_global[:, :, :2], p=2, dim=2)
        sigma_ = sigma[1] + self.ee_global[:, :, 2] * sigma[2]
        feet_slip = torch.sum(feet_move * feet_low / sigma_, dim=1)
        return torch.exp(-torch.square(feet_slip))

    def _reward_feet_slip_h(self, sigma):
        feet_too_low = self.ee_global[:, :, 2] < sigma[0]
        feet_off_ground_when_too_low = torch.sum(self.ee_global[:, :, 2] * feet_too_low, dim=1)
        return torch.exp(-torch.square(feet_off_ground_when_too_low / sigma[1]))

    def _reward_feet_slip_v(self, sigma):
        feet_low = self.ee_global[:, :, 2] < sigma[0]
        feet_vel_xy = torch.norm(self.ee_vel_global[:, :, :2], p=2, dim=2)
        feet_slip_v = torch.sum(feet_vel_xy * feet_low, dim=1)
        return torch.exp(-torch.square(feet_slip_v / sigma[1]))

    def _reward_dof_vel(self, sigma):
        return torch.exp(-torch.square(torch.norm(self.dof_vel, p=2, dim=1) / sigma))

    def _reward_dof_acc(self, sigma):
        return torch.exp(-torch.square(torch.norm(self.dof_acc, p=2, dim=1) / sigma))

    def _reward_stand_still(self, sigma):
        not_stand = torch.norm(self.dof_pos - self.default_dof_pos, p=2, dim=1) * (
                torch.norm(self.commands, dim=1) < 0.1)
        return torch.exp(-torch.square(not_stand / sigma))

    def _reward_stand_still_h(self, sigma):
        feet_height = self.ee_global[:, :, 2]
        feet_off_ground_when_stand = torch.sum(feet_height, dim=-1) * (torch.norm(self.commands, dim=1) < 0.1)
        return torch.exp(-torch.square(feet_off_ground_when_stand / sigma))

    def _reward_torques(self, sigma):
        return torch.exp(-torch.square(torch.norm(self.torques, p=2, dim=1) / sigma))

    def _reward_feet_contact_force(self, sigma):
        return torch.exp(-torch.square(torch.norm(self.feet_contact_force, p=2, dim=1) / sigma))
