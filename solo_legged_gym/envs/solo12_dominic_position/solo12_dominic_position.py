import torch
import numpy as np
import sys
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import torch_rand_float, quat_rotate_inverse, get_euler_xyz, quat_apply, quat_rotate

from solo_legged_gym.envs import BaseTask
from solo_legged_gym.utils import class_to_dict, get_quat_yaw, wrap_to_pi, quat_apply_yaw, torch_rand_float_ring


class Solo12DOMINOPosition(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.init_done = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_commands = self.cfg.commands.num_commands
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        self.commands_in_base = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                            device=self.device, requires_grad=False)

        self.remaining_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.remaining_check_time = self.cfg.env.remaining_check_time_s / self.cfg.env.episode_length_s

        self.num_skills = self.cfg.env.num_skills
        self.skills = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.add_noise = self.cfg.observations.add_noise
        if self.add_noise:
            self.noise_scale_vec = self._get_noise_scale_vec()
        self.push_robots = self.cfg.domain_rand.push_robots
        if self.push_robots:
            self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        self._set_default_dof_pos()
        self._prepare_reward()

    def _init_buffers(self):
        super()._init_buffers()
        self.scale_joint_target = torch.tensor(self.cfg.control.scale_joint_target, device=self.device)

        self.num_features = self.cfg.env.num_features
        self.feature_buf = torch.zeros(self.num_envs, self.num_features, device=self.device, dtype=torch.float)
        self.joint_lower_limits = torch.tensor(self.cfg.control.joint_lower_limits, device=self.device)
        self.joint_upper_limits = torch.tensor(self.cfg.control.joint_upper_limits, device=self.device)
        # self.HAA_indices = torch.tensor(
        #     [i for i in range(self.num_dof) if "HAA" not in self.dof_names[i]],
        #     device=self.device,
        #     requires_grad=False,
        # )
        # self.KFE_indices = torch.tensor(
        #     [i for i in range(self.num_dof) if "KFE" not in self.dof_names[i]],
        #     device=self.device,
        #     requires_grad=False,
        # )
        # self.HFE_indices = torch.tensor(
        #     [i for i in range(self.num_dof) if "HFE" in self.dof_names[i]],
        #     device=self.device,
        #     requires_grad=False,
        # )
        self.torque_limits[:] = self.cfg.control.torque_limits

        self.base_terrain_heights = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.base_target_terrain_heights = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                       requires_grad=False)
        self.joint_targets = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.last_joint_targets = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.joint_targets_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.torques = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        # self.total_power = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.total_torque = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.last_ee_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.ee_vel_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.last_ee_vel_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.ee_acc_global = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device,
                                         requires_grad=False)

        # self.ee_terrain_heights = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        self.actuator_lag_buffer = torch.zeros(self.num_envs, self.cfg.domain_rand.actuator_lag_steps + 1, 12,
                                               dtype=torch.float, device=self.device, requires_grad=False)
        self.actuator_lag_index = torch.randint(low=0, high=self.cfg.domain_rand.actuator_lag_steps,
                                                size=[self.num_envs], device=self.device)

        self.ee_check_px, self.ee_check_py = torch.meshgrid(torch.arange(-1, 2, device=self.device),
                                                            torch.arange(-1, 2, device=self.device))

        self.base_check_px, self.base_check_py = torch.meshgrid(torch.arange(-1, 2, device=self.device),
                                                                torch.arange(-1, 2, device=self.device))

        self.play_skill = torch.arange(self.cfg.env.num_skills, device=self.device, requires_grad=False)
        self.eval_skill = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.draw_height_colors = [(0.75, 0, 0),
                                   (0, 0.75, 0),
                                   (0, 0, 0.75),
                                   (0.75, 0.75, 0),
                                   (0, 0.75, 0.75),
                                   (0.75, 0, 0.75),
                                   (0.75, 0.75, 0.75),
                                   (0.5, 0.75, 0),
                                   (0.5, 0, 0.75),
                                   (0.75, 0.5, 0),
                                   (0, 0.5, 0.75),
                                   (0.75, 0, 0.5),
                                   (0, 0.75, 0.5),
                                   ]
        self.draw_body_colors = [(0.95, 0.5, 0.5),
                                 (0.5, 0.95, 0.5),
                                 (0.5, 0.5, 0.95),
                                 (0.95, 0.95, 0.5),
                                 (0.5, 0.95, 0.95),
                                 (0.95, 0.5, 0.95),
                                 (0.95, 0.95, 0.95),
                                 (0.75, 0.95, 0.5),
                                 (0.75, 0.5, 0.95),
                                 (0.95, 0.75, 0.5),
                                 (0.5, 0.75, 0.95),
                                 (0.95, 0.5, 0.75),
                                 (0.5, 0.95, 0.75),
                                 ]

        # for evaluating the successor features for distance measurement
        self.init_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.base_face_direction = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        if len(env_ids) == 0:
            return

        self._update_env_origin(env_ids)  # curriculum
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_skills(env_ids)
        self._refresh_quantities()

        # reset buffers
        self.last_dof_vel[env_ids] = 0.
        self.last_ee_global[env_ids] = 0.
        self.last_ee_vel_global[env_ids] = 0.
        self.actuator_lag_buffer[env_ids] = 0.

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
        self.extras["terminate"] = self.terminate_buf

    def reset(self):
        """ Reset all robots"""
        # till now the robot should be uniformly placed in all terrains.
        if (not self.cfg.env.play) and (not self.cfg.env.evaluation):
            self._init_env_origins()
            self.reset_idx(torch.arange(self.num_envs, device=self.device))
            self._update_remaining_time()
            self._update_commands_in_base()

            # store the initial observation for evaluating the successor features
            self.init_obs_buf, _, _, _, _, _ = self.step(torch.zeros(self.num_envs,
                                                                     self.num_actions,
                                                                     device=self.device,
                                                                     requires_grad=False))

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_remaining_time()
        self._update_commands_in_base()

        _, _, _, _, _, _ = self.step(torch.zeros(self.num_envs,
                                                 self.num_actions,
                                                 device=self.device,
                                                 requires_grad=False))

        self.init_done = True

    def pre_physics_step(self, actions):
        self.actions = actions
        self.joint_targets = (actions * self.scale_joint_target.unsqueeze(0)).to(self.device)

    def step(self, actions, pause=False):
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        self.total_torque[:] = 0
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.joint_targets).view(self.torques.shape)
            total_torque_ = torch.sum(torch.square(self.torques), dim=1)
            # self.total_power += total_power_
            self.total_torque += total_torque_
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        return self.obs_buf, self.skills, self.feature_buf, self.group_rew_buf, self.reset_buf, self.extras

    def render(self, sync_frame_time=True):
        # Overwrite with lines
        # fetch results
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        if self.cfg.viewer.record_camera_imgs:
            ref_pos = [self.root_states[self.image_env, 0].item() + self.cfg.viewer.camera_pos_b[0],
                       self.root_states[self.image_env, 1].item() + self.cfg.viewer.camera_pos_b[1],
                       self.root_states[self.image_env, 2].item() + self.cfg.viewer.camera_pos_b[2]]
            ref_lookat = [self.root_states[self.image_env, 0].item(),
                          self.root_states[self.image_env, 1].item(),
                          self.root_states[self.image_env, 2].item()]
            cam_pos = gymapi.Vec3(ref_pos[0], ref_pos[1], ref_pos[2])
            cam_target = gymapi.Vec3(ref_lookat[0], ref_lookat[1], ref_lookat[2])

            self.gym.set_camera_location(self.camera_sensors, self.envs[self.image_env], cam_pos, cam_target)

        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_overview" and evt.value > 0:
                    self.overview = not self.overview
                    self.viewer_set = False

            # step graphics
            if self.enable_viewer_sync:
                if not self.viewer_set:
                    if self.overview:
                        self._set_camera(self.cfg.viewer.overview_pos, self.cfg.viewer.overview_lookat)
                    else:
                        ref_pos = [
                            self.root_states[self.cfg.viewer.ref_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                            self.root_states[self.cfg.viewer.ref_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                            self.root_states[self.cfg.viewer.ref_env, 2].item() + self.cfg.viewer.ref_pos_b[2]]
                        ref_lookat = [self.root_states[self.cfg.viewer.ref_env, 0].item(),
                                      self.root_states[self.cfg.viewer.ref_env, 1].item(),
                                      self.root_states[self.cfg.viewer.ref_env, 2].item()]
                        self._set_camera(ref_pos, ref_lookat)
                    self.viewer_set = True
            else:
                self.gym.poll_viewer_events(self.viewer)

        if self.cfg.viewer.record_camera_imgs or (self.viewer and self.enable_viewer_sync):
            self.gym.step_graphics(self.sim)

            if self.cfg.viewer.record_camera_imgs:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                self.camera_image = self.gym.get_camera_image(self.sim, self.envs[self.image_env],
                                                              self.camera_sensors,
                                                              gymapi.IMAGE_COLOR).reshape((self.camera_props.height,
                                                                                           self.camera_props.width,
                                                                                           4))
                self.gym.end_access_image_tensors(self.sim)

            if self.viewer and self.enable_viewer_sync:
                self._draw_target()
                if self.cfg.terrain.measure_height and self.cfg.env.play and self.cfg.env.plot_heights:
                    self._draw_heights()
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
                self.gym.clear_lines(self.viewer)

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

        if self.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_robots()

        env_ids_reset = self.reset_buf.nonzero().flatten()
        self.reset_idx(env_ids_reset)

        self._update_remaining_time()
        self._update_commands_in_base()
        if self.cfg.terrain.measure_height:
            self._measure_height()

        self.compute_observations()
        self.compute_features()

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_joint_targets[:] = self.joint_targets[:]
        self.last_ee_global[:] = self.ee_global[:]
        self.last_ee_vel_global[:] = self.ee_vel_global[:]

    def _check_termination(self):
        if self.cfg.env.evaluation:
            self.reset_buf[:] = False
            return

        self.terminate_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.terminate_buf |= self.root_states[:, 2] < self.base_terrain_heights + self.cfg.rewards.base_height_danger
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = torch.logical_or(self.terminate_buf, self.time_out_buf)

    def compute_observations(self):
        self.obs_buf = torch.cat((self.base_lin_vel,  # 3
                                  self.base_ang_vel,  # 3
                                  (self.dof_pos - self.default_dof_pos),  # 12
                                  self.dof_vel,  # 12
                                  self.projected_gravity,  # 3
                                  ), dim=-1)
        if self.cfg.terrain.measure_height:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.25 - self.measured_height, -1, 1.)
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        self.obs_buf = torch.cat((
            self.obs_buf,
            self.actions,
            self.commands_in_base,
            self.remaining_time.unsqueeze(-1),
        ), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # clip the observation if needed
        if self.cfg.observations.clip_obs:
            self.obs_buf = torch.clip(self.obs_buf, -self.cfg.observations.clip_limit, self.cfg.observations.clip_limit)

    def compute_features(self):
        # FL, FR, HL, HR
        self.feature_buf = torch.cat((
            # self.root_states[:, 2:3],  # 1
            self.base_lin_vel / torch.norm(self.base_lin_vel, dim=1, keepdim=True),  # 3
            # self.ee_vel_global[:, :, 2],  # 4
            # self.base_lin_vel,  # 3
            # (self.dof_pos - self.default_dof_pos),  # 12
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

        num_height_points = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
        if self.cfg.terrain.measure_height:
            noise_vec[33:33 + num_height_points] = noise_scales.height_measurements

        # noise_vec[30 + num_height_points:
        #           30 + num_height_points + self.num_actions] = noise_scales.actions
        noise_vec[33 + num_height_points + self.num_actions:
                  33 + num_height_points + self.num_actions + self.cfg.commands.num_commands] = noise_scales.commands

        return noise_vec * noise_level

    def _refresh_quantities(self):
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        base_global = self.root_states[:, :2].clone()
        base_global += self.terrain.cfg.border_size
        base_global = (base_global / self.terrain.cfg.horizontal_scale).long()
        base_px = base_global[:, 0]
        base_py = base_global[:, 1]
        base_terrain_px = torch.clip(base_px, 2, self.height_samples.shape[0] - 2)
        base_terrain_py = torch.clip(base_py, 2, self.height_samples.shape[1] - 2)

        base_terrain_px = base_terrain_px.unsqueeze(1) + torch.flatten(self.base_check_px)
        base_terrain_py = base_terrain_py.unsqueeze(1) + torch.flatten(self.base_check_py)
        base_heights = self.height_samples[base_terrain_px, base_terrain_py]
        base_heights = torch.min(base_heights, dim=1)[0]
        self.base_terrain_heights = base_heights * self.terrain.cfg.vertical_scale

        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 1))
        # for i in range(self.num_envs):
        #     heights = self.base_target_terrain_heights[i].cpu().numpy()
        #     height_points = self.root_states[i, :2].cpu().numpy()
        #     x = height_points[0]
        #     y = height_points[1]
        #     z = heights
        #     sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        self.ee_global = self.rigid_body_state[:, self.feet_indices, 0:3]
        self.ee_vel_global = self.rigid_body_state[:, self.feet_indices, 7:10]

        self.joint_targets_rate = torch.norm(self.last_joint_targets - self.joint_targets, p=2, dim=1)
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.ee_acc_global = (self.last_ee_vel_global - self.ee_vel_global) / self.dt

        # ee_globals = self.ee_global.clone()
        # ee_globals += self.terrain.cfg.border_size
        # ee_globals = (ee_globals / self.terrain.cfg.horizontal_scale).long()
        # ee_px = ee_globals[:, :, 0].view(-1)
        # ee_py = ee_globals[:, :, 1].view(-1)
        # ee_terrain_px = torch.clip(ee_px, 2, self.height_samples.shape[0] - 2)
        # ee_terrain_py = torch.clip(ee_py, 2, self.height_samples.shape[1] - 2)
        #
        # ee_terrain_px = ee_terrain_px.unsqueeze(1) + torch.flatten(self.ee_check_px)
        # ee_terrain_py = ee_terrain_py.unsqueeze(1) + torch.flatten(self.ee_check_py)
        # ee_terrain_heights = self.height_samples[ee_terrain_px, ee_terrain_py]
        # ee_terrain_heights = torch.min(ee_terrain_heights, dim=1)[0]
        # self.ee_terrain_heights = ee_terrain_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 1))
        # for i in range(self.num_envs):
        #     heights = self.ee_terrain_heights[i].cpu().numpy()
        #     height_points = self.ee_global[i, :, :2].cpu().numpy()
        #     for j in range(heights.shape[0]):
        #         x = height_points[j, 0]
        #         y = height_points[j, 1]
        #         z = heights[j]
        #         sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _resample_commands(self, env_ids):
        # get current position
        if self.cfg.env.play:
            self.commands[env_ids, 0] = self.env_origins[env_ids, 0] + self.cfg.terrain.play_target[0]
            self.commands[env_ids, 1] = self.env_origins[env_ids, 1] + self.cfg.terrain.play_target[1]
            sampled_yaw = np.pi
            # sampled_yaw = np.pi / 2
            # sampled_yaw = 0.0
            # sampled_yaw = -np.pi / 2
        else:
            base_pos = self.root_states[env_ids, 0:2]  # x and y in global frame
            sampled_yaw = torch_rand_float(self.command_ranges["yaw"][0], self.command_ranges["yaw"][1],
                                           (len(env_ids), 1), device=self.device).squeeze(1)
            sampled_direction = torch_rand_float(self.command_ranges["direction"][0],
                                                 self.command_ranges["direction"][1],
                                                 (len(env_ids), 1),
                                                 device=self.device).squeeze(1)
            sampled_radius = torch_rand_float_ring(self.command_ranges["radius"][0],
                                                   self.command_ranges["radius"][1], (len(env_ids), 1),
                                                   device=self.device).squeeze(1)
            self.commands[env_ids, 0] = base_pos[:, 0] + sampled_radius * torch.cos(sampled_direction)
            self.commands[env_ids, 1] = base_pos[:, 1] + sampled_radius * torch.sin(sampled_direction)

        command_global = self.commands[env_ids, :2].clone()
        command_global += self.terrain.cfg.border_size
        command_global = (command_global / self.terrain.cfg.horizontal_scale).long()
        command_px = command_global[:, 0]
        command_py = command_global[:, 1]
        command_px = torch.clip(command_px, 0, self.height_samples.shape[0] - 2)
        command_py = torch.clip(command_py, 0, self.height_samples.shape[1] - 2)

        command_height_0 = self.height_samples[command_px, command_py]
        command_height_1 = self.height_samples[command_px + 1, command_py]
        command_height_2 = self.height_samples[command_px, command_py + 1]

        command_heights = torch.max(torch.cat((
            command_height_0.unsqueeze(1),
            command_height_1.unsqueeze(1),
            command_height_2.unsqueeze(1),
        ), dim=1), dim=1)[0]
        command_terrain_heights = command_heights * self.terrain.cfg.vertical_scale
        self.commands[env_ids, 2] = command_terrain_heights + self.cfg.rewards.base_height_target
        self.commands[env_ids, 3] = sampled_yaw

    def _update_remaining_time(self):
        self.remaining_time = (self.max_episode_length - self.episode_length_buf) / self.max_episode_length

    def _update_commands_in_base(self):
        target_pos_in_global = self.commands[:, 0:3] - self.root_states[:, 0:3]
        target_pos_in_base = quat_rotate_inverse(get_quat_yaw(self.base_quat), target_pos_in_global)

        forward_global = quat_apply(self.base_quat, self.forward_vec)
        base_yaw = torch.atan2(forward_global[:, 1], forward_global[:, 0])
        target_yaw_in_base = wrap_to_pi(self.commands[:, 3] - base_yaw)

        self.commands_in_base[:, 0:3] = target_pos_in_base
        self.commands_in_base[:, 3] = target_yaw_in_base

    def _draw_target(self):
        for i in range(self.num_envs):
            if self.cfg.env.plot_target:
                box_pos = gymapi.Transform(gymapi.Vec3(self.commands[i, 0], self.commands[i, 1], self.commands[i, 2]),
                                           gymapi.Quat.from_euler_zyx(0, 0, self.commands[i, 3]))
                if self.remaining_time[i] < self.remaining_check_time:
                    color_ = (1, 0, 0)
                else:
                    color_ = (1, 1, 0)
                box_geom = gymutil.WireframeBoxGeometry(0.3, 0.15, self.remaining_time[i] * 0.4, color=color_)
                gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[i], box_pos)

    def _draw_heights(self):
        for i in range(self.num_envs):
            color_ = self.draw_height_colors[self.skills[i].item()]
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 8, 8, None, color=color_)
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_height[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _resample_skills(self, env_ids):
        self.skills[env_ids] = torch.randint(low=0, high=self.num_skills, size=(len(env_ids),), device=self.device)

        if self.cfg.env.play:
            self.skills[env_ids] = self.play_skill[env_ids]

        if self.cfg.env.evaluation:
            self.skills[env_ids] = self.eval_skill[env_ids]

        if self.cfg.env.plot_colors:
            for i in range(len(env_ids)):
                self.gym.set_rigid_body_color(self.envs[env_ids[i]], 0, 0, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(self.draw_body_colors[self.skills[env_ids[i]]][0],
                                                          self.draw_body_colors[self.skills[env_ids[i]]][1],
                                                          self.draw_body_colors[self.skills[env_ids[i]]][2]))

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
            joint_target_positions = joint_targets_ + self.default_dof_pos
            joint_target_positions = torch.clamp(joint_target_positions, min=self.joint_lower_limits,
                                                 max=self.joint_upper_limits)
            torques = self.p_gains * (
                    joint_target_positions - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (joint_targets_ - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = joint_targets_
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if self.cfg.env.play:
            self.terrain_cols = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.terrain_rows = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.terrain_origins = torch.from_numpy(self.terrain.sub_terrain_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_rows, self.terrain_cols]
        else:
            if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
                self.terrain_cols = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,),
                                                  device=self.device)
                self.terrain_rows = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,),
                                                  device=self.device)
                self.terrain_origins = torch.from_numpy(self.terrain.sub_terrain_origins).to(self.device).to(
                    torch.float)
                self.env_origins[:] = self.terrain_origins[self.terrain_rows, self.terrain_cols]
            else:
                # create a grid of robots
                num_cols = np.floor(np.sqrt(self.num_envs))
                num_rows = np.ceil(self.num_envs / num_cols)
                xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
                spacing = self.cfg.env.env_spacing
                self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
                self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
                self.env_origins[:, 2] = 0.

    def _init_env_origins(self):
        if self.cfg.terrain.train_all_together == 0:
            self.terrain_cols = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,),
                                              device=self.device)
            self.terrain_rows = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,),
                                              device=self.device)
        elif self.cfg.terrain.train_all_together == 1:
            self.terrain_cols = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.terrain_rows = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,),
                                              device=self.device)
        elif self.cfg.terrain.train_all_together == 2:
            self.terrain_cols = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.terrain_rows = torch.randint(0, int(0.5 * self.terrain.cfg.num_rows),
                                              (self.num_envs,), device=self.device)
        elif self.cfg.terrain.train_all_together == 3:
            self.terrain_cols = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.terrain_rows = torch.randint(0, int(0.5 * self.terrain.cfg.num_rows),
                                              (self.num_envs,), device=self.device)
            self.terrain_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            raise NotImplementedError
        self.env_origins[:] = self.terrain_origins[self.terrain_rows, self.terrain_cols]

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos

        if not self.cfg.env.play:
            self.dof_pos[env_ids] += torch_rand_float(-0.5, 0.5, (len(env_ids), 12), device=self.device)

        self.dof_vel[env_ids] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]

        if self.cfg.env.play:
            self.root_states[env_ids, 0] += self.cfg.terrain.play_init[0]
            self.root_states[env_ids, 1] += self.cfg.terrain.play_init[1]
            self.root_states[env_ids, 5] = 1.0  # z
            self.root_states[env_ids, 6] = 0.0  # w
        else:
            sampled_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
            self.root_states[env_ids, 5] = torch.sin(sampled_yaw / 2)  # z
            self.root_states[env_ids, 6] = torch.cos(sampled_yaw / 2)  # w
            self.root_states[env_ids, :2] += torch_rand_float(-self.cfg.terrain.init_range,
                                                              self.cfg.terrain.init_range,
                                                              (len(env_ids), 2), device=self.device)

        base_global = self.root_states[env_ids, :2].clone()
        base_global += self.terrain.cfg.border_size
        base_global = (base_global / self.terrain.cfg.horizontal_scale).long()
        base_px = base_global[:, 0]
        base_py = base_global[:, 1]
        base_px = torch.clip(base_px, 2, self.height_samples.shape[0] - 2)
        base_py = torch.clip(base_py, 2, self.height_samples.shape[1] - 2)

        base_px = base_px.unsqueeze(1) + torch.flatten(self.base_check_px)
        base_py = base_py.unsqueeze(1) + torch.flatten(self.base_check_py)
        base_heights = self.height_samples[base_px, base_py]
        base_heights = torch.max(base_heights, dim=1)[0]
        self.root_states[env_ids, 2] += base_heights * self.terrain.cfg.vertical_scale

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

    def _update_env_origin(self, env_ids):
        if self.cfg.env.play:
            return

        if not self.init_done:
            self.env_origins[:] = self.terrain_origins[self.terrain_rows, self.terrain_cols]
            return

        if self.cfg.terrain.train_all_together == 0:
            self.terrain_cols[env_ids] = torch.randint(0, self.cfg.terrain.num_cols, (len(env_ids),),
                                                       device=self.device)
            self.terrain_rows[env_ids] = torch.randint(0, self.cfg.terrain.num_rows, (len(env_ids),),
                                                       device=self.device)
        elif self.cfg.terrain.train_all_together == 1:
            self.terrain_rows[env_ids] = torch.randint(0, self.cfg.terrain.num_rows, (len(env_ids),),
                                                       device=self.device)
            pos_distance = torch.norm(self.commands_in_base[env_ids, 0:3], dim=1, p=2)
            move_up = (pos_distance < 0.25)
            move_down = (pos_distance > 2.0) * ~move_up
            self.terrain_cols[env_ids] += 1 * move_up - 1 * move_down
            self.terrain_cols[env_ids] = torch.where(
                torch.Tensor(self.terrain_cols[env_ids] >= self.cfg.terrain.num_cols),
                torch.randint_like(input=self.terrain_cols[env_ids],
                                   high=self.cfg.terrain.num_cols),
                torch.clip(input=self.terrain_cols[env_ids],
                           min=0))
        elif self.cfg.terrain.train_all_together == 2:
            pos_distance = torch.norm(self.commands_in_base[env_ids, 0:3], dim=1, p=2)
            move_up = (pos_distance < 0.25)
            move_down = (pos_distance > 2.0) * ~move_up
            self.terrain_cols[env_ids] += 1 * move_up - 1 * move_down
            finish = torch.Tensor(self.terrain_cols[env_ids] >= self.cfg.terrain.num_cols)
            self.terrain_cols[env_ids] = torch.where(finish,
                                                     torch.randint_like(input=self.terrain_cols[env_ids],
                                                                        high=self.cfg.terrain.num_cols),
                                                     torch.clip(input=self.terrain_cols[env_ids],
                                                                min=0))
            self.terrain_rows[env_ids] = torch.where(finish,
                                                     torch.randint_like(input=self.terrain_rows[env_ids],
                                                                        high=self.cfg.terrain.num_rows),
                                                     self.terrain_rows[env_ids])
        elif self.cfg.terrain.train_all_together == 3:
            pos_distance = torch.norm(self.commands_in_base[env_ids, 0:3], dim=1, p=2)
            move_up = (pos_distance < 0.25)
            move_down = (pos_distance > 2.0) * ~move_up
            after_move = (self.terrain_cols[env_ids] + 1 * move_up - 1 * move_down).to(torch.int32)
            self.terrain_finished[env_ids[after_move >= self.cfg.terrain.num_cols]] = True
            self.terrain_cols[env_ids] = torch.where(self.terrain_finished[env_ids],
                                                     torch.randint_like(input=self.terrain_cols[env_ids],
                                                                        high=self.cfg.terrain.num_cols),
                                                     torch.clip(input=after_move, min=0))

            self.terrain_rows[env_ids] = torch.where(self.terrain_finished[env_ids],
                                                     torch.randint_like(input=self.terrain_rows[env_ids],
                                                                        high=self.cfg.terrain.num_rows),
                                                     self.terrain_rows[env_ids])
        else:
            raise NotImplementedError
        self.env_origins[:] = self.terrain_origins[self.terrain_rows, self.terrain_cols]

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
        self.reward_num_groups = self.cfg.rewards.num_groups
        self.reward_groups = {}
        for i in range(self.reward_num_groups):
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
        self.group_rew_buf = torch.ones(self.num_envs, self.reward_num_groups, dtype=torch.float, device=self.device,
                                        requires_grad=False)

    def compute_reward(self):
        self.rew_buf[:] = 1.0
        for group_name, terms in self.reward_groups.items():
            group_idx = int(group_name)

            if group_idx == 0:  # special treatment for tracking reward since it is given at the end of the episode
                self.group_rew_buf[:, group_idx] = 0.0
                for i in range(len(terms)):
                    reward_name = terms[i]
                    reward_function = getattr(self, '_reward_' + reward_name)
                    reward_sigma = eval(self.reward_terms[reward_name])[1]
                    term_reward = reward_function(reward_sigma)
                    assert torch.isnan(term_reward).sum() == 0
                    self.episode_term_sums[reward_name] += term_reward
                    self.group_rew_buf[:, group_idx] += term_reward
            else:
                self.group_rew_buf[:, group_idx] = 1.0
                for i in range(len(terms)):
                    reward_name = terms[i]
                    reward_function = getattr(self, '_reward_' + reward_name)
                    reward_sigma = eval(self.reward_terms[reward_name])[1]
                    term_reward = reward_function(reward_sigma)
                    assert torch.isnan(term_reward).sum() == 0
                    self.episode_term_sums[reward_name] += term_reward
                    self.group_rew_buf[:, group_idx] *= term_reward

            assert torch.isnan(self.group_rew_buf[:, group_idx]).sum() == 0
            self.episode_group_sums[group_name] += self.group_rew_buf[:, group_idx]

    # ------------------------------------------------------------------------------------------------------------------
    # reward group 0 ---------------------------------------------------------------------------------------------------
    def _reward_posi(self, sigma):
        pos_error = torch.norm(self.commands_in_base[:, 0:3], dim=1, p=2)
        rew = 1 / (1 + sigma * pos_error)
        return rew * (self.remaining_time < self.remaining_check_time)

    def _reward_yawi(self, sigma):
        pos_distance = torch.norm(self.commands_in_base[:, 0:3], dim=1, p=2)
        yaw_error = torch.abs(self.commands_in_base[:, 3])
        rew = 1 / (1 + sigma[0] * yaw_error)
        return rew * (self.remaining_time < self.remaining_check_time) * (pos_distance < sigma[1])

    # reward group 1 ---------------------------------------------------------------------------------------------------
    def _reward_stall_pos(self, sigma):  # do not stand still when target is far
        distance = torch.norm(self.commands_in_base[:, 0:3], dim=1, p=2)
        base_vel = torch.norm(self.base_lin_vel, dim=-1, p=2)
        base_vel_low = torch.clip(sigma[0] - base_vel, min=0.0, max=None) * (distance > sigma[1])
        return torch.exp(-torch.square(base_vel_low / sigma[2]))

    def _reward_joint_targets_rate(self, sigma):  # action should be smooth
        return torch.exp(-torch.square(self.joint_targets_rate / sigma))

    def _reward_feet_acc(self, sigma):  # feet should not vibrate
        feet_acc_error = torch.sum(torch.norm(self.ee_acc_global, p=2, dim=-1), dim=-1)
        return torch.exp(-torch.square(feet_acc_error / sigma))

    def _reward_contact(self, sigma):  # contact forces of the base, shoulder, upper legs should be small
        contact_sum = torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1), dim=1)
        return torch.exp(-torch.square(contact_sum / sigma))

    def _reward_feet_contact(self, sigma):  # contact forces of the feet horizontally
        feet_contact_sum = torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, 0:2], dim=-1), dim=1)
        return torch.exp(-torch.square(feet_contact_sum / sigma))

    # def _reward_dof_acc(self, sigma):
    #     return torch.exp(-torch.square(torch.norm(self.dof_acc, p=2, dim=1) / sigma))

    def _reward_torques(self, sigma):
        return torch.exp(-torch.square(self.total_torque / sigma))

    def _reward_gravity(self, sigma):
        gravity_difference = torch.norm(self.projected_gravity - self.gravity_vec, p=2, dim=1)
        return torch.exp(-torch.square(gravity_difference / sigma))

    # def _reward_ang_xy(self, sigma):
    #     ang_xy = torch.stack(list(get_euler_xyz(self.base_quat)[:2]), dim=1)
    #     ang_xy = torch.norm(ang_xy, p=2, dim=1)
    #     return torch.exp(-torch.square(ang_xy / sigma))

    # reward group 2 ---------------------------------------------------------------------------------------------------
    def _reward_move_towards(self, sigma):  # move towards target
        target_pos_in_base_normalized = self.commands_in_base[:, 0:3] / (
                torch.norm(self.commands_in_base[:, 0:3], dim=-1, keepdim=True) + 1e-8)
        base_lin_vel_normalized = self.base_lin_vel / (
                torch.norm(self.base_lin_vel, dim=-1, keepdim=True) + 1e-8)
        norm_rew = torch.sum(target_pos_in_base_normalized * base_lin_vel_normalized, dim=-1) / 2 + 0.5
        return torch.clip(norm_rew, min=None, max=sigma) / sigma

    def _reward_face_towards(self, sigma):  # face towards target
        target_pos_in_base_normalized = self.commands_in_base[:, 0:3] / (
                torch.norm(self.commands_in_base[:, 0:3], dim=-1, keepdim=True) + 1e-8)
        norm_rew = torch.sum(target_pos_in_base_normalized * self.base_face_direction, dim=-1) / 2 + 0.5
        return torch.clip(norm_rew, min=None, max=sigma) / sigma

    def _reward_joint_default(self, sigma):  # joints should be close to default
        joint_deviation = torch.norm(self.dof_pos - self.default_dof_pos, p=2, dim=1)
        return torch.clip(torch.exp(-torch.square(joint_deviation / sigma[0])), min=None, max=sigma[1]) / sigma[1]

    # def _reward_feet_slip(self, sigma):  # feet should not slip
    #     feet_low = self.ee_global[:, :, 2] < self.ee_terrain_heights + sigma[0]
    #     feet_move = torch.norm(self.ee_global[:, :, :2] - self.last_ee_global[:, :, :2], p=2, dim=2)
    #     sigma_ = sigma[1] + self.ee_global[:, :, 2] * sigma[2]
    #     feet_slip = torch.sum(feet_move * feet_low / sigma_, dim=1)
    #     return torch.exp(-torch.square(feet_slip))

    # def _reward_feet_slip_h(self, sigma):
    #     feet_too_low = self.ee_global[:, :, 2] < sigma[0]
    #     feet_off_ground_when_too_low = torch.sum(self.ee_global[:, :, 2] * feet_too_low, dim=1)
    #     return torch.exp(-torch.square(feet_off_ground_when_too_low / sigma[1]))

    # def _reward_feet_slip_v(self, sigma):
    #     feet_low = self.ee_global[:, :, 2] < sigma[0]
    #     feet_vel_xy = torch.norm(self.ee_vel_global[:, :, :2], p=2, dim=2)
    #     feet_slip_v = torch.sum(feet_vel_xy * feet_low, dim=1)
    #     return torch.exp(-torch.square(feet_slip_v / sigma[1]))
    #
    # def _reward_dof_vel(self, sigma):
    #     return torch.exp(-torch.square(torch.norm(self.dof_vel, p=2, dim=1) / sigma))
    #

    # def _reward_feet_height(self, sigma):
    #     distance = torch.norm(self.commands_in_base[:, 0:3], dim=1, p=2)
    #     feet_height_error = torch.norm(self.ee_global[:, :, 2] - (self.ee_target_terrain_heights + sigma[0]), p=2,
    #     dim=1)
    #     feet_height_error *= distance > sigma[2]
    #     return torch.exp(-torch.square(feet_height_error / sigma[1]))
    #

    # def _reward_stall_yaw(self, sigma):
    #     yaw_distance = torch.abs(self.commands_in_base[:, 2])
    #     base_ang_vel = self.base_ang_vel[:, 2]
    #     base_ang_vel_threshold = -sigma[0] * torch.sign(self.commands_in_base[:, 3])
    #     base_ang_vel_low = torch.clip(base_ang_vel_threshold - base_ang_vel, min=0.0, max=None) * (
    #             base_ang_vel_threshold < 0) + \
    #                        torch.clip(base_ang_vel - base_ang_vel_threshold, min=0.0, max=None) * (
    #                                base_ang_vel_threshold >= 0)
    #     base_ang_vel_low *= (yaw_distance > sigma[1])
    #     return torch.exp(-torch.square(base_ang_vel_low / sigma[2]))

    # def _reward_lin_z(self, sigma):
    #     lin_z_error = self.root_states[:, 2] - (self.base_target_terrain_heights +
    #     self.cfg.rewards.base_height_target)
    #     return torch.exp(-torch.square(lin_z_error / sigma))

    # def _reward_lin_vel_z(self, sigma):
    #     lin_vel_z = self.base_lin_vel[:, 2]
    #     return torch.exp(-torch.square(lin_vel_z / sigma))
    #
    # def _reward_lin_acc_z(self, sigma):
    #     lin_vel_z = self.base_lin_vel[:, 2]
    #     last_lin_vel_z = self.last_root_vel[:, 2]
    #     lin_acc_z = torch.abs(lin_vel_z - last_lin_vel_z)
    #     return torch.exp(-torch.square(lin_acc_z / sigma))
    #
    # def _reward_ang_vel_xy(self, sigma):
    #     ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], p=2, dim=1)
    #     return torch.exp(-torch.square(ang_vel_xy / sigma))
    #
    # def _reward_ang_acc_xy(self, sigma):
    #     ang_vel_xy = self.base_ang_vel[:, :2]
    #     last_ang_vel_xy = self.last_root_vel[:, 3:5]
    #     ang_acc_xy = torch.norm(ang_vel_xy - last_ang_vel_xy, p=2, dim=1)
    #     return torch.exp(-torch.square(ang_acc_xy / sigma))

    # def _reward_pos(self, sigma):
    #     pos_error = torch.clip(torch.norm(self.commands_in_base[:, 0:3], dim=1, p=2), min=None, max=2 * sigma)
    #     rew = torch.exp(-torch.square(pos_error / sigma))
    #     return rew * (self.remaining_time < self.remaining_check_time)
    #
    # def _reward_posl(self, sigma):
    #     pos_error = torch.norm(self.commands_in_base[:, 0:3], dim=1, p=2)
    #     max_pos_error = sigma
    #     rew = (max_pos_error - torch.abs(pos_error)) / max_pos_error
    #     return rew * (self.remaining_time < self.remaining_check_time)

    # def _reward_yawl(self, sigma):
    #     yaw_error = torch.abs(self.commands_in_base[:, 2])
    #     max_yaw_error = sigma[0]
    #     rew = (max_yaw_error - torch.abs(yaw_error)) / max_yaw_error
    #     return rew * (self.remaining_time < self.remaining_check_time)

    # def _reward_yaw(self, sigma):
    #     yaw_error = torch.clip(torch.abs(self.commands_in_base[:, 3]), min=None, max=2 * sigma)
    #     rew = torch.exp(-torch.square(yaw_error / sigma))
    #     return rew * (self.remaining_time < self.remaining_check_time)
