from isaacgym.torch_utils import to_torch, torch_rand_float, get_axis_params, quat_rotate_inverse
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import sys
import torch
import os
from solo_legged_gym import ROOT_DIR


# Base class for RL tasks
class BaseTask:

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless

        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        self.graphics_device_id = self.sim_device_id
        if self.headless:
            self.graphics_device_id = -1

        # parse env config
        self._parse_cfg()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id,
                                       self.graphics_device_id,
                                       self.physics_engine,
                                       self.sim_params)
        self._create_ground_plane()
        self._create_envs()
        self.gym.prepare_sim(self.sim)

        self._init_buffers()
        self._set_camera_recording()

        self.enable_viewer_sync = True
        self.overview = self.cfg.viewer.overview
        self.viewer = None
        self.debug_viz = False

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            if self.overview:
                self._set_camera(self.cfg.viewer.overview_pos, self.cfg.viewer.overview_lookat)
            else:
                ref_pos = [self.root_states[self.cfg.viewer.ref_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                           self.root_states[self.cfg.viewer.ref_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                           self.cfg.viewer.ref_pos_b[2]]
                ref_lookat = [self.root_states[self.cfg.viewer.ref_env, 0].item(),
                              self.root_states[self.cfg.viewer.ref_env, 1].item(),
                              0.2]
                self._set_camera(ref_pos, ref_lookat)
            self.viewer_set = True
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_B, "toggle_overview")

    def _set_camera_recording(self):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = self.cfg.viewer.camera_horizontal_fov
        camera_props.width = self.cfg.viewer.camera_width
        camera_props.height = self.cfg.viewer.camera_height
        self.camera_props = camera_props
        self.image_env = self.cfg.viewer.camera_env
        self.camera_sensors = self.gym.create_camera_sensor(self.envs[self.image_env], self.camera_props)
        self.camera_image = np.zeros((self.camera_props.height, self.camera_props.width, 4), dtype=np.uint8)

    def _get_img(self):
        self.camera_image = self.gym.get_camera_image(self.sim, self.envs[self.image_env], self.camera_sensors,
                                                      gymapi.IMAGE_COLOR).reshape((self.camera_props.height,
                                                                                   self.camera_props.width, 4))

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def compute_observations(self):
        raise NotImplementedError

    def get_observations(self):
        return self.obs_buf

    def render(self, sync_frame_time=True):
        # fetch results
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        if self.cfg.viewer.record_camera_imgs:
            ref_pos = [self.root_states[self.image_env, 0].item() + self.cfg.viewer.camera_pos_b[0],
                       self.root_states[self.image_env, 1].item() + self.cfg.viewer.camera_pos_b[1],
                       self.cfg.viewer.camera_pos_b[2]]
            ref_lookat = [self.root_states[self.image_env, 0].item(),
                          self.root_states[self.image_env, 1].item(),
                          0.2]
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
                if self.cfg.env.play:
                    ref_pos = [self.root_states[self.cfg.viewer.ref_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                               self.root_states[self.cfg.viewer.ref_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                               self.cfg.viewer.ref_pos_b[2]]
                    ref_lookat = [self.root_states[self.cfg.viewer.ref_env, 0].item(),
                                  self.root_states[self.cfg.viewer.ref_env, 1].item(),
                                  0.2]
                    self._set_camera(ref_pos, ref_lookat)
                else:
                    if not self.viewer_set:
                        if self.overview:
                            self._set_camera(self.cfg.viewer.overview_pos, self.cfg.viewer.overview_lookat)
                        else:
                            ref_pos = [
                                self.root_states[self.cfg.viewer.ref_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                                self.root_states[self.cfg.viewer.ref_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                                self.cfg.viewer.ref_pos_b[2]]
                            ref_lookat = [self.root_states[self.cfg.viewer.ref_env, 0].item(),
                                          self.root_states[self.cfg.viewer.ref_env, 1].item(),
                                          0.2]
                            self._set_camera(ref_pos, ref_lookat)
                        self.viewer_set = True
            else:
                self.gym.poll_viewer_events(self.viewer)

        if self.cfg.viewer.record_camera_imgs or (self.viewer and self.enable_viewer_sync):
            self.gym.step_graphics(self.sim)

            if self.cfg.viewer.record_camera_imgs:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                self._get_img()
                self.gym.end_access_image_tensors(self.sim)

            if self.viewer and self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)

    # ------------------------------------------------------------------------------------------------------------------

    def _parse_cfg(self):
        self.num_envs = self.cfg.env.num_envs
        self.num_obs = self.cfg.env.num_observations
        self.num_actions = self.cfg.env.num_actions
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(root=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + \
                               self.cfg.init_state.rot + \
                               self.cfg.init_state.lin_vel + \
                               self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim,
                                             gymapi.Vec3(0., 0., 0.),
                                             gymapi.Vec3(0., 0., 0.),
                                             int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _process_rigid_shape_props(self, props, env_id):
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_rigid_body_props(self, props):
        # randomize mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _init_buffers(self):
        # basic buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_body_jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        # contact_forces shape: num_envs, num_bodies, xyz axis
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.rigid_body_jacobian = gymtorch.wrap_tensor(rigid_body_jacobian)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

    def _set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)
