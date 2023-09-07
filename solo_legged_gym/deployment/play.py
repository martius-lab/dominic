from solo_legged_gym import ROOT_DIR
from isaacgym.torch_utils import (
    quat_rotate_inverse, quat_apply,
)
import os
import csv
import torch
from solo_legged_gym.deployment.tracker import Tracker
from robot_interfaces_solo import solo12
from pynput import keyboard
import argparse
import numpy as np
import torch.nn.functional as func

from solo_legged_gym.utils import get_quat_yaw, wrap_to_pi, quat_apply_yaw

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

TEST_SKILLS = 0
MAX_EPISODE_LENGTH = 6.0
# RUN_NAME = '1_controllability2/92'
RUN_NAME = 'noisy'


# TERRAIN = 0.18  # 0.105
# size_x = 0.9  # 0.8
# size_y = 1.0  # 0.8
TERRAIN = 0.105
size_x = 0.85  # 0.8
size_y = 0.85  # 0.8
# TARGET = [0.0, 0.0, 0.23 + TERRAIN]
TARGET = [0.0, -1.0, 0.23]

MIN_SMOOTH = 0.1  # 0.0: no smooth, 1.0: full smooth (will not move)
KP = 2.5
KD = 0.1

LOG_DATA = False
LOG_FILE_NAME = "log_data.csv"


class Solo12Controller:
    def __init__(self) -> None:
        log_root = os.path.join(ROOT_DIR, "logs", 'solo12_domino_position')
        run_name = RUN_NAME
        self.policy_path = os.path.join(log_root, run_name, "exported", "policies", "policy.pt")
        # server_ip = socket.gethostbyname('enp0s25')
        self.tracker = Tracker('192.168.10.1')

        # load robot configuration
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "config_file",
            type=str,
            help="YAML file with Solo12 driver configuration.",
        )
        args = parser.parse_args()
        config = solo12.Config.from_file(args.config_file)

        robot_data = solo12.SingleProcessData()

        # The backend takes care of communication with the robot hardware.
        self.robot_backend = solo12.create_real_backend(robot_data, config)
        # Initializes the robot (e.g. performs homing).
        self.robot_backend.initialize()

        self.platform = solo12.Frontend(robot_data)
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=None)

        self.actions = torch.zeros(12, dtype=torch.float, requires_grad=False)
        self.commands = torch.zeros(4, dtype=torch.float, requires_grad=False)
        self.commands_in_base = torch.zeros(4, dtype=torch.float, requires_grad=False)

        self.forward_vec = torch.tensor([1., 0., 0.]).repeat((1, 1))

        self.skills = torch.ones(1, dtype=torch.long, requires_grad=False) * TEST_SKILLS

        self.height_points = self._init_height_points()
        self.measured_height = torch.zeros(11 * 11, dtype=torch.float, requires_grad=False)

        self.max_episode_length_s = MAX_EPISODE_LENGTH
        self.dt = 0.02
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.current_step = 0
        self.run_towards = False

        self.tracker_states = torch.zeros(13, dtype=torch.float, requires_grad=False)
        self.default_dof_pos = torch.tensor(
            [0.0, np.pi / 4, -np.pi / 2,
             0.0, np.pi / 4, -np.pi / 2,
             0.0, -np.pi / 4, np.pi / 2,
             0.0, -np.pi / 4, np.pi / 2],
            dtype=torch.float, requires_grad=False)
        self.dof_lower_limits = torch.tensor([-0.9, -np.pi / 2, -np.pi,
                                              -0.9, -np.pi / 2, -np.pi,
                                              -0.9, -np.pi / 2, -np.pi,
                                              -0.9, -np.pi / 2, -np.pi], dtype=torch.float, requires_grad=False)
        self.dof_upper_limits = torch.tensor([0.9, np.pi / 2, np.pi,
                                              0.9, np.pi / 2, np.pi,
                                              0.9, np.pi / 2, np.pi,
                                              0.9, np.pi / 2, np.pi], dtype=torch.float, requires_grad=False)
        self.decimation = 4

        self.kp = KP
        self.kd = KD
        self.smooth_coeff = 1.0
        if LOG_DATA:
            self.prepare_log_file()

    def prepare_log_file(self):
        self.log_path = os.path.join(os.getcwd(), LOG_FILE_NAME)
        self.step_counter = 0
        header = ['step']
        header.extend([f'desired_joint_pos_{i}' for i in range(12)])
        header.extend([f'actual_joint_pos_{i}' for i in range(12)])
        header.extend([f'actual_joint_vel_{i}' for i in range(12)])
        header.extend([f'desired_joint_torque_{i}' for i in range(12)])
        header.extend([f'actual_joint_torque_{i}' for i in range(12)])
        with open(self.log_path, 'w+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_data(self, desired_dof_pos, actual_dof_pos, actual_dof_vel, desired_dof_torques, actual_dof_torques):
        data = [self.step_counter]
        data.extend(desired_dof_pos)
        data.extend(actual_dof_pos)
        data.extend(actual_dof_vel)
        data.extend(desired_dof_torques)
        data.extend(actual_dof_torques)

        self.step_counter += 1
        with open(self.log_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def _on_press(self, key):
        try:
            if key.char == "s":
                self.run_towards = True
                print(f'Start!')
            elif key.char == "d":
                self.run_towards = False
                print(f'Stop!')

        except AttributeError:
            pass

    def run(self):
        while True:
            obs_skills = torch.cat((self.observations.detach(), self.encode_skills(self.skills).squeeze()), dim=-1)
            self.step(self.policy(obs_skills.unsqueeze(0)).detach().squeeze())

    def step(self, actions):
        if self.run_towards:
            # print(self.smooth_coeff)
            self.actions = self.smooth_coeff * self.actions + (1 - self.smooth_coeff) * actions
            self.smooth_coeff -= 0.05
            self.smooth_coeff = np.clip(self.smooth_coeff, MIN_SMOOTH, 1.0)
        else:
            self.actions[:] = 0.0
        # print(self.actions.detach().cpu().numpy().transpose())
        joint_targets = self.actions * 0.25
        for _ in range(self.decimation):
            dof_action, dof_target_pos = self._compute_actions(joint_targets)
            self.t = self.platform.append_desired_action(dof_action)

        if LOG_DATA:
            dof_info = self.platform.get_observation(self.t)
            self.log_data(dof_target_pos.squeeze().tolist(),
                          dof_info.joint_positions,
                          dof_info.joint_velocities,
                          dof_info.joint_target_torques,
                          dof_info.joint_torques)

        if self.run_towards:
            self.commands[0:3] = torch.tensor(TARGET, dtype=torch.float, requires_grad=False)
            self.commands[3] = -np.pi / 2
            self.current_step += 1
        else:
            forward_global = quat_apply(self.tracker_states[3:7], self.forward_vec)
            base_yaw = torch.atan2(forward_global[:, 1], forward_global[:, 0]).squeeze()
            self.commands[0:3] = self.tracker_states[0:3]
            self.commands[3] = base_yaw

        self._compute_observations()

    def initialize(self):
        self.policy = torch.jit.load(self.policy_path)
        self.tracker.connect()
        self.listener.start()

        dof_zero_action = self.default_dof_pos.clone().view(12, 1).float()
        dof_action = solo12.Action()
        dof_action.joint_positions = dof_zero_action.cpu().numpy()
        dof_action.joint_position_gains = np.ones(12) * 1.0
        dof_action.joint_velocity_gains = np.ones(12) * self.kd

        for _ in range(100):
            self.t = self.platform.append_desired_action(dof_action)
            self._compute_observations()
        print("[Solo 12] Initialization finished.")

    def _compute_actions(self, joint_targets):
        # control_type = "P"
        # dof_info = self.platform.get_observation(self.t)
        # dof_pos = self._adapt_dofs(dof_info.joint_positions)
        # dof_vel = self._adapt_dofs(dof_info.joint_velocities)
        #
        # if control_type == "P":
        #     torques = self.kp * (
        #             joint_targets + self.default_dof_pos - dof_pos) - self.kd * dof_vel
        # elif control_type == "V":
        #     raise NotImplementedError
        # elif control_type == "T":
        #     raise NotImplementedError
        # else:
        #     raise NameError(f"Unknown controller type: {control_type}")
        # torques = torch.clip(torques, -1.0, 1.0)
        # apply_torques = torques.clone().view(12, 1).float()
        # print(apply_torques.cpu().numpy())
        # dof_action = solo12.Action()
        # dof_action.joint_torques = apply_torques.cpu().numpy()
        # dof_action.joint_position_gains = np.ones(12) * 0.0
        # dof_action.joint_velocity_gains = np.ones(12) * 0.0

        dof_action = solo12.Action()
        dof_target = joint_targets + self.default_dof_pos
        dof_target = torch.clamp(dof_target, self.dof_lower_limits, self.dof_upper_limits)
        dof_value_reordered = dof_target.clone().view(12, 1).float()
        dof_action.joint_positions = dof_value_reordered.cpu().numpy()
        # print(dof_value_reordered.cpu().numpy().transpose())
        dof_action.joint_position_gains = np.ones(12) * self.kp
        dof_action.joint_velocity_gains = np.ones(12) * self.kd

        return dof_action, dof_value_reordered.cpu().numpy()

    def _adapt_dofs(self, dof_item):
        # # Adapt from Isaac to real when dir = 0
        # if dir == 0:
        #     dof_value_reordered = dof_item.clone().view(12, 1).float()
        #     dof_adapted = solo12.Action()
        #     dof_adapted.joint_positions = dof_value_reordered.cpu().numpy()
        #     dof_adapted.joint_position_gains = np.ones(12) * self.kp
        #     dof_adapted.joint_velocity_gains = np.ones(12) * self.kd
        # # Adapt from real to Isaac when dir = 1
        # elif dir == 1:
        dof_item = torch.tensor(dof_item, dtype=torch.float, requires_grad=False)
        dof_adapted = dof_item
        return dof_adapted

    def _compute_observations(self):
        self.tracker_states[:] = self.tracker.get_root_states()
        self._measure_height()

        base_lin_vel = self.tracker.base_local_lin_vel
        base_ang_vel = self.tracker.base_local_ang_vel
        dof_info = self.platform.get_observation(self.t)
        dof_pos = self._adapt_dofs(dof_info.joint_positions)
        dof_vel = self._adapt_dofs(dof_info.joint_velocities)

        heights = torch.clip(self.tracker_states[2] - 0.25 - self.measured_height, -1, 1.)
        # heights = self.measured_height
        # print(self.tracker_states[2])

        self._update_remaining_time()
        self._update_commands_in_base()

        # print(self.remaining_time)

        self.observations = torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                (dof_pos - self.default_dof_pos),
                dof_vel,
                heights,
                self.actions,
                self.commands_in_base,
                torch.tensor(self.remaining_time, dtype=torch.float).unsqueeze(0)
            ),
            dim=-1,
        )

    def _measure_height(self):
        if TERRAIN <= 0.01:
            self.measured_height[:] = 0
        else:
            self.measured_height[:] = 0
            points = (quat_apply_yaw(self.tracker_states[3:7].unsqueeze(0).repeat(1, self.num_height_points),
                                     self.height_points.unsqueeze(0)) + (self.tracker_states[:3]).unsqueeze(
                0).unsqueeze(1)).squeeze(0)

            on_box = (points[:, 0] > -size_x / 2) & (points[:, 0] < size_x / 2) & (points[:, 1] > -size_y / 2) & (
                        points[:, 1] < size_y / 2)
            self.measured_height[on_box] = TERRAIN
            # print("__________________________")
            # print(points[:, 0].flip(0).reshape((11, 11)).detach().cpu().numpy())
            # print("__________________________")
            # print(points[:, 1].flip(0).reshape((11, 11)).detach().cpu().numpy())
            # print("__________________________")
            print(self.measured_height.flip(0).reshape((11, 11)).detach().cpu().numpy())

    def _init_height_points(self):
        y = torch.tensor(list((np.arange(11) - (11 - 1) / 2) / 10), requires_grad=False)
        x = torch.tensor(list((np.arange(11) - (11 - 1) / 2) / 10), requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_height_points, 3, requires_grad=False)
        points[:, 0] = grid_x.flatten()
        points[:, 1] = grid_y.flatten()
        return points

    def encode_skills(self, skills):
        return func.one_hot(skills, num_classes=8).squeeze(1)

    def _update_remaining_time(self):
        self.remaining_time = (self.max_episode_length - self.current_step) / self.max_episode_length
        self.remaining_time = np.clip(self.remaining_time, 0.0, 1.0)

    def _update_commands_in_base(self):
        print(f'robot global position: {self.tracker_states[0:3].detach().cpu().numpy()}')

        target_pos_in_global = self.commands[0:3] - self.tracker_states[0:3]
        target_pos_in_base = quat_rotate_inverse(get_quat_yaw(self.tracker_states[3:7]),
                                                 target_pos_in_global.unsqueeze(0)).squeeze()

        forward_global = quat_apply(self.tracker_states[3:7].unsqueeze(0), self.forward_vec).squeeze()
        base_yaw = torch.atan2(forward_global[1], forward_global[0])
        print(f'robot global yaw: {base_yaw.detach().cpu().numpy()}')

        target_yaw_in_base = wrap_to_pi(self.commands[3] - base_yaw)

        self.commands_in_base[0:3] = target_pos_in_base
        self.commands_in_base[3] = target_yaw_in_base

        print(f'command in base: {self.commands_in_base.detach().cpu().numpy()}')


if __name__ == "__main__":
    controller = Solo12Controller()
    controller.initialize()
    controller.run()
