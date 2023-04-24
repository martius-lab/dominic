from solo_legged_gym import ROOT_DIR
from isaacgym.torch_utils import (
    quat_rotate_inverse,
    to_torch,
    get_axis_params,
)
import os
import torch
from solo_legged_gym.deployment.tracker import Tracker
from robot_interfaces_solo import solo12
import socket
from pynput import keyboard
import argparse


class Solo12Controller:
    def __init__(self) -> None:
        log_root = os.path.join(ROOT_DIR, "logs", 'solo12_vanilla')
        run_name = "20230419_104356_test"
        self.policy_path = os.path.join(log_root, run_name, "exported", "policies", "policy.pt")
        server_ip = socket.gethostbyname('octavius')
        self.tracker = Tracker(server_ip)

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
        self.robot_backend = solo12.create_backend(robot_data, config)
        # Initializes the robot (e.g. performs homing).
        self.robot_backend.initialize()

        self.platform = solo12.Frontend(robot_data)
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=None)

        self.actions = torch.zeros(12, dtype=torch.float, requires_grad=False)
        self.commands = torch.zeros(3, dtype=torch.float, requires_grad=False)
        self.tracker_states = torch.zeros(13, dtype=torch.float, requires_grad=False)
        self.default_dof_pos = torch.tensor([0.05, 0.6, -1.4, -0.05, 0.6, -1.4, 0.05, -0.6,  1.4, -0.05, -0.6, 1.4],
                                            dtype=torch.float, requires_grad=False)
        self.decimation = 4

        self.kp = 5.0
        self.kd = 0.1

    def _on_press(self, key):
        try:
            if key.char == "w":
                self.commands[0] += 0.1
                print('Current commands x_lin_vel: {:.2f}'.format(self.commands[0].item()))
            elif key.char == "s":
                self.commands[0] -= 0.1
                print('Current commands x_lin_vel: {:.2f}'.format(self.commands[0].item()))
            elif key.char == "a":
                self.commands[1] -= 0.1
                print('Current commands y_lin_vel: {:.2f}'.format(self.commands[1].item()))
            elif key.char == "d":
                self.commands[1] += 0.1
                print('Current commands y_lin_vel: {:.2f}'.format(self.commands[1].item()))
            elif key.char == "q":
                self.commands[2] -= 0.1
                print('Current commands z_ang_vel: {:.2f}'.format(self.commands[2].item()))
            elif key.char == "e":
                self.commands[2] += 0.1
                print('Current commands z_ang_vel: {:.2f}'.format(self.commands[2].item()))
        except AttributeError:
            pass

    def run(self, num_steps):
        for _ in range(num_steps):
            self.actions = self.policy(self.observations.detach()).view(12)
            self.step(self.actions.detach())

    def step(self, actions):
        self.actions = actions
        # self.actions[:] = 0.0
        for _ in range(self.decimation):
            dof_pos = actions * 0.25 + self.default_dof_pos
            self.t = self.platform.append_desired_action(self._adapt_dofs(dof_pos, dir=0))
        self._compute_observations()

    def initialize(self):
        self.policy = torch.jit.load(self.policy_path)
        self.tracker.connect()
        self.listener.start()
        dof_zero_action = self._adapt_dofs(self.default_dof_pos, dir=0)
        for _ in range(100):
            dof_zero_action.joint_position_gains = (
                        torch.ones(12, dtype=torch.float, requires_grad=False) * 1.0).numpy()
            self.t = self.platform.append_desired_action(dof_zero_action)
            self._compute_observations()
        print("[Solo 12] Initialization finished.")

    def _adapt_dofs(self, dof_item, dir=0):
        # Adapt from Isaac to real when dir = 0
        if dir == 0:
            dof_value_reordered = dof_item.clone().view(12, 1).float()
            dof_adapted = solo12.Action()
            dof_adapted.joint_positions = dof_value_reordered.cpu().numpy()
            dof_adapted.joint_position_gains = (
                        torch.ones(12, dtype=torch.float, requires_grad=False) * self.kp).numpy()
            dof_adapted.joint_velocity_gains = (
                        torch.ones(12, dtype=torch.float, requires_grad=False) * self.kd).numpy()
        # Adapt from real to Isaac when dir = 1
        elif dir == 1:
            dof_item = torch.tensor(dof_item, dtype=torch.float, requires_grad=False)
            dof_adapted = dof_item
        return dof_adapted

    def _compute_observations(self):
        self.tracker_states[:] = self.tracker.get_root_states()
        base_lin_vel = self.tracker.base_local_lin_vel
        base_ang_vel = self.tracker.base_local_ang_vel
        base_quat = self.tracker_states[3:7]
        gravity_vec = to_torch(get_axis_params(-1.0, 2), device='cpu')
        projected_gravity = quat_rotate_inverse(base_quat.unsqueeze(0), gravity_vec.unsqueeze(0)).squeeze(0)
        dof_info = self.platform.get_observation(self.t)
        dof_pos = self._adapt_dofs(dof_info.joint_positions, dir=1)
        dof_vel = self._adapt_dofs(dof_info.joint_velocities, dir=1)

        self.observations = torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                (dof_pos - self.default_dof_pos),
                dof_vel,
                projected_gravity,
                self.actions,
                self.commands,
            ),
            dim=-1,
        )


if __name__ == "__main__":
    num_steps = 5e6
    controller = Solo12Controller()
    controller.initialize()
    controller.run(num_steps)
