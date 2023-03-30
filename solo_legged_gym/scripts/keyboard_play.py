from isaacgym import gymapi
from isaacgym.torch_utils import get_euler_xyz

from solo_legged_gym import ROOT_DIR
from solo_legged_gym.envs import task_registry
from solo_legged_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, get_load_path
import os
import numpy as np
import torch
import threading
import csv

EXPORT_POLICY = True
LOG_DATA = True
REAL_TIME = False


class keyboard_play:

    def __init__(self, args):
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 1.e4
        env_cfg.env.play = True
        env_cfg.env.debug = False
        env_cfg.viewer.overview = False
        env_cfg.observations.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.commands.change_commands = False

        self.env = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.obs = self.env.get_observations()
        self.runner = task_registry.make_alg_runner(env=self.env, name=args.task, args=args, train_cfg=train_cfg)

        load_path = get_load_path(
            os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name),
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        print(f"Loading model from: {load_path}")
        self.runner.load(load_path)
        self.policy = self.runner.get_inference_policy(device=self.env.device)

        # export policy as a jit module and as onnx model (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(
                ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "policies",
            )
            name = "policy"
            export_policy_as_jit(self.runner.algorithm.actor_critic, self.runner.obs_normalizer, path, filename=f"{name}.pt")
            export_policy_as_onnx(self.runner.algorithm.actor_critic, self.runner.obs_normalizer, path, filename=f"{name}.onnx")
            print("Exported policy to: ", path)

        if LOG_DATA:
            self.prepare_log_file(load_path)

        self.register_keyboard()

    def register_keyboard(self):
        self.keyboard_control = {
            "linear velocity X increment: 0.1 m/s": "w",
            "linear velocity X increment: -0.1 m/s": "s",
            "angular velocity YAW increment: 0.1 rad/s": "q",
            "angular velocity YAW increment: -0.1 rad/s": "e",
            "reset command": "x",
            "reset robot": "r",
        }

        for action, key in self.keyboard_control.items():
            key_enum = getattr(gymapi.KeyboardInput, f"KEY_{key.upper()}")
            self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, key_enum, key)

    def play(self):
        if REAL_TIME:
            threading.Timer(1 / 50, self.play).start()
            self.step()
            self.update_keyboard_command()
        else:
            while True:
                self.step()
                self.update_keyboard_command()

    def step(self):
        # self.obs, _, _, _ = self.env.step(torch.zeros(1, 16, device=self.env.device))
        self.obs, _, _, _ = self.env.step(self.policy(self.obs.detach()).detach())
        if LOG_DATA:
            self.log_data()

    def prepare_log_file(self, load_path):
        path = os.path.join(os.path.dirname(load_path), "logged_data")
        if not os.path.exists(path):
            os.makedirs(path)
        self.log_path = os.path.join(path, "log_data.csv")
        self.step_counter = 0
        header = ['step',
                  'command_x', 'command_az',
                  'base_x', 'base_y', 'base_z', 'base_ax', 'base_ay', 'base_az',
                  'base_vel_x', 'base_vel_y', 'base_vel_z',
                  'base_avel_x', 'base_avel_y', 'base_avel_z',
                  'dphase_FL', 'dphase_FR', 'dphase_RL', 'dphase_RR',
                  'phase_FL', 'phase_FR', 'phase_RL', 'phase_RR',
                  'contact_FL', 'contact_FR', 'contact_RL', 'contact_RR',
                  'joint_targets_rate',
                  'cot',
                  ]
        with open(self.log_path, 'w+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_data(self):
        data = [self.step_counter]
        data.extend(self.env.commands[0, :].tolist())
        data.extend(self.env.root_states[0, :3].tolist())
        data.extend(torch.stack(get_euler_xyz(self.env.base_quat), dim=1)[0, :].tolist())
        data.extend(self.env.base_lin_vel[0, :].tolist())
        data.extend(self.env.base_ang_vel[0, :].tolist())
        data.extend(self.env.delta_phases[0, :].tolist())
        data.extend(self.env.phases[0, :].tolist())
        data.extend(map(lambda x: 1 if x else 0, self.env.ee_contact[0, :].tolist()))
        data.append(self.env.joint_targets_rate[0].item())
        data.append(self.env.cot[0].item())

        self.step_counter += 1
        with open(self.log_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def update_keyboard_command(self):
        events = self.env.gym.query_viewer_action_events(self.env.viewer)
        for event in events:
            if event.action == "x" and event.value > 0:
                self.env.commands[:, :] = 0
            elif event.action == "r" and event.value > 0:
                self.env.reset()
            elif event.action == "w" and event.value > 0:
                self.env.commands[:, 0] += 0.1
            elif event.action == "s" and event.value > 0:
                self.env.commands[:, 0] -= 0.1
            elif event.action == "q" and event.value > 0:
                self.env.commands[:, 1] += 0.1
            elif event.action == "e" and event.value > 0:
                self.env.commands[:, 1] -= 0.1

            if event.value > 0 and event.action in list(self.keyboard_control.values()):
                self.env.commands = torch.round(self.env.commands * 10) / 10
                self.env.commands *= (torch.norm(self.env.commands[:, :2], dim=1) >= 0.1).unsqueeze(1)
                print(list(self.keyboard_control.keys())[list(self.keyboard_control.values()).index(event.action)] +
                      ", now the command is " +
                      np.array2string(self.env.commands[0, :].cpu().detach().numpy().astype(float)))


if __name__ == "__main__":
    args = get_args()
    kp = keyboard_play(args)
    kp.play()

