from isaacgym import gymapi
from isaacgym.torch_utils import get_euler_xyz
import torch.nn.functional as func

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
np.set_printoptions(precision=2)


class keyboard_play:

    def __init__(self, args):
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        env_cfg.env.num_envs = 5
        env_cfg.env.play = True
        env_cfg.env.debug = False
        env_cfg.observations.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.actuator_lag = True
        env_cfg.domain_rand.randomize_actuator_lag = False
        env_cfg.domain_rand.actuator_lag_steps = 3
        env_cfg.commands.change_commands = False

        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.env = env
        self.runner = task_registry.make_alg_runner(env=self.env, name=args.task, args=args, env_cfg=env_cfg, train_cfg=train_cfg)
        self.obs = self.env.get_observations()

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
                os.path.dirname(load_path),
                "exported",
                "policies",
            )
            name = "policy"
            export_policy_as_jit(self.env.num_skills,
                                 self.runner.policy.num_hidden_dim,
                                 self.runner.policy.policy_latent_layers,
                                 self.runner.policy.masks,
                                 self.runner.policy.action_mean_net,
                                 self.runner.obs_normalizer,
                                 path, filename=f"{name}.pt")
            export_policy_as_onnx(self.env.num_skills,
                                  self.runner.policy.num_hidden_dim,
                                  self.runner.policy.policy_latent_layers,
                                  self.runner.policy.masks,
                                  self.runner.policy.action_mean_net,
                                  self.runner.obs_normalizer,
                                  path, filename=f"{name}.onnx")
            print("--------------------------")
            print("Exported policy to: ", path)
            policy_jit_path = os.path.join(
                os.path.dirname(load_path),
                "exported",
                "policies",
                "policy.pt"
            )
            policy_jit = torch.jit.load(policy_jit_path)
            test_obs = torch.rand(1, env_cfg.env.num_observations)
            test_skill = torch.zeros(1, 1).type(torch.long)
            test_encoded_skill = self.encode_skills(test_skill)

            print("loaded policy test output: ")
            print(self.policy((test_obs.to("cuda:0"), test_encoded_skill.to("cuda:0"))))
            print("loaded jit policy test output: ")
            print(policy_jit(torch.concat((test_obs, test_encoded_skill), dim=-1)))
            print("--------------------------")

        if LOG_DATA:
            self.prepare_log_file(load_path)

        self.register_keyboard()

    def register_keyboard(self):
        self.keyboard_control = {
            "reset robot": "r",
        }
        self.skill_control = {}
        for i in range(self.env.num_skills):
            key = "skill " + str(i)
            value = str(i)
            self.skill_control[key] = value

        for action, key in self.keyboard_control.items():
            key_enum = getattr(gymapi.KeyboardInput, f"KEY_{key.upper()}")
            self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, key_enum, key)

        for action, key in self.skill_control.items():
            key_enum = getattr(gymapi.KeyboardInput, f"KEY_{key.upper()}")
            self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, key_enum, key)

    def play(self):
        while True:
            self.step()

    def step(self):
        # self.obs, _, _, _ = self.env.step(torch.zeros(1, 16, device=self.env.device))

        obs_skills = (self.obs.detach(), self.encode_skills(self.env.skills))

        self.obs, _, _, _, _, _ = self.env.step(self.policy(obs_skills).detach())
        self.update_keyboard_command()
        if LOG_DATA:
            self.log_data()

    def prepare_log_file(self, load_path):
        path = os.path.join(os.path.dirname(load_path), "logged_data")
        if not os.path.exists(path):
            os.makedirs(path)
        self.log_path = os.path.join(path, "log_data.csv")
        self.step_counter = 0
        header = ['step',
                  'command_x', 'command_y', 'command_az',
                  'base_x', 'base_y', 'base_z', 'base_ax', 'base_ay', 'base_az',
                  'base_vel_x', 'base_vel_y', 'base_vel_z',
                  'base_avel_x', 'base_avel_y', 'base_avel_z',
                  'contact_FL', 'contact_FR', 'contact_RL', 'contact_RR',
                  'skill',
                  'feet_contact_force',
                  'joint_targets_rate', 'torques', 'dof_vel', 'dof_acc'
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
        data.extend(map(lambda x: 1 if x else 0, self.env.ee_contact[0, :].tolist()))
        data.append(self.env.skills[0].item())
        data.append(torch.norm(self.env.feet_contact_force[0, :], p=2).item())
        data.append(self.env.joint_targets_rate[0].item())
        data.append(torch.norm(self.env.torques[0, :], p=2).item())
        data.append(torch.norm(self.env.dof_vel[0, :], p=2).item())
        data.append(torch.norm(self.env.dof_acc[0, :], p=2).item())

        self.step_counter += 1
        with open(self.log_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def update_keyboard_command(self):
        events = self.env.gym.query_viewer_action_events(self.env.viewer)
        for event in events:
            if event.action == "r" and event.value > 0:
                self.env.reset()
            elif event.action in list(self.skill_control.values()) and event.value > 0:
                self.env.skills[:] = int(event.action)

            if event.value > 0 and event.action in list(self.skill_control.values()):
                print(list(self.skill_control.keys())[list(self.skill_control.values()).index(event.action)])

    def encode_skills(self, skills):
        return func.one_hot(skills, num_classes=self.env.num_skills).squeeze(1)


if __name__ == "__main__":
    args = get_args()
    kp = keyboard_play(args)
    kp.play()

