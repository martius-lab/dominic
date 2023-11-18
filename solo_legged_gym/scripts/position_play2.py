import json
import threading

from isaacgym import gymapi
import torch.nn.functional as func

from solo_legged_gym import ROOT_DIR
from solo_legged_gym.envs import task_registry
from solo_legged_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, get_load_path, \
    update_cfgs_from_dict
import os
import numpy as np
import torch

EXPORT_POLICY = False
REAL_TIME = True
np.set_printoptions(precision=2)


class keyboard_play:

    def __init__(self, args):
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        train_cfg.runner.load_run = -1
        train_cfg.runner.checkpoint = -1

        load_path = get_load_path(
            os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name),
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        print(f"Loading model from: {load_path}")

        load_config_path = os.path.join(os.path.dirname(load_path), f"{train_cfg.runner.experiment_name}.json")
        if os.path.isfile(load_config_path):
            f = open(load_config_path)
            load_config = json.load(f)
            update_cfgs_from_dict(env_cfg, train_cfg, load_config)

        env_cfg.env.num_envs = 8
        env_cfg.env.play = True
        env_cfg.env.plot_heights = False
        env_cfg.env.plot_colors = True
        env_cfg.env.debug = False
        env_cfg.env.episode_length_s = 10
        env_cfg.viewer.overview = True
        env_cfg.viewer.overview_pos = [3.5, 6.5, 2.0]  # [m]
        env_cfg.viewer.overview_lookat = [2, 4, 0]  # [m]
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.init_range = 0.5
        env_cfg.terrain.params = [0.2]
        env_cfg.terrain.play_terrain = "box2"
        env_cfg.terrain.play_init = [-0.5, 0.0]
        env_cfg.terrain.play_target = [-3.5, 0.0]
        env_cfg.terrain.border_size = 5

        env_cfg.observations.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.actuator_lag = True
        env_cfg.domain_rand.randomize_actuator_lag = False
        env_cfg.domain_rand.actuator_lag_steps = 3
        env_cfg.commands.change_commands = False

        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.env = env
        self.runner = task_registry.make_alg_runner(env=self.env, name=args.task, args=args, env_cfg=env_cfg,
                                                    train_cfg=train_cfg)
        self.obs = self.env.get_observations()
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

        self.register_keyboard()

    def register_keyboard(self):
        self.keyboard_control = {
            "reset robot": "r",
            "all skills": "a",
        }

        self.skill_control = {}
        for i in range(self.env.num_skills):
            key = "skill " + str(i)
            value = str(i)
            self.skill_control[key] = value

        for action, key in self.skill_control.items():
            key_enum = getattr(gymapi.KeyboardInput, f"KEY_{key.upper()}")
            self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, key_enum, key)

        for action, key in self.keyboard_control.items():
            key_enum = getattr(gymapi.KeyboardInput, f"KEY_{key.upper()}")
            self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, key_enum, key)

    def play(self):
        if REAL_TIME:
            threading.Timer(1 / 50, self.play).start()
            self.step()
        else:
            while True:
                self.step()

    def step(self):
        # self.obs, _, _, _ = self.env.step(torch.zeros(1, 16, device=self.env.device))

        obs_skills = (self.obs.detach(), self.encode_skills(self.env.skills))

        self.obs, _, _, _, _, _ = self.env.step(self.policy(obs_skills).detach())
        self.update_keyboard_command()

    def update_keyboard_command(self):
        events = self.env.gym.query_viewer_action_events(self.env.viewer)
        for event in events:
            if event.action == "r" and event.value > 0:
                self.env.reset()
            elif event.action == "a" and event.value > 0:
                self.env.play_skill = torch.arange(self.env.cfg.env.num_skills, device=self.env.device,
                                                   requires_grad=False)
            elif event.action in list(self.skill_control.values()) and event.value > 0:
                self.env.play_skill[:] = int(event.action)

            if event.value > 0 and event.action in list(self.skill_control.values()):
                print(list(self.skill_control.keys())[list(self.skill_control.values()).index(event.action)])

    def encode_skills(self, skills):
        return func.one_hot(skills, num_classes=self.env.num_skills).squeeze(1)


if __name__ == "__main__":
    args = get_args()
    kp = keyboard_play(args)
    kp.play()
