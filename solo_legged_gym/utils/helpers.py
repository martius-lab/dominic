from isaacgym import gymapi
from isaacgym import gymutil
import os
import sys
import copy
import numpy as np
import random
import torch
import argparse


def get_args() -> argparse.Namespace:
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "a1_vanilla",
            "help": "Start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--dv",
            "action": "store_false",
            "default": True,
            "help": "Disable viewer",
        },
        {
            "name": "--w",
            "action": "store_true",
            "default": False,
            "help": "Turn on Weights and Bias writer",
        }
    ]
    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy using IsaacGym", custom_parameters=custom_parameters)
    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    args.headless = False
    args.device = "cuda:0"
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if hasattr(attr, "__dict__"):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)
    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        if 'wandb' in runs: runs.remove('wandb')
        if 'cluster' in runs: runs.remove('cluster')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_env_cfg_from_args(env_cfg, args):
    if args.dv is not None:
        env_cfg.viewer.enable_viewer = args.dv
    return env_cfg


def update_train_cfg_from_args(train_cfg, args):
    if args.w is not None:
        train_cfg.runner.wandb = args.w
    return train_cfg


def export_policy_as_jit(policy_latent, action, normalizer, path, filename="policy.pt"):
    policy_exporter = TorchPolicyExporter(policy_latent, action, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(policy_latent, action, normalizer, path, filename="policy.onnx"):
    policy_exporter = OnnxPolicyExporter(policy_latent, action, normalizer)
    policy_exporter.export(path, filename)


class TorchPolicyExporter(torch.nn.Module):
    def __init__(self, policy_latent, action, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.policy_latent = copy.deepcopy(policy_latent)
        self.action = copy.deepcopy(action)

    def forward(self, x):
        return self.action(self.policy_latent(self.normalizer(x)))

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy_latent, action, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.policy_latent = copy.deepcopy(policy_latent)
        self.action = copy.deepcopy(action)

    def forward(self, x):
        return self.action(self.policy_latent(self.normalizer(x)))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.policy_latent[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def merge_config_args_into_cmd_line(args):
    # arguments that don't need a value in command line
    store_true_keywords = ["dv", "w"]
    for k, v in args.items():
        if v is None:
            continue
        elif k in store_true_keywords:
            if v is True:
                sys.argv.append(f"--{k}")
        else:
            sys.argv.append(f"--{k}={v}")


def update_cfgs_from_dict(env_cfg, train_cfg, update_cfg):
    update_class_from_dict(env_cfg, update_cfg["solo_legged_gym"]["env_cfg"])
    update_class_from_dict(train_cfg, update_cfg["solo_legged_gym"]["train_cfg"])
