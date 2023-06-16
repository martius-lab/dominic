
from solo_legged_gym.envs import task_registry
from solo_legged_gym.utils import get_args, class_to_dict

# python
import argparse
import yaml
import os
from torch.multiprocessing import Process, set_start_method

try:
    set_start_method("spawn")
except RuntimeError as e:
    print(e)


def train(args, env_cfg, train_cfg):
    # initialize the runner and env
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg, train_cfg=train_cfg)
    avg_score =runner.learn()


def train_batch(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    for i in [0.5, 0.7, 0.9]:
        for j in ["auto_0.5", "auto_1", "auto_2"]:
            for k in [0.5, 1.0, 2.0]:
                # for l in [[0.9, 0.99], [0.99, 0.999], [0.999, 0.9999]]:
                # hyperparams to run over
                train_cfg.algorithm.alpha = i
                train_cfg.algorithm.clip_lagrange = j
                train_cfg.algorithm.sigmoid_scale = k
                # train_cfg.algorithm.avg_values_decay_factor = l[0]
                # train_cfg.algorithm.avg_features_decay_factor = l[1]
                # run name
                train_cfg.runner.run_name = f"grid_search_0_{i}_{j}_{k}"
                # launch process
                p = Process(target=train, args=(args, env_cfg, train_cfg))
                p.start()
                p.join()
                p.kill()
                print(f">>> Run grid_search_0_{i}_{j}_{k} done!")


if __name__ == "__main__":
    # set_np_formatting()
    args = get_args()
    train_batch(args)
