
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
    runner.learn()


def train_batch(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    for i in range(5):
        # hyperparams to run over
        seed = 23 * i + 17
        train_cfg.seed = seed
        env_cfg.seed = seed
        # run name
        train_cfg.runner.run_name = f"_no_timeouts_{i}"
        # launch process
        p = Process(target=train, args=(args, env_cfg, train_cfg))
        p.start()
        p.join()
        p.kill()
        print(f">>> Run {i} done!")


if __name__ == "__main__":
    # set_np_formatting()
    args = get_args()
    train_batch(args)
