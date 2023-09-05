from solo_legged_gym.envs import *  # needed
import os
import sys
import time
import faulthandler

from cluster import cluster_main


@cluster_main
def train(id, working_dir, **kwargs):
    faulthandler.enable()
    import isaacgym
    import torch
    from solo_legged_gym.utils import (
        get_args,
        merge_config_args_into_cmd_line,
        task_registry,
        update_cfgs_from_dict,
    )

    # remove json from command line argument if present
    sys.argv = [sys.argv[0]]
    # we need these for gymparse to work correctly
    merge_config_args_into_cmd_line(kwargs["solo_legged_gym"]["args"])
    # get args from gymparse
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    update_cfgs_from_dict(env_cfg, train_cfg, kwargs)
    train_cfg.runner.run_name += "_" + str(id)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner = task_registry.make_alg_runner(
        env=env,
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        name=args.task,
        log_root=working_dir,
        args=args,
    )
    avg_score = runner.learn()

    try:
        assert avg_score is not None
    except AssertionError:
        print("Error during metrics return, your ppo agent might not be up to date.")
    metrics = {"score": avg_score}

    return metrics


if __name__ == "__main__":
    train()
