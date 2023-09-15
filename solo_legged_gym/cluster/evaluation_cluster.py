from solo_legged_gym.envs import *  # needed
import os
import sys
import time

from cluster import cluster_main


@cluster_main
def train(id, working_dir, **kwargs):
    import isaacgym
    import torch
    import torch.nn.functional as func
    from solo_legged_gym.utils import (
        get_args,
        merge_config_args_into_cmd_line,
        task_registry,
        update_cfgs_from_dict,
        get_load_path,
    )
    import allogger
    import logging

    allogger.basic_configure(
        logdir=working_dir,
        default_outputs=["hdf"],
        manual_flush=True)
    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO},
                                      default_outputs=['hdf'])

    # remove json from command line argument if present
    sys.argv = [sys.argv[0]]
    # we need these for gymparse to work correctly
    merge_config_args_into_cmd_line(kwargs["solo_legged_gym"]["args"])
    # get args from gymparse
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    update_cfgs_from_dict(env_cfg, train_cfg, kwargs)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    eval_number = id
    ROOT_DIR = os.path.join("/is/rg/al/Data/solo12_data/blm_579_alpha_l0/working_directories")
    load_path = get_load_path(
        ROOT_DIR,
        load_run=str(eval_number),
        checkpoint=-1,
    )
    print(f"Loading model from: {load_path}")

    env_cfg.seed = 2023  # evaluation seed
    env_cfg.env.num_envs = 4096  # evaluation seed
    env_cfg.env.evaluation = True
    env_cfg.env.plot_heights = False
    env_cfg.env.plot_target = False
    env_cfg.env.plot_colors = True
    env_cfg.terrain.train_all_together = 0

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    runner = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg,
                                           train_cfg=train_cfg)
    obs = env.get_observations()
    runner.load(load_path)
    policy = runner.get_inference_policy(device=env.device)

    feature_history = torch.zeros(int(env.max_episode_length),
                                  env.num_envs,
                                  env.num_features,
                                  device=env.device,
                                  dtype=torch.float)
    feat_gamma = 0.95
    successor = torch.zeros(int(env.max_episode_length),
                            env.num_envs,
                            env.num_features,
                            device=env.device,
                            dtype=torch.float)

    mean_init_sf = torch.zeros(8, env.num_features, device=env.device, dtype=torch.float)
    mean_every_sf = torch.zeros(8, env.num_features, device=env.device, dtype=torch.float)

    nearest_init_dist = torch.zeros(8, device=env.device, dtype=torch.float)
    all_init_dist = torch.zeros(8, 8, device=env.device, dtype=torch.float)

    nearest_every_dist = torch.zeros(8, device=env.device, dtype=torch.float)
    all_every_dist = torch.zeros(8, 8, device=env.device, dtype=torch.float)

    for i in range(8):
        env.eval_skill[:] = i
        successor[:] = 0
        env.reset()

        for j in range(int(env.max_episode_length)):
            obs_skills = (obs.detach(), func.one_hot(env.skills, num_classes=env.num_skills).squeeze(1))

            obs, skills, features, _, done, _ = env.step(policy(obs_skills).detach())
            feature_history[j] = features

        for k in reversed(range(int(env.max_episode_length))):
            if k == int(env.max_episode_length) - 1:
                successor[k] = feature_history[k]
            else:
                successor[k] = feature_history[k] + feat_gamma * successor[k + 1]

        mean_init_sf[i] = torch.mean(successor[0], dim=0)
        mean_every_sf[i] = torch.mean(torch.mean(successor, dim=1), dim=0)

    for i in range(8):
        init_sf = mean_init_sf[i]
        every_sf = mean_every_sf[i]

        for j in range(8):
            all_init_dist[i, j] = torch.dist(init_sf, mean_init_sf[j])  # default p=2
            all_every_dist[i, j] = torch.dist(every_sf, mean_every_sf[j])

            nearest_init_dist[i], _ = torch.kthvalue(all_init_dist[i, :], k=2)
            nearest_every_dist[i], _ = torch.kthvalue(all_every_dist[i, :], k=2)

        avg_nearest_init_dist = torch.mean(nearest_init_dist)
        avg_init_dist = torch.mean(all_init_dist)
        avg_nearest_every_dist = torch.mean(nearest_every_dist)
        avg_every_dist = torch.mean(all_every_dist)

        logger.log(avg_nearest_init_dist.detach().cpu().numpy(), "avg_nearest_init_dist", to_hdf=True)
        logger.log(avg_init_dist.detach().cpu().numpy(), "avg_init_dist", to_hdf=True)
        logger.log(avg_nearest_every_dist.detach().cpu().numpy(), "avg_nearest_every_dist", to_hdf=True)
        logger.log(avg_every_dist.detach().cpu().numpy(), "avg_every_dist", to_hdf=True)
        allogger.get_root().flush(children=True)
        allogger.close()

    return 0.0


if __name__ == "__main__":
    train()
