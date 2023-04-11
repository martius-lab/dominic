from solo_legged_gym.envs import *  # needed
from solo_legged_gym.utils import task_registry, get_args
from cluster import cluster_main


@cluster_main
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner.learn()


if __name__ == '__main__':
    args = get_args()
    train(args)
