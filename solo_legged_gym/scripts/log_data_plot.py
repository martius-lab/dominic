from solo_legged_gym import ROOT_DIR
from solo_legged_gym.envs import task_registry
from solo_legged_gym.utils import get_args, get_load_path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

START = 20
END = -1


def plot(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    load_path = get_load_path(
        os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name),
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    log_path = os.path.join(os.path.dirname(load_path), "logged_data", "log_data.csv")
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(nrows=4, ncols=3)
    df.command_x.iloc[START:END].plot(ax=axes[0, 0], label='command_x')
    df.base_vel_x.iloc[START:END].plot(ax=axes[0, 0], label='base_vel_x')
    df.command_y.iloc[START:END].plot(ax=axes[0, 1], label='command_y')
    df.base_vel_y.iloc[START:END].plot(ax=axes[0, 1], label='base_vel_y')
    df.command_az.iloc[START:END].plot(ax=axes[0, 2], label='command_az')
    df.base_avel_z.iloc[START:END].plot(ax=axes[0, 2], label='base_avel_z')
    df.joint_targets_rate.iloc[START:END].plot(ax=axes[2, 0], label='joint_targets_rate')

    for ax in fig.get_axes():
        ax.legend()
        ax.grid()

    # data_columns = list(df)
    # n = len(data_columns)
    # ncols = int(np.ceil(np.sqrt(n)))
    # nrows= int(np.ceil(n / ncols))
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    #
    # for i in range(n):
    #     row = int(i // np.ceil(np.sqrt(n)))
    #     col = int(i % np.ceil(np.sqrt(n)))
    #     data = df.loc[:, data_columns[i]]
    #     data.iloc[START:END].plot(ax=axes[row, col], label=data_columns[i])
    #     axes[row, col].legend()
    #     axes[row, col].grid()

    plt.show()


def hex_to_rgb(values):
    rgbs = []
    for value in values:
        value = value.lstrip('#')
        lv = len(value)
        rgbs.append(tuple(int(value[i:i + lv // 3], 16) / 256 for i in range(0, lv, lv // 3)))
    return rgbs


if __name__ == "__main__":
    args = get_args()
    plot(args)
