import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def get_quat_yaw(quat):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_yaw


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


@torch.jit.script
def torch_rand_float_ring(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return torch.sqrt((upper ** 2 - lower ** 2) * torch.rand(*shape, device=device) + lower ** 2)
