# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def swing_up(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for swinging the pole up and keeping the cart near center."""
    asset: Articulation = env.scene[asset_cfg.name]

    cart_pos       = asset.data.joint_pos[:, 0]                    # shape: (num_envs,)
    pole_joint_pos = wrap_to_pi(asset.data.joint_pos[:, 1])        # shape: (num_envs,)

    cart_reward = torch.cos(cart_pos * torch.pi / 4.8)             # ✅ torch, works for any batch size
    pole_reward = (torch.cos(pole_joint_pos) + 1.0) / 2.0         # ✅ torch, works for any batch size

    reward = cart_reward * pole_reward                              # shape: (num_envs,)
    return reward                                                   # ✅ return directly, no wrapping needed