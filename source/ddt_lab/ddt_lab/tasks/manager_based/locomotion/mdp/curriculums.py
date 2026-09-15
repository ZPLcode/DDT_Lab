# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def diagonal_spin_yaw_speed_levels(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    command_name: str = "base_velocity",
    levels: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    tracking_threshold: float = 0.7,
    required_success_rate: float = 0.6,
    evaluation_episodes: int = 4096,
) -> dict[str, float]:
    """Raise the fixed yaw command after stable tracking at the current level.

    A completed episode is successful when its time-normalized yaw-tracking
    reward reaches ``tracking_threshold`` of the term's maximum. Early
    termination therefore counts against progression. After advancing, one
    episode duration is left as a cooldown so every environment can receive
    the new command before the next level is evaluated.
    """
    if not levels:
        raise ValueError("Yaw-speed curriculum requires at least one level.")

    command_cfg = env.command_manager.get_term(command_name).cfg

    if not hasattr(env, "_diagonal_spin_curriculum_level"):
        env._diagonal_spin_curriculum_level = 0
        env._diagonal_spin_curriculum_successes = 0
        env._diagonal_spin_curriculum_episodes = 0
        env._diagonal_spin_curriculum_success_rate = 0.0
        env._diagonal_spin_curriculum_level_step = env.common_step_counter
    elif env.common_step_counter - env._diagonal_spin_curriculum_level_step >= env.max_episode_length:
        episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
        reward_weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        normalized_tracking = episode_sums / (env.max_episode_length_s * reward_weight)

        env._diagonal_spin_curriculum_successes += int(
            torch.count_nonzero(normalized_tracking >= tracking_threshold).item()
        )
        env._diagonal_spin_curriculum_episodes += int(normalized_tracking.numel())

        if env._diagonal_spin_curriculum_episodes >= evaluation_episodes:
            success_rate = (
                env._diagonal_spin_curriculum_successes
                / env._diagonal_spin_curriculum_episodes
            )
            env._diagonal_spin_curriculum_success_rate = success_rate
            if (
                success_rate >= required_success_rate
                and env._diagonal_spin_curriculum_level < len(levels) - 1
            ):
                env._diagonal_spin_curriculum_level += 1
                env._diagonal_spin_curriculum_level_step = env.common_step_counter

            env._diagonal_spin_curriculum_successes = 0
            env._diagonal_spin_curriculum_episodes = 0

    yaw_command = float(levels[env._diagonal_spin_curriculum_level])
    command_cfg.ranges.ang_vel_z = (yaw_command, yaw_command)
    return {
        "level": float(env._diagonal_spin_curriculum_level),
        "yaw_command": yaw_command,
        "success_rate": float(env._diagonal_spin_curriculum_success_rate),
    }
