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
    lift_reward_term_name: str,
    support_reward_term_name: str,
    command_name: str = "base_velocity",
    levels: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.0),
    lift_activation_level: int = 4,
    lift_success_thresholds: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.2),
    support_success_threshold: float = 0.9,
    support_body_count: int = 2,
    illegal_contact_term_name: str = "illegal_contact",
    tracking_threshold: float = 0.7,
    required_success_rate: float = 0.6,
    evaluation_episodes: int = 4096,
) -> dict[str, float]:
    """Advance spin stages only after their required behavior is stable.

    Before ``lift_activation_level``, success requires yaw tracking only. At
    and after that level it additionally requires the requested air pair,
    both support contacts, and no illegal-contact termination. Early
    termination naturally reduces the time-normalized reward scores. After an
    advance, one episode duration is left as a cooldown so every environment
    can receive the new stage before the next evaluation window begins.
    """
    if not levels:
        raise ValueError("Yaw-speed curriculum requires at least one level.")
    if not 0 <= lift_activation_level < len(levels):
        raise ValueError("lift_activation_level must index the levels sequence.")
    if len(lift_success_thresholds) != len(levels):
        raise ValueError("lift_success_thresholds must have one value per level.")
    if any(not 0.0 <= value <= 1.0 for value in lift_success_thresholds):
        raise ValueError("Lift success thresholds must be in [0, 1].")
    if not 0.0 <= support_success_threshold <= 1.0:
        raise ValueError("support_success_threshold must be in [0, 1].")
    if support_body_count <= 0:
        raise ValueError("support_body_count must be positive.")
    if not 0.0 <= tracking_threshold <= 1.0:
        raise ValueError("tracking_threshold must be in [0, 1].")
    if not 0.0 <= required_success_rate <= 1.0:
        raise ValueError("required_success_rate must be in [0, 1].")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive.")

    command_cfg = env.command_manager.get_term(command_name).cfg

    if not hasattr(env, "_diagonal_spin_curriculum_level"):
        env._diagonal_spin_curriculum_level = 0
        env._diagonal_spin_curriculum_successes = 0
        env._diagonal_spin_tracking_successes = 0
        env._diagonal_spin_lift_successes = 0
        env._diagonal_spin_support_successes = 0
        env._diagonal_spin_clean_episodes = 0
        env._diagonal_spin_curriculum_episodes = 0
        env._diagonal_spin_curriculum_success_rate = 0.0
        env._diagonal_spin_tracking_success_rate = 0.0
        env._diagonal_spin_lift_success_rate = 0.0
        env._diagonal_spin_support_success_rate = 0.0
        env._diagonal_spin_clean_rate = 0.0
        env._diagonal_spin_curriculum_level_step = env.common_step_counter
    elif env.common_step_counter - env._diagonal_spin_curriculum_level_step >= env.max_episode_length:
        tracking_weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        lift_weight = env.reward_manager.get_term_cfg(lift_reward_term_name).weight
        support_weight = env.reward_manager.get_term_cfg(support_reward_term_name).weight
        if tracking_weight <= 0.0 or lift_weight <= 0.0 or support_weight <= 0.0:
            raise ValueError("Curriculum reward terms must have positive weights.")

        tracking_score = env.reward_manager._episode_sums[reward_term_name][env_ids] / (
            env.max_episode_length_s * tracking_weight
        )
        lift_score = env.reward_manager._episode_sums[lift_reward_term_name][env_ids] / (
            env.max_episode_length_s * lift_weight
        )
        support_score = env.reward_manager._episode_sums[support_reward_term_name][env_ids] / (
            env.max_episode_length_s * support_weight * support_body_count
        )

        level = env._diagonal_spin_curriculum_level
        tracking_ok = tracking_score >= tracking_threshold
        if level >= lift_activation_level:
            lift_ok = lift_score >= lift_success_thresholds[level]
            support_ok = support_score >= support_success_threshold
            clean = ~env.termination_manager.get_term(illegal_contact_term_name)[env_ids]
        else:
            lift_ok = torch.ones_like(tracking_ok)
            support_ok = torch.ones_like(tracking_ok)
            clean = torch.ones_like(tracking_ok)

        successes = tracking_ok & lift_ok & support_ok & clean
        env._diagonal_spin_curriculum_successes += int(torch.count_nonzero(successes).item())
        env._diagonal_spin_tracking_successes += int(torch.count_nonzero(tracking_ok).item())
        env._diagonal_spin_lift_successes += int(torch.count_nonzero(lift_ok).item())
        env._diagonal_spin_support_successes += int(torch.count_nonzero(support_ok).item())
        env._diagonal_spin_clean_episodes += int(torch.count_nonzero(clean).item())
        env._diagonal_spin_curriculum_episodes += int(tracking_score.numel())

        if env._diagonal_spin_curriculum_episodes >= evaluation_episodes:
            num_episodes = env._diagonal_spin_curriculum_episodes
            success_rate = (
                env._diagonal_spin_curriculum_successes
                / num_episodes
            )
            env._diagonal_spin_curriculum_success_rate = success_rate
            env._diagonal_spin_tracking_success_rate = (
                env._diagonal_spin_tracking_successes / num_episodes
            )
            env._diagonal_spin_lift_success_rate = (
                env._diagonal_spin_lift_successes / num_episodes
            )
            env._diagonal_spin_support_success_rate = (
                env._diagonal_spin_support_successes / num_episodes
            )
            env._diagonal_spin_clean_rate = (
                env._diagonal_spin_clean_episodes / num_episodes
            )
            if (
                success_rate >= required_success_rate
                and env._diagonal_spin_curriculum_level < len(levels) - 1
            ):
                env._diagonal_spin_curriculum_level += 1
                env._diagonal_spin_curriculum_level_step = env.common_step_counter

            env._diagonal_spin_curriculum_successes = 0
            env._diagonal_spin_tracking_successes = 0
            env._diagonal_spin_lift_successes = 0
            env._diagonal_spin_support_successes = 0
            env._diagonal_spin_clean_episodes = 0
            env._diagonal_spin_curriculum_episodes = 0

    level = env._diagonal_spin_curriculum_level
    yaw_command = float(levels[level])
    command_cfg.ranges.ang_vel_z = (yaw_command, yaw_command)
    return {
        "level": float(level),
        "yaw_command": yaw_command,
        "success_rate": float(env._diagonal_spin_curriculum_success_rate),
        "tracking_success_rate": float(env._diagonal_spin_tracking_success_rate),
        "lift_success_rate": float(env._diagonal_spin_lift_success_rate),
        "support_success_rate": float(env._diagonal_spin_support_success_rate),
        "clean_rate": float(env._diagonal_spin_clean_rate),
        "lift_threshold": float(lift_success_thresholds[level]),
    }
