# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1 flat-ground diagonal two-wheel spin task.

Every episode starts from a normal four-wheel stance. The policy lifts FR/RL,
keeps the diagonal FL/RR pair in contact, and spins about the world z-axis.
The objective is fixed, so no gait-mode one-hot is needed.
"""

import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .flat_env_cfg import D1FlatEnvCfg
from .rough_env_cfg import CostsCfg, CurriculumCfg, TerminationsCfg


# The base stays level while one diagonal wheel pair is lifted.
TARGET_GRAVITY = (0.0, 0.0, -1.0)
SUPPORT_WHEELS = ["FL_foot", "RR_foot"]
AIR_WHEELS = ["FR_foot", "RL_foot"]


@configclass
class D1DiagonalSpinRewardsCfg:
    """Rewards for lifting one diagonal pair and spinning on the other."""

    # Keep the body level while FR/RL are lifted and FL/RR support the robot.
    orientation = RewTerm(
        func=mdp.projected_gravity_target_l2,
        weight=-20.0,
        params={"target_gravity": TARGET_GRAVITY},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.35},
    )
    air_wheel_height = RewTerm(
        func=mdp.selected_body_height_exp_at_min_ang_vel_z,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=AIR_WHEELS),
            # D1 wheel radius is 0.087 m, so a 0.18 m wheel-center target
            # leaves about 9.3 cm of ground clearance without over-folding.
            "target_height": 0.18,
            "std": 0.08,
            "min_ang_vel_z": 2.0,
            "transition_width": 0.10,
        },
    )
    air_wheels_both_air = RewTerm(
        func=mdp.selected_bodies_air_with_support_at_min_ang_vel_z,
        weight=5.0,
        params={
            "air_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=AIR_WHEELS
            ),
            "support_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEELS
            ),
            "min_ang_vel_z": 2.0,
            "transition_width": 0.10,
            "min_air_time": 0.02,
            "contact_threshold": 1.0,
        },
    )

    # Zero planar velocity keeps the spin centered. Yaw tracking is active from
    # reset so the policy first builds speed on all four wheels.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_at_gravity_target,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "target_gravity": TARGET_GRAVITY,
            "gate_std": 0.5,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            # The curriculum keeps adjacent targets close enough for this
            # narrow kernel to provide a dense, discriminative signal.
            "std": 0.5,
        },
    )

    # FL/RR remain the support pair throughout both phases.
    support_wheel_contacts = RewTerm(
        func=mdp.selected_body_contact_count,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=SUPPORT_WHEELS),
            "threshold": 1.0,
        },
    )
    # FR/RL contacts are rewarded while speed is low, then smoothly become a
    # penalty. This explicitly teaches four-wheel acceleration before lift-off.
    air_wheel_contacts = RewTerm(
        func=mdp.selected_body_contact_count_by_ang_vel_z_phase,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=AIR_WHEELS),
            "threshold": 1.0,
            "min_ang_vel_z": 2.0,
            "transition_width": 0.10,
            "low_speed_scale": 1.0,
            "high_speed_scale": -2.0,
        },
    )

    # Generic actuator and safety terms, aligned with RobotLab.
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_(hip|thigh|calf)_joint"]
            )
        },
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_(hip|thigh|calf)_joint"]
            )
        },
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_(hip|thigh|calf)_joint"]
            )
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2_no_upright, weight=-0.05)
    undesired_contacts = RewTerm(
        func=mdp.selected_body_contact_count,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh"]),
            "threshold": 1.0,
        },
    )


@configclass
class D1DiagonalSpinTerminationsCfg(TerminationsCfg):
    """Terminate when a non-wheel body hits the ground."""

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["^(?!.*_foot).*"]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class D1DiagonalSpinCurriculumCfg(CurriculumCfg):
    """Learn stable four-wheel yaw tracking before increasing spin speed."""

    yaw_speed_levels = CurrTerm(
        func=mdp.diagonal_spin_yaw_speed_levels,
        params={
            "reward_term_name": "track_ang_vel_z",
            "levels": (0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0),
            "tracking_threshold": 0.7,
            "required_success_rate": 0.6,
            "evaluation_episodes": 4096,
        },
    )


@configclass
class D1DiagonalSpinFlatEnvCfg(D1FlatEnvCfg):
    """Train D1 to lift FR/RL and spin on the FL/RR diagonal pair."""

    rewards: D1DiagonalSpinRewardsCfg = D1DiagonalSpinRewardsCfg()
    terminations: D1DiagonalSpinTerminationsCfg = D1DiagonalSpinTerminationsCfg()
    curriculum: D1DiagonalSpinCurriculumCfg = D1DiagonalSpinCurriculumCfg()
    costs: CostsCfg = CostsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0

        # Keep a single fixed lift objective, then spin in place on FL/RR. A
        # command lasts for the complete episode so the policy
        # is not asked to reverse a high-speed spin halfway through.
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (3.0, 3.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Every reset is the normal, level four-wheel D1 stance.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.60)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.50,
            ".*_foot_joint": 0.0,
        }
        self.scene.robot.init_state.joint_vel = {".*": 0.0}

        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.16, -0.13),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.10, 0.10),
            },
            "velocity_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (-0.05, 0.05),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.98, 1.02)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Initial transition training is deliberately easy. Re-enable these
        # after the policy can spin up and lift the diagonal pair reliably.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_base_com = None
        self.events.randomize_actuator_gains = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.2)


@configclass
class D1DiagonalSpinFlatEnvCfg_PLAY(D1DiagonalSpinFlatEnvCfg):
    """Small deterministic scene for inspecting spin-up and diagonal lift."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        # Evaluate the final target directly; curriculum is training-only.
        self.curriculum.yaw_speed_levels = None
        # Playback also starts from the normal four-wheel stance.
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.05, 0.05)
        self.events.reset_base.params["pose_range"]["z"] = (-0.16, -0.13)
