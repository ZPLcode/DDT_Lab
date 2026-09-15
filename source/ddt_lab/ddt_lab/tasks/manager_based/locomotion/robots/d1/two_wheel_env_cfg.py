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
from ddt_lab.managers import CostTermCfg
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
        # Smoothly favor a level chassis. Excessive tilt is also handled as
        # an NP3O cost with a gradually tightening tolerance below.
        weight=-5.0,
        params={"target_gravity": TARGET_GRAVITY},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.40},
    )
    air_wheel_height = RewTerm(
        func=mdp.selected_body_height_exp_in_diagonal_lift_phase,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=AIR_WHEELS),
            # D1 wheel radius is 0.087 m.  A 0.18 m wheel-center target gives
            # about 9.3 cm of visible clearance, matching the tucked pose.
            "target_height": 0.18,
            "std": 0.08,
            "activation_level": 4,
            "grace_period_s": 2.0,
            "transition_width_s": 1.0,
        },
    )
    air_wheels_both_air = RewTerm(
        func=mdp.selected_bodies_air_with_support_in_diagonal_lift_phase,
        weight=5.0,
        params={
            "air_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=AIR_WHEELS
            ),
            "support_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SUPPORT_WHEELS
            ),
            "activation_level": 4,
            "grace_period_s": 2.0,
            "transition_width_s": 1.0,
            "min_air_time": 0.02,
            "contact_threshold": 1.0,
        },
    )

    # Zero planar velocity keeps the spin centered. Yaw tracking is active from
    # reset so the policy first builds speed on all four wheels.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_at_gravity_target,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "target_gravity": TARGET_GRAVITY,
            "gate_std": 0.5,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=10.0,
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
    # FR/RL contacts are rewarded through the four-wheel curriculum. From
    # level 4 onward, each episode gets 2 s to spin up before this reward fades
    # to zero. Height and air-time rewards then teach lift without a large
    # contact-penalty cliff.
    air_wheel_contacts = RewTerm(
        func=mdp.selected_body_contact_count_by_diagonal_lift_phase,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=AIR_WHEELS),
            "threshold": 1.0,
            "activation_level": 4,
            "grace_period_s": 2.0,
            "transition_width_s": 1.0,
            "spinup_scale": 1.0,
            "lift_scale": 0.0,
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
    """Terminate only when the chassis hits the ground."""

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*base_link"]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class D1DiagonalSpinCurriculumCfg(CurriculumCfg):
    """Learn four-wheel spin, then hold at 2 rad/s to learn diagonal lift."""

    yaw_speed_levels = CurrTerm(
        func=mdp.diagonal_spin_yaw_speed_levels,
        params={
            "reward_term_name": "track_ang_vel_z",
            "lift_reward_term_name": "air_wheels_both_air",
            "support_reward_term_name": "support_wheel_contacts",
            "levels": (0.5, 1.0, 1.5, 2.0, 2.0),
            "lift_activation_level": 4,
            "lift_success_thresholds": (0.0, 0.0, 0.0, 0.0, 0.20),
            "support_success_threshold": 0.90,
            "support_body_count": len(SUPPORT_WHEELS),
            "illegal_contact_term_name": "illegal_contact",
            "tracking_threshold": 0.7,
            "required_success_rate": 0.6,
            "evaluation_episodes": 4096,
        },
    )


@configclass
class D1DiagonalSpinCostsCfg(CostsCfg):
    """Safety constraints for diagonal-spin training."""

    hip_posture = CostTermCfg(
        func=mdp.hip_pos_l2,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    body_tilt_limit = CostTermCfg(
        func=mdp.body_tilt_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={
            # NP3O collects 24 environment steps per learning iteration:
            # allow 10 degrees initially, then linearly tighten the dead zone
            # to 5 degrees by iteration 3k.
            "initial_limit_rad": math.radians(10.0),
            "final_limit_rad": math.radians(5.0),
            "start_step": 0,
            "end_step": 72_000,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class D1DiagonalSpinFlatEnvCfg(D1FlatEnvCfg):
    """Train D1 to lift FR/RL and spin on the FL/RR diagonal pair."""

    rewards: D1DiagonalSpinRewardsCfg = D1DiagonalSpinRewardsCfg()
    terminations: D1DiagonalSpinTerminationsCfg = D1DiagonalSpinTerminationsCfg()
    curriculum: D1DiagonalSpinCurriculumCfg = D1DiagonalSpinCurriculumCfg()
    costs: D1DiagonalSpinCostsCfg = D1DiagonalSpinCostsCfg()

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
        self.commands.base_velocity.ranges.ang_vel_z = (2.0, 2.0)
        # Playback also starts from the normal four-wheel stance.
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.05, 0.05)
        self.events.reset_base.params["pose_range"]["z"] = (-0.16, -0.13)
