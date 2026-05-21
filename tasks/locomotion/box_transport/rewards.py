"""Reward orchestration — phase-1 velocity-tracking reward + manipulation milestones.

Architecture (see env.py and events.update_autocmd):

    * The HIGH-LEVEL controller (``events.update_autocmd``) writes a velocity
      command into ``env._commands_dict["base_velocity"]`` every step, aimed
      at the box (if not lifted) or the target (if lifted), with magnitude
      tapering to zero within ``cfg.autocmd_stop_distance`` of the goal.

    * The LOW-LEVEL policy was warm-started from a velocity-tracking
      checkpoint. We give it the SAME tracking reward it was trained on:

          + track_lin_vel_xy_yaw_frame_exp   (yaw-frame xy velocity match)
          + track_ang_vel_z_world_exp        (yaw-rate match)
          + feet_air_time_positive_biped     (stepping incentive, only when cmd≠0)

      The "walk-to-box" / "walk-to-target" behaviour emerges from the policy
      tracking the auto-derived command — no separate dense distance shaping.
      When the policy is within stop_distance of the active goal, command
      magnitude is zero, so the tracking reward pays the policy for being
      STILL — exactly when it needs to grasp / release.

    * Dense staged manipulation reward layers on top:

          + reach        (per-arm, each palm → its own box face)
          + squeeze      (peaks when BOTH palms are precisely on the faces)
          + lift_shaping (continuous box-height shaping)
          + lift         (one-shot, box.z first crosses BOX_LIFT_Z)
          + carry        (continuous, gated on lifted; box → target xy)
          + place_bonus  (continuous, box xy within tol of target AND on table)
          + success      (one-shot terminal, success criterion met)
          − drop         (one-shot, box hits the floor; from terminations)
          − termination  (-pen_termination, only when robot fell; read from
                          env._fell_buf, NOT generic mdp.is_terminated, so
                          box-drop / success don't double-fire)

    * Locomotion regularizers (see env_cfg pen_* fields).

The lift one-shot bonus is gated by the per-env latch ``env._lift_achieved``
so it fires once per episode and can't be farmed by bouncing the box. It
uses CURRENT box state to detect the rising-edge transition; the latch is
reset in ``BoxTransportEnv._reset_idx``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from .env import BoxTransportEnv


def compute_reward(env: "BoxTransportEnv") -> torch.Tensor:
    cfg = env.cfg
    s = env._scfg

    # ── Locomotion tracking (against auto-derived command) ──────────────
    track_lin_xy_yaw = locomotion_mdp.rewards.track_lin_vel_xy_yaw_frame_exp(
        env, std=cfg.rew_track_lin_vel_std,
        command_name="base_velocity", asset_cfg=s["robot_all"],
    )
    track_ang_z = locomotion_mdp.rewards.track_ang_vel_z_world_exp(
        env, std=cfg.rew_track_ang_vel_std,
        command_name="base_velocity", asset_cfg=s["robot_all"],
    )
    feet_air_time = locomotion_mdp.rewards.feet_air_time_positive_biped(
        env, command_name="base_velocity",
        threshold=cfg.rew_feet_air_time_threshold,
        sensor_cfg=s["feet_contact"],
    )

    # ── Geometry shortcuts ────────────────────────────────────────────
    box_pos_w    = env.box.data.root_pos_w
    box_quat_w   = env.box.data.root_quat_w
    target_pos_w = env._target_pos_w
    l_palm_w = env.robot.data.body_pos_w[:, env._left_palm_body_id]
    r_palm_w = env.robot.data.body_pos_w[:, env._right_palm_body_id]

    # ── Stage 1 — REACH (per-arm to box faces) + SQUEEZE ───────────────
    # Per-side grasp targets: the left palm is guided to the box's +y face,
    # the right palm to the -y face. The face points are computed in the
    # BOX frame (quat_apply with the box orientation) so they track the
    # box's faces even as it rotates — e.g. when the robot turns 90° while
    # carrying it.
    he = env._box_half_extent
    face_off = torch.zeros(env.num_envs, 3, device=env.device)
    face_off[:, 1] = he
    left_face_w  = box_pos_w + quat_apply(box_quat_w,  face_off)   # +y face
    right_face_w = box_pos_w + quat_apply(box_quat_w, -face_off)   # -y face
    l_to_face = (l_palm_w - left_face_w).norm(dim=-1)
    r_to_face = (r_palm_w - right_face_w).norm(dim=-1)

    # reach: per-arm exp(-d/σ) sum — broad, non-vanishing guidance so each
    # arm independently has a gradient toward its own face.
    std_r = cfg.rew_reach_std
    reach = 0.5 * (
        torch.exp(-l_to_face / std_r) + torch.exp(-r_to_face / std_r)
    )
    # squeeze: exp(-max(l,r)/σ) with a tight σ — peaks ONLY when BOTH palms
    # are precisely on their faces (the bimanual squeeze pose). The max()
    # means a single lagging arm tanks it — it can't be farmed one-handed.
    squeeze = torch.exp(-torch.maximum(l_to_face, r_to_face) / cfg.rew_squeeze_std)

    # ── Stage 2 — LIFT (dense height shaping + one-shot bonus) ─────────
    box_z = box_pos_w[:, 2]
    lift_shaping = torch.clip(
        box_z - env._box_rest_z, 0.0, cfg.rew_lift_shaping_height
    ) / cfg.rew_lift_shaping_height               # 0..1, dense
    lifted_now = box_z > env._box_lift_z
    lift_event = lifted_now & (~env._lift_achieved)
    env._lift_achieved = env._lift_achieved | lifted_now
    lift = lift_event.float()

    # ── Stage 3 — CARRY (dense, gated on currently lifted) ─────────────
    # exp kernel on the box→target xy distance — pulls the HELD box toward
    # the target table. Zero when the box isn't currently up.
    box_xy_err = (box_pos_w[:, :2] - target_pos_w[:, :2]).norm(dim=-1)
    carry = lifted_now.float() * torch.exp(-box_xy_err / cfg.rew_carry_std)

    # ── Stage 4 — PLACE bonus (continuous) ─────────────────────────────
    # Box xy near the per-env target AND z below "well above the table".
    near_xy = box_xy_err < cfg.rew_place_distance_tol
    near_z  = box_z < (env._target_z + 0.10)
    place_bonus = (near_xy & near_z).float()

    # ── Terminal events (one-shot, from terminations.compute_dones) ────
    success    = env._success_buf.float()       # box held on target → win
    drop       = env._box_dropped_buf.float()   # box hit the floor
    term_robot = env._fell_buf.float()          # robot fell

    # ── Locomotion regularizers (kept, reduced weights via cfg) ──────
    lin_vel_z    = mdp.lin_vel_z_l2(env, asset_cfg=s["robot_all"])
    ang_vel_xy   = mdp.ang_vel_xy_l2(env, asset_cfg=s["robot_all"])
    flat_orient  = mdp.flat_orientation_l2(env, asset_cfg=s["robot_all"])
    action_rate  = mdp.action_rate_l2(env)
    dof_torques  = mdp.joint_torques_l2(env, asset_cfg=s["hips_and_knees"])
    dof_acc      = mdp.joint_acc_l2(env, asset_cfg=s["hips_and_knees"])
    pos_limits_ankle = mdp.joint_pos_limits(env, asset_cfg=s["ankles"])
    feet_slide   = locomotion_mdp.rewards.feet_slide(
        env, sensor_cfg=s["feet_contact"], asset_cfg=s["feet_robot"],
    )
    dev_hip   = mdp.joint_deviation_l1(env, asset_cfg=s["hip_yaw_roll"])
    dev_arms  = mdp.joint_deviation_l1(env, asset_cfg=s["arms"])
    dev_hands = mdp.joint_deviation_l1(env, asset_cfg=s["hands"])
    dev_waist = mdp.joint_deviation_l1(env, asset_cfg=s["waist"])

    # ── Torso-upright penalty ──────────────────────────────────────────
    # Gravity projected into the torso_link frame: [0,0,-1] when the torso
    # is vertical, xy-magnitude grows as it leans. Catches a backward torso
    # lean even when the pelvis itself stays level — flat_orientation_l2
    # above only sees the pelvis.
    torso_quat = env.robot.data.body_quat_w[:, env._torso_body_id]    # (N, 4)
    grav_w = torch.tensor(
        [0.0, 0.0, -1.0], device=env.device,
    ).expand(env.num_envs, 3)
    torso_grav = quat_apply_inverse(torso_quat, grav_w)              # (N, 3)
    torso_upright = torso_grav[:, :2].square().sum(dim=-1)            # (N,)

    # ── Weighted sum ──────────────────────────────────────────────────
    reward = (
        + cfg.rew_track_lin_vel_xy * track_lin_xy_yaw
        + cfg.rew_track_ang_vel_z  * track_ang_z
        + cfg.rew_feet_air_time    * feet_air_time
        + cfg.rew_reach            * reach
        + cfg.rew_squeeze          * squeeze
        + cfg.rew_lift_shaping     * lift_shaping
        + cfg.rew_carry            * carry
        + cfg.rew_lift             * lift
        + cfg.rew_place_bonus      * place_bonus
        + cfg.rew_success          * success
        - cfg.pen_drop             * drop
        - cfg.pen_termination      * term_robot
        - cfg.pen_lin_vel_z        * lin_vel_z
        - cfg.pen_ang_vel_xy       * ang_vel_xy
        - cfg.pen_flat_orientation * flat_orient
        - cfg.pen_torso_upright    * torso_upright
        - cfg.pen_action_rate      * action_rate
        - cfg.pen_dof_torques      * dof_torques
        - cfg.pen_dof_acc          * dof_acc
        - cfg.pen_dof_pos_limits   * pos_limits_ankle
        - cfg.pen_feet_slide       * feet_slide
        - cfg.pen_joint_dev_hip    * dev_hip
        - cfg.pen_joint_dev_arms   * dev_arms
        - cfg.pen_joint_dev_hands  * dev_hands
        - cfg.pen_joint_dev_waist  * dev_waist
    )

    # ── Logging to extras (trainer flattens these to wandb) ────────────
    extras = env.extras
    extras["reward/track_lin_vel_xy"]    = track_lin_xy_yaw
    extras["reward/track_ang_vel_z"]     = track_ang_z
    extras["reward/feet_air_time"]       = feet_air_time
    extras["reward/reach"]               = reach
    extras["reward/squeeze"]             = squeeze
    extras["reward/lift_shaping"]        = lift_shaping
    extras["reward/carry"]               = carry
    extras["reward/lift"]                = lift
    extras["reward/place_bonus"]         = place_bonus
    extras["reward/success"]             = success
    extras["reward/drop"]                = drop
    extras["reward/termination"]         = term_robot
    extras["reward/lin_vel_z"]           = lin_vel_z
    extras["reward/ang_vel_xy"]          = ang_vel_xy
    extras["reward/flat_orientation"]    = flat_orient
    extras["reward/torso_upright"]       = torso_upright
    extras["reward/action_rate"]         = action_rate
    extras["reward/dof_torques"]         = dof_torques
    extras["reward/dof_acc"]             = dof_acc
    extras["reward/dof_pos_limits_ankle"] = pos_limits_ankle
    extras["reward/feet_slide"]          = feet_slide
    extras["reward/joint_dev_hip"]       = dev_hip
    extras["reward/joint_dev_arms"]      = dev_arms
    extras["reward/joint_dev_hands"]     = dev_hands
    extras["reward/joint_dev_waist"]     = dev_waist
    extras["reward/total"]               = reward

    # Task diagnostics — milestone progression and geometry.
    extras["task/dist_to_box"]      = (env.robot.data.root_pos_w[:, :2] - box_pos_w[:, :2]).norm(dim=-1)
    extras["task/dist_to_target"]   = (env.robot.data.root_pos_w[:, :2] - target_pos_w[:, :2]).norm(dim=-1)
    extras["task/box_height_z"]     = box_pos_w[:, 2]
    extras["task/box_xy_err"]       = box_xy_err
    extras["task/cmd_lin_x"]        = env._commands_dict["base_velocity"][:, 0]
    extras["task/cmd_lin_y"]        = env._commands_dict["base_velocity"][:, 1]
    extras["task/cmd_ang_z"]        = env._commands_dict["base_velocity"][:, 2]
    extras["task/lifted_now"]       = lifted_now.float()
    extras["task/lifted"]           = env._lift_achieved.float()
    extras["task/gripped"]          = (
        (l_to_face < 0.10) & (r_to_face < 0.10)
    ).float()
    extras["task/success_steps"]    = env._success_step_count.float()

    return reward
