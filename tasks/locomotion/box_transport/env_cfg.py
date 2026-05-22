"""Box-transport env config.

Builds on the velocity-tracking scene (free-base G1, flat ground, contact
sensors) with two tables and a graspable box. Same 43-DoF action space as
locomotion so a locomotion checkpoint can be warm-started.

Assets are primitive ``CuboidCfg`` (table + box) — same pattern as the
reorient task, no Nucleus dependency. Swap to USD assets later if needed.

Reward / observation / termination knobs are exposed as fields here so they
can be overridden from ``configs/task/box_transport.yaml``.
"""
from __future__ import annotations

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from gymnasium.spaces import Box

from assets.robots.g1_cfg import G1_FREE_CFG


# --- Body name constants (shared with the locomotion env) ----------------
ROOT_BODY = "pelvis"
TORSO_BODY = "torso_link"
LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
LEFT_PALM_BODY = "left_hand_palm_link"
RIGHT_PALM_BODY = "right_hand_palm_link"


# --- Scene layout constants (meters, world frame) ------------------------
# Robot spawns at the world origin (0,0,*) facing +x. L-shaped task: walk
# +x to the start table, grab the box, turn 90° and walk -y to the target.
TABLE_SIZE = (1.0, 1.0, 0.70)            # (sx, sy, sz). Top at 0.70 m → box
                                         # centre at 0.82 m, which is the G1's
                                         # natural hand height (pelvis 0.74 m,
                                         # shoulders ~1.1 m). A box at hand
                                         # height makes the bimanual grasp a
                                         # forward reach, NOT a deep squat —
                                         # the single biggest lever for making
                                         # the lift learnable.
TABLE_TOP_Z = TABLE_SIZE[2]              # 0.70 m

START_TABLE_POS  = (2.5,  0.0, TABLE_SIZE[2] / 2)   # 2.5 m forward of spawn
# Target table sits at the SAME x as where the robot stops to grab
# (~1.8 m), straight along -y — so after the robot grabs the box and
# turns 90° right, the target table is directly in front of it. Pulled
# in close (~0.9 m carry) so the focus this round is the grasp+lift, not
# a long carry.
TARGET_TABLE_POS = (1.8, -1.6, TABLE_SIZE[2] / 2)

BOX_SIZE = (0.24, 0.24, 0.24)            # 24 cm cube (was 30) — easier bimanual grip
BOX_MASS = 0.8                           # kg (was 1.5) — scaled down with the box

# Box sits toward the FRONT (robot-facing, smaller-x) edge of the start
# table so the robot can reach it without standing inside the footprint.
BOX_SPAWN_POS = (
    START_TABLE_POS[0] - 0.30,             # x = 2.20  (box front face 0.08 m in)
    START_TABLE_POS[1],
    TABLE_TOP_Z + BOX_SIZE[2] / 2 + 0.01,  # 1 cm above the table surface
)

# Target: sampled near the FRONT (robot-facing, +y) edge of the target
# table — same reasoning, the robot reaches over from in front.
TARGET_ANCHOR_POS = (
    TARGET_TABLE_POS[0],
    TARGET_TABLE_POS[1] + TABLE_SIZE[1] / 2 - 0.20,   # y = -2.20 (0.20 m onto table)
    TABLE_TOP_Z + BOX_SIZE[2] / 2 + 0.001,
)
# Uniform half-range sampled around TARGET_ANCHOR_POS each reset.
TARGET_XY_HALF_RANGE = (0.12, 0.10)
TARGET_Z = TABLE_TOP_Z + BOX_SIZE[2] / 2 + 0.001  # tiny epsilon over the surface

# Drop threshold: episode ends when box.z falls below this (hit floor).
# Box CENTER resting on the floor is at BOX_SIZE/2 = 0.12; 0.20 fires once
# the box is clearly off any table and almost on the floor.
BOX_DROP_Z = 0.20

# Box rest height (center) when sitting on a table top.
BOX_REST_Z = TABLE_TOP_Z + BOX_SIZE[2] / 2          # = 0.82 m
# Lift threshold: box center must rise 10 cm above rest to count as lifted
# — otherwise the flag would fire at spawn before the policy acts.
BOX_LIFT_Z = BOX_REST_Z + 0.10                      # = 0.92 m

# Box half-extent (m) — used to place the per-side grasp targets on the
# box's +y / -y faces for the reach / squeeze reward.
BOX_HALF_EXTENT = BOX_SIZE[1] / 2        # 0.12 m


# --- Phase-A "assisted grasp" constants ----------------------------------
# Two-phase warm-start chain (see scripts/sbatch + warm_start.py):
#   Phase A — assisted_grasp_start=True: the robot spawns standing AT the
#     start table with its arms pre-posed around the box, so the policy
#     learns squeeze → lift → carry → place without the hard walk-in +
#     reach-discovery. Warm-started from the velocity-tracking checkpoint.
#   Phase B — assisted_grasp_start=False (default, the full task): warm-
#     started from the Phase-A checkpoint, which already knows the grasp
#     and carry; Phase B only adds the walk-in and the reach-from-default.
#
# Robot standing pose for Phase A — at the start table, the spot the
# autocmd stops it at. env-local; faces +x toward the box.
ASSISTED_ROBOT_AT_TABLE = (1.80, 0.0, 0.74)
# Pre-posed arm joints (rad) for Phase A — the vetted "ready to manipulate
# a table object" pose from the working reorient task (g1_cfg
# _INIT_JOINT_POS_MANIPULATION): shoulder_pitch -0.9, shoulder_roll 0.4,
# elbow 0.6. 28 values in actuated order: left arm(7), right arm(7), left
# hand(7), right hand(7). Per arm: shoulder_pitch, _roll, _yaw, elbow,
# wrist_roll, _pitch, _yaw. Hands open (0).
ASSISTED_GRASP_ARM_POSE = (
    -0.9, 0.4, 0.0, 0.6, 0.0, 0.0, 0.0,       # left arm
    -0.9, 0.4, 0.0, 0.6, 0.0, 0.0, 0.0,       # right arm
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # left hand (open)
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # right hand (open)
)


# --- Scene --------------------------------------------------------------
@configclass
class BoxTransportSceneCfg(InteractiveSceneCfg):
    """G1 free-base + flat plane + two tables + graspable box."""

    # Flat ground (matches locomotion).
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot (free-base, light-squat init pose, 43 DoF actuated).
    robot = G1_FREE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Two tables — kinematic, primitive cuboid. Wood-ish tan colour.
    start_table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StartTable",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.58, 0.40)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=START_TABLE_POS),
    )

    target_table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetTable",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # Slightly different colour so it's distinguishable in videos.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.50, 0.60)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TARGET_TABLE_POS),
    )

    # The box the robot must transport.
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # Cardboard-ish brown.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.45, 0.30)),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            # High friction (1.5) so a moderate bimanual squeeze holds the
            # box — makes the lift physically easier to discover.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN_POS),
    )

    # Contact sensors over every body — used for foot air-time, illegal-contact
    # termination, and palm/forearm-to-box contact detection.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Lights.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(1.0, 1.0, 1.0)),
    )


# --- Env cfg -------------------------------------------------------------
@configclass
class BoxTransportEnvCfg(DirectRLEnvCfg):
    scene: BoxTransportSceneCfg = BoxTransportSceneCfg(
        num_envs=4096,
        env_spacing=8.0,        # wider than locomotion: tables + walk corridor
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=10 * 2 ** 15),
    )

    decimation: int = 4
    episode_length_s: float = 30.0       # 50 Hz * 30 s = 1500 control steps

    # ── Spaces ───────────────────────────────────────────────────────────
    # Obs layout (164 dims):
    #   [0:141]    locomotion proprioception (kept identical for warm-start;
    #              the velocity_commands slot is auto-derived from the task
    #              phase — see env._pre_physics_step).
    #   [141:144]  box_pos_rel    (box - robot root, in robot base frame)
    #   [144:148]  box_quat       (world; orientation isn't base-frame critical)
    #   [148:151]  box_lin_vel_b  (box linear vel, robot base frame)
    #   [151:154]  target_pos_rel (target - robot root, in robot base frame)
    #   [154:157]  l_palm_to_face (left grasp face - left_palm, base frame)
    #   [157:160]  r_palm_to_face (right grasp face - right_palm, base frame)
    #   [160:163]  box_to_target  (target - box, world frame)
    #   [163]      lifted_flag    (1 if box.z > BOX_LIFT_Z this step)
    observation_space = Box(low=-np.inf, high=np.inf, shape=(164,))
    state_space = 0
    action_space = Box(low=-1.0, high=1.0, shape=(43,))

    # ── Action ──────────────────────────────────────────────────────────
    action_scale: float = 0.5             # same as locomotion (offset-from-default)

    # ── Velocity command derivation ─────────────────────────────────────
    # The locomotion obs slot for velocity_commands is filled by an auto-
    # derived signal: when not lifted, point toward the box; when lifted,
    # point toward the target. Magnitude in [0, max_vel_x]; ang_z via P-control
    # on heading error toward the active goal (same as locomotion's heading
    # command). This keeps the warm-started policy fed with the same kind of
    # input it was trained on.
    autocmd_lin_vel_max:   float = 1.0
    autocmd_ang_vel_max:   float = 1.0
    autocmd_heading_stiffness: float = 0.5
    # Distance to the active goal at which command magnitude reaches zero.
    # 0.40 m → pelvis stops ~0.2 m in front of the table face: the box is
    # then a ~0.4 m forward reach (within G1 arm range from a slight lean),
    # while the legs still stay clear of the table footprint.
    autocmd_stop_distance: float = 0.40

    # ── Reset noise (robot spawn) ──────────────────────────────────────
    reset_pose_x_range:  tuple = (-0.20, 0.20)
    reset_pose_y_range:  tuple = (-0.20, 0.20)
    reset_pose_yaw_range: tuple = (-0.3, 0.3)   # mostly forward-facing
    reset_joint_pos_scale_range: tuple = (1.0, 1.0)
    reset_joint_vel_range: tuple = (0.0, 0.0)

    # Box-spawn xy randomization (uniform half-range). Small — the box is
    # near the table edge, so large noise would push it off.
    reset_box_xy_range: float = 0.05
    # Target xy randomization (uniform half-range around TARGET_ANCHOR_POS).
    reset_target_xy_half: tuple = TARGET_XY_HALF_RANGE

    # ── Reward weights ───────────────────────────────────────────────────
    # Architecture:
    #   * Navigation = phase-1 velocity-tracking reward against the
    #     auto-derived command from ``events.update_autocmd``.
    #   * Manipulation = dense staged shaping, strong enough to dominate
    #     locomotion once the robot is at the box:
    #         reach → squeeze → lift → carry → place → success
    #     Each stage has a non-vanishing dense gradient into the next.

    # Locomotion tracking (mirror of velocity_tracking.yaml defaults).
    rew_track_lin_vel_xy:       float = 1.0
    rew_track_lin_vel_std:      float = 0.5
    rew_track_ang_vel_z:        float = 1.0
    rew_track_ang_vel_std:      float = 0.5
    rew_feet_air_time:          float = 0.75
    rew_feet_air_time_threshold:float = 0.4

    # ── Manipulation reward — dense staged shaping ───────────────────────
    # Staged dense gradient: reach → squeeze → grip-force → lift → carry →
    # place → success. Weights are MODERATED from the earlier desperate
    # values (reach was 16, lift_shaping 12) so the manipulation reward no
    # longer ~10x-swamps the locomotion reward — the warm-started gait was
    # collapsing under that pressure (PPO fell). Manipulation still leads
    # at the box (~25/step vs ~2-3/step locomotion) but no longer crushes
    # the gait.

    # Stage 1 — REACH (per-arm to box faces) + SQUEEZE (both palms on
    # faces) + GRIP-FORCE (the actual press).
    #   reach: each palm pulled toward ITS box face via a per-arm
    #     exp(-d/σ) sum — broad, non-vanishing guidance.
    #   squeeze: exp(-max(l,r)/σ_sq), tight σ — peaks only when BOTH palms
    #     are precisely on their faces (the geometric squeeze pose).
    #   grip_force: the palm↔box contact-force magnitude (capped). This is
    #     the DIRECT reward for pressing the box — without it the only
    #     signal for "press hard" was the indirect lift term, and the
    #     squeeze→lift gap was never crossed. This is the key new term.
    rew_reach:               float = 8.0
    rew_reach_std:           float = 0.35
    rew_squeeze:             float = 6.0
    rew_squeeze_std:         float = 0.12
    rew_grip_force:          float = 5.0
    grip_force_cap:          float = 10.0    # N — reward saturates at a firm grip

    # Stage 2 — LIFT. Dense height shaping clip(box.z-rest, 0, H)/H, always
    # on, plus the one-shot crossing bonus.
    rew_lift_shaping:        float = 8.0
    rew_lift_shaping_height: float = 0.30    # m of rise that saturates the term
    rew_lift:                float = 120.0   # one-shot: box.z first crosses BOX_LIFT_Z

    # Stage 3 — CARRY. Dense exp(-box_to_target_xy / σ) kernel, gated on the
    # box being currently lifted — pulls the held box toward the target.
    rew_carry:               float = 6.0
    rew_carry_std:           float = 0.8

    # Stage 4 — PLACE. Per-step bonus on target + one-shot terminal success.
    rew_place_bonus:         float = 12.0    # per-step while box on target within tol
    rew_success:             float = 300.0   # one-shot terminal: success criterion met
    rew_place_distance_tol:  float = 0.15

    # Negative — discourage dropping.
    pen_drop:                float = 50.0    # one-shot when box hits the floor

    # ── Phase-A assisted-grasp start (opt-in; default off = full task) ──
    # When True, training resets spawn the robot AT the start table with
    # its arms pre-posed around the box (ASSISTED_GRASP_ARM_POSE). Used for
    # the Phase-A warm-start run; Phase B (the full task) leaves it False.
    assisted_grasp_start:    bool  = False

    # Locomotion regularizers — RESTORED to phase-1 (velocity_tracking)
    # values. The earlier 0.5× reduction degraded gait quality (weird /
    # jumpy gaits in the from-scratch runs). Matching phase 1 also keeps
    # the warm-started policy inside the reward landscape it was trained
    # on. Arm / hand deviation stays very low — those joints must move
    # freely for the manipulation.
    pen_termination:        float = 200.0    # robot fell
    pen_lin_vel_z:          float = 0.2
    pen_ang_vel_xy:         float = 0.05
    pen_flat_orientation:   float = 1.0      # pelvis upright
    pen_action_rate:        float = 0.005
    pen_dof_torques:        float = 2.0e-6
    pen_dof_acc:            float = 1.0e-7
    pen_dof_pos_limits:     float = 1.0
    pen_feet_slide:         float = 0.1
    pen_joint_dev_hip:      float = 0.1
    pen_joint_dev_arms:     float = 0.02     # very low — arms must reach
    pen_joint_dev_hands:    float = 0.01
    # POSTURE: the 0.3 waist-deviation penalty did NOT fix the backward
    # lean. Bumped to 0.6 AND backed by a direct torso-upright term
    # (pen_torso_upright) — gravity projected into the torso_link frame,
    # which catches a torso lean even when the pelvis itself is level.
    pen_joint_dev_waist:    float = 0.6
    pen_torso_upright:      float = 1.0

    # ── Termination ──────────────────────────────────────────────────────
    # Robot fell (same 3-signal check as locomotion).
    termination_base_height:   float = 0.4
    termination_gravity_z:     float = -0.4
    illegal_contact_threshold: float = 1.0
    # Box on the floor. See BOX_DROP_Z comment — box center below ~0.20
    # means it is essentially at floor level (rests at 0.15).
    box_drop_z:                float = BOX_DROP_Z
    # Success: box on target table within tolerance for N consecutive steps.
    success_steps_required:    int   = 25      # 0.5 s at 50 Hz

    # ── Observation noise ─────────────────────────────────────────────
    obs_noise_base_lin_vel:    float = 0.1
    obs_noise_base_ang_vel:    float = 0.2
    obs_noise_projected_grav:  float = 0.05
    obs_noise_joint_pos:       float = 0.01
    obs_noise_joint_vel:       float = 1.5
    obs_noise_box_pos:         float = 0.01
    obs_noise_enabled:         bool  = True

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Match locomotion: tick the contact sensor every physics step so
        # foot-air-time / illegal-contact detection don't alias.
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
