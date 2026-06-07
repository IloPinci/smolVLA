import numpy as np
import torch
from dataclasses import dataclass, field

@dataclass
class OracleConfig:
    # Noise
    pos_noise_sigma: float  = 0.004
    rot_noise_sigma: float  = np.deg2rad(2)  

    # Trajectory
    steps_per_segment: int  = 30
    pregrasp_clearance: float = 0.12
    lift_height: float        = 0.15
    carry_clearance: float    = 0.15

    # Gripper — position control only (one-sided jaw)
    gripper_open: float       = 0.8
    # FIX #4: tighter close position + higher kp/kv in two_camera_setup.py
    gripper_close_safe: float = 0.18   # was 0.22 — jaw now presses harder against cube

    # IK tolerances
    ik_pos_tol: float = 1e-4
    ik_rot_tol: float = 1e-4

    # Settle steps per waypoint
    settle_steps: list = field(default_factory=lambda: [
        10,   # WP0  hover
        50,   # WP1  descend        — extra settle so arm is truly still before close
        100,  # WP2  close gripper  # FIX #3/#4: longer settle lets contact force build fully
        25,   # WP3  lift
        25,   # WP4  carry
        60,   # WP5  release
    ])


# Joint index aliases matching so101 XML DOF ordering
_ARM_DOFS     = np.arange(5)
_GRIPPER_DOF  = np.array([5])


class SO101Oracle:
    def __init__(self, so101, scene, cameras=None, cfg: OracleConfig = OracleConfig()):
        self.robot  = so101
        self.scene  = scene
        self.cfg    = cfg
        self._rng   = np.random.default_rng()

        # Terminal kinematic link used as IK target
        self.end_effector = so101.get_link("moving_jaw_so101_v1")

        # Downward-facing gripper orientation (180 deg about Y)
        self.grasp_quat = np.array([0.707107, 0.0, -0.707107, 0.0])

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_episode(self, cube, target_zone, table_height,
                is_success_fn, check_sub_goals_fn) -> bool:

        cube_pos   = cube.get_pos().cpu().numpy()
        target_pos = target_zone.get_pos().cpu().numpy()

        waypoints = self._compute_waypoints(cube_pos, target_pos, table_height)
        waypoints.insert(2, waypoints[1].copy())

        gripper_targets = [
            self.cfg.gripper_open,        # WP0  hover
            self.cfg.gripper_open,        # WP1  descend
            self.cfg.gripper_close_safe,  # WP2  close (arm stationary)
            self.cfg.gripper_close_safe,  # WP3  lift
            self.cfg.gripper_close_safe,  # WP4  carry
            # WP5 release is handled separately below — not in this list
        ]

        prev_qpos = self.robot.get_dofs_position()
        success   = False

        # Run WP0–WP4 as normal (gripper stays closed throughout motion)
        for wp_pos, g_target, s_steps in zip(
                waypoints[:-1], gripper_targets, self.cfg.settle_steps[:-1]):

            prev_qpos, success = self._execute_segment(
                wp_pos, g_target, s_steps, prev_qpos,
                cube, target_zone, table_height, check_sub_goals_fn,
            )
            if success:
                break

        # WP5 — dedicated release sequence (only runs if not already succeeded)
        if not success:
            success = self._execute_release_segment(
                waypoints[-1],          # release position above target
                self.cfg.settle_steps[-1],
                prev_qpos,
                cube, target_zone, table_height,
                is_success_fn,
            )

        if not success:
            success = is_success_fn(cube, target_zone, table_height)

        return success



    def _execute_release_segment(self, target_cart_pos: np.ndarray,
                              settle_steps: int,
                              prev_qpos: torch.Tensor,
                              cube, target_zone, table_height,
                              is_success_fn) -> bool:
        """
        Three-phase release:
        1. Move arm to release position — gripper stays CLOSED (cube held).
        2. Open gripper — arm stationary, cube drops.
        3. Settle — wait for cube to stop bouncing, then check success once.
        """
        cfg = self.cfg

        yaw_noise  = self._rng.normal(0, cfg.rot_noise_sigma)
        noisy_quat = self._perturb_quat_z(self.grasp_quat, yaw_noise)

        # IK for the release Cartesian position
        target_qpos = self.robot.inverse_kinematics(
            link    = self.end_effector,
            pos     = target_cart_pos,
            quat    = noisy_quat,
            pos_tol = cfg.ik_pos_tol,
            rot_tol = cfg.ik_rot_tol,
        )

        # ── Phase 1: move arm to release position, gripper stays CLOSED ──────
        steps  = cfg.steps_per_segment
        alphas = torch.linspace(0.0, 1.0, steps,
                                device=target_qpos.device).unsqueeze(1)

        # Keep gripper closed during arm motion
        closed_qpos        = target_qpos.clone()
        closed_qpos[5]     = prev_qpos[5]          # carry over closed position
        interp_configs     = prev_qpos + alphas * (closed_qpos - prev_qpos)
        interp_configs[:, 5] = prev_qpos[5]        # belt-and-suspenders: lock gripper col

        for interp_qpos in interp_configs:
            self.robot.control_dofs_position(interp_qpos[_ARM_DOFS], dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(interp_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        # Hold arm still for a few steps so it fully settles before releasing
        arm_settle_qpos    = closed_qpos.clone()
        for _ in range(15):
            self.robot.control_dofs_position(arm_settle_qpos[_ARM_DOFS], dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(arm_settle_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        # ── Phase 2: open gripper — arm stationary ────────────────────────────
        open_qpos      = closed_qpos.clone()
        open_qpos[5]   = cfg.gripper_open

        for _ in range(20):     # give jaw time to actually swing open
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS], dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        # ── Phase 3: settle — wait for cube to stop bouncing ─────────────────
        for _ in range(settle_steps):
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS], dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        # Single success check after everything has settled
        return is_success_fn(cube, target_zone, table_height)


    # ------------------------------------------------------------------ #
    #  Waypoint computation                                                #
    # ------------------------------------------------------------------ #

    def _compute_waypoints(self, cube_pos: np.ndarray,
                           target_pos: np.ndarray,
                           table_height: float) -> list:
        cfg = self.cfg

        # Physical offset from the IK link origin to the jaw contact surface
        gripper_length = 0.06

        # FIX #1/#2: removed x-offset (was 0.015) so the static jaw doesn't
        # sweep through the cube volume during descent. y_offset kept small
        # to centre the cube in the jaw gap.
        x_offset = 0.016
        y_offset = -0.000

        wp0 = cube_pos   + np.array([x_offset, y_offset, cfg.pregrasp_clearance + gripper_length])
        # FIX #3: raise descent Z from 0.010 to 0.018 so jaw closes at
        # cube mid-height (~half of 0.03 cube = 0.015), not below equator.
        wp1 = cube_pos   + np.array([x_offset, y_offset, 0.01 + gripper_length])
        wp2 = cube_pos   + np.array([x_offset, y_offset, cfg.lift_height + gripper_length])
        wp3 = target_pos + np.array([0.0,      0.0,      cfg.carry_clearance + gripper_length])
        # FIX #6: raise release Z so gripper tip doesn't contact the target
        # cylinder. Cube (0.03 tall) is released ~3 cm above target surface.
        wp4 = target_pos + np.array([x_offset,      0.0,      0.04 + gripper_length])

        waypoints = []
        for nominal in [wp0, wp1, wp2, wp3, wp4]:
            noise    = self._rng.normal(0, cfg.pos_noise_sigma, size=3)
            noise[2] = abs(noise[2])   # never push Z below nominal
            noise[0] *= 0.3            # FIX #3: tighter lateral scatter
            noise[1] *= 0.3
            waypoints.append(nominal + noise)

        return waypoints

    # ------------------------------------------------------------------ #
    #  Segment execution                                                   #
    # ------------------------------------------------------------------ #

    def _execute_segment(self, target_cart_pos: np.ndarray,
                     gripper_target: float,
                     settle_steps: int,
                     prev_qpos: torch.Tensor,
                     cube, target_zone, table_height,
                     check_sub_goals_fn):

        cfg = self.cfg

        yaw_noise  = self._rng.normal(0, cfg.rot_noise_sigma)
        noisy_quat = self._perturb_quat_z(self.grasp_quat, yaw_noise)

        target_qpos = self.robot.inverse_kinematics(
            link    = self.end_effector,
            pos     = target_cart_pos,
            quat    = noisy_quat,
            pos_tol = cfg.ik_pos_tol,
            rot_tol = cfg.ik_rot_tol,
        )

        target_qpos[5] = gripper_target

        steps  = cfg.steps_per_segment
        alphas = torch.linspace(0.0, 1.0, steps,
                                device=target_qpos.device).unsqueeze(1)
        interp_configs = prev_qpos + alphas * (target_qpos - prev_qpos)

        # ── FIX: freeze gripper during the arm-motion phase ──────────────────
        # interp_configs already blends gripper from prev→target across all
        # steps, which causes the jaw to open mid-air on the carry→release leg.
        # Lock column 5 to the *current* gripper position for the whole
        # interpolation phase; only switch to gripper_target in the settle phase.
        interp_configs[:, 5] = prev_qpos[5]
        # ─────────────────────────────────────────────────────────────────────

        success = False

        def _step(qpos: torch.Tensor) -> bool:
            self.robot.control_dofs_position(
                qpos[_ARM_DOFS],
                dofs_idx_local=_ARM_DOFS,
            )
            self.robot.control_dofs_position(
                qpos[_GRIPPER_DOF],
                dofs_idx_local=_GRIPPER_DOF,
            )
            self.scene.step()
            sub = check_sub_goals_fn(self.robot, cube, target_zone, table_height)
            return sub["placed"]

        # Interpolation phase — arm moves, gripper stays locked
        for interp_qpos in interp_configs:
            if _step(interp_qpos):
                return interp_qpos, True

        # Settle phase — arm holds position, gripper now actuates to target
        for _ in range(settle_steps):
            if _step(target_qpos):
                return target_qpos, True

        return target_qpos, False

    # ------------------------------------------------------------------ #
    #  Quaternion utility                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _perturb_quat_z(base_quat: np.ndarray, angle_rad: float) -> np.ndarray:
        half = angle_rad / 2.0
        dq   = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        w1, x1, y1, z1 = base_quat
        w2, x2, y2, z2 = dq
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])


# ====================================================================== #
#  Demonstration collection                                               #
# ====================================================================== #

def collect_demonstrations(so101, scene, cameras, cube, target_zone,
                           table_height, is_success_fn, check_sub_goals_fn,
                           n_episodes: int = 150,
                           output_dir: str = "demos/") -> list:

    cfg    = OracleConfig()
    oracle = SO101Oracle(so101, scene, cameras, cfg)

    results   = []
    successes = 0
    home_pose = np.zeros(so101.n_dofs)

    for ep in range(n_episodes):

        # FIX #7: set_dofs_position TELEPORTS joints to home instantly.
        # control_dofs_position only sets motor targets — without this the
        # arm physically stays at whatever pose the last episode ended at.
        so101.set_dofs_position(home_pose)
        so101.set_dofs_velocity(np.zeros(so101.n_dofs))
        so101.control_dofs_position(home_pose)

        # Reset cube
        cube.set_pos(np.array([0.3, 0.0, table_height + 0.015]))
        cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

        # Let physics settle
        for _ in range(20):
            so101.set_dofs_velocity(np.zeros(so101.n_dofs))
            so101.control_dofs_position(home_pose)
            scene.step()

        success = oracle.run_episode(
            cube, target_zone, table_height,
            is_success_fn, check_sub_goals_fn,
        )

        label = "PASS" if success else "FAIL"
        print(f"[{ep+1:3d}/{n_episodes}] {label}")
        if success:
            successes += 1
        results.append(success)

    print(f"\nCollection complete: {successes/n_episodes:.1%} "
          f"({successes}/{n_episodes})")
    return results