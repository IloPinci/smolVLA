import numpy as np
import torch
from dataclasses import dataclass, field

@dataclass
class OracleConfig:
    # Noise
    pos_noise_sigma: float  = 0.002
    rot_noise_sigma: float  = np.deg2rad(2)

    # Trajectory
    steps_per_segment: int  = 30
    pregrasp_clearance: float = 0.12
    lift_height: float        = 0.15
    carry_clearance: float    = 0.15

    # Gripper — position control only (one-sided jaw)
    gripper_open: float       = 0.8
    # Jaw stops here and kp does the squeezing.
    # Tune upward (0.22, 0.25 …) if jaw clips through cube geometry,
    # tune downward if it doesn't make firm contact.
    gripper_close_safe: float = 0.22

    # IK tolerances
    ik_pos_tol: float = 1e-4
    ik_rot_tol: float = 1e-4

    # Settle steps per waypoint
    settle_steps: list = field(default_factory=lambda: [
        10,   # WP0  hover          — move fast
        40,   # WP1  descend        — let arm settle
        80,   # WP2  close gripper  — long: contact force must stabilise before arm moves
        20,   # WP3  lift
        20,   # WP4  carry
        50,   # WP5  release        — let cube settle on target
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

        # Duplicate WP1 (descent endpoint) so the arm holds still while the
        # gripper actuates.  This becomes the new WP2; original WP2 (lift)
        # shifts to WP3, etc.
        waypoints.insert(2, waypoints[1].copy())

        # All waypoints use position control — force mode is unsafe for a
        # one-sided jaw (no opposing surface → jaw tunnels through cube).
        # kp on the gripper DOF provides squeeze force once the jaw contacts
        # the cube and can no longer reach gripper_close_safe.
        gripper_targets = [
            self.cfg.gripper_open,        # WP0  hover
            self.cfg.gripper_open,        # WP1  descend
            self.cfg.gripper_close_safe,  # WP2  close (arm stationary)
            self.cfg.gripper_close_safe,  # WP3  lift
            self.cfg.gripper_close_safe,  # WP4  carry
            self.cfg.gripper_open,        # WP5  release
        ]

        prev_qpos = self.robot.get_dofs_position()
        success   = False

        for wp_pos, g_target, s_steps in zip(
                waypoints, gripper_targets, self.cfg.settle_steps):

            prev_qpos, success = self._execute_segment(
                wp_pos, g_target, s_steps, prev_qpos,
                cube, target_zone, table_height, check_sub_goals_fn,
            )
            if success:
                break

        if not success:
            success = is_success_fn(cube, target_zone, table_height)

        return success

    # ------------------------------------------------------------------ #
    #  Waypoint computation                                                #
    # ------------------------------------------------------------------ #

    def _compute_waypoints(self, cube_pos: np.ndarray,
                           target_pos: np.ndarray,
                           table_height: float) -> list:
        cfg = self.cfg

        # Physical offset from the IK link origin to the jaw contact surface
        gripper_length = 0.06
        y_offset       = -0.006

        wp0 = cube_pos   + np.array([0.015, y_offset, cfg.pregrasp_clearance + gripper_length])
        wp1 = cube_pos   + np.array([0.015, y_offset, 0.010 + gripper_length])
        wp2 = cube_pos   + np.array([0.015, y_offset, cfg.lift_height + gripper_length])
        wp3 = target_pos + np.array([0.0,   0.0,      cfg.carry_clearance + gripper_length])
        wp4 = target_pos + np.array([0.0,   0.0,      0.02 + gripper_length])

        waypoints = []
        for nominal in [wp0, wp1, wp2, wp3, wp4]:
            noise    = self._rng.normal(0, cfg.pos_noise_sigma, size=3)
            noise[2] = abs(noise[2])   # never push Z below nominal
            noise[0] *= 0.4            # dampen lateral scatter for one-sided jaw
            noise[1] *= 0.4
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

        # Noisy yaw perturbation around the grasp orientation
        yaw_noise  = self._rng.normal(0, cfg.rot_noise_sigma)
        noisy_quat = self._perturb_quat_z(self.grasp_quat, yaw_noise)

        target_qpos = self.robot.inverse_kinematics(
            link    = self.end_effector,
            pos     = target_cart_pos,
            quat    = noisy_quat,
            pos_tol = cfg.ik_pos_tol,
            rot_tol = cfg.ik_rot_tol,
        )

        # Bake gripper target into qpos so interpolation carries it correctly
        target_qpos[5] = gripper_target

        # Linear interpolation in joint space
        steps  = cfg.steps_per_segment
        alphas = torch.linspace(0.0, 1.0, steps,
                                device=target_qpos.device).unsqueeze(1)
        interp_configs = prev_qpos + alphas * (target_qpos - prev_qpos)

        success = False

        def _step(qpos: torch.Tensor) -> bool:
            """Command one full joint configuration and advance the sim."""
            # Arm: position control for precise IK tracking
            self.robot.control_dofs_position(
                qpos[_ARM_DOFS],
                dofs_idx_local=_ARM_DOFS,
            )
            # Gripper: position control — kp acts as implicit squeeze spring
            self.robot.control_dofs_position(
                qpos[_GRIPPER_DOF],
                dofs_idx_local=_GRIPPER_DOF,
            )
            self.scene.step()
            sub = check_sub_goals_fn(self.robot, cube, target_zone, table_height)
            return sub["placed"]

        # Interpolation phase
        for interp_qpos in interp_configs:
            if _step(interp_qpos):
                return interp_qpos, True

        # Static settle phase — hold target until PD transients decay
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

        # Reset arm
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