import os
import numpy as np
import genesis as gs
import torch
from dataclasses import dataclass, field
from typing import Optional


# ─── Oracle configuration ─────────────────────────────────────────────────────

@dataclass
class OracleConfig:
    # Gaussian noise added to each waypoint position (metres)
    pos_noise_sigma:  float = 0.005          # 0.5 cm per plan spec
    # Gaussian noise added to each waypoint orientation (radians)
    rot_noise_sigma:  float = np.deg2rad(2)  # 2° per plan spec

    # Physics steps spent moving between consecutive waypoints
    steps_per_segment: int = 30

    # Extra settling steps after reaching each waypoint
    # (PD controller needs time to converge — same reason Genesis docs add 100 steps)
    settle_steps: int = 40

    # Gripper DOF joint limits from XML:
    #   range="-0.17453 1.74533"
    gripper_open:  float = -0.17   # open position (near lower limit)
    gripper_close: float =  1.40   # closed enough to grip a 3 cm cube

    # Waypoint heights relative to the table surface
    pregrasp_clearance: float = 0.10   # 10 cm above cube
    lift_height:        float = 0.15   # 15 cm above grasp pose
    carry_clearance:    float = 0.15   # 15 cm above target zone

    # IK solver tolerance — tighten if the arm undershoots waypoints
    ik_pos_tol: float = 1e-4
    ik_rot_tol: float = 1e-4
    


@dataclass
class EpisodeRecord:
    """Stores everything needed to build one LeRobot dataset episode."""
    observations_front: list = field(default_factory=list)   # (H, W, 3) uint8
    observations_top:   list = field(default_factory=list)
    states:             list = field(default_factory=list)   # (6,) float32
    actions:            list = field(default_factory=list)   # (6,) float32
    sub_goals:          list = field(default_factory=list)   # dict per step
    success:            bool = False
    total_steps:        int  = 0
    task:               str  = "Pick up the red block and place it on the target."


# ─── DOF index map for SO-101 ─────────────────────────────────────────────────
# From the XML actuator order:
#   0: shoulder_pan  1: shoulder_lift  2: elbow_flex
#   3: wrist_flex    4: wrist_roll     5: gripper
ARM_DOFS     = np.arange(5)   # indices 0-4
GRIPPER_DOF  = np.array([5])  # index 5


# ─── Oracle class ─────────────────────────────────────────────────────────────

class SO101Oracle:
    """
    IK-based scripted oracle for pick-and-place on the SO-101 arm in Genesis.

    Usage:
        oracle = SO101Oracle(so101, scene, cameras, cfg)
        record = oracle.run_episode(cube_pos, target_pos)
    """

    def __init__(self, so101, scene, cameras: dict,
                 cfg: OracleConfig = OracleConfig()):
        self.robot   = so101
        self.scene   = scene
        self.cameras = cameras
        self.cfg     = cfg

        # End-effector: the moving jaw — closest link to the object
        # "moving_jaw_so101_v1" is the terminal link in the XML kinematic chain
        self.end_effector = so101.get_link("moving_jaw_so101_v1")

        # Downward-pointing gripper orientation (wxyz).
        # (0, 1, 0, 0) = 180° around Y → gripper Z points down, matching Genesis
        # convention used in all IK examples. Adjust if your home pose differs.
        self.grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])

        self._rng = np.random.default_rng()   # seeded externally via gs.init(seed=)

    # ── Public entry point ────────────────────────────────────────────────────

    def run_episode(self,
                    cube,
                    target_zone,
                    table_height: float,
                    is_success_fn,
                    check_sub_goals_fn) -> EpisodeRecord:
        """
        Execute one full pick-and-place episode and return the recorded data.

        Args:
            cube:              Genesis entity for the cube
            target_zone:       Genesis entity for the target cylinder
            table_height:      float, metres
            is_success_fn:     callable(cube, target_zone, table_height) → bool
            check_sub_goals_fn: callable(so101, cube, target_zone, table_height) → dict
        """
        record = EpisodeRecord()

        cube_pos   = cube.get_pos().cpu().numpy()
        target_pos = target_zone.get_pos().cpu().numpy()

        # Compute all 5 Cartesian waypoints once, then add noise
        waypoints = self._compute_waypoints(cube_pos, target_pos, table_height)

        # Gripper state per waypoint: open for pre-grasp/grasp approach,
        # close at grasp, keep closed through carry, open at place
        gripper_states = [
            self.cfg.gripper_open,    # WP0 — home / pregrasp
            self.cfg.gripper_open,    # WP1 — descend to grasp
            self.cfg.gripper_close,   # WP2 — lift (gripper closes here)
            self.cfg.gripper_close,   # WP3 — carry over target
            self.cfg.gripper_open,    # WP4 — place (release)
        ]

        # Execute segment by segment
        prev_qpos = self.robot.get_dofs_position()

        for wp_idx, (wp_pos, g_state) in enumerate(zip(waypoints, gripper_states)):
            prev_qpos = self._execute_segment(
                wp_pos, g_state, prev_qpos,
                record, cube, target_zone, table_height,
                is_success_fn, check_sub_goals_fn,
            )

            # Early exit if placed successfully
            if record.success:
                break

        # Final check after all waypoints complete
        if not record.success:
            record.success = is_success_fn(cube, target_zone, table_height)

        record.total_steps = len(record.states)
        return record

    # ── Waypoint computation ──────────────────────────────────────────────────

    def _compute_waypoints(self,
                           cube_pos:   np.ndarray,
                           target_pos: np.ndarray,
                           table_height: float) -> list:
        """
        Build 5 Cartesian waypoints with per-waypoint Gaussian noise.

        WP0  pre-grasp  — 10 cm above cube (gripper open, approaching)
        WP1  grasp      — at cube surface
        WP2  lift       — 15 cm above grasp
        WP3  carry      — 15 cm above target zone
        WP4  place      — at target zone surface
        """
        cfg = self.cfg
        sz  = cfg.pos_noise_sigma

        wp0_nominal = cube_pos   + np.array([0, 0, cfg.pregrasp_clearance])
        wp1_nominal = cube_pos   + np.array([0, 0, 0.015])  # cube half-height
        wp2_nominal = cube_pos   + np.array([0, 0, cfg.lift_height])
        wp3_nominal = target_pos + np.array([0, 0, cfg.carry_clearance])
        wp4_nominal = target_pos + np.array([0, 0, 0.016])  # rest on target

        waypoints = []
        for nominal in [wp0_nominal, wp1_nominal, wp2_nominal,
                        wp3_nominal, wp4_nominal]:
            noise    = self._rng.normal(0, sz, size=3)
            noise[2] = abs(noise[2])   # never push below table on Z
            waypoints.append(nominal + noise)

        return waypoints

# ── Segment execution ─────────────────────────────────────────────────────

    def _execute_segment(self,
                         target_cart_pos: np.ndarray,
                         gripper_state:   float,
                         prev_qpos:       torch.Tensor, # Note: Now expects a Tensor
                         record:          EpisodeRecord,
                         cube, target_zone, table_height,
                         is_success_fn, check_sub_goals_fn) -> torch.Tensor:
        """
        Solve IK, interpolate, and step physics 100% on the GPU.
        Converts to NumPy only when appending to the recording buffer.
        """
        cfg = self.cfg

        # Add orientation noise
        rot_noise_angle = self._rng.normal(0, cfg.rot_noise_sigma)
        noisy_quat = self._perturb_quat_z(self.grasp_quat, rot_noise_angle)

        # 1. Solve IK (Returns a GPU Tensor)
        target_qpos = self.robot.inverse_kinematics(
            link     = self.end_effector,
            pos      = target_cart_pos,
            quat     = noisy_quat,
            pos_tol  = cfg.ik_pos_tol,
            rot_tol  = cfg.ik_rot_tol,
        )

        # 2. Overwrite the gripper DOF directly on the GPU tensor (index 5)
        target_qpos[5] = gripper_state

        # 3. Calculate interpolation trajectory 100% on the GPU using PyTorch
        steps = cfg.steps_per_segment
        alphas = torch.linspace(0.0, 1.0, steps, device=target_qpos.device).unsqueeze(1)
        
        # Matrix math: creates shape (steps, 6) with all intermediate joint states
        interp_configs = prev_qpos + alphas * (target_qpos - prev_qpos)

        # 4. Execution loop
        for interp_qpos in interp_configs:
            self.robot.control_dofs_position(interp_qpos)
            self.scene.step()
            
            # Send a CPU copy to the recorder for the LeRobot dataset
            self._record_step(record, cube, target_zone, table_height,
                              interp_qpos.cpu().numpy(), is_success_fn, check_sub_goals_fn)
            if record.success:
                return interp_qpos

        # 5. Settling steps — hold target_qpos while the PD controller converges
        for _ in range(cfg.settle_steps):
            self.robot.control_dofs_position(target_qpos)
            self.scene.step()
            
            self._record_step(record, cube, target_zone, table_height,
                              target_qpos.cpu().numpy(), is_success_fn, check_sub_goals_fn)
            if record.success:
                return target_qpos

        return target_qpos


    # ── Per-step recording ────────────────────────────────────────────────────

    def _record_step(self,
                     record:   EpisodeRecord,
                     cube, target_zone, table_height,
                     commanded_qpos: np.ndarray,
                     is_success_fn, check_sub_goals_fn):
        """Capture cameras, joint state, action, and sub-goals for one step."""
        rgb_front, _, _, _ = self.cameras["front"].render()
        rgb_top,   _, _, _ = self.cameras["top"].render()
        joint_pos          = self.robot.get_dofs_position().cpu().numpy()

        record.observations_front.append(rgb_front)
        record.observations_top.append(rgb_top)
        record.states.append(joint_pos.astype(np.float32))

        # Action = the commanded position we just sent (not the current state).
        # LeRobot trains on absolute joint targets, not deltas, for position control.
        record.actions.append(commanded_qpos.astype(np.float32))

        sub = check_sub_goals_fn(self.robot, cube, target_zone, table_height)
        record.sub_goals.append(sub)

        if sub["placed"] and not record.success:
            record.success = True

    # ── Orientation noise helper ──────────────────────────────────────────────

    @staticmethod
    def _perturb_quat_z(base_quat: np.ndarray, angle_rad: float) -> np.ndarray:
        """
        Compose base_quat with a small rotation around world Z.
        Keeps the gripper pointing downward while varying yaw slightly.
        """
        half   = angle_rad / 2.0
        dq     = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # wxyz, rot around Z
        w1,x1,y1,z1 = base_quat
        w2,x2,y2,z2 = dq
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])


# ─── Demo collection loop ─────────────────────────────────────────────────────

def collect_demonstrations(so101, scene, cameras, cube, target_zone,
                            table_height, is_success_fn, check_sub_goals_fn,
                            n_episodes: int = 150,
                            output_dir: str = "demos/") -> list:
    """
    Run the oracle n_episodes times and return a list of EpisodeRecord objects.

    Gate check: the oracle must achieve >=95% success rate.
    If it does not, something is wrong with the IK configuration
    or waypoint heights — fix before collecting the full dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg    = OracleConfig()
    oracle = SO101Oracle(so101, scene, cameras, cfg)

    records   = []
    successes = 0

    # Home pose: all joints at zero.
    # Run the arm there before the first episode.
    home_pose = np.zeros(so101.n_dofs)

    for ep in range(n_episodes):
        # ── Reset ────────────────────────────────────────────────────────────
        # Arm back to home
        so101.control_dofs_position(home_pose)

        # Cube back to spawn; zero all velocity state
        cube.set_pos(np.array([0.25, 0.0, table_height + 0.015]))
        cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

        # Settle for 20 steps so residual forces dissipate
        for _ in range(20):
            so101.control_dofs_position(home_pose)
            scene.step()

        # ── Run episode ───────────────────────────────────────────────────────
        record = oracle.run_episode(
            cube, target_zone, table_height,
            is_success_fn, check_sub_goals_fn,
        )

        if record.success:
            successes += 1
            print(f"[{ep+1:3d}/{n_episodes}] ✓  steps={record.total_steps}")
        else:
            print(f"[{ep+1:3d}/{n_episodes}] ✗  FAILED  steps={record.total_steps}")

        records.append(record)

        # ── Early gate check at episode 20 ───────────────────────────────────
        if ep == 19:
            early_rate = successes / 20
            print(f"\n--- Gate check after 20 episodes: {early_rate:.0%} ---")
            if early_rate < 0.95:
                print("FAIL: Oracle success rate below 95%. Check waypoint heights and IK config.")
                print("Aborting collection — fix oracle before running full 150 episodes.")
                return records

    final_rate = successes / n_episodes
    print(f"\nCollection complete: {final_rate:.1%} ({successes}/{n_episodes})")

    if final_rate < 0.95:
        print("WARNING: Final rate below 95% target. Consider increasing settle_steps or adjusting waypoints.")

    return records