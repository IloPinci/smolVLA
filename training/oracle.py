"""
oracle.py  —  SO-101 scripted oracle + demonstration recorder

collect_demonstrations() now records every step of every successful
episode and writes a LeRobot-compatible HDF5 dataset to output_dir.

Dataset layout (one HDF5 file per episode):
    /data/
        observation.images.context     uint8 [T, 256, 256, 3]
        observation.images.wrist       uint8 [T, 256, 256, 3]
        observation.state              float32 [T, 6]   joint angles
        action                         float32 [T, 6]    joint deltas + gripper
    /meta/
        success                        bool scalar
        episode_id                     int scalar
        language_instruction           str
        task_success_rate (filled later by analysis scripts)

A top-level meta.json in output_dir tracks episode count and success rate.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OracleConfig:
    # Noise
    pos_noise_sigma: float = 0.004
    rot_noise_sigma: float = np.deg2rad(2)

    # Trajectory
    steps_per_segment: int    = 30
    pregrasp_clearance: float = 0.12
    lift_height: float        = 0.15
    carry_clearance: float    = 0.15

    # Gripper (position control, one-sided jaw)
    gripper_open: float       = 0.8
    gripper_close_safe: float = 0.18

    # IK tolerances
    ik_pos_tol: float = 1e-4
    ik_rot_tol: float = 1e-4

    # Settle steps per waypoint
    settle_steps: list = field(default_factory=lambda: [
        10,   # WP0 hover
        50,   # WP1 descend
        100,  # WP2 close gripper
        25,   # WP3 lift
        25,   # WP4 carry
        60,   # WP5 release
    ])


# DOF aliases
_ARM_DOFS    = np.arange(5)
_GRIPPER_DOF = np.array([5])

# Fixed language instruction (swap colour word in colour-perturbation runs)
LANGUAGE_INSTRUCTION = "Pick up the red block and place it on the green target."


# ══════════════════════════════════════════════════════════════════════════════
#  Oracle policy
# ══════════════════════════════════════════════════════════════════════════════

class SO101Oracle:
    def __init__(self, so101, scene, cameras=None, cfg: OracleConfig = OracleConfig()):
        self.robot    = so101
        self.scene    = scene
        self.cameras  = cameras         
        self.cfg      = cfg
        self._rng     = np.random.default_rng()

        self.end_effector = so101.get_link("moving_jaw_so101_v1")
        self.grasp_quat   = np.array([0.707107, 0.0, -0.707107, 0.0])

        # Frame buffer filled during run_episode (cleared at episode start)
        self._frames: list[dict] = []
        self._last_qpos: torch.Tensor | None = None 

    # ── Public API ────────────────────────────────────────────────────────────

    def run_episode(self, cube, target_zone, table_height,
                    is_success_fn, check_sub_goals_fn) -> bool:
        """Run one episode; record frames if cameras are provided."""
        self._frames = []
        self._last_qpos = None
        
        # Per-episode latch dict for sub-goal tracking.
        # Passed through to check_sub_goals_fn so `lifted` stays True
        # once the block leaves the table, regardless of gripper distance.
        self._subgoal_latch: dict = {"lifted": False}

        cube_pos   = cube.get_pos().cpu().numpy()
        target_pos = target_zone.get_pos().cpu().numpy()

        waypoints = self._compute_waypoints(cube_pos, target_pos, table_height)
        waypoints.insert(2, waypoints[1].copy())  # insert stationary close-gripper wp

        gripper_targets = [
            self.cfg.gripper_open,
            self.cfg.gripper_open,
            self.cfg.gripper_close_safe,
            self.cfg.gripper_close_safe,
            self.cfg.gripper_close_safe,
        ]

        prev_qpos = self.robot.get_dofs_position()
        success   = False

        for wp_pos, g_target, s_steps in zip(
                waypoints[:-1], gripper_targets, self.cfg.settle_steps[:-1]):
            prev_qpos, success = self._execute_segment(
                wp_pos, g_target, s_steps, prev_qpos,
                cube, target_zone, table_height, check_sub_goals_fn,
                self._subgoal_latch,
            )
            if success:
                break

        if not success:
            success = self._execute_release_segment(
                waypoints[-1],
                self.cfg.settle_steps[-1],
                prev_qpos,
                cube, target_zone, table_height,
                is_success_fn,
            )

        if not success:
            success = is_success_fn(cube, target_zone, table_height)

        return success

    def get_recorded_frames(self) -> list[dict]:
        """Return the step-by-step recording from the last run_episode call."""
        return self._frames

    # ── Segment execution ─────────────────────────────────────────────────────

    def _record_step(self, qpos: torch.Tensor):
        """Capture one observation+action frame if cameras are attached."""
        if self.cameras is None:
            return

        # we "rename" the cameras so they fit with what the 
        KEY_REMAP = {
            "context": "camera1",
            "wrist":   "camera2",
            "top":     "camera3",
}

        frame = {}

        ATTACHED_CAM_KEYS = {"wrist"}
        for key, cam in self.cameras.items():
            if key in ATTACHED_CAM_KEYS:
                cam.move_to_attach()
            rgb, _, _, _ = cam.render()
            remapped_key = KEY_REMAP.get(key, key)
            frame[f"observation.images.{remapped_key}"] = rgb

        # State: 6 DOFs (5 arm + 1 gripper)  
        frame["observation.state"] = qpos[:6].cpu().numpy().astype(np.float32)

        # Action: per-step delta from the PREVIOUS FRAME, not segment start
        truly_prev = self._last_qpos if self._last_qpos is not None else qpos
        delta = (qpos - truly_prev).cpu().numpy().astype(np.float32)
        action = np.concatenate([delta[:5], [qpos[5].item()]])  # 5 deltas + abs gripper
        frame["action"] = action

        self._last_qpos = qpos.clone()   # update for next step
        self._frames.append(frame)

    def _execute_segment(self, target_cart_pos, gripper_target, settle_steps,
                         prev_qpos, cube, target_zone, table_height, check_sub_goals_fn, latch: dict):
        cfg = self.cfg

        noisy_quat = self._perturb_quat_z(
            self.grasp_quat, self._rng.normal(0, cfg.rot_noise_sigma)
        )
        target_qpos = self.robot.inverse_kinematics(
            link=self.end_effector,
            pos=target_cart_pos,
            quat=noisy_quat,
            pos_tol=cfg.ik_pos_tol,
            rot_tol=cfg.ik_rot_tol,
        )
        target_qpos[5] = gripper_target

        alphas = torch.linspace(0.0, 1.0, cfg.steps_per_segment,
                                device=target_qpos.device).unsqueeze(1)
        interp_configs = prev_qpos + alphas * (target_qpos - prev_qpos)
        interp_configs[:, 5] = prev_qpos[5]   # freeze gripper during arm motion

        success = False

        def _step(qpos):
            self.robot.control_dofs_position(qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(qpos)
            return check_sub_goals_fn(self.robot, cube, target_zone, table_height, latch)["placed"]

        for interp_qpos in interp_configs:
            if _step(interp_qpos):
                return interp_qpos, True

        for _ in range(settle_steps):
            if _step(target_qpos):
                return target_qpos, True

        return target_qpos, False

    def _execute_release_segment(self, target_cart_pos, settle_steps,
                                 prev_qpos, cube, target_zone, table_height, is_success_fn):
        cfg = self.cfg

        noisy_quat = self._perturb_quat_z(
            self.grasp_quat, self._rng.normal(0, cfg.rot_noise_sigma)
        )
        target_qpos = self.robot.inverse_kinematics(
            link=self.end_effector,
            pos=target_cart_pos,
            quat=noisy_quat,
            pos_tol=cfg.ik_pos_tol,
            rot_tol=cfg.ik_rot_tol,
        )

        # Phase 1: move arm to release position, gripper stays CLOSED
        closed_qpos       = target_qpos.clone()
        closed_qpos[5]    = prev_qpos[5]
        alphas = torch.linspace(0.0, 1.0, cfg.steps_per_segment,
                                device=target_qpos.device).unsqueeze(1)
        interp = prev_qpos + alphas * (closed_qpos - prev_qpos)
        interp[:, 5] = prev_qpos[5]

        for interp_qpos in interp:
            self.robot.control_dofs_position(interp_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(interp_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(interp_qpos)

        for _ in range(15):   # arm settle
            self.robot.control_dofs_position(closed_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(closed_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(closed_qpos)

        # Phase 2: open gripper, arm stationary
        open_qpos    = closed_qpos.clone()
        open_qpos[5] = cfg.gripper_open

        for _ in range(20):
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(open_qpos)

        # Phase 3: settle — cube drops onto target
        for _ in range(settle_steps):
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(open_qpos)

        return is_success_fn(cube, target_zone, table_height)

    # ── Waypoint computation ──────────────────────────────────────────────────

    def _compute_waypoints(self, cube_pos, target_pos, table_height):
        cfg            = self.cfg
        gripper_length = 0.06
        x_offset       = 0.016
        y_offset       = -0.000

        wp0 = cube_pos   + np.array([x_offset, y_offset, cfg.pregrasp_clearance + gripper_length])
        wp1 = cube_pos   + np.array([x_offset, y_offset, 0.01 + gripper_length])
        wp2 = cube_pos   + np.array([x_offset, y_offset, cfg.lift_height + gripper_length])
        wp3 = target_pos + np.array([0.0,      0.0,      cfg.carry_clearance + gripper_length])
        wp4 = target_pos + np.array([x_offset, 0.0,      0.04 + gripper_length])

        waypoints = []
        for nominal in [wp0, wp1, wp2, wp3, wp4]:
            noise    = self._rng.normal(0, cfg.pos_noise_sigma, size=3)
            noise[2] = abs(noise[2])
            noise[0] *= 0.3
            noise[1] *= 0.3
            waypoints.append(nominal + noise)
        return waypoints

    # ── Quaternion utility ────────────────────────────────────────────────────

    @staticmethod
    def _perturb_quat_z(base_quat, angle_rad):
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


# ══════════════════════════════════════════════════════════════════════════════
#  HDF5 writer
# ══════════════════════════════════════════════════════════════════════════════

def _save_episode_hdf5(frames: list[dict], episode_id: int,
                       success: bool, output_dir: Path,
                       prefix: str = "episode_"):
    """
    Write one episode to <output_dir>/episode_<id>.hdf5.

    Schema (LeRobot-compatible):
        /data/observation.images.context   uint8  [T, H, W, 3]
        /data/observation.images.wrist     uint8  [T, H, W, 3]
        /data/observation.state            float32 [T, 6]
        /data/action                       float32 [T, 6]
        /meta/success                      bool
        /meta/episode_id                   int64
        /meta/language_instruction         str (stored as bytes)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}{episode_id:04d}.hdf5"

    # Discover camera keys from the first frame
    img_keys = [k.replace("observation.images.", "")
                for k in frames[0] if k.startswith("observation.images.")]
 
    state = np.stack([f["observation.state"] for f in frames])   # [T, 6]
    acts  = np.stack([f["action"]            for f in frames])   # [T, 6]
 
    with h5py.File(path, "w") as hf:
        data = hf.create_group("data")
 
        for key in img_keys:
            imgs = np.stack([f[f"observation.images.{key}"] for f in frames])
            data.create_dataset(f"observation.images.{key}", data=imgs,
                                dtype="uint8", compression="gzip", compression_opts=4)
 
        data.create_dataset("observation.state", data=state, dtype="float32")
        data.create_dataset("action",            data=acts,  dtype="float32")
 
        meta = hf.create_group("meta")
        meta.create_dataset("success",              data=bool(success))
        meta.create_dataset("episode_id",           data=episode_id)
        meta.create_dataset("language_instruction", data=LANGUAGE_INSTRUCTION.encode())
 
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Demonstration collection  
# ══════════════════════════════════════════════════════════════════════════════

def collect_demonstrations(so101, scene, cameras, cube, target_zone,
                           table_height, is_success_fn, check_sub_goals_fn,
                           n_episodes: int = 200,
                           output_dir: str = "demos/baseline/") -> list:
    """
    Run the oracle for n_episodes, record each episode, and save to HDF5.
 
    Only SUCCESSFUL episodes are saved to disk (failed episodes produce
    no training signal and inflate dataset size).  Failed episodes are
    still counted toward the success-rate log.
 
    Returns a list of bool (True = success) for each episode.
    """
    cfg      = OracleConfig()
    oracle   = SO101Oracle(so101, scene, cameras, cfg)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # we define a new folder for the failures 
    fail_dir = Path(output_dir).parent / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
 
    home_pose = np.zeros(so101.n_dofs)
    results   = []
    successes = 0
    saved_eps = 0
    saved_fails = 0 
 
    for ep in range(n_episodes):
        # ── Randomise cube and target positions ────────────────────────────
        # Cube:   centre (0.25, 0.00) ± 5 cm  — within comfortable IK reach
        # Target: centre (0.15, -0.15) ± 6 cm — different region of workspace
        # Constraint: cube and target must be ≥ 12 cm apart (XY) so the
        #             oracle never has to place on top of the spawn position.
        MIN_SEP = 0.12
        for _ in range(50):   # rejection-sample up to 50 times
            cube_xy   = np.array([0.25, 0.00]) + np.random.uniform(-0.05,  0.05, 2)
            target_xy = np.array([0.15, -0.15]) + np.random.uniform(-0.06, 0.06, 2)
            if np.linalg.norm(cube_xy - target_xy) >= MIN_SEP:
                break
 
        # ── Reset ──────────────────────────────────────────────────────────
        so101.set_dofs_position(home_pose)
        so101.set_dofs_velocity(np.zeros(so101.n_dofs))
        so101.control_dofs_position(home_pose)
 
        cube.set_pos(np.array([cube_xy[0], cube_xy[1], table_height + 0.015]))
        cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        target_zone.set_pos(np.array([target_xy[0], target_xy[1], table_height + 0.001]))
 
        for _ in range(20):
            so101.set_dofs_velocity(np.zeros(so101.n_dofs))
            so101.control_dofs_position(home_pose)
            scene.step()
 
        # ── Run oracle (recording happens inside run_episode) ───────────
        success = oracle.run_episode(
            cube, target_zone, table_height,
            is_success_fn, check_sub_goals_fn,
        )
 
        label = "PASS ✓" if success else "FAIL ✗"
        print(f"[{ep + 1:3d}/{n_episodes}] {label}", flush=True)
 
        results.append(success)
        frames = oracle.get_recorded_frames()

        if success:
            successes += 1
            if frames:
                _save_episode_hdf5(frames, saved_eps, success, out_dir)
                saved_eps += 1
        # the same for saving the fails so we can understand what went wrong
        else:
            if frames:
                _save_episode_hdf5(frames, saved_fails, success, fail_dir,
                                   prefix="fail_")
                saved_fails += 1


    # ── Write meta.json for success
    meta = {
        "n_episodes_attempted": n_episodes,
        "n_episodes_saved":     saved_eps,
        "success_rate":         round(successes / n_episodes, 4),
        "language_instruction": LANGUAGE_INSTRUCTION,
        "camera_keys":          ["camera1", "camera2", "camera3"],
        "state_dim":            6,
        "action_dim":           6,
        "image_resolution":     [256, 256],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


    # ── Write meta.json for failures 
    fail_meta = {
        "n_episodes_attempted": n_episodes,
        "n_failures_saved":     saved_fails,
        "failure_rate":         round(saved_fails / n_episodes, 4),
        "language_instruction": LANGUAGE_INSTRUCTION,
        "camera_keys":          ["camera1", "camera2", "camera3"],
        "state_dim":            6,
        "action_dim":           6,
        "image_resolution":     [256, 256],
    }
    with open(fail_dir / "meta.json", "w") as f:
        json.dump(fail_meta, f, indent=2)
    

    print(f"\n{'─'*50}")
    print(f"Collection complete: {successes}/{n_episodes} successful "
          f"({successes/n_episodes:.1%})")
    print(f"Saved {saved_eps} episodes to {out_dir}")
    print(f"{'─'*50}")

    return results