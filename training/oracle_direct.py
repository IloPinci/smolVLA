"""
oracle_direct.py  —  SO-101 scripted oracle + DIRECT LeRobot v3.0 writer
=========================================================================

Writes episodes DIRECTLY into the LeRobot v3.0 format during collection
using the official LeRobotDataset API. No convert step, no v2.1 intermediate.

Key API flow (v3.0):
    dataset = LeRobotDataset.create(repo_id, fps, root, features)
    for each episode:
        dataset.add_frame(frame_dict)   # called once per timestep
        dataset.save_episode()          # called at episode end
    dataset.finalize()                  # MUST be called before push_to_hub
    dataset.push_to_hub()              # optional — omit for local-only

Dataset layout written on the fly (v3.0 — multi-episode files):
    <output_dir>/
        data/
            chunk-000/
                file-000000.parquet     ← many episodes per file
        videos/
            chunk-000/
                observation.images.camera1/
                    file-000000.mp4     ← many episodes per file
                observation.images.camera2/
                    file-000000.mp4
                observation.images.camera3/
                    file-000000.mp4
        meta/
            info.json
            stats.json
            tasks.jsonl
            episodes/
                chunk-000/
                    episode_000000.parquet  ← one row per episode

Usage (identical to before — just swap the import):
    from oracle_direct import SO101Oracle, OracleConfig, collect_demonstrations

Action format — ABSOLUTE joint positions (6-DOF):
    action[:5]  — arm joint absolute positions (radians)
    action[5]   — gripper absolute position
    This matches SmolVLA's expected input from the SO-101 dataset.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from scene_params import language_instruction_for, DEFAULT_CUBE_COLOR

# ── LeRobot v3 API ────────────────────────────────────────────────────────────
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    raise ImportError(
        "lerobot not found. Install from source with SmolVLA extras:\n"
        "  pip install -e '.[smolvla]'  (inside lerobot repo)\n"
        f"Original error: {e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_KEYS          = ["camera1", "camera2", "camera3"]
STATE_DIM            = 6
ACTION_DIM           = 6
FPS                  = 10          # collect at 10 fps (every 10th sim step at dt=0.01)
LANGUAGE_INSTRUCTION = language_instruction_for(DEFAULT_CUBE_COLOR)

# Camera key remap: oracle internal name → LeRobot observation key suffix
_CAM_REMAP = {
    "context": "camera1",
    "wrist":   "camera2",
    "top":     "camera3",
}

_ARM_DOFS    = np.arange(5)
_GRIPPER_DOF = np.array([5])

# LeRobot v3 features dict — describes every key in each frame dict
# Shapes match what LeRobotDataset.create() expects.
# Images: [H, W, C] uint8 — the API handles stacking across time internally.
def make_features(img_h: int = 256, img_w: int = 256) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {
                "motors": [f"motor_{i}" for i in range(STATE_DIM)]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {
                "motors": [f"motor_{i}" for i in range(ACTION_DIM)]
            },
        },
    }
    for cam_key in CAMERA_KEYS:
        features[f"observation.images.{cam_key}"] = {
            "dtype":  "video",
            "shape":  (img_h, img_w, 3),
            "names":  ["height", "width", "channel"],
        }
    return features


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

    # Settle steps per waypoint (sim steps, not recorded frames)
    settle_steps: list = field(default_factory=lambda: [
        10,   # WP0 hover
        50,   # WP1 descend
        100,  # WP2 close gripper
        25,   # WP3 lift
        25,   # WP4 carry
        60,   # WP5 release
    ])

    # Record 1 frame every N sim steps (keeps episode length manageable)
    record_every_n_steps: int = 10


# ══════════════════════════════════════════════════════════════════════════════
#  Oracle policy
# ══════════════════════════════════════════════════════════════════════════════

class SO101Oracle:
    def __init__(self, so101, scene, cameras=None, cfg: OracleConfig = OracleConfig(),
                 language_instruction: str = LANGUAGE_INSTRUCTION):
        self.robot                = so101
        self.scene                = scene
        self.cameras              = cameras
        self.cfg                  = cfg
        self.language_instruction = language_instruction   # embedded in every frame dict
        self._rng                 = np.random.default_rng()

        self.end_effector = so101.get_link("moving_jaw_so101_v1")
        self.grasp_quat   = np.array([0.707107, 0.0, -0.707107, 0.0])

        self._frames: list[dict]              = []
        self._last_qpos: torch.Tensor | None  = None
        self._step_counter: int               = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def run_episode(self, cube, target_zone, table_height,
                    is_success_fn, check_sub_goals_fn) -> bool:
        """Run one episode; frame buffer is available via get_recorded_frames()."""
        self._frames        = []
        self._last_qpos     = None
        self._step_counter  = 0
        self._subgoal_latch: dict = {"lifted": False}

        cube_pos   = cube.get_pos().cpu().numpy()
        target_pos = target_zone.get_pos().cpu().numpy()

        waypoints = self._compute_waypoints(cube_pos, target_pos, table_height)
        waypoints.insert(2, waypoints[1].copy())   # stationary close-gripper WP

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
        """Return step-by-step frame dicts from the last run_episode call."""
        return self._frames

    # ── Frame recording ────────────────────────────────────────────────────

    def _record_step(self, qpos: torch.Tensor):
        """
        Capture one frame if cameras are attached and we're on a record step.

        Frame dict keys (matching make_features()):
            observation.images.camera1  — uint8 [H, W, 3]
            observation.images.camera2  — uint8 [H, W, 3]
            observation.images.camera3  — uint8 [H, W, 3]
            observation.state           — float32 [6]
            action                      — float32 [6]  absolute joint positions
        """
        self._step_counter += 1

        if self.cameras is None:
            return

        if self._step_counter % self.cfg.record_every_n_steps != 0:
            return

        frame = {}

        ATTACHED_CAM_KEYS = {"wrist"}
        for key, cam in self.cameras.items():
            if key in ATTACHED_CAM_KEYS:
                cam.move_to_attach()
            rgb, _, _, _ = cam.render()                          # uint8 [H, W, 3]
            remapped_key = _CAM_REMAP.get(key, key)
            frame[f"observation.images.{remapped_key}"] = rgb

        # State: 6 DOFs as float32
        frame["observation.state"] = qpos[:6].cpu().numpy().astype(np.float32)

        # Action: absolute joint positions (SmolVLA SO-101 convention)
        frame["action"] = qpos[:6].cpu().numpy().astype(np.float32)

        # Task string — required by LeRobotDataset.add_frame() in v3.0
        frame["task"] = self.language_instruction

        self._frames.append(frame)

    # ── Segment execution ──────────────────────────────────────────────────

    def _execute_segment(self, target_cart_pos, gripper_target, settle_steps,
                         prev_qpos, cube, target_zone, table_height,
                         check_sub_goals_fn, latch: dict):
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
        interp_configs       = prev_qpos + alphas * (target_qpos - prev_qpos)
        interp_configs[:, 5] = prev_qpos[5]   # freeze gripper during arm motion

        def _step(qpos) -> bool:
            self.robot.control_dofs_position(qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(qpos)
            return check_sub_goals_fn(
                self.robot, cube, target_zone, table_height, latch
            )["placed"]

        for interp_qpos in interp_configs:
            if _step(interp_qpos):
                return interp_qpos, True

        for _ in range(settle_steps):
            if _step(target_qpos):
                return target_qpos, True

        return target_qpos, False

    def _execute_release_segment(self, target_cart_pos, settle_steps,
                                 prev_qpos, cube, target_zone, table_height,
                                 is_success_fn):
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

        # Phase 1: move arm to release position, gripper stays closed
        closed_qpos    = target_qpos.clone()
        closed_qpos[5] = prev_qpos[5]
        alphas = torch.linspace(0.0, 1.0, cfg.steps_per_segment,
                                device=target_qpos.device).unsqueeze(1)
        interp       = prev_qpos + alphas * (closed_qpos - prev_qpos)
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

    # ── Waypoint computation ──────────────────────────────────────────────

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

    # ── Quaternion utility ────────────────────────────────────────────────

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
#  collect_demonstrations  —  writes directly to LeRobot v3.0
# ══════════════════════════════════════════════════════════════════════════════

def collect_demonstrations(
    so101,
    scene,
    cameras,
    cube,
    target_zone,
    table_height,
    is_success_fn,
    check_sub_goals_fn,
    n_episodes: int     = 200,
    output_dir: str     = "demos/lerobot/",
    fps: int            = FPS,
    repo_id: str        = "local/genesis_pickplace",
    push_to_hub: bool   = False,
    save_failures: bool = True,
    cube_color: str     = DEFAULT_CUBE_COLOR,
    img_h: int          = 256,
    img_w: int          = 256,
) -> list[bool]:
    """
    Run the oracle for n_episodes and write each episode directly into a
    LeRobot v3.0 dataset tree using the official LeRobotDataset API.

    Successful episodes → <output_dir>
    Failed episodes     → <parent of output_dir>/failures_lerobot/  (if save_failures=True)

    The official write flow is:
        dataset = LeRobotDataset.create(...)
        for each episode:
            dataset.add_frame(frame_dict)
            dataset.save_episode()
        dataset.finalize()         # ← mandatory, closes parquet writers
        dataset.push_to_hub()      # ← optional

    Returns a list of bool (True = success) per attempted episode.
    """
    cfg                  = OracleConfig()
    language_instruction = language_instruction_for(cube_color)
    # Pass instruction to oracle so it embeds it in every frame dict (required by v3 API)
    oracle               = SO101Oracle(so101, scene, cameras, cfg, language_instruction)
    features             = make_features(img_h, img_w)

    out_dir  = Path(output_dir).resolve()
    fail_dir = out_dir.parent / "failures_lerobot"

    # ── Create LeRobotDataset for successes ───────────────────────────────────
    print(f"[info] Creating LeRobot v3.0 dataset at: {out_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=str(out_dir),
        features=features,
        use_videos=True,
        image_writer_threads=4,   # encode mp4 frames in parallel
    )

    # ── Optionally create a second dataset for failures ───────────────────────
    fail_dataset = None
    if save_failures:
        print(f"[info] Creating failures dataset at: {fail_dir}")
        fail_dataset = LeRobotDataset.create(
            repo_id=repo_id + "_failures",
            fps=fps,
            root=str(fail_dir),
            features=features,
            use_videos=True,
            image_writer_threads=4,
        )

    home_pose = np.zeros(so101.n_dofs)
    results: list[bool] = []
    successes = 0

    print(f"\nCollecting {n_episodes} episodes  (fps={fps}, cameras={CAMERA_KEYS})")
    print(f"Task: \"{language_instruction}\"\n")

    for ep in range(n_episodes):
        # ── Randomise cube & target positions ────────────────────────────────
        MIN_SEP = 0.12
        for _ in range(50):
            cube_xy   = np.array([0.24, 0.00])  + np.random.uniform(-0.04,  0.04, 2)
            target_xy = np.array([0.15, -0.15]) + np.random.uniform(-0.05,  0.05, 2)
            if np.linalg.norm(cube_xy - target_xy) >= MIN_SEP:
                break

        # ── Reset environment ─────────────────────────────────────────────────
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

        # ── Run oracle (recording happens inside) ─────────────────────────────
        success = oracle.run_episode(
            cube, target_zone, table_height,
            is_success_fn, check_sub_goals_fn,
        )

        label = "PASS ✓" if success else "FAIL ✗"
        frames = oracle.get_recorded_frames()
        print(f"[{ep + 1:3d}/{n_episodes}] {label}  T={len(frames)}", flush=True)

        results.append(success)

        if not frames:
            print(f"           [warn] no frames recorded — skipping")
            continue

        # ── Write frames to the appropriate dataset ───────────────────────────
        target_ds = dataset if success else fail_dataset

        if target_ds is not None:
            for frame in frames:
                target_ds.add_frame(frame)

            # task is already embedded per-frame via frame["task"] in add_frame()
            target_ds.save_episode()

        if success:
            successes += 1

    # ── Finalize — mandatory before push_to_hub or loading for training ───────
    print(f"\n[info] Finalizing dataset (flushing parquet writers)…")
    dataset.finalize()
    if fail_dataset is not None:
        fail_dataset.finalize()

    # ── Optional push to Hub ──────────────────────────────────────────────────
    if push_to_hub:
        print(f"[info] Pushing to Hub as '{repo_id}' …")
        dataset.push_to_hub()
        if fail_dataset is not None:
            fail_dataset.push_to_hub()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Collection complete: {successes}/{n_episodes} successful "
          f"({successes / max(n_episodes, 1):.1%})")
    print(f"Dataset written to : {out_dir}")
    if save_failures:
        print(f"Failures written to: {fail_dir}")
    print(f"{'─' * 50}")
    print(f"\n[next] Fine-tune with:")
    print(f"       lerobot-train \\")
    print(f"         --policy.path=lerobot/smolvla_base \\")
    print(f"         --dataset.repo_id={repo_id} \\")
    print(f"         --dataset.root={out_dir} \\")
    print(f"         --batch_size=64 \\")
    print(f"         --steps=20000 \\")
    print(f"         --output_dir=outputs/train/smolvla_genesis \\")
    print(f"         --policy.device=cuda")

    return results