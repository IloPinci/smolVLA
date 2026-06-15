"""
oracle_direct.py  —  SO-101 scripted oracle + DIRECT LeRobot v2.1 writer
=========================================================================

Drop-in replacement for oracle.py that writes episodes DIRECTLY into the
LeRobot v2.1 parquet + video layout during collection — no convert_demos.py
step needed.

Dataset layout written on the fly:
    <output_dir>/
        data/chunk-000/
            episode_000000.parquet
            episode_000001.parquet
            ...
        videos/chunk-000/
            observation.images.camera1/episode_000000.mp4
            observation.images.camera2/episode_000000.mp4
            observation.images.camera3/episode_000000.mp4
            ...
        meta/
            info.json
            episodes.jsonl
            tasks.jsonl

Usage (identical to before — just swap the import):
    from oracle_direct import SO101Oracle, OracleConfig, collect_demonstrations

Key changes vs oracle.py:
  - _save_episode_hdf5 is gone; replaced by _write_episode_lerobot()
  - collect_demonstrations writes parquet + mp4 per episode, then finalises
    meta/ at the end.
  - The HDF5 intermediate is never created.
  - action shape is [T, 6]: first 5 are arm joint DELTAS, index 5 is the
    ABSOLUTE gripper position — exactly what SmolVLA expects.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from scene_params import language_instruction_for, DEFAULT_CUBE_COLOR


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_KEYS          = ["camera1", "camera2", "camera3"]
STATE_DIM            = 6
ACTION_DIM           = 6
CHUNK_SIZE           = 1000        # episodes per chunk folder
FPS                  = 30          # video frame rate written into the dataset
LANGUAGE_INSTRUCTION = language_instruction_for(DEFAULT_CUBE_COLOR)

# Camera key remap: oracle internal name → LeRobot observation key suffix
_CAM_REMAP = {
    "context": "camera1",
    "wrist":   "camera2",
    "top":     "camera3",
}

_ARM_DOFS    = np.arange(5)
_GRIPPER_DOF = np.array([5])


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


# ══════════════════════════════════════════════════════════════════════════════
#  Oracle policy  (unchanged logic from oracle.py)
# ══════════════════════════════════════════════════════════════════════════════

class SO101Oracle:
    def __init__(self, so101, scene, cameras=None, cfg: OracleConfig = OracleConfig()):
        self.robot   = so101
        self.scene   = scene
        self.cameras = cameras
        self.cfg     = cfg
        self._rng    = np.random.default_rng()

        self.end_effector = so101.get_link("moving_jaw_so101_v1")
        self.grasp_quat   = np.array([0.707107, 0.0, -0.707107, 0.0])

        self._frames: list[dict]         = []
        self._last_qpos: torch.Tensor | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def run_episode(self, cube, target_zone, table_height,
                    is_success_fn, check_sub_goals_fn) -> bool:
        self._frames     = []
        self._last_qpos  = None
        self._subgoal_latch: dict = {"lifted": False}

        cube_pos   = cube.get_pos().cpu().numpy()
        target_pos = target_zone.get_pos().cpu().numpy()

        waypoints = self._compute_waypoints(cube_pos, target_pos, table_height)
        waypoints.insert(2, waypoints[1].copy())

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
        return self._frames

    # ── Frame recording ────────────────────────────────────────────────────

    def _record_step(self, qpos: torch.Tensor):
        """Capture one observation+action frame."""
        if self.cameras is None:
            return

        frame = {}

        ATTACHED_CAM_KEYS = {"wrist"}
        for key, cam in self.cameras.items():
            if key in ATTACHED_CAM_KEYS:
                cam.move_to_attach()
            rgb, _, _, _ = cam.render()
            remapped_key = _CAM_REMAP.get(key, key)
            frame[f"observation.images.{remapped_key}"] = rgb   # uint8 [H,W,3]

        frame["observation.state"] = qpos[:6].cpu().numpy().astype(np.float32)

        truly_prev = self._last_qpos if self._last_qpos is not None else qpos
        delta      = (qpos - truly_prev).cpu().numpy().astype(np.float32)
        # action: 5 arm deltas + 1 absolute gripper position
        action = np.concatenate([delta[:5], [qpos[5].item()]])
        frame["action"] = action

        self._last_qpos = qpos.clone()
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
        interp_configs[:, 5] = prev_qpos[5]

        def _step(qpos):
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

        for _ in range(15):
            self.robot.control_dofs_position(closed_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(closed_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(closed_qpos)

        open_qpos    = closed_qpos.clone()
        open_qpos[5] = cfg.gripper_open

        for _ in range(20):
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(open_qpos)

        for _ in range(settle_steps):
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
            self._record_step(open_qpos)

        return is_success_fn(cube, target_zone, table_height)

    # ── Waypoints ──────────────────────────────────────────────────────────

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
#  Direct LeRobot v2.1 writer
# ══════════════════════════════════════════════════════════════════════════════

def _write_episode_lerobot(
    frames: list[dict],
    ep_idx: int,
    out_dir: Path,
    global_frame_offset: int,
    fps: int = FPS,
    task_idx: int = 0,
) -> int:
    """
    Write one episode's parquet file + one MP4 per camera into the
    LeRobot v2.1 directory tree.

    Returns the number of frames written (T).
    """
    T         = len(frames)
    chunk_str = f"chunk-{ep_idx // CHUNK_SIZE:03d}"
    ep_str    = f"episode_{ep_idx:06d}"

    # ── Parquet ───────────────────────────────────────────────────────────────
    state  = np.stack([f["observation.state"] for f in frames])  # [T, 6]
    action = np.stack([f["action"]            for f in frames])  # [T, 6]

    parquet_dir = out_dir / "data" / chunk_str
    parquet_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "timestamp":         (np.arange(T, dtype=np.float32) / fps).tolist(),
        "frame_index":       np.arange(T, dtype=np.int64).tolist(),
        "episode_index":     np.full(T, ep_idx, dtype=np.int64).tolist(),
        "task_index":        np.full(T, task_idx, dtype=np.int64).tolist(),
        # global index: unique across the whole dataset, never resets
        "index":             np.arange(
                                 global_frame_offset,
                                 global_frame_offset + T,
                                 dtype=np.int64,
                             ).tolist(),
        # store as list-of-arrays so pyarrow encodes them as fixed-size lists
        "observation.state": list(state),
        "action":            list(action),
        "next.done":         np.concatenate(
                                 [np.zeros(T - 1, dtype=bool), [True]]
                             ).tolist(),
    })
    df.to_parquet(parquet_dir / f"{ep_str}.parquet", index=False)

    # ── Videos ────────────────────────────────────────────────────────────────
    for cam_key in CAMERA_KEYS:
        obs_key   = f"observation.images.{cam_key}"
        vid_dir   = out_dir / "videos" / chunk_str / obs_key
        vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path  = vid_dir / f"{ep_str}.mp4"

        img_frames = np.stack([f[obs_key] for f in frames])   # [T, H, W, 3] uint8
        iio.imwrite(
            str(vid_path),
            img_frames,
            fps=fps,
            codec="libx264",
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )

    return T


def _finalise_meta(
    out_dir: Path,
    episodes_meta: list[dict],
    fps: int,
    repo_id: str,
    val_frac: float = 0.1,
    language_instruction: str = LANGUAGE_INSTRUCTION,
):
    """
    Write meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl after all
    episodes have been collected.
    """
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    n_eps        = len(episodes_meta)
    n_val        = max(1, int(math.floor(n_eps * val_frac)))
    n_train      = n_eps - n_val
    total_frames = sum(e["length"] for e in episodes_meta)

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": language_instruction}) + "\n")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")

    H, W = 256, 256

    # Build features dict
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": [f"motor_{i}" for i in range(STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [f"motor_{i}" for i in range(ACTION_DIM)],
        },
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "next.done":     {"dtype": "bool",    "shape": [1], "names": None},
    }

    for cam_key in CAMERA_KEYS:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": [H, W, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps":          float(fps),
                "video.codec":        "h264",
                "video.pix_fmt":      "yuv420p",
                "video.is_depth_map": False,
                "has_audio":          False,
            },
        }

    info = {
        "codebase_version": "v2.1",          # lerobot 0.5.x expects this
        "robot_type":       "so101",
        "total_episodes":   n_eps,
        "total_frames":     total_frames,
        "total_tasks":      1,
        "total_videos":     n_eps * len(CAMERA_KEYS),
        "total_chunks":     math.ceil(n_eps / CHUNK_SIZE),
        "chunks_size":      CHUNK_SIZE,
        "fps":              fps,
        "splits": {
            "train": f"0:{n_train}",
            "val":   f"{n_train}:{n_eps}",
        },
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features":   features,
        "repo_id":    repo_id,
        "tasks":      [language_instruction],
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n── Dataset meta written ──────────────────────────────────")
    print(f"  Total episodes : {n_eps}  (train {n_train} / val {n_val})")
    print(f"  Total frames   : {total_frames}")
    print(f"  Output         : {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  collect_demonstrations  —  direct v2.1 edition
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
    val_frac: float     = 0.1,
    repo_id: str        = "local/genesis_pickplace",
    save_failures: bool = True,
    cube_color: str     = DEFAULT_CUBE_COLOR,
) -> list[bool]:
    """
    Run the oracle for n_episodes and write each SUCCESSFUL episode directly
    into a LeRobot v2.1 dataset tree (parquet + mp4) — no HDF5 intermediary,
    no convert_demos.py step.

    Failed episodes are optionally saved to <output_dir>/../failures_lerobot/
    for debugging, using the same layout.

    Returns a list of bool (True = success) for each attempted episode.
    """
    cfg    = OracleConfig()
    oracle = SO101Oracle(so101, scene, cameras, cfg)

    language_instruction = language_instruction_for(cube_color)

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fail_dir = out_dir.parent / "failures_lerobot"
    if save_failures:
        fail_dir.mkdir(parents=True, exist_ok=True)

    home_pose = np.zeros(so101.n_dofs)
    results: list[bool] = []

    # Running counters
    successes           = 0
    saved_eps           = 0       # successful episodes index
    saved_fails         = 0       # failed episodes index
    global_frame_offset = 0       # total frames written to success dataset
    fail_frame_offset   = 0       # total frames written to failure dataset

    episodes_meta: list[dict] = []
    fail_episodes_meta: list[dict] = []

    for ep in range(n_episodes):
        # ── Randomise cube & target positions ────────────────────────────────
        MIN_SEP = 0.12
        for _ in range(50):
            cube_xy   = np.array([0.25, 0.00])  + np.random.uniform(-0.05, 0.05, 2)
            target_xy = np.array([0.15, -0.15]) + np.random.uniform(-0.06, 0.06, 2)
            if np.linalg.norm(cube_xy - target_xy) >= MIN_SEP:
                break

        # ── Reset ─────────────────────────────────────────────────────────────
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

        # ── Run oracle ────────────────────────────────────────────────────────
        success = oracle.run_episode(
            cube, target_zone, table_height,
            is_success_fn, check_sub_goals_fn,
        )

        label = "PASS ✓" if success else "FAIL ✗"
        print(f"[{ep + 1:3d}/{n_episodes}] {label}", flush=True)
        results.append(success)

        frames = oracle.get_recorded_frames()

        # ── Write episode directly to LeRobot layout ─────────────────────────
        if success and frames:
            successes += 1
            T = _write_episode_lerobot(
                frames, saved_eps, out_dir,
                global_frame_offset, fps=fps,
            )
            episodes_meta.append({
                "episode_index": saved_eps,
                "tasks":         [LANGUAGE_INSTRUCTION],
                "length":        T,
            })
            global_frame_offset += T
            saved_eps           += 1

        elif not success and frames and save_failures:
            T = _write_episode_lerobot(
                frames, saved_fails, fail_dir,
                fail_frame_offset, fps=fps,
            )
            fail_episodes_meta.append({
                "episode_index": saved_fails,
                "tasks":         [LANGUAGE_INSTRUCTION],
                "length":        T,
            })
            fail_frame_offset += T
            saved_fails       += 1

    # ── Finalise meta files ───────────────────────────────────────────────────
    if episodes_meta:
        _finalise_meta(out_dir, episodes_meta, fps, repo_id, val_frac,
                       language_instruction=language_instruction)

    if save_failures and fail_episodes_meta:
        _finalise_meta(fail_dir, fail_episodes_meta, fps,
                       repo_id + "_failures", val_frac=0.0,
                       language_instruction=language_instruction)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Collection complete: {successes}/{n_episodes} successful "
          f"({successes / n_episodes:.1%})")
    print(f"Saved {saved_eps} episodes  →  {out_dir}")
    if save_failures:
        print(f"Saved {saved_fails} failures  →  {fail_dir}")
    print(f"{'─' * 50}")
    print(f"\n[next] Fine-tune directly with:")
    print(f"       python finetune.py --dataset_dir {out_dir}")

    return results