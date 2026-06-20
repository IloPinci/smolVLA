"""
oracle_direct.py  —  SO-101 scripted oracle + DIRECT LeRobot v3.0 writer
=========================================================================

Writes episodes DIRECTLY into the LeRobot v3.0 format during collection
using the official LeRobotDataset API.  No convert step, no v2.1 intermediate.

Camera architecture
-------------------
Two camera dicts are accepted:

  cameras         — TRAINING cameras (256x256).  These are rendered every
                    record_every_n_steps sim steps and written into the
                    LeRobot dataset as observation frames.  Subject to
                    CameraPerturbation.blackout — frames are zeroed before
                    storage when the relevant mode is active.

  witness_cameras — WITNESS cameras (1280x720, optional).  Rendered at the
                    same cadence as training cameras but written ONLY to
                    MP4 review videos, never into the dataset.  Always
                    record the true, unperturbed view.

Key API flow (v3.0):
    dataset = LeRobotDataset.create(repo_id, fps, root, features)
    for each episode:
        dataset.add_frame(frame_dict)   # called once per recorded timestep
        dataset.save_episode()          # called at episode end
    dataset.finalize()                  # MUST be called before push_to_hub
    dataset.push_to_hub()              # optional — omit for local-only

Action format — ABSOLUTE joint positions (6-DOF):
    action[:5]  — arm joint absolute positions (radians)
    action[5]   — gripper absolute position
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from scene_params import (
    language_instruction_for, DEFAULT_CUBE_COLOR,
    CameraPerturbation, NOMINAL_PERTURBATION,
)

# ── LeRobot v3 API ─────────────────────────────────────────────────────────
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
FPS                  = 10
LANGUAGE_INSTRUCTION = language_instruction_for(DEFAULT_CUBE_COLOR)

_CAM_REMAP = {
    "context": "camera1",
    "wrist":   "camera2",
    "top":     "camera3",
}
# Reverse map: LeRobot key → role (for blackout lookup)
_KEY_TO_ROLE = {v: k for k, v in _CAM_REMAP.items()}

_ARM_DOFS    = np.arange(5)
_GRIPPER_DOF = np.array([5])


def make_features(img_h: int = 256, img_w: int = 256) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {"motors": [f"motor_{i}" for i in range(STATE_DIM)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {"motors": [f"motor_{i}" for i in range(ACTION_DIM)]},
        },
    }
    for cam_key in CAMERA_KEYS:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channel"],
        }
    return features


# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OracleConfig:
    pos_noise_sigma:   float = 0.004
    rot_noise_sigma:   float = np.deg2rad(2)
    steps_per_segment: int   = 30
    pregrasp_clearance: float = 0.12
    lift_height:        float = 0.15
    carry_clearance:    float = 0.15
    gripper_open:       float = 0.8
    gripper_close_safe: float = 0.18
    ik_pos_tol:         float = 1e-4
    ik_rot_tol:         float = 1e-4
    settle_steps: list = field(default_factory=lambda: [10, 50, 100, 25, 25, 60])
    record_every_n_steps: int = 10


# ══════════════════════════════════════════════════════════════════════════════
#  Oracle policy
# ══════════════════════════════════════════════════════════════════════════════

class SO101Oracle:
    """
    Scripted oracle that records training frames AND optional witness frames.

    Parameters
    ----------
    cameras : dict
        Training cameras {"context": cam, "wrist": cam, "top": cam}.
        Rendered at IMG_RES and written into the dataset.
    witness_cameras : dict or None
        High-res witness cameras {"witness_context": cam, ...}.
        Rendered at the same cadence but NEVER written to the dataset.
    perturbation : CameraPerturbation
        Controls blackout applied to training frames post-render.
        Position/tilt perturbations are already baked into the camera
        objects by attach_cameras() in two_camera_setup.py.
    """

    def __init__(self, so101, scene,
                 cameras=None,
                 witness_cameras=None,
                 perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
                 cfg: OracleConfig = OracleConfig(),
                 language_instruction: str = LANGUAGE_INSTRUCTION):
        self.robot                = so101
        self.scene                = scene
        self.cameras              = cameras
        self.witness_cameras      = witness_cameras   # NEW — high-res, never in dataset
        self.perturbation         = perturbation
        self.cfg                  = cfg
        self.language_instruction = language_instruction
        self._rng                 = np.random.default_rng()

        self.end_effector = so101.get_link("moving_jaw_so101_v1")
        self.grasp_quat   = np.array([0.707107, 0.0, -0.707107, 0.0])

        self._frames:         list[dict] = []   # training frames (written to dataset)
        self._witness_frames: list[dict] = []   # witness frames (written to video only)
        self._last_qpos:      torch.Tensor | None = None
        self._step_counter:   int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def run_episode(self, cube, target_zone, table_height,
                    is_success_fn, check_sub_goals_fn) -> bool:
        self._frames         = []
        self._witness_frames = []
        self._last_qpos      = None
        self._step_counter   = 0
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
                waypoints[-1], self.cfg.settle_steps[-1], prev_qpos,
                cube, target_zone, table_height, is_success_fn,
            )

        if not success:
            success = is_success_fn(cube, target_zone, table_height)

        return success

    def get_recorded_frames(self) -> list[dict]:
        """Training frames to be written into the LeRobot dataset."""
        return self._frames

    def get_witness_frames(self) -> list[dict]:
        """
        Witness frames for human review.  Each dict has keys:
            "witness_context"  — uint8 [H, W, 3]   (1280×720)
            "witness_wrist"    — uint8 [H, W, 3]   (if witness wrist cam attached)
            "witness_top"      — uint8 [H, W, 3]
        """
        return self._witness_frames

    # ── Frame recording ────────────────────────────────────────────────────

    def _record_step(self, qpos: torch.Tensor):
        self._step_counter += 1
        if self._step_counter % self.cfg.record_every_n_steps != 0:
            return
        if self.cameras is None:
            return

        # ── Training frame ──────────────────────────────────────────────────
        frame = {}
        ATTACHED = {"wrist"}

        for role, cam in self.cameras.items():
            if role in ATTACHED:
                cam.move_to_attach()
            rgb, _, _, _ = cam.render()                      # uint8 [H, W, 3]

            cam_key = _CAM_REMAP.get(role, role)             # e.g. "camera1"
            # Apply blackout perturbation if configured
            rgb = self.perturbation.apply_blackout(rgb, cam_key)
            frame[f"observation.images.{cam_key}"] = rgb

        actual_qpos = self.robot.get_dofs_position()
        frame["observation.state"] = actual_qpos[:6].cpu().numpy().astype(np.float32)
        frame["action"]            = qpos[:6].cpu().numpy().astype(np.float32)
        frame["task"]              = self.language_instruction
        self._frames.append(frame)

        # ── Witness frame (high-res, no blackout) ───────────────────────────
        if self.witness_cameras:
            wf = {}
            for w_role, w_cam in self.witness_cameras.items():
                if "wrist" in w_role:
                    w_cam.move_to_attach()
                rgb_w, _, _, _ = w_cam.render()
                wf[w_role] = rgb_w
            self._witness_frames.append(wf)

    # ── Segment execution ──────────────────────────────────────────────────

    def _execute_segment(self, target_cart_pos, gripper_target, settle_steps,
                         prev_qpos, cube, target_zone, table_height,
                         check_sub_goals_fn, latch: dict):
        cfg = self.cfg
        noisy_quat = self._perturb_quat_z(
            self.grasp_quat, self._rng.normal(0, cfg.rot_noise_sigma)
        )
        target_qpos = self.robot.inverse_kinematics(
            link=self.end_effector, pos=target_cart_pos, quat=noisy_quat,
            pos_tol=cfg.ik_pos_tol, rot_tol=cfg.ik_rot_tol,
        )
        target_qpos[5] = gripper_target

        alphas = torch.linspace(0.0, 1.0, cfg.steps_per_segment,
                                device=target_qpos.device).unsqueeze(1)
        interp_configs       = prev_qpos + alphas * (target_qpos - prev_qpos)
        interp_configs[:, 5] = prev_qpos[5]

        def _step(qpos) -> bool:
            self._record_step(qpos)
            self.robot.control_dofs_position(qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()
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
            link=self.end_effector, pos=target_cart_pos, quat=noisy_quat,
            pos_tol=cfg.ik_pos_tol, rot_tol=cfg.ik_rot_tol,
        )

        closed_qpos    = target_qpos.clone()
        closed_qpos[5] = prev_qpos[5]
        alphas = torch.linspace(0.0, 1.0, cfg.steps_per_segment,
                                device=target_qpos.device).unsqueeze(1)
        interp       = prev_qpos + alphas * (closed_qpos - prev_qpos)
        interp[:, 5] = prev_qpos[5]

        for interp_qpos in interp:
            self._record_step(interp_qpos)
            self.robot.control_dofs_position(interp_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(interp_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        for _ in range(15):
            self._record_step(closed_qpos)
            self.robot.control_dofs_position(closed_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(closed_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        open_qpos    = closed_qpos.clone()
        open_qpos[5] = cfg.gripper_open

        for _ in range(20):
            self._record_step(open_qpos)
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

        for _ in range(settle_steps):
            self._record_step(open_qpos)
            self.robot.control_dofs_position(open_qpos[_ARM_DOFS],    dofs_idx_local=_ARM_DOFS)
            self.robot.control_dofs_position(open_qpos[_GRIPPER_DOF], dofs_idx_local=_GRIPPER_DOF)
            self.scene.step()

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
#  Witness video writer
# ══════════════════════════════════════════════════════════════════════════════

def _save_witness_video(
    witness_frames: list[dict],
    episode_id: int,
    success: bool,
    witness_dir: Path,
    fps: int = 10,
    prefix: str = "episode_",
):
    """
    Save one high-res MP4 per witness camera for a single episode.

    Output files:
        <witness_dir>/context/<prefix><id:04d>_{pass|fail}.mp4
        <witness_dir>/wrist/<prefix><id:04d>_{pass|fail}.mp4   (if present)
        <witness_dir>/top/<prefix><id:04d>_{pass|fail}.mp4
        <witness_dir>/combined/<prefix><id:04d>_{pass|fail}.mp4
    """
    if not witness_frames:
        return

    label     = "pass" if success else "fail"
    stem      = f"{prefix}{episode_id:04d}_{label}"
    cam_roles = list(witness_frames[0].keys())   # e.g. ["witness_context", "witness_top"]

    for role in cam_roles:
        short = role.replace("witness_", "")     # "context", "wrist", "top"
        out   = witness_dir / short
        out.mkdir(parents=True, exist_ok=True)

        frames_np = np.stack([wf[role] for wf in witness_frames])  # [T, H, W, 3]
        iio.imwrite(
            str(out / f"{stem}.mp4"),
            frames_np,
            fps=fps,
            codec="libx264",
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )

    # Combined tile (side-by-side, whatever roles are available)
    combined_dir = witness_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_frames = []
    for wf in witness_frames:
        row = np.concatenate([wf[r] for r in cam_roles], axis=1)   # tile horizontally
        combined_frames.append(row)
    iio.imwrite(
        str(combined_dir / f"{stem}.mp4"),
        np.stack(combined_frames),
        fps=fps,
        codec="libx264",
        output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )


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
    n_episodes: int              = 200,
    output_dir: str              = "demos/lerobot/",
    fps: int                     = FPS,
    repo_id: str                 = "local/genesis_pickplace",
    push_to_hub: bool            = False,
    save_failures: bool          = True,
    cube_color: str              = DEFAULT_CUBE_COLOR,
    img_h: int                   = 256,
    img_w: int                   = 256,
    # New parameters -----------------------------------------------------------
    witness_cameras: dict | None = None,
    perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
) -> list[bool]:
    """
    Run the oracle for n_episodes, write training frames into a LeRobot v3.0
    dataset, and optionally save high-res witness videos for human review.

    Parameters
    ----------
    witness_cameras : dict or None
        High-res Genesis camera objects (from attach_witness_cameras_with_robot).
        When supplied, one MP4 is written per camera per episode to
        <output_dir>/../witness_videos/.
        Pass None to disable witness recording.
    perturbation : CameraPerturbation
        Controls which training camera frames are blacked out before
        being written into the dataset.  Position/tilt perturbations must
        be pre-applied at attach_cameras() time in two_camera_setup.py.
    """
    cfg                  = OracleConfig()
    language_instruction = language_instruction_for(cube_color)
    oracle               = SO101Oracle(
        so101, scene, cameras,
        witness_cameras=witness_cameras,
        perturbation=perturbation,
        cfg=cfg,
        language_instruction=language_instruction,
    )
    features = make_features(img_h, img_w)

    out_dir  = Path(output_dir).resolve()
    fail_dir = out_dir.parent / "failures_lerobot"

    # Witness video output dir  (always at a fixed location relative to out_dir)
    witness_dir = out_dir.parent / "witness_videos"
    if witness_cameras:
        witness_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Witness videos will be written to: {witness_dir}")

    print(f"[info] Creating LeRobot v3.0 dataset at: {out_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=str(out_dir),
        features=features, use_videos=True, image_writer_threads=4,
    )

    fail_dataset = None
    if save_failures:
        print(f"[info] Creating failures dataset at: {fail_dir}")
        fail_dataset = LeRobotDataset.create(
            repo_id=repo_id + "_failures", fps=fps, root=str(fail_dir),
            features=features, use_videos=True, image_writer_threads=4,
        )

    home_pose = np.zeros(so101.n_dofs)
    results: list[bool] = []
    successes  = 0
    saved_ep   = 0
    saved_fail = 0

    print(f"\nCollecting {n_episodes} episodes  "
          f"(fps={fps}, perturbation={perturbation.label})\n")

    for ep in range(n_episodes):
        MIN_SEP = 0.12
        for _ in range(50):
            cube_xy   = np.array([0.24, 0.00])  + np.random.uniform(-0.04,  0.04, 2)
            target_xy = np.array([0.15, -0.15]) + np.random.uniform(-0.05,  0.05, 2)
            if np.linalg.norm(cube_xy - target_xy) >= MIN_SEP:
                break

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

        success = oracle.run_episode(
            cube, target_zone, table_height, is_success_fn, check_sub_goals_fn,
        )

        label   = "PASS ✓" if success else "FAIL ✗"
        frames  = oracle.get_recorded_frames()
        wframes = oracle.get_witness_frames()
        print(f"[{ep + 1:3d}/{n_episodes}] {label}  T={len(frames)}", flush=True)

        results.append(success)

        if not frames:
            print(f"           [warn] no frames recorded — skipping")
            continue

        # ── Write training frames into the LeRobot dataset ─────────────────
        target_ds = dataset if success else fail_dataset
        if target_ds is not None:
            for frame in frames:
                target_ds.add_frame(frame)
            target_ds.save_episode()

        # ── Write witness video ────────────────────────────────────────────
        if witness_cameras and wframes:
            ep_index = saved_ep if success else saved_fail
            _save_witness_video(
                wframes,
                episode_id=ep_index,
                success=success,
                witness_dir=witness_dir,
                fps=fps,
            )

        if success:
            successes += 1
            saved_ep  += 1
        else:
            saved_fail += 1

    # ── Finalize ────────────────────────────────────────────────────────────
    print(f"\n[info] Finalizing dataset (flushing parquet writers)…")
    dataset.finalize()
    if fail_dataset is not None:
        fail_dataset.finalize()

    if push_to_hub:
        print(f"[info] Pushing to Hub as '{repo_id}' …")
        dataset.push_to_hub()
        if fail_dataset is not None:
            fail_dataset.push_to_hub()

    print(f"\n{'─' * 50}")
    print(f"Collection complete: {successes}/{n_episodes} successful "
          f"({successes / max(n_episodes, 1):.1%})")
    print(f"Dataset written to : {out_dir}")
    if save_failures:
        print(f"Failures written to: {fail_dir}")
    if witness_cameras:
        print(f"Witness videos     : {witness_dir}")
    print(f"{'─' * 50}")

    return results