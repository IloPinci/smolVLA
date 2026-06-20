"""
zeroshot_eval.py
----------------
Run N rollouts of a SmolVLA checkpoint in the Genesis scene and record
Task Success Rate (TSR) and Partial Success Score (PSS).

Grasp detection
---------------
"lifted" is determined by the privileged GraspDetector (simulator
ground-truth), not by bare cube-elevation. A step only counts as a
verified grasp when the gripper is actually closed, the cube is elevated
above the table, AND the cube's position relative to the jaw has stayed
stable over several consecutive steps (i.e. it's moving WITH the jaw, not
just incidentally near it). This is diagnostic-only and is never seen by
the policy.

Camera architecture
-------------------
Two separate camera rigs are used during evaluation:

  TRAINING cameras  (256×256)
      Passed to the policy agent on every step. Subject to all
      CameraPerturbation axes (position shift baked into the Genesis
      camera objects, tilt baked into the Genesis camera objects,
      blackout applied post-render — before the frame reaches the
      policy or gets buffered into a video).

  WITNESS cameras   (1280×720)
      Rendered at the same cadence but NEVER passed to the policy.
      Always record the true, unperturbed view. Saved as separate MP4s
      for human review alongside the policy-facing rollout videos, so
      you can tell what the policy actually saw vs. what really happened
      (e.g. when running a blackout perturbation).

Usage
-----
    # Nominal evaluation (no perturbation)
    python zeroshot_eval.py --checkpoint lerobot/smolvla_base \
        --dataset_dir demos/lerobot/ --n 25 --out results/zeroshot.csv

    # Perturbed: black out the wrist camera
    python zeroshot_eval.py --checkpoint lerobot/smolvla_base \
        --dataset_dir demos/lerobot/ --n 25 \
        --blackout_mode wrist_only \
        --out results/eval_wrist_blackout.csv

    # Perturbed: shift the context camera 10 cm to the right
    python zeroshot_eval.py ... \
        --pos_target context --pos_offset 0.0 0.10 0.0

    # Perturbed: tilt the context camera 15 degrees
    python zeroshot_eval.py ... \
        --tilt_target context --tilt_deg 15.0

    # Combined perturbation
    python zeroshot_eval.py ... \
        --blackout_mode wrist_only \
        --pos_target context --pos_offset 0.05 0.0 0.0 \
        --tilt_target context --tilt_deg 10.0

Requirements: genesis, lerobot (with SmolVLA), torch, transformers
"""

import argparse
import csv
import os
import sys
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from transformers import AutoTokenizer

from scene_params import (
    IMG_RES, language_instruction_for, DEFAULT_CUBE_COLOR,
    CameraPerturbation, NOMINAL_PERTURBATION,
)

# ── Genesis ───────────────────────────────────────────────────────────────────
import genesis as gs

# ── LeRobot / SmolVLA ─────────────────────────────────────────────────────────
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError:
    print("[error] lerobot not installed or SmolVLA policy not found.")
    print("        Run: pip install -e '.[smolvla]' inside your lerobot clone.")
    sys.exit(1)

# ── Your environment helpers (must be importable from this script's directory)
import importlib.util, pathlib

def _import_from(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_script_dir = pathlib.Path(__file__).parent
_setup      = _import_from("two_camera_setup", _script_dir / "two_camera_setup.py")

build_environment                 = _setup.build_environment
attach_cameras                    = _setup.attach_cameras
attach_witness_cameras_with_robot  = _setup.attach_witness_cameras_with_robot
is_success                        = _setup.is_success
check_sub_goals                   = _setup.check_sub_goals


# ══════════════════════════════════════════════════════════════════════════════
#  Language instruction (must match your dataset)
# ══════════════════════════════════════════════════════════════════════════════

LANGUAGE_INSTRUCTION = language_instruction_for(DEFAULT_CUBE_COLOR)

# ══════════════════════════════════════════════════════════════════════════════
#  Policy wrapper
# ══════════════════════════════════════════════════════════════════════════════

class SmolVLAAgent:
    """
    Thin wrapper that converts Genesis observations into the dict format
    SmolVLAPolicy expects and returns a numpy action array.
    """
    def __init__(self, checkpoint: str, dataset_dir: str, device: str = "cuda"):
        self.device = device
        print(f"[info] loading SmolVLA checkpoint: {checkpoint}")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, device=device)
        self.policy.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )

        # Load normalization stats
        stats_path = Path(dataset_dir) / "meta" / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"stats.json not found at {stats_path}")
        with open(stats_path) as f:
            raw = json.load(f)

        self.state_mean = np.array(raw["observation.state"]["mean"], dtype=np.float32)
        self.state_std  = np.array(raw["observation.state"]["std"],  dtype=np.float32)
        self.act_mean   = np.array(raw["action"]["mean"],            dtype=np.float32)
        self.act_std    = np.array(raw["action"]["std"],             dtype=np.float32)

        print(f"[info] normalization stats loaded from {stats_path}")
        print(f"       state_mean : {np.round(self.state_mean, 3)}")
        print(f"       state_std  : {np.round(self.state_std,  3)}")
        print(f"       action_mean: {np.round(self.act_mean,   3)}")
        print(f"       action_std : {np.round(self.act_std,    3)}")
        print("[info] SmolVLA loaded successfully.")

    def reset(self):
        """Call at the start of every episode to clear action chunk buffer."""
        self.policy.reset()

    @torch.no_grad()
    def act(self, context_rgb, wrist_rgb, top_rgb,
            joint_state, language_instruction=LANGUAGE_INSTRUCTION):

        def img_to_tensor(arr):
            t = torch.from_numpy(arr.copy()).float() / 255.0      # [H,W,3] -> [0,1]
            return t.permute(2, 0, 1).unsqueeze(0).to(self.device) # [1,3,H,W]

        # Normalize state
        norm_state = (joint_state - self.state_mean) / (self.state_std + 1e-8)
        state_t = torch.from_numpy(norm_state).float().unsqueeze(0).to(self.device)

        enc = self.tokenizer(
            [language_instruction],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )

        obs = {
            "observation.images.camera1":          img_to_tensor(context_rgb),
            "observation.images.camera2":          img_to_tensor(wrist_rgb),
            "observation.images.camera3":          img_to_tensor(top_rgb),
            "observation.state":                   state_t,
            "observation.language.tokens":         enc["input_ids"].to(self.device),
            "observation.language.attention_mask": enc["attention_mask"].bool().to(self.device),
        }

        action_t = self.policy.select_action(obs)
        if action_t.ndim == 3:
            action_t = action_t[:, 0, :]

        action_norm = action_t.squeeze(0).cpu().numpy()

        # Unnormalize output — model outputs normalized values
        action = action_norm * self.act_std + self.act_mean
        return action.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Scene builder
# ══════════════════════════════════════════════════════════════════════════════

def build_scene(xml_path: str, perturbation: CameraPerturbation = NOMINAL_PERTURBATION):
    """
    Build the Genesis scene with both the TRAINING camera rig (policy-facing,
    subject to `perturbation`) and the WITNESS camera rig (always nominal,
    high-res, human-review-only).

    Returns
    -------
    scene, so101, cube, target_zone, training_cams, witness_cams, table_height
    """
    gs.init(backend=gs.gpu, seed=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=16),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=100,
            tolerance=1e-9,
            constraint_timeconst=0.006,
            enable_self_collision=True,
            box_box_detection=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            show_cameras=False,
            ambient_light=(0.25, 0.25, 0.25),
            lights=[
                {"type": "directional", "dir": (-0.5, -0.5, -1.0),
                 "intensity": 8.0, "color": (1.0, 0.97, 0.90)},
                {"type": "directional", "dir": (0.5, 0.5, -0.5),
                 "intensity": 2.5, "color": (0.6, 0.7, 1.0)},
            ],
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    table_height = 0.8
    cube, target_zone = build_environment(scene, table_height)

    so101 = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, table_height))
    )

    # Training cameras — position/tilt perturbation baked in here.
    training_cams = attach_cameras(scene, so101, table_height, perturbation=perturbation)

    # Witness cameras — always nominal, always high-res, never fed to the policy.
    witness_cams = attach_witness_cameras_with_robot(scene, so101, table_height)

    scene.build()

    # Joint gains — identical to collection setup
    arm_dofs = np.arange(5)
    so101.set_dofs_kp(np.array([4000, 4000, 3000, 2000, 2000]), dofs_idx_local=arm_dofs)
    so101.set_dofs_kv(np.array([400,  400,  300,  200,  200]),  dofs_idx_local=arm_dofs)
    gripper_dof = np.array([5])
    so101.set_dofs_kp(np.array([800.0]), dofs_idx_local=gripper_dof)
    so101.set_dofs_kv(np.array([80.0]),  dofs_idx_local=gripper_dof)
    so101.set_dofs_force_range(
        lower=np.array([-100.0]), upper=np.array([100.0]),
        dofs_idx_local=gripper_dof,
    )
    for link_name in ["gripper", "moving_jaw_so101_v1"]:
        so101.get_link(link_name).set_friction(5.0)

    return scene, so101, cube, target_zone, training_cams, witness_cams, table_height


# ══════════════════════════════════════════════════════════════════════════════
#  Episode reset
# ══════════════════════════════════════════════════════════════════════════════

def reset_episode(scene, so101, cube, target_zone, table_height, rng):
    home = np.zeros(so101.n_dofs)

    # Randomise cube + target (same distribution as collection)
    MIN_SEP = 0.12
    for _ in range(50):
        cube_xy   = np.array([0.25, 0.00]) + rng.uniform(-0.05,  0.05, 2)
        target_xy = np.array([0.15, -0.15]) + rng.uniform(-0.06, 0.06, 2)
        if np.linalg.norm(cube_xy - target_xy) >= MIN_SEP:
            break

    so101.set_dofs_position(home)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home)

    cube.set_pos(np.array([cube_xy[0], cube_xy[1], table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    target_zone.set_pos(np.array([target_xy[0], target_xy[1], table_height + 0.001]))

    for _ in range(20):
        so101.set_dofs_velocity(np.zeros(so101.n_dofs))
        so101.control_dofs_position(home)
        scene.step()

    return cube_xy, target_xy


# ══════════════════════════════════════════════════════════════════════════════
#  Run one rollout
# ══════════════════════════════════════════════════════════════════════════════

MAX_STEPS    = 50       # ~30 frames per oracle episode; 50 gives slight extra margin
HOLD_STEPS   = 10        # must match oracle's record_every_n_steps
RECORD_EVERY = 5         # buffer 1 frame per 5 policy steps for video export
ARM_DOFS     = np.arange(5)
GRIPPER_DOF  = np.array([5])
GRIPPER_CLOSE_TARGET = 0.18


class GraspDetector:
    """
    Privileged (simulator ground-truth) grasp detector for diagnostics only.
    Never seen by the policy — same role as check_sub_goals(), just stricter.

    A step counts as "grasped" only when ALL of:
      - gripper DOF is actually closed (not mid-transition)
      - cube is elevated above the table
      - cube's position relative to the jaw has stayed stable over `window`
        consecutive steps (i.e. it's moving WITH the jaw, not just near it)
    """
    def __init__(self, table_height, jaw_link_name="moving_jaw_so101_v1",
                 gripper_close_target=GRIPPER_CLOSE_TARGET, gripper_closed_tol=0.05,
                 near_radius=0.08, elevation_thresh=0.02,
                 window=5, rel_pos_tol=0.01):
        self.table_height         = table_height
        self.jaw_link_name        = jaw_link_name
        self.gripper_close_target = gripper_close_target
        self.gripper_closed_tol   = gripper_closed_tol
        self.near_radius          = near_radius
        self.elevation_thresh     = elevation_thresh
        self.window               = window
        self.rel_pos_tol          = rel_pos_tol
        self._rel_pos_history     = []

    def reset(self):
        self._rel_pos_history = []

    def step(self, so101, cube):
        jaw_pos   = so101.get_link(self.jaw_link_name).get_pos()
        cube_pos  = cube.get_pos()
        gripper_q = so101.get_dofs_position()[5].item()

        dist        = torch.norm(jaw_pos - cube_pos).item()
        near        = dist < self.near_radius
        is_closed   = abs(gripper_q - self.gripper_close_target) < self.gripper_closed_tol
        is_elevated = cube_pos[2].item() > (self.table_height + self.elevation_thresh)

        rel_pos = (cube_pos - jaw_pos).cpu().numpy()
        self._rel_pos_history.append(rel_pos)
        if len(self._rel_pos_history) > self.window:
            self._rel_pos_history.pop(0)

        co_moving = False
        if len(self._rel_pos_history) == self.window:
            spread    = np.std(self._rel_pos_history, axis=0).max()
            co_moving = spread < self.rel_pos_tol

        grasped = is_closed and is_elevated and co_moving
        return {
            "near": near, "grasped": grasped, "dist": dist,
            "is_closed": is_closed, "is_elevated": is_elevated, "co_moving": co_moving,
        }


def run_rollout(
    agent,
    scene,
    so101,
    cube,
    target_zone,
    training_cams: dict,
    witness_cams: dict,
    table_height: float,
    perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
    cube_xy=None,
    target_xy=None,
) -> dict:
    """
    Run one episode of the SmolVLA policy and return a results dict.

    Action interpretation:
        action[:5]  — arm joint absolute positions
        action[5]   — gripper absolute position

    Camera handling:
        Training cameras are rendered, perturbation.apply_blackout() is
        applied (a no-op under nominal conditions), and ONLY those
        (possibly blacked-out) frames are passed to the policy and buffered
        into the "policy" rollout video. Witness cameras are rendered at the
        same cadence, always nominal, and buffered separately into the
        "witness" rollout video — this is the ground truth the policy never
        sees.

    Grasp detection:
        "lifted" comes from the privileged GraspDetector (verified grasp:
        gripper closed + cube elevated + co-moving with the jaw), not bare
        elevation. "reached" comes from the detector's near-block check.
        Both latch True for the rest of the episode once triggered.
    """
    agent.reset()
    grasp_detector = GraspDetector(table_height=table_height)
    grasp_detector.reset()
    latch = {"lifted": False, "reached": False}

    context_cam = training_cams["context"]
    wrist_cam   = training_cams["wrist"]
    top_cam     = training_cams["top"]

    reached = lifted = placed = False
    step = 0

    # Frame buffers — policy-facing (training, post-blackout) and witness (true view)
    train_vid = {"context": [], "wrist": [], "top": []}
    wit_vid   = {k: [] for k in witness_cams}

    while step < MAX_STEPS:
        # ── Render training cameras ─────────────────────────────────────────
        ctx_rgb,   _, _, _ = context_cam.render()
        wrist_cam.move_to_attach()               # immediately before wrist render
        wrist_rgb, _, _, _ = wrist_cam.render()
        top_rgb,   _, _, _ = top_cam.render()

        # Apply blackout to training frames BEFORE they reach the policy
        ctx_rgb   = perturbation.apply_blackout(ctx_rgb,   "camera1")
        wrist_rgb = perturbation.apply_blackout(wrist_rgb, "camera2")
        top_rgb   = perturbation.apply_blackout(top_rgb,   "camera3")

        # ── Render witness cameras (always true view) ───────────────────────
        for w_role, w_cam in witness_cams.items():
            if "wrist" in w_role:
                w_cam.move_to_attach()
            w_rgb, _, _, _ = w_cam.render()
            if step % RECORD_EVERY == 0:
                wit_vid[w_role].append(w_rgb.copy())

        # Buffer training frames (post-blackout) for the policy-facing video
        if step % RECORD_EVERY == 0:
            train_vid["context"].append(ctx_rgb.copy())
            train_vid["wrist"].append(wrist_rgb.copy())
            train_vid["top"].append(top_rgb.copy())

        # ── Policy action ────────────────────────────────────────────────────
        qpos   = so101.get_dofs_position().cpu().numpy()
        action = agent.act(ctx_rgb, wrist_rgb, top_rgb, qpos[:6])

        # ── Apply — absolute positions, HOLD_STEPS sim steps each ───────────
        arm_target     = action[:5]
        gripper_target = np.array([action[5]])
        for _ in range(HOLD_STEPS):
            so101.control_dofs_position(arm_target,     dofs_idx_local=ARM_DOFS)
            so101.control_dofs_position(gripper_target, dofs_idx_local=GRIPPER_DOF)
            scene.step()

        # ── Sub-goals ─────────────────────────────────────────────────────────
        sg = check_sub_goals(so101, cube, target_zone, table_height)   # only "placed" is used
        gd = grasp_detector.step(so101, cube)

        if gd["near"]:
            latch["reached"] = True
        if gd["grasped"]:
            latch["grasped"] = True

        reached = latch["reached"]
        lifted  = latch.get("grasped", False)   # verified grasp, not bare elevation
        placed  = sg["placed"]

        if placed:
            break
        step += 1

    success = is_success(cube, target_zone, table_height)
    pss     = 0.2 * reached + 0.4 * lifted + 1.0 * placed

    return {
        "success":    success,
        "pss":        round(pss, 3),
        "steps":      step,
        "reached":    reached,
        "lifted":     lifted,
        "placed":     placed,
        "cube_xy":    cube_xy.tolist() if cube_xy is not None else None,
        "target_xy":  target_xy.tolist() if target_xy is not None else None,
        # Frame buffers — stripped before CSV write
        "_train_vid": train_vid,
        "_wit_vid":   wit_vid,
    }


def _save_episode_videos(
    r: dict,
    tag: str,
    policy_video_dir: Path,
    witness_video_dir: Path,
    fps: int = 15,
):
    """
    Write MP4s for one episode.

    policy_video_dir  — frames the policy actually saw (may include
                        blackout artefacts from the active perturbation).
    witness_video_dir — true high-res witness view, always unperturbed.
    """
    def _write(frames, path):
        if frames:
            iio.imwrite(
                str(path), np.stack(frames), fps=fps,
                codec="libx264",
                output_params=["-crf", "22", "-pix_fmt", "yuv420p"],
            )

    train_vid = r["_train_vid"]
    wit_vid   = r["_wit_vid"]

    # ── Policy (training-camera) videos ─────────────────────────────────────
    for role in ("context", "wrist", "top"):
        (policy_video_dir / role).mkdir(parents=True, exist_ok=True)
        _write(train_vid[role], policy_video_dir / role / f"{tag}.mp4")

    combined_policy = [
        np.concatenate([c, w, t], axis=1)
        for c, w, t in zip(train_vid["context"], train_vid["wrist"], train_vid["top"])
    ]
    (policy_video_dir / "combined").mkdir(parents=True, exist_ok=True)
    _write(combined_policy, policy_video_dir / "combined" / f"{tag}.mp4")

    # ── Witness videos ───────────────────────────────────────────────────────
    w_roles = list(wit_vid.keys())
    for w_role in w_roles:
        short = w_role.replace("witness_", "")
        (witness_video_dir / short).mkdir(parents=True, exist_ok=True)
        _write(wit_vid[w_role], witness_video_dir / short / f"{tag}.mp4")

    if len(w_roles) > 1 and wit_vid[w_roles[0]]:
        combined_wit = [
            np.concatenate([wit_vid[r][i] for r in w_roles], axis=1)
            for i in range(len(wit_vid[w_roles[0]]))
        ]
        (witness_video_dir / "combined").mkdir(parents=True, exist_ok=True)
        _write(combined_wit, witness_video_dir / "combined" / f"{tag}.mp4")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",         default="so101_arm/so101_new_calib.xml")
    parser.add_argument("--checkpoint",  default="lerobot/smolvla_base",
                        help="HuggingFace repo or local path to SmolVLA checkpoint")
    parser.add_argument("--n",           type=int, default=25, help="number of rollouts")
    parser.add_argument("--out",         default="results/zeroshot.csv")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--dataset_dir", default="demos/lerobot/",
                        help="Dataset used for fine-tuning (must contain meta/stats.json)")

    # ── Camera perturbation CLI args ─────────────────────────────────────────
    parser.add_argument(
        "--blackout_mode", default="none",
        choices=["none", "3-cam", "3rd_black", "all_black", "wrist_only", "context_only"],
        help="Which training camera(s) to black out (zeroed frame, never touches policy input).",
    )
    parser.add_argument(
        "--pos_target", default=None, choices=[None, "context", "wrist", "top"],
        help="Which training camera to shift in world space.",
    )
    parser.add_argument(
        "--pos_offset", nargs=3, type=float, default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="World-frame (dx, dy, dz) offset in metres for --pos_target camera.",
    )
    parser.add_argument(
        "--tilt_target", default=None, choices=[None, "context", "wrist", "top"],
        help="Which training camera to tilt (rotate look direction around world-Z).",
    )
    parser.add_argument(
        "--tilt_deg", type=float, default=0.0,
        help="Tilt angle in degrees for --tilt_target camera.",
    )
    args = parser.parse_args()

    # ── Build perturbation config ─────────────────────────────────────────────
    perturbation = CameraPerturbation(
        blackout_mode    = args.blackout_mode,
        position_target  = args.pos_target,
        position_offset  = tuple(args.pos_offset),
        tilt_target      = args.tilt_target,
        tilt_deg         = args.tilt_deg,
    )
    print(f"[perturbation] {perturbation.label}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ── Build scene ───────────────────────────────────────────────────────────
    xml_path = os.path.abspath(args.xml)
    scene, so101, cube, target_zone, training_cams, witness_cams, table_height = \
        build_scene(xml_path, perturbation=perturbation)

    # ── Load policy ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent  = SmolVLAAgent(checkpoint=args.checkpoint, dataset_dir=args.dataset_dir, device=device)

    # ── Run rollouts ──────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed)
    records = []

    results_dir        = Path(args.out).parent
    policy_video_dir   = results_dir / "videos" / "policy"
    witness_video_dir  = results_dir / "videos" / "witness"
    policy_video_dir.mkdir(parents=True, exist_ok=True)
    witness_video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {args.n} zero-shot rollouts  (checkpoint: {args.checkpoint})\n")
    print(f"{'ep':>4}  {'result':>6}  {'pss':>5}  {'steps':>6}")
    print("─" * 40)

    MAX_SUCCESS_VIDEOS = 5
    success_video_count = 0

    for ep in range(args.n):
        cube_xy, target_xy = reset_episode(scene, so101, cube, target_zone, table_height, rng)
        r = run_rollout(
            agent, scene, so101, cube, target_zone,
            training_cams, witness_cams, table_height,
            perturbation=perturbation,
            cube_xy=cube_xy, target_xy=target_xy,
        )
        r["episode"] = ep

        res_str = "PASS ✓" if r["success"] else "FAIL ✗"
        print(f"{ep:4d}  {res_str:>6}  {r['pss']:>5.3f}  {r['steps']:>6}")

        # ── Save videos ──────────────────────────────────────────────────────
        save_video = False
        if r["success"] and success_video_count < MAX_SUCCESS_VIDEOS:
            tag = f"ep{ep:02d}_pass"
            save_video = True
            success_video_count += 1
        elif not r["success"]:
            tag = f"ep{ep:02d}_fail"
            save_video = True

        if save_video:
            _save_episode_videos(r, tag, policy_video_dir, witness_video_dir)

        # Strip frame buffers before storing in records (keep CSV clean)
        r.pop("_train_vid")
        r.pop("_wit_vid")
        records.append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_success = sum(r["success"] for r in records)
    tsr       = n_success / args.n
    mean_pss  = np.mean([r["pss"] for r in records])

    print("\n" + "─" * 40)
    print(f"Perturbation  : {perturbation.label}")
    print(f"Zero-shot TSR : {n_success}/{args.n} = {tsr:.1%}")
    print(f"Mean PSS      : {mean_pss:.3f}")
    print(f"Policy  videos: {policy_video_dir}")
    print(f"Witness videos: {witness_video_dir}")
    print("─" * 40)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[ok] results saved → {args.out}")

    # ── Paper-ready line ──────────────────────────────────────────────────────
    print(f"\nPaper entry:  Zero-shot {perturbation.label} TSR = {tsr:.1%}  (N={args.n})")


if __name__ == "__main__":
    main()