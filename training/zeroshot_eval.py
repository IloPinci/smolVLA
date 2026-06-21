"""
zeroshot_eval.py
----------------
Run N rollouts of a SmolVLA checkpoint in the Genesis scene and record
Task Success Rate (TSR) and Partial Success Score (PSS).

Changes from previous version
------------------------------
Three new perturbation axes are exposed via CLI (and via ScenePerturbation):

  --cube_color   <color>      Change the block colour (red/blue/green/yellow).
                               Also changes the language instruction fed to
                               the policy so it stays internally consistent.

  --cube_pos     <preset>     Fix cube + target spawn to a named preset from
                               CUBE_POSITIONS (e.g. workspace_edge, far_left).
                               Omit to use the normal randomised distribution.

  --spheres      <preset>     Add distractor spheres from SPHERE_PRESETS
                               (e.g. two_flanking, high_clutter).  The spheres
                               are re-placed at the start of every episode.

Usage
-----
    # Nominal
    python zeroshot_eval.py --checkpoint lerobot/smolvla_base \
        --dataset_dir demos/lerobot/ --n 25

    # Blue cube
    python zeroshot_eval.py ... --cube_color blue

    # Fixed OOD spawn position
    python zeroshot_eval.py ... --cube_pos workspace_edge

    # High clutter
    python zeroshot_eval.py ... --spheres high_clutter

    # Combined: blue cube + far_left spawn + two flanking spheres
    python zeroshot_eval.py ... \
        --cube_color blue --cube_pos far_left --spheres two_flanking

    # Still works with camera perturbations too
    python zeroshot_eval.py ... --blackout_mode wrist_only --spheres single_left
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from transformers import AutoTokenizer

from scene_params import (
    IMG_RES, language_instruction_for, DEFAULT_CUBE_COLOR, CUBE_COLORS,
    CameraPerturbation, NOMINAL_PERTURBATION,
    ScenePerturbation, NOMINAL_SCENE,
    CUBE_POSITIONS, SPHERE_PRESETS,
    SphereConfig,
    BLACKOUT_MODES, 
)

import genesis as gs

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError:
    print("[error] lerobot not installed or SmolVLA policy not found.")
    sys.exit(1)

import importlib.util
import pathlib

def _import_from(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_script_dir = pathlib.Path(__file__).parent
_setup      = _import_from("two_camera_setup", _script_dir / "two_camera_setup.py")

build_environment                  = _setup.build_environment
attach_cameras                     = _setup.attach_cameras
attach_witness_cameras_with_robot  = _setup.attach_witness_cameras_with_robot
is_success                         = _setup.is_success
check_sub_goals                    = _setup.check_sub_goals


# ══════════════════════════════════════════════════════════════════════════════
#  Policy wrapper  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SmolVLAAgent:
    def __init__(self, checkpoint: str, dataset_dir: str, device: str = "cuda",
                 language_instruction: str | None = None):
        self.device = device
        self._default_instruction = (
            language_instruction or language_instruction_for(DEFAULT_CUBE_COLOR)
        )
        print(f"[info] Loading SmolVLA checkpoint: {checkpoint}")
        self.policy    = SmolVLAPolicy.from_pretrained(checkpoint, device=device)
        self.policy.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )

        stats_path = Path(dataset_dir) / "meta" / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"stats.json not found at {stats_path}")
        with open(stats_path) as f:
            raw = json.load(f)

        self.state_mean = np.array(raw["observation.state"]["mean"], dtype=np.float32)
        self.state_std  = np.array(raw["observation.state"]["std"],  dtype=np.float32)
        self.act_mean   = np.array(raw["action"]["mean"],            dtype=np.float32)
        self.act_std    = np.array(raw["action"]["std"],             dtype=np.float32)
        print("[info] SmolVLA loaded successfully.")

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def act(self, context_rgb, wrist_rgb, top_rgb, joint_state,
        language_instruction=None, absent_cameras: set[str] | None = None):

        instruction = language_instruction or self._default_instruction
        absent_cameras = absent_cameras or set()

        def img_to_tensor(arr):
            t = torch.from_numpy(arr.copy()).float() / 255.0
            return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        norm_state = (joint_state - self.state_mean) / (self.state_std + 1e-8)
        state_t    = torch.from_numpy(norm_state).float().unsqueeze(0).to(self.device)
        enc = self.tokenizer(
            [instruction], return_tensors="pt",
            padding=True, truncation=True, max_length=64,
        )
        obs = {}

        if "camera1" not in absent_cameras and context_rgb is not None:
            obs["observation.images.camera1"] = img_to_tensor(context_rgb)
        if "camera2" not in absent_cameras and wrist_rgb is not None:
            obs["observation.images.camera2"] = img_to_tensor(wrist_rgb)
        if "camera3" not in absent_cameras and top_rgb is not None:
            obs["observation.images.camera3"] = img_to_tensor(top_rgb)
        obs["observation.state"] = state_t
        obs["observation.language.tokens"] = enc["input_ids"].to(self.device)
        obs["observation.language.attention_mask"] = enc["attention_mask"].bool().to(self.device)

        action_t    = self.policy.select_action(obs)
        if action_t.ndim == 3:
            action_t = action_t[:, 0, :]
        action_norm = action_t.squeeze(0).cpu().numpy()
        return (action_norm * self.act_std + self.act_mean).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Scene builder
# ══════════════════════════════════════════════════════════════════════════════

def build_scene(
    xml_path: str,
    perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
    scene_perturbation: ScenePerturbation = NOMINAL_SCENE,
):
    """
    Build the Genesis scene.

    Returns
    -------
    scene, so101, cube, target_zone, sphere_entities,
    training_cams, witness_cams, table_height
    """
    gs.init(backend=gs.gpu, seed=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=16),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=100, tolerance=1e-9,
            constraint_timeconst=0.006,
            enable_self_collision=True, box_box_detection=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False, show_cameras=False,
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
    cube, target_zone, sphere_entities = build_environment(
        scene, table_height,
        cube_color=scene_perturbation.cube_color,
        spheres=scene_perturbation.spheres,
    )

    so101 = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, table_height))
    )

    training_cams = attach_cameras(scene, so101, table_height, perturbation=perturbation)
    witness_cams  = attach_witness_cameras_with_robot(scene, so101, table_height)

    scene.build()

    arm_dofs    = np.arange(5)
    gripper_dof = np.array([5])
    so101.set_dofs_kp(np.array([4000, 4000, 3000, 2000, 2000]), dofs_idx_local=arm_dofs)
    so101.set_dofs_kv(np.array([400,  400,  300,  200,  200]),  dofs_idx_local=arm_dofs)
    so101.set_dofs_kp(np.array([800.0]), dofs_idx_local=gripper_dof)
    so101.set_dofs_kv(np.array([80.0]),  dofs_idx_local=gripper_dof)
    so101.set_dofs_force_range(
        lower=np.array([-100.0]), upper=np.array([100.0]),
        dofs_idx_local=gripper_dof,
    )
    for link_name in ["gripper", "moving_jaw_so101_v1"]:
        so101.get_link(link_name).set_friction(5.0)

    return (scene, so101, cube, target_zone, sphere_entities,
            training_cams, witness_cams, table_height)


# ══════════════════════════════════════════════════════════════════════════════
#  Episode reset  (extended for scene perturbation)
# ══════════════════════════════════════════════════════════════════════════════

def reset_episode(
    scene, so101, cube, target_zone, table_height, rng,
    scene_perturbation: ScenePerturbation = NOMINAL_SCENE,
    sphere_entities: list | None = None,
):
    home = np.zeros(so101.n_dofs)
    cube_xy, target_xy = scene_perturbation.resolve_spawn(rng)

    so101.set_dofs_position(home)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home)

    cube.set_pos(np.array([cube_xy[0], cube_xy[1], table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    target_zone.set_pos(np.array([target_xy[0], target_xy[1], table_height + 0.001]))

    # Reposition distractor spheres
    if sphere_entities and scene_perturbation.spheres:
        for entity, cfg_s in zip(sphere_entities, scene_perturbation.spheres):
            z = table_height + cfg_s.radius
            entity.set_pos(np.array([cfg_s.xy[0], cfg_s.xy[1], z]))
            entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(20):
        so101.set_dofs_velocity(np.zeros(so101.n_dofs))
        so101.control_dofs_position(home)
        scene.step()

    return cube_xy, target_xy


# ══════════════════════════════════════════════════════════════════════════════
#  GraspDetector  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

MAX_STEPS            = 50
HOLD_STEPS           = 10
RECORD_EVERY         = 5
ARM_DOFS             = np.arange(5)
GRIPPER_DOF          = np.array([5])
GRIPPER_CLOSE_TARGET = 0.18


class GraspDetector:
    def __init__(self, table_height, jaw_link_name="moving_jaw_so101_v1",
                 gripper_close_target=GRIPPER_CLOSE_TARGET, gripper_closed_tol=0.05,
                 near_radius=0.08, elevation_thresh=0.02, window=5, rel_pos_tol=0.01):
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
        dist      = torch.norm(jaw_pos - cube_pos).item()
        near      = dist < self.near_radius
        is_closed = abs(gripper_q - self.gripper_close_target) < self.gripper_closed_tol
        is_elev   = cube_pos[2].item() > (self.table_height + self.elevation_thresh)

        rel_pos = (cube_pos - jaw_pos).cpu().numpy()
        self._rel_pos_history.append(rel_pos)
        if len(self._rel_pos_history) > self.window:
            self._rel_pos_history.pop(0)

        co_moving = False
        if len(self._rel_pos_history) == self.window:
            spread    = np.std(self._rel_pos_history, axis=0).max()
            co_moving = spread < self.rel_pos_tol

        grasped = is_closed and is_elev and co_moving
        return {"near": near, "grasped": grasped, "dist": dist,
                "is_closed": is_closed, "is_elevated": is_elev, "co_moving": co_moving}


# ══════════════════════════════════════════════════════════════════════════════
#  Run one rollout
# ══════════════════════════════════════════════════════════════════════════════

def run_rollout(
    agent,
    scene,
    so101,
    cube,
    target_zone,
    training_cams: dict,
    witness_cams: dict,
    table_height: float,
    perturbation: CameraPerturbation       = NOMINAL_PERTURBATION,
    scene_perturbation: ScenePerturbation  = NOMINAL_SCENE,
    cube_xy=None,
    target_xy=None,
) -> dict:
    agent.reset()
    grasp_detector = GraspDetector(table_height=table_height)
    grasp_detector.reset()
    latch = {"lifted": False, "reached": False}

    context_cam = training_cams["context"]
    wrist_cam   = training_cams["wrist"]
    top_cam     = training_cams["top"]

    reached = lifted = placed = False
    step    = 0

    train_vid = {"context": [], "wrist": [], "top": []}
    wit_vid   = {k: [] for k in witness_cams}

    # Language instruction reflects the cube colour for this episode
    lang_instr = scene_perturbation.language_instruction

    while step < MAX_STEPS:

        ctx_rgb = None
        if not perturbation.camera_is_absent("camera1"):
            ctx_rgb, _, _, _ = context_cam.render()
            ctx_rgb = perturbation.apply_blackout(ctx_rgb, "camera1")

        wrist_rgb = None
        if not perturbation.camera_is_absent("camera2"):
            wrist_cam.move_to_attach()
            wrist_rgb, _, _, _ = wrist_cam.render()
            wrist_rgb = perturbation.apply_blackout(wrist_rgb, "camera2")

        top_rgb = None
        if not perturbation.camera_is_absent("camera3"):
            top_rgb, _, _, _ = top_cam.render()
            top_rgb = perturbation.apply_blackout(top_rgb, "camera3")

        for w_role, w_cam in witness_cams.items():
            if "wrist" in w_role:
                w_cam.move_to_attach()
            w_rgb, _, _, _ = w_cam.render()
            if step % RECORD_EVERY == 0:
                wit_vid[w_role].append(w_rgb.copy())

        if step % RECORD_EVERY == 0:
            if ctx_rgb is not None:
                train_vid["context"].append(ctx_rgb.copy())
            if wrist_rgb is not None:
                train_vid["wrist"].append(wrist_rgb.copy())
            if top_rgb is not None:
                train_vid["top"].append(top_rgb.copy())

        qpos   = so101.get_dofs_position().cpu().numpy()

        absent = {k for k in ["camera1","camera2","camera3"]
          if perturbation.camera_is_absent(k)}
        
        action = agent.act(ctx_rgb, wrist_rgb, top_rgb, qpos[:6],
                   language_instruction=lang_instr,
                   absent_cameras=absent)

        arm_target     = action[:5]
        gripper_target = np.array([action[5]])
        for _ in range(HOLD_STEPS):
            so101.control_dofs_position(arm_target,     dofs_idx_local=ARM_DOFS)
            so101.control_dofs_position(gripper_target, dofs_idx_local=GRIPPER_DOF)
            scene.step()

        sg = check_sub_goals(so101, cube, target_zone, table_height)
        gd = grasp_detector.step(so101, cube)

        if gd["near"]:
            latch["reached"] = True
        if gd["grasped"]:
            latch["grasped"] = True

        reached = latch["reached"]
        lifted  = latch.get("grasped", False)
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
        "_train_vid": train_vid,
        "_wit_vid":   wit_vid,
    }


def _save_episode_videos(r, tag, policy_video_dir, witness_video_dir, fps=15):
    def _write(frames, path):
        if frames:
            iio.imwrite(str(path), np.stack(frames), fps=fps,
                        codec="libx264",
                        output_params=["-crf", "22", "-pix_fmt", "yuv420p"])

    train_vid = r["_train_vid"]
    wit_vid   = r["_wit_vid"]

    for role in ("context", "wrist", "top"):
        (policy_video_dir / role).mkdir(parents=True, exist_ok=True)
        _write(train_vid[role], policy_video_dir / role / f"{tag}.mp4")

    present = [train_vid[r] for r in ("context", "wrist", "top") if train_vid[r]]
    if len(present) > 1:
        combined_policy = [
            np.concatenate(frames, axis=1)
            for frames in zip(*present)
        ]
        _write(combined_policy, policy_video_dir / "combined" / f"{tag}.mp4")

    for w_role in wit_vid:
        short = w_role.replace("witness_", "")
        (witness_video_dir / short).mkdir(parents=True, exist_ok=True)
        _write(wit_vid[w_role], witness_video_dir / short / f"{tag}.mp4")

    w_roles = list(wit_vid.keys())
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
    parser.add_argument("--checkpoint",  default="lerobot/smolvla_base")
    parser.add_argument("--n",           type=int, default=25)
    parser.add_argument("--out",         default="results/zeroshot.csv")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--dataset_dir", default="demos/lerobot/")

    # ── Camera perturbation ───────────────────────────────────────────────────
    parser.add_argument(
    "--blackout_mode", default="none",
    choices=list(BLACKOUT_MODES), 
    )

    parser.add_argument("--pos_target",  default=None,
                        choices=[None, "context", "wrist", "top"])
    parser.add_argument("--pos_offset",  nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("DX", "DY", "DZ"))
    parser.add_argument("--tilt_target", default=None,
                        choices=[None, "context", "wrist", "top"])
    parser.add_argument("--tilt_deg",    type=float, default=0.0)

    # ── Scene perturbation (NEW) ──────────────────────────────────────────────
    parser.add_argument(
        "--cube_color", default=DEFAULT_CUBE_COLOR,
        choices=list(CUBE_COLORS.keys()),
        help="Cube colour — also changes the language instruction to the policy.",
    )
    parser.add_argument(
        "--cube_pos", default=None,
        choices=[None] + list(CUBE_POSITIONS.keys()),
        help="Fix cube/target spawn to a named preset (omit for random spawn).",
    )
    parser.add_argument(
        "--spheres", default="none",
        choices=list(SPHERE_PRESETS.keys()),
        help="Distractor sphere layout from SPHERE_PRESETS.",
    )

    args = parser.parse_args()

    # ── Build perturbation objects ────────────────────────────────────────────
    perturbation = CameraPerturbation(
        blackout_mode   = args.blackout_mode,
        position_target = args.pos_target,
        position_offset = tuple(args.pos_offset),
        tilt_target     = args.tilt_target,
        tilt_deg        = args.tilt_deg,
    )
    scene_perturbation = ScenePerturbation(
        cube_color           = args.cube_color,
        cube_position_preset = args.cube_pos,
        spheres              = SPHERE_PRESETS[args.spheres],
    )

    print(f"[camera perturbation] {perturbation.label}")
    print(f"[scene  perturbation] {scene_perturbation.label}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    xml_path = os.path.abspath(args.xml)
    (scene, so101, cube, target_zone, sphere_entities,
     training_cams, witness_cams, table_height) = build_scene(
        xml_path, perturbation=perturbation,
        scene_perturbation=scene_perturbation,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent  = SmolVLAAgent(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset_dir,
        device=device,
        language_instruction=scene_perturbation.language_instruction,
    )

    rng     = np.random.default_rng(args.seed)
    records = []

    results_dir       = Path(args.out).parent
    policy_video_dir  = results_dir / "videos" / "policy"
    witness_video_dir = results_dir / "videos" / "witness"
    policy_video_dir.mkdir(parents=True, exist_ok=True)
    witness_video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {args.n} rollouts  (checkpoint: {args.checkpoint})\n")
    print(f"{'ep':>4}  {'result':>6}  {'pss':>5}  {'steps':>6}")
    print("─" * 40)

    MAX_SUCCESS_VIDEOS  = 5
    success_video_count = 0

    for ep in range(args.n):
        cube_xy, target_xy = reset_episode(
            scene, so101, cube, target_zone, table_height, rng,
            scene_perturbation=scene_perturbation,
            sphere_entities=sphere_entities,
        )
        r = run_rollout(
            agent, scene, so101, cube, target_zone,
            training_cams, witness_cams, table_height,
            perturbation=perturbation,
            scene_perturbation=scene_perturbation,
            cube_xy=cube_xy, target_xy=target_xy,
        )
        r["episode"] = ep

        res_str = "PASS ✓" if r["success"] else "FAIL ✗"
        print(f"{ep:4d}  {res_str:>6}  {r['pss']:>5.3f}  {r['steps']:>6}")

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

        r.pop("_train_vid")
        r.pop("_wit_vid")
        records.append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_success = sum(r["success"] for r in records)
    tsr       = n_success / args.n
    mean_pss  = np.mean([r["pss"] for r in records])

    print("\n" + "─" * 40)
    print(f"Camera perturbation : {perturbation.label}")
    print(f"Scene  perturbation : {scene_perturbation.label}")
    print(f"Zero-shot TSR       : {n_success}/{args.n} = {tsr:.1%}")
    print(f"Mean PSS            : {mean_pss:.3f}")
    print("─" * 40)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[ok] Results saved → {args.out}")
    print(f"\nPaper entry: {scene_perturbation.label} | "
          f"{perturbation.label} TSR = {tsr:.1%}  (N={args.n})")


if __name__ == "__main__":
    main()