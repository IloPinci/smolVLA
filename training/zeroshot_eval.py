"""
zeroshot_eval.py
----------------
Run 25 rollouts of the unmodified SmolVLA base checkpoint in your Genesis
scene and record Task Success Rate (TSR) and Partial Success Score (PSS).

Usage:
    python zeroshot_eval.py \
        --xml    so101_arm/so101_new_calib.xml \
        --demos  demos/baseline/ \
        --n      25 \
        --out    results/zeroshot.csv

Requirements: genesis, lerobot (with SmolVLA), torch, h5py
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from transformers import AutoTokenizer
from scene_params import IMG_RES, language_instruction_for, DEFAULT_CUBE_COLOR


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
# Adjust the import path if your file structure is different.
import importlib.util, pathlib

def _import_from(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_script_dir = pathlib.Path(__file__).parent
_setup      = _import_from("two_camera_setup", _script_dir / "two_camera_setup.py")

build_environment  = _setup.build_environment
attach_cameras     = _setup.attach_cameras
is_success         = _setup.is_success
check_sub_goals    = _setup.check_sub_goals


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
    def __init__(self, checkpoint: str = "lerobot/smolvla_base", device: str = "cuda"):
        self.device = device
        print(f"[info] loading SmolVLA checkpoint: {checkpoint}")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, device=device)
        self.policy.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        print("[info] SmolVLA loaded successfully.")

    def reset(self):
        """Call at the start of every episode to clear action chunk buffer."""
        self.policy.reset()

    @torch.no_grad()
    def act(self, context_rgb: np.ndarray, wrist_rgb: np.ndarray, top_rgb: np.ndarray,
            joint_state: np.ndarray, language_instruction: str = LANGUAGE_INSTRUCTION) -> np.ndarray:
        """
        Parameters
        ----------
        context_rgb  : uint8 [256, 256, 3]
        wrist_rgb    : uint8 [256, 256, 3]
        joint_state  : float32 [6]   joint angles (radians)

        Returns
        -------
        action : float32 [6]   6 joint deltas + gripper absolute position
        """
        def to_tensor(arr, dtype=torch.float32):
            t = torch.from_numpy(arr.copy()).to(dtype=dtype, device=self.device)
            return t.unsqueeze(0)   # add batch dim

        # Images: [1, C, H, W] float32 in [0, 1]
        ctx_t   = to_tensor(context_rgb).permute(0, 3, 1, 2) / 255.0
        wrist_t = to_tensor(wrist_rgb).permute(0, 3, 1, 2)   / 255.0
        top_t   = to_tensor(top_rgb).permute(0, 3, 1, 2)     / 255.0

        # Tokenize the language instruction
        enc = self.tokenizer(
            [language_instruction],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )

        obs = {
            "observation.images.camera1":          ctx_t,
            "observation.images.camera2":          wrist_t,
            "observation.images.camera3":          top_t,
            "observation.state":                   to_tensor(joint_state),
            #! tokenized language
            "observation.language.tokens":         enc["input_ids"].to(self.device),
            "observation.language.attention_mask": enc["attention_mask"].bool().to(self.device),
        }

        action_t = self.policy.select_action(obs)  # [1, action_dim] or [1, chunk, action_dim]

        # Handle chunked output — take first action in chunk
        if action_t.ndim == 3:
            action_t = action_t[:, 0, :]

        return action_t.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Genesis scene builder  (mirrors two_camera_setup.py / main())
# ══════════════════════════════════════════════════════════════════════════════

def build_scene(xml_path: str):
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

    cameras = attach_cameras(scene, so101, table_height)
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

    return scene, so101, cube, target_zone, cameras, table_height


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

MAX_STEPS    = 300   # 30 sim-seconds at dt=0.01
ARM_DOFS     = np.arange(5)
GRIPPER_DOF  = np.array([5])
RECORD_EVERY = 5   # save 1 frame per 5 steps → ~600 frames max per episode

def _save_episode_videos(r: dict, tag: str, video_dir: Path, fps: int = 15):
    """
    Save three side-by-side MP4s per episode:
      <tag>_context.mp4   — context (third-person) camera
      <tag>_wrist.mp4     — wrist camera
      <tag>_combined.mp4  — all three cameras tiled horizontally
    """
    

    def _write(frames, path):
        if frames:
            iio.imwrite(str(path), frames, fps=fps, codec="libx264",
                        output_params=["-crf", "22", "-pix_fmt", "yuv420p"])

    _write(r["vid_context"], video_dir / f"{tag}_context.mp4")
    _write(r["vid_wrist"],   video_dir / f"{tag}_wrist.mp4")
    _write(r["vid_top"],     video_dir / f"{tag}_top.mp4")

    # Combined: tile context | wrist | top horizontally
    combined = []
    for ctx, wri, top in zip(r["vid_context"], r["vid_wrist"], r["vid_top"]):
        combined.append(np.concatenate([ctx, wri, top], axis=1))
    _write(combined, video_dir / f"{tag}_combined.mp4")


def run_rollout(agent, scene, so101, cube, target_zone,
                cameras, table_height,
                cube_xy=None, target_xy=None) -> dict:
    """
    Run one episode of the SmolVLA policy and return a results dict.
    Action interpretation:
        action[:5]  — arm joint deltas (added to current position)
        action[5]   — gripper delta (index 5 in the 7-dim vector)
        action[6]   — gripper absolute position (we use this for control)
    """
    agent.reset()
    latch = {"lifted": False}

    top_cam     = cameras["top"]
    wrist_cam   = cameras["wrist"]
    context_cam = cameras["context"]

    # Sub-goal tracking
    reached = False
    lifted  = False
    placed  = False
    step    = 0

    arm_dofs    = np.arange(5)
    gripper_dof = np.array([5])

    # Frame buffers — one list per camera
    vid_context = []
    vid_wrist   = []
    vid_top     = []

    while step < MAX_STEPS:
        # ── Render observations ───────────────────────────────────────────────
        wrist_cam.move_to_attach()
        ctx_rgb,   _, _, _ = context_cam.render()
        wrist_rgb, _, _, _ = wrist_cam.render()
        top_rgb,   _, _, _ = top_cam.render()

        qpos = so101.get_dofs_position().cpu().numpy()    # [6]

        # ── Policy inference ──────────────────────────────────────────────────
        action = agent.act(ctx_rgb, wrist_rgb, top_rgb, qpos[:6])  # [7]

        # ── DEBUG (remove after diagnosis) ───────────────────────────────────
        if step < 5:
            print(f"\n[DEBUG step {step}]")
            print(f"  qpos now     = {qpos[:6].round(3)}")
            print(f"  raw action   = {action.round(3)}")
            print(f"  arm_target   = {(qpos[:5] + action[:5]).round(3)}")
            print(f"  gripper_tgt  = {action[5]:.4f}")
            print(f"  ctx_rgb mean = {ctx_rgb.mean():.1f}  wrist mean = {wrist_rgb.mean():.1f}")


        # ── Apply action ──────────────────────────────────────────────────────
        # Arm: current absolute without delta
        arm_target     = action[:5]
        # Gripper: absolute position from action[5]
        gripper_target = np.array([action[5]])


        HOLD_STEPS = 10

        for _ in range(HOLD_STEPS):
            so101.control_dofs_position(arm_target,     dofs_idx_local=arm_dofs)
            so101.control_dofs_position(gripper_target, dofs_idx_local=gripper_dof)
            scene.step()

        # Record frames every N steps
        if step % RECORD_EVERY == 0:
            vid_context.append(ctx_rgb.copy())
            vid_wrist.append(wrist_rgb.copy())
            vid_top.append(top_rgb.copy())

        # ── Sub-goal tracking ─────────────────────────────────────────────────
        sg = check_sub_goals(so101, cube, target_zone, table_height, _latch=latch)
        reached = reached or sg["near_block"]
        lifted  = lifted  or sg["lifted"]
        placed  = sg["placed"]

        if placed:
            break

        step += 1

    success = is_success(cube, target_zone, table_height)
    pss     = 0.2 * reached + 0.4 * lifted + 1.0 * placed

    return {
        "success":  success,
        "pss":      round(pss, 3),
        "steps":    step,
        "reached":  reached,
        "lifted":   lifted,
        "placed":   placed,
        "cube_xy":   cube_xy.tolist() if cube_xy is not None else None,
        "target_xy": target_xy.tolist() if target_xy is not None else None,
        "vid_context": vid_context,
        "vid_wrist":   vid_wrist,
        "vid_top":     vid_top,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",        default="so101_arm/so101_new_calib.xml")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base",
                        help="HuggingFace repo or local path to SmolVLA checkpoint")
    parser.add_argument("--n",          type=int, default=25, help="number of rollouts")
    parser.add_argument("--out",        default="results/zeroshot.csv")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ── Build scene ───────────────────────────────────────────────────────────
    xml_path = os.path.abspath(args.xml)
    scene, so101, cube, target_zone, cameras, table_height = build_scene(xml_path)

    # ── Load policy ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent  = SmolVLAAgent(checkpoint=args.checkpoint, device=device)

    # ── Run rollouts ──────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed)
    records = []

    print(f"\nRunning {args.n} zero-shot rollouts  (checkpoint: {args.checkpoint})\n")
    print(f"{'ep':>4}  {'result':>6}  {'pss':>5}  {'steps':>6}  sub-goals")
    print("─" * 55)


    MAX_SUCCESS_VIDEOS = 5
    success_video_count = 0
    video_dir = Path(args.out).parent / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)


    for ep in range(args.n):
        cube_xy, target_xy = reset_episode(scene, so101, cube, target_zone, table_height, rng)
        r = run_rollout(agent, scene, so101, cube, target_zone,
                        cameras, table_height,
                        cube_xy=cube_xy, target_xy=target_xy)
        r["episode"] = ep

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
            _save_episode_videos(r, tag, video_dir)

        # Strip frame data before storing in records (keep CSV clean)
        r.pop("vid_context"); r.pop("vid_wrist"); r.pop("vid_top")
        records.append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_success = sum(r["success"] for r in records)
    tsr       = n_success / args.n
    mean_pss  = np.mean([r["pss"] for r in records])

    print("\n" + "─" * 55)
    print(f"Zero-shot TSR : {n_success}/{args.n} = {tsr:.1%}")
    print(f"Mean PSS      : {mean_pss:.3f}")
    print("─" * 55)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[ok] results saved → {args.out}")

    # ── Paper-ready line ──────────────────────────────────────────────────────
    print(f"\nPaper entry:  Zero-shot nominal TSR = {tsr:.1%}  (N={args.n})")


if __name__ == "__main__":
    main()