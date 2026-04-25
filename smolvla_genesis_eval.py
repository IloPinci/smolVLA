"""
smolVLA + Genesis Simulation Loop
===================================
Tested working inference stack:
  - SmolVLAPolicy (lerobot/smolvla_base)
  - AutoTokenizer (HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
  - Genesis 0.4.6 physics simulator (Franka Panda — MJCF format)

Run:
    conda activate lerobot
    python3 smolvla_genesis_eval.py
"""

import numpy as np
import torch
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

VLM_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
POLICY_REPO  = "lerobot/smolvla_base"
DEVICE       = "cuda"

# Absolute path confirmed from your Genesis assets
PANDA_MJCF = (
    "/home/snape/miniforge3/envs/lerobot/lib/python3.12"
    "/site-packages/genesis/assets/xml/franka_sim/franka_panda.xml"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD SMOLVLA
# ─────────────────────────────────────────────────────────────────────────────
def load_smolvla():
    print("🚀 Loading smolVLA...")
    policy = SmolVLAPolicy.from_pretrained(POLICY_REPO, device=DEVICE)
    policy.eval()
    print("📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(VLM_BACKBONE)
    state_dim = policy.config.input_features["observation.state"].shape[0]
    print(f"   ✅ Ready  |  state_dim={state_dim}")
    return policy, tokenizer, state_dim


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD OBSERVATION DICT
# ─────────────────────────────────────────────────────────────────────────────
def build_obs(pixels, state, lang_tokens, lang_mask):
    return {
        "observation.images.camera1":          pixels,
        "observation.state":                   state,
        "observation.language.tokens":         lang_tokens,
        "observation.language.attention_mask": lang_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GENESIS SCENE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def build_scene():
    import genesis as gs

    gs.init(backend=gs.cuda)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,                  # 50 Hz physics
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(1.5, 0.5, 1.2),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=True,
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Franka Panda — MJCF format (Genesis ships .xml, not .urdf)
    robot = scene.add_entity(
        gs.morphs.MJCF(                         # ← MJCF, not URDF
            file=PANDA_MJCF,
            pos=(0.0, 0.0, 0.0),
        )
    )

    # Red target block
    target = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.45, 0.0, 0.02),
        ),
        surface=gs.surfaces.Rough(color=(0.9, 0.2, 0.2)),
    )

    # Off-screen camera — 256×256 to match smolVLA input
    cam = scene.add_camera(
        res=(256, 256),
        pos=(0.5, 0.0, 1.6),
        lookat=(0.35, 0.0, 0.2),
        fov=55,
        GUI=False,
    )

    scene.build()
    return scene, robot, target, cam


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cam_to_tensor(cam) -> torch.Tensor:
    """Render camera → (1, 3, 256, 256) float32 [0,1] on DEVICE."""
    rgb = cam.render(rgb=True)[0]                       # (H, W, 3) uint8
    t   = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(DEVICE)


def robot_state_tensor(robot, state_dim: int) -> torch.Tensor:
    """Joint positions → (1, state_dim) float32 on DEVICE."""
    qpos  = robot.get_dofs_position().cpu().numpy()     # (n_dof,)
    state = np.zeros(state_dim, dtype=np.float32)
    n     = min(len(qpos), state_dim)
    state[:n] = qpos[:n]
    return torch.from_numpy(state).unsqueeze(0).to(DEVICE)


def apply_action(robot, action_np: np.ndarray):
    """Send one action step to the robot joints."""
    n_dof   = robot.n_dofs
    targets = np.zeros(n_dof, dtype=np.float32)
    n       = min(len(action_np), n_dof)
    targets[:n] = action_np[:n]
    robot.control_dofs_position(targets)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(n_steps: int = 300,
        instruction_text: str = "pick up the red block and place it on the green mat"):

    # ── Load model ───────────────────────────────────────────────────────────
    policy, tokenizer, state_dim = load_smolvla()

    # Tokenize once — instruction is fixed for the whole episode
    enc = tokenizer(
        [instruction_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    lang_tokens = enc["input_ids"].to(DEVICE)
    lang_mask   = enc["attention_mask"].bool().to(DEVICE)

    # ── Build Genesis scene ──────────────────────────────────────────────────
    scene, robot, target, cam = build_scene()
    print(f"\n🌍 Scene ready  |  robot DOFs: {robot.n_dofs}")
    print(f"   Task : \"{instruction_text}\"")
    print(f"   Steps: {n_steps} × 20 ms = {n_steps * 0.02:.1f} s simulated\n")

    # ── Control loop ─────────────────────────────────────────────────────────
    action_chunk = None
    chunk_cursor = 0

    for step in range(n_steps):

        # Re-query the policy whenever the previous chunk is exhausted
        if action_chunk is None or chunk_cursor >= action_chunk.shape[0]:
            pixels = cam_to_tensor(cam)
            state  = robot_state_tensor(robot, state_dim)
            obs    = build_obs(pixels, state, lang_tokens, lang_mask)

            with torch.no_grad():
                action_t = policy.select_action(obs)        # (1, 6)

            action_chunk = action_t.squeeze(0).cpu().numpy()
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[np.newaxis, :]  # (1, 6)
            chunk_cursor = 0

        apply_action(robot, action_chunk[chunk_cursor])
        chunk_cursor += 1
        scene.step()

        if step % 50 == 0:
            a = action_chunk[chunk_cursor - 1]
            print(f"   step {step:4d}  |  action = [{', '.join(f'{v:+.3f}' for v in a)}]")

    print("\n✅ Simulation finished.")
    import genesis as gs
    gs.destroy()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(
        n_steps=300,
        instruction_text="pick up the red block and place it on the green mat",
    )