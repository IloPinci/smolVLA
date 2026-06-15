"""
scene_params.py
----------------
Shared constants and helpers for scene construction, color/instruction
lookups, and perturbation-condition logic (used by oracle_direct.py,
two_camera_setup.py, verify_dataset.py, zeroshot_eval.py, and the future
perturbation_eval.py).
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Image resolution — single source of truth
# ══════════════════════════════════════════════════════════════════════════
# Matches the existing 200-episode dataset (cameras already render at 256x256,
# verify_dataset.IMG_SHAPE expects 256x256, and the zero-shot SmolVLA agent
# was evaluated on 256x256 inputs). Keeping this at 256 avoids re-collection
# and avoids a silent resize mismatch in the eval pipeline.
IMG_RES = (256, 256)   # (H, W)


# ══════════════════════════════════════════════════════════════════════════
#  Cube color -> (RGB, instruction word) lookup
# ══════════════════════════════════════════════════════════════════════════
CUBE_COLORS = {
    "red":    {"rgb": (1.0, 0.0, 0.0), "word": "red"},
    "blue":   {"rgb": (0.0, 0.0, 1.0), "word": "blue"},
    "green":  {"rgb": (0.0, 1.0, 0.0), "word": "green"},
    "yellow": {"rgb": (1.0, 1.0, 0.0), "word": "yellow"},
}

DEFAULT_CUBE_COLOR = "red"


def language_instruction_for(cube_color: str = DEFAULT_CUBE_COLOR) -> str:
    """
    Return the SmolVLA language instruction for a given cube color.

    For cube_color="red" (the default), this returns exactly the string
    already stored in meta/language_instruction for the existing
    200-episode dataset — nothing needs to change for nominal.
    """
    entry = CUBE_COLORS.get(cube_color, CUBE_COLORS[DEFAULT_CUBE_COLOR])
    return f"Pick up the {entry['word']} block and place it on the green target."


# ══════════════════════════════════════════════════════════════════════════
#  Home-pose perturbation (robot_initial_state axis) — used in Phase C
# ══════════════════════════════════════════════════════════════════════════
def perturb_home_pose(home_pose: np.ndarray, sigma: float,
                       rng: np.random.Generator | None = None,
                       clip: float = 0.15) -> np.ndarray:
    """
    Return a copy of home_pose with clipped Gaussian noise added per joint.
    sigma <= 0 returns home_pose unchanged (nominal condition).
    """
    if sigma <= 0.0:
        return home_pose.copy()
    rng = rng or np.random.default_rng()
    noise = np.clip(rng.normal(0.0, sigma, size=home_pose.shape), -clip, clip)
    return home_pose + noise


# ══════════════════════════════════════════════════════════════════════════
#  Camera blackout (camera-dropout axis) — used in Phase C
# ══════════════════════════════════════════════════════════════════════════
BLACKOUT_MODES = {"none", "3-cam", "3rd_black", "all_black", "wrist_only", "context_only"}

# camera1 = context, camera2 = wrist, camera3 = top (matches oracle_direct._CAM_REMAP)
_CAM_ROLE = {"camera1": "context", "camera2": "wrist", "camera3": "top"}


def apply_camera_blackout(frame: np.ndarray, cam_key: str, mode: str = "none") -> np.ndarray:
    """
    Given a rendered RGB uint8 [H, W, 3] frame and its camera key
    ("camera1"/"camera2"/"camera3"), return it unchanged or zeroed out
    depending on `mode`. "none"/"3-cam" = unchanged (nominal).
    """
    if mode not in BLACKOUT_MODES:
        raise ValueError(f"Unknown camera_blackout_mode: {mode!r}")
    if mode in ("none", "3-cam"):
        return frame

    role = _CAM_ROLE.get(cam_key)
    if mode == "all_black":
        return np.zeros_like(frame)
    if mode == "3rd_black" and role == "context":
        return np.zeros_like(frame)
    if mode == "wrist_only" and role != "wrist":
        return np.zeros_like(frame)
    if mode == "context_only" and role != "context":
        return np.zeros_like(frame)
    return frame


# ══════════════════════════════════════════════════════════════════════════
#  Out-of-distribution block/target positions (block_position axis)
# ══════════════════════════════════════════════════════════════════════════
# Training distribution: cube_xy = (0.25, 0.00) +/- (0.05, 0.05)
#                         target_xy = (0.15, -0.15) +/- (0.06, 0.06)
# Each entry below is chosen to sit outside those ranges.
# NOTE: verify IK reachability before using these in the Phase C sweep.
OOD_POSITIONS = {
    "near_left":      {"cube_xy": (0.20,  0.12), "target_xy": (0.15, -0.15)},
    "far_left":       {"cube_xy": (0.32,  0.14), "target_xy": (0.15, -0.15)},
    "near_right":     {"cube_xy": (0.20, -0.13), "target_xy": (0.15, -0.15)},
    "workspace_edge": {"cube_xy": (0.34,  0.00), "target_xy": (0.15, -0.15)},
    "target_shift":   {"cube_xy": (0.25,  0.00), "target_xy": (0.05, -0.22)},
}