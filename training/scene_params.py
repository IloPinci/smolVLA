"""
scene_params.py
----------------
Shared constants and helpers for scene construction, color/instruction
lookups, perturbation-condition logic, and the CameraPerturbation config
object used by two_camera_setup.py, oracle_direct.py, and zeroshot_eval.py.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Image resolution — single source of truth (training cameras)
# ══════════════════════════════════════════════════════════════════════════
# Matches the existing 200-episode dataset (cameras already render at 256×256,
# verify_dataset.IMG_SHAPE expects 256×256, and the zero-shot SmolVLA agent
# was evaluated on 256×256 inputs). Keeping this at 256 avoids re-collection
# and avoids a silent resize mismatch in the eval pipeline.
IMG_RES = (256, 256)   # (H, W)  — used for TRAINING cameras only

# Resolution for the high-res witness cameras (never written to the dataset).
WITNESS_RES = (1280, 720)   # (H, W) — for human review only


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
OOD_POSITIONS = {
    "near_left":      {"cube_xy": (0.20,  0.12), "target_xy": (0.15, -0.15)},
    "far_left":       {"cube_xy": (0.32,  0.14), "target_xy": (0.15, -0.15)},
    "near_right":     {"cube_xy": (0.20, -0.13), "target_xy": (0.15, -0.15)},
    "workspace_edge": {"cube_xy": (0.34,  0.00), "target_xy": (0.15, -0.15)},
    "target_shift":   {"cube_xy": (0.25,  0.00), "target_xy": (0.05, -0.22)},
}


# ══════════════════════════════════════════════════════════════════════════
#  CameraPerturbation — single config object for all camera perturbations
# ══════════════════════════════════════════════════════════════════════════

# Which training camera is perturbed.  "context" = camera1, "wrist" = camera2,
# "top" = camera3.  None means no per-camera position/tilt change.
CameraTarget = Literal["context", "wrist", "top"]

@dataclass
class CameraPerturbation:
    """
    Describes all camera perturbations for one experimental condition.

    Blackout
    --------
    blackout_mode : str
        One of BLACKOUT_MODES.  Applied frame-by-frame in post-render to
        the *training* camera images written into the dataset.
        "none" / "3-cam"  → all cameras nominal (no blackout).
        "3rd_black"        → context (camera1) zeroed out.
        "wrist_only"       → only wrist (camera2) passes through.
        "context_only"     → only context (camera1) passes through.
        "all_black"        → every training camera zeroed.
        NOTE: blackout is applied ONLY to the tensors passed to the policy /
              stored in the dataset.  Witness cameras always record the true view.

    Position shift
    --------------
    position_target : str or None
        Which training camera to move ("context", "wrist", "top", or None).
    position_offset : (dx, dy, dz) in metres
        World-frame translation added to the camera's nominal position.
        Ignored when position_target is None.

    Tilt
    ----
    tilt_target : str or None
        Which training camera to tilt ("context", "wrist", "top", or None).
    tilt_deg : float
        Rotation of the look direction around world-Z, in degrees.
        Ignored when tilt_target is None.

    Examples
    --------
    Nominal (no perturbation):
        CameraPerturbation()

    Black out the wrist camera:
        CameraPerturbation(blackout_mode="wrist_only")
        # → context + top still pass through; wrist frames are all-black

    Shift the context camera 10 cm to the right and 5 cm up:
        CameraPerturbation(
            position_target="context",
            position_offset=(0.0, 0.10, 0.05),
        )

    Tilt the top camera 15 degrees:
        CameraPerturbation(
            tilt_target="top",
            tilt_deg=15.0,
        )

    Combined — shifted context + blacked-out wrist:
        CameraPerturbation(
            blackout_mode="wrist_only",
            position_target="context",
            position_offset=(0.05, 0.0, 0.0),
        )
    """

    # Blackout ----------------------------------------------------------------
    blackout_mode: str = "none"   # see BLACKOUT_MODES

    # Position shift ----------------------------------------------------------
    position_target: Optional[str] = None          # "context" | "wrist" | "top" | None
    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Tilt --------------------------------------------------------------------
    tilt_target: Optional[str] = None              # "context" | "wrist" | "top" | None
    tilt_deg: float = 0.0

    def __post_init__(self):
        if self.blackout_mode not in BLACKOUT_MODES:
            raise ValueError(
                f"blackout_mode must be one of {BLACKOUT_MODES}, "
                f"got {self.blackout_mode!r}"
            )
        valid_targets = {None, "context", "wrist", "top"}
        if self.position_target not in valid_targets:
            raise ValueError(f"position_target must be one of {valid_targets}")
        if self.tilt_target not in valid_targets:
            raise ValueError(f"tilt_target must be one of {valid_targets}")

    @property
    def label(self) -> str:
        """Short human-readable tag for file-naming / logging."""
        parts = []
        if self.blackout_mode not in ("none", "3-cam"):
            parts.append(f"blackout={self.blackout_mode}")
        if self.position_target is not None:
            dx, dy, dz = self.position_offset
            parts.append(f"pos_{self.position_target}=({dx:.2f},{dy:.2f},{dz:.2f})")
        if self.tilt_target is not None:
            parts.append(f"tilt_{self.tilt_target}={self.tilt_deg:.1f}deg")
        return "_".join(parts) if parts else "nominal"

    def apply_blackout(self, frame: np.ndarray, cam_key: str) -> np.ndarray:
        """
        Apply blackout to a uint8 [H, W, 3] training frame.
        cam_key is the LeRobot key suffix, e.g. "camera1".
        """
        return apply_camera_blackout(frame, cam_key, self.blackout_mode)

    def position_offset_for(self, role: str) -> Tuple[float, float, float]:
        """
        Return the world-frame (dx, dy, dz) offset for `role` ("context"/"wrist"/"top").
        Returns (0, 0, 0) if this role is not the position_target.
        """
        if self.position_target == role:
            return self.position_offset
        return (0.0, 0.0, 0.0)

    def tilt_deg_for(self, role: str) -> float:
        """
        Return the tilt in degrees for `role`. Returns 0.0 if not targeted.
        """
        if self.tilt_target == role:
            return self.tilt_deg
        return 0.0


# Convenience: the nominal (no perturbation) instance
NOMINAL_PERTURBATION = CameraPerturbation()