"""
scene_params.py
----------------
Shared constants and helpers for scene construction, color/instruction
lookups, perturbation-condition logic, and the CameraPerturbation config
object used by two_camera_setup.py, oracle_direct.py, and zeroshot_eval.py.

NEW in this version
-------------------
• CUBE_POSITIONS  — named in-distribution and OOD spawn positions for the
                    cube (replaces the earlier ad-hoc OOD_POSITIONS dict).
• SphereConfig    — dataclass describing one distractor sphere (position,
                    radius, colour, whether it can roll/fall).
• SPHERE_PRESETS  — ready-made sphere layouts for Phase-C experiments.
• ScenePerturbation — top-level config object that bundles:
                      - cube_color  (str from CUBE_COLORS)
                      - cube_position_preset  (str from CUBE_POSITIONS or None
                                               to keep oracle randomisation)
                      - spheres  (list[SphereConfig])
                      Camera perturbations live in CameraPerturbation as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, List
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Image resolution — single source of truth (training cameras)
# ══════════════════════════════════════════════════════════════════════════
IMG_RES     = (256, 256)    # (H, W)  — TRAINING cameras
WITNESS_RES = (1280, 720)   # (H, W)  — witness cameras (never in dataset)


# ══════════════════════════════════════════════════════════════════════════
#  Cube color → (RGB, instruction word) lookup
# ══════════════════════════════════════════════════════════════════════════
CUBE_COLORS = {
    "red":    {"rgb": (1.0, 0.0, 0.0), "word": "red"},
    "blue":   {"rgb": (0.0, 0.0, 1.0), "word": "blue"},
    "green":  {"rgb": (0.0, 1.0, 0.0), "word": "green"},
    "yellow": {"rgb": (1.0, 1.0, 0.0), "word": "yellow"},
}

DEFAULT_CUBE_COLOR = "red"


def language_instruction_for(cube_color: str = DEFAULT_CUBE_COLOR) -> str:
    """Return the SmolVLA language instruction for a given cube color."""
    entry = CUBE_COLORS.get(cube_color, CUBE_COLORS[DEFAULT_CUBE_COLOR])
    return f"Pick up the {entry['word']} block and place it on the green target."


# ══════════════════════════════════════════════════════════════════════════
#  Cube spawn positions
#
#  Each entry gives a *fixed* (cube_xy, target_xy) pair, bypassing the
#  oracle's normal rejection-sampling randomisation.  Use cube_position_preset
#  inside ScenePerturbation to select one.
#
#  Layout:
#      "nominal"        — centre of the training distribution (no OOD)
#      "near_left/right"  — moderate lateral shift
#      "far_left/right"   — large lateral shift (near IK boundary)
#      "workspace_edge"   — cube near the far reach of the arm
#      "target_shift"     — target relocated, cube at nominal centre
#      "close_to_target"  — cube starts very close to the target zone
# ══════════════════════════════════════════════════════════════════════════
CUBE_POSITIONS: dict[str, dict] = {
    # in-distribution anchor
    "nominal":        {"cube_xy": (0.25,  0.00), "target_xy": (0.15, -0.15)},
    # lateral shifts  (OOD)
    "near_left":      {"cube_xy": (0.20,  0.12), "target_xy": (0.15, -0.15)},
    "far_left":       {"cube_xy": (0.30,  0.14), "target_xy": (0.15, -0.15)},
    "near_right":     {"cube_xy": (0.20, -0.13), "target_xy": (0.15, -0.15)},
    "far_right":      {"cube_xy": (0.30, -0.14), "target_xy": (0.15, -0.15)},
    # reach extremes
    "workspace_edge": {"cube_xy": (0.32,  0.00), "target_xy": (0.15, -0.15)},
    # target relocated
    "target_shift":   {"cube_xy": (0.25,  0.00), "target_xy": (0.05, -0.22)},
    # cube already close to the target (tests placing without long carry)
    "close_to_target":{"cube_xy": (0.17, -0.12), "target_xy": (0.15, -0.15)},
}

# Backwards-compat alias (old name used in some analysis scripts)
OOD_POSITIONS = CUBE_POSITIONS


# ══════════════════════════════════════════════════════════════════════════
#  Distractor sphere configuration
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SphereConfig:
    """
    Describes one distractor sphere placed on the table.

    Parameters
    ----------
    xy : (float, float)
        World-frame (x, y) centre position. The sphere is placed just
        above the table surface automatically by build_environment().
    radius : float
        Sphere radius in metres.  0.02-0.04 m is visually salient but
        small enough not to obstruct most grasps.
    color : (R, G, B) floats in [0, 1]
        Surface colour passed to gs.surfaces.Rough.
    mass_kg : float
        Mass used for the Rigid material. Set to 0.0 for a fixed sphere
        (does not roll when touched).  Non-zero values make it dynamic.
    friction : float
        Surface friction coefficient.
    label : str
        Human-readable label for logging/file-naming.
    """
    xy:       Tuple[float, float] = (0.10, 0.10)
    radius:   float               = 0.025
    color:    Tuple[float, float, float] = (0.8, 0.5, 0.0)   # orange
    mass_kg:  float               = 0.05     # >0 → dynamic; 0 → fixed
    friction: float               = 1.0
    label:    str                 = "sphere"


# ── Ready-made sphere layouts ──────────────────────────────────────────────
# Each entry is a list of SphereConfig objects to pass to build_environment().
SPHERE_PRESETS: dict[str, List[SphereConfig]] = {
    # No distractors — baseline / nominal
    "none": [],

    # One orange sphere left of the cube path
    "single_left": [
        SphereConfig(xy=(0.18,  0.10), radius=0.025, color=(0.9, 0.45, 0.0),
                     label="distractor_left"),
    ],

    # One orange sphere right of the cube path
    "single_right": [
        SphereConfig(xy=(0.18, -0.10), radius=0.025, color=(0.9, 0.45, 0.0),
                     label="distractor_right"),
    ],

    # Two spheres flanking the typical cube→target corridor
    "two_flanking": [
        SphereConfig(xy=(0.20,  0.09), radius=0.025, color=(0.9, 0.45, 0.0),
                     label="flank_left"),
        SphereConfig(xy=(0.20, -0.09), radius=0.025, color=(0.6, 0.0,  0.8),
                     label="flank_right"),
    ],

    # Three spheres scattered across the workspace — high-clutter condition
    "high_clutter": [
        SphereConfig(xy=(0.18,  0.11), radius=0.025, color=(0.9, 0.45, 0.0),
                     label="clutter_a"),
        SphereConfig(xy=(0.22, -0.10), radius=0.022, color=(0.1, 0.5,  0.9),
                     label="clutter_b"),
        SphereConfig(xy=(0.12, -0.05), radius=0.028, color=(0.2, 0.8,  0.2),
                     label="clutter_c"),
    ],

    # One large sphere close to the target zone (tests final-placement OOD)
    "near_target": [
        SphereConfig(xy=(0.12, -0.18), radius=0.030, color=(0.8, 0.1,  0.1),
                     label="near_target"),
    ],

    # Same colour as the cube — maximum visual confusion
    "color_match": [
        SphereConfig(xy=(0.20,  0.08), radius=0.025, color=(1.0, 0.0, 0.0),
                     label="red_match"),
    ],
}


# ══════════════════════════════════════════════════════════════════════════
#  ScenePerturbation — top-level experiment config
#
#  This bundles the three new perturbation axes so callers only need to
#  pass one object instead of multiple loose kwargs.
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenePerturbation:
    """
    Top-level perturbation config for one experimental condition.

    cube_color : str
        Key into CUBE_COLORS.  Changing this simultaneously changes the
        rendered block colour AND the language instruction passed to the
        policy.  Defaults to "red" (matches the training distribution).

    cube_position_preset : str or None
        Key into CUBE_POSITIONS that fixes the cube and target XY
        coordinates for every episode.  When None (default) the oracle's
        normal rejection-sampling randomisation is used instead.

    spheres : list[SphereConfig]
        Distractor spheres to add to the scene.  Pass [] for no
        distractors.  Use SPHERE_PRESETS[name] for predefined layouts.

    Examples
    --------
    # Nominal (no perturbation)
    ScenePerturbation()

    # Change cube to blue
    ScenePerturbation(cube_color="blue")

    # Fix cube at workspace_edge OOD position
    ScenePerturbation(cube_position_preset="workspace_edge")

    # Add two flanking spheres
    ScenePerturbation(spheres=SPHERE_PRESETS["two_flanking"])

    # Combined: yellow cube, far_left position, high clutter
    ScenePerturbation(
        cube_color="yellow",
        cube_position_preset="far_left",
        spheres=SPHERE_PRESETS["high_clutter"],
    )
    """
    cube_color:             str                    = DEFAULT_CUBE_COLOR
    cube_position_preset:   Optional[str]          = None
    spheres:                List[SphereConfig]     = field(default_factory=list)

    def __post_init__(self):
        if self.cube_color not in CUBE_COLORS:
            raise ValueError(
                f"cube_color must be one of {list(CUBE_COLORS)}, "
                f"got {self.cube_color!r}"
            )
        if (self.cube_position_preset is not None
                and self.cube_position_preset not in CUBE_POSITIONS):
            raise ValueError(
                f"cube_position_preset must be one of "
                f"{list(CUBE_POSITIONS)} or None, "
                f"got {self.cube_position_preset!r}"
            )

    @property
    def label(self) -> str:
        """Short human-readable tag for file-naming / logging."""
        parts = []
        if self.cube_color != DEFAULT_CUBE_COLOR:
            parts.append(f"color={self.cube_color}")
        if self.cube_position_preset is not None:
            parts.append(f"pos={self.cube_position_preset}")
        sphere_labels = [s.label for s in self.spheres]
        if sphere_labels:
            parts.append("spheres=" + "+".join(sphere_labels))
        return "_".join(parts) if parts else "nominal"

    @property
    def language_instruction(self) -> str:
        return language_instruction_for(self.cube_color)

    def resolve_spawn(
        self,
        rng: np.random.Generator,
        min_sep: float = 0.12,
        max_tries: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (cube_xy, target_xy) as (2,) float arrays.

        If cube_position_preset is set, returns the fixed values from
        CUBE_POSITIONS (no randomisation, no rejection sampling).
        Otherwise falls back to the oracle's normal random distribution.
        """
        if self.cube_position_preset is not None:
            preset = CUBE_POSITIONS[self.cube_position_preset]
            return (
                np.array(preset["cube_xy"],   dtype=float),
                np.array(preset["target_xy"], dtype=float),
            )

        # Normal randomised spawn (same distribution as the oracle)
        for _ in range(max_tries):
            cube_xy   = np.array([0.24, 0.00]) + rng.uniform(-0.04,  0.04, 2)
            target_xy = np.array([0.15, -0.15]) + rng.uniform(-0.05,  0.05, 2)
            if np.linalg.norm(cube_xy - target_xy) >= min_sep:
                return cube_xy, target_xy

        # Fallback if rejection sampling exhausted (should be extremely rare)
        return np.array([0.24, 0.00]), np.array([0.15, -0.15])


# Convenience: the nominal (no scene perturbation) instance
NOMINAL_SCENE = ScenePerturbation()


# ══════════════════════════════════════════════════════════════════════════
#  Home-pose perturbation (robot_initial_state axis)
# ══════════════════════════════════════════════════════════════════════════
def perturb_home_pose(
    home_pose: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
    clip: float = 0.15,
) -> np.ndarray:
    """Add clipped Gaussian noise to home_pose. sigma <= 0 → unchanged."""
    if sigma <= 0.0:
        return home_pose.copy()
    rng = rng or np.random.default_rng()
    noise = np.clip(rng.normal(0.0, sigma, size=home_pose.shape), -clip, clip)
    return home_pose + noise


# ══════════════════════════════════════════════════════════════════════════
#  Camera blackout helpers
# ══════════════════════════════════════════════════════════════════════════
BLACKOUT_MODES = {
    "none", "3-cam", "3rd_black", "all_black",
    "wrist_only", "context_only", "top_only",   
    "no_top", "no_wrist", "no_context",
    "absent_wrist", "absent_context", "absent_top",
    "absent_all",         
}

_CAM_ROLE = {"camera1": "context", "camera2": "wrist", "camera3": "top"}

def is_camera_absent(cam_key: str, mode: str) -> bool:
    role = _CAM_ROLE.get(cam_key)
    if mode == "absent_all":
        return True
    return mode == f"absent_{role}"


def apply_camera_blackout(
    frame: np.ndarray, cam_key: str, mode: str = "none"
) -> np.ndarray:
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
    if mode == "top_only" and role != "top":
        return np.zeros_like(frame)

    # two-camera combos: name says which ONE camera is blacked out
    if mode == "no_top" and role == "top":
        return np.zeros_like(frame)
    if mode == "no_wrist" and role == "wrist":
        return np.zeros_like(frame)
    if mode == "no_context" and role == "context":
        return np.zeros_like(frame)

    return frame


# ══════════════════════════════════════════════════════════════════════════
#  CameraPerturbation — unchanged from v1, kept for full backwards compat
# ══════════════════════════════════════════════════════════════════════════
CameraTarget = Literal["context", "wrist", "top"]


@dataclass
class CameraPerturbation:
    """
    Camera-only perturbations (blackout, position shift, tilt).
    Scene-level perturbations (cube color, positions, spheres) live in
    ScenePerturbation above.
    """
    blackout_mode:    str                         = "none"
    position_target:  Optional[str]               = None
    position_offset:  Tuple[float, float, float]  = (0.0, 0.0, 0.0)
    tilt_target:      Optional[str]               = None
    tilt_deg:         float                       = 0.0

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
        return apply_camera_blackout(frame, cam_key, self.blackout_mode)

    def camera_is_absent(self, cam_key: str) -> bool:
        return is_camera_absent(cam_key, self.blackout_mode)

    def position_offset_for(self, role: str) -> Tuple[float, float, float]:
        if self.position_target == role:
            return self.position_offset
        return (0.0, 0.0, 0.0)

    def tilt_deg_for(self, role: str) -> float:
        if self.tilt_target == role:
            return self.tilt_deg
        return 0.0


# Convenience instance
NOMINAL_PERTURBATION = CameraPerturbation()