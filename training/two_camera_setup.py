"""
two_camera_setup.py
-------------------
Scene construction, camera attachment, and success-check helpers.

Changes from previous version
------------------------------
• build_environment() now accepts:
    - cube_color      : str  — key into CUBE_COLORS (default "red")
    - spheres         : list[SphereConfig]  — distractor spheres to add
  and returns (cube, target_zone, sphere_entities) so callers can
  reposition spheres between episodes if needed.

• reset_episode() has been extended with an optional spheres argument
  that repositions distractor spheres to their configured XY locations.
  (Sphere Z is recalculated from table_height + radius automatically.)

Camera architecture
-------------------
TRAINING cameras  (returned by attach_cameras)
    Rendered at IMG_RES (256×256). Subject to CameraPerturbation axes.

WITNESS cameras   (returned by attach_witness_cameras_with_robot)
    Rendered at WITNESS_RES (1280×720). Never in the dataset.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import genesis as gs
import torch

from scene_params import (
    CUBE_COLORS, DEFAULT_CUBE_COLOR, IMG_RES, WITNESS_RES,
    CameraPerturbation, NOMINAL_PERTURBATION,
    SphereConfig,
)

_ROLE_TO_CAM_KEY = {"context": "camera1", "wrist": "camera2", "top": "camera3"}


# ══════════════════════════════════════════════════════════════════════════════
#  Environment
# ══════════════════════════════════════════════════════════════════════════════

def build_environment(
    scene,
    table_height: float,
    cube_color: str = DEFAULT_CUBE_COLOR,
    spheres: List[SphereConfig] | None = None,
):
    """
    Build the table, cube, target zone, and optional distractor spheres.

    Parameters
    ----------
    scene : gs.Scene
        The Genesis scene to add entities to.
    table_height : float
        Height of the table surface in world coordinates.
    cube_color : str
        Key into CUBE_COLORS.  Changes both the rendered colour and (via
        scene_params.language_instruction_for) the language instruction.
    spheres : list[SphereConfig] or None
        Distractor spheres to add.  Each sphere is placed at
        (xy[0], xy[1], table_height + radius) so it sits on the table.
        Pass [] or None for no distractors.

    Returns
    -------
    cube : gs.Entity
    target_zone : gs.Entity
    sphere_entities : list[gs.Entity]
        Empty list when no spheres are requested.
    """
    if spheres is None:
        spheres = []

    # Validate cube colour
    if cube_color not in CUBE_COLORS:
        raise ValueError(
            f"cube_color must be one of {list(CUBE_COLORS)}, got {cube_color!r}"
        )
    cube_rgb = CUBE_COLORS[cube_color]["rgb"]

    table_width       = 1.0
    table_depth       = 1.0
    surface_thickness = 0.05
    leg_thickness     = 0.05

    # ── Tabletop ──────────────────────────────────────────────────────────────
    scene.add_entity(
        gs.morphs.Box(
            size=(table_width, table_depth, surface_thickness),
            pos=(0.0, 0.0, table_height - surface_thickness / 2.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(friction=0.01, coup_friction=0.1, coup_softness=0.001),
    )

    # ── Table legs ────────────────────────────────────────────────────────────
    for x in [-table_width / 2 + leg_thickness / 2, table_width / 2 - leg_thickness / 2]:
        for y in [-table_depth / 2 + leg_thickness / 2, table_depth / 2 - leg_thickness / 2]:
            scene.add_entity(
                gs.morphs.Box(
                    size=(leg_thickness, leg_thickness, table_height - surface_thickness),
                    pos=(x, y, (table_height - surface_thickness) / 2.0),
                    fixed=True,
                )
            )

    # ── Cube (colour-parameterised) ───────────────────────────────────────────
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.03, 0.03, 0.03),
            pos=(0.25, 0.0, table_height + 0.015),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
            friction=3.0,
            coup_friction=1.5,
            coup_softness=0.0001,
        ),
        surface=gs.surfaces.Rough(color=cube_rgb),
    )

    # ── Target zone (visual only) ─────────────────────────────────────────────
    target_zone = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.05,
            height=0.001,
            pos=(0.15, -0.15, table_height + 0.001),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(color=(0.0, 1.0, 0.0)),
    )

    # ── Distractor spheres ────────────────────────────────────────────────────
    sphere_entities: list = []
    for cfg in spheres:
        z = table_height + cfg.radius   # rest on the table surface
        is_fixed = (cfg.mass_kg <= 0.0)
        entity = scene.add_entity(
            gs.morphs.Sphere(
                radius=cfg.radius,
                pos=(cfg.xy[0], cfg.xy[1], z),
                fixed=is_fixed,
            ),
            material=gs.materials.Rigid(
                rho=cfg.mass_kg / ((4.0 / 3.0) * np.pi * cfg.radius ** 3)
                    if not is_fixed else 1000.0,   # rho back-calculated from mass
                friction=cfg.friction,
            ),
            surface=gs.surfaces.Rough(color=cfg.color),
        )
        sphere_entities.append(entity)
        print(f"[scene] Added sphere '{cfg.label}' "
              f"at ({cfg.xy[0]:.3f}, {cfg.xy[1]:.3f}, {z:.3f})  "
              f"r={cfg.radius}  fixed={is_fixed}")

    return cube, target_zone, sphere_entities


# ══════════════════════════════════════════════════════════════════════════════
#  Nominal camera positions (single source of truth)
# ══════════════════════════════════════════════════════════════════════════════

def _context_nominal(table_height: float):
    pos    = np.array([0.4, -0.4, table_height + 0.55])
    lookat = np.array([0.2,  0.0, table_height + 0.05])
    return pos, lookat


def _top_nominal(table_height: float):
    pos    = np.array([0.0, 0.0, table_height + 1.0])
    lookat = np.array([0.0, 0.0, table_height + 0.1])
    return pos, lookat


# ══════════════════════════════════════════════════════════════════════════════
#  Training cameras
# ══════════════════════════════════════════════════════════════════════════════

def attach_cameras(
    scene,
    so101,
    table_height: float,
    perturbation: CameraPerturbation = NOMINAL_PERTURBATION,
    context_offset: tuple = (0.0, 0.0, 0.0),
    context_tilt_deg: float = 0.0,
) -> dict:
    """
    Add the three TRAINING cameras to the scene and return them as a dict.
    Keys: {"context": <cam>, "wrist": <cam>, "top": <cam>}
    """
    # ── Context camera ────────────────────────────────────────────────────────
    ctx_pos_nom, ctx_look_nom = _context_nominal(table_height)

    pert_ctx_offset = perturbation.position_offset_for("context")
    ctx_pos = (ctx_pos_nom + np.array(pert_ctx_offset, dtype=float)
               if any(v != 0.0 for v in pert_ctx_offset)
               else ctx_pos_nom + np.array(context_offset, dtype=float))

    pert_ctx_tilt = perturbation.tilt_deg_for("context")
    ctx_tilt_deg  = pert_ctx_tilt if pert_ctx_tilt != 0.0 else context_tilt_deg
    ctx_lookat    = _apply_tilt(ctx_pos, ctx_pos_nom, ctx_look_nom, ctx_tilt_deg)

    context_cam = scene.add_camera(
        res=IMG_RES, pos=tuple(ctx_pos), lookat=tuple(ctx_lookat), fov=60, GUI=False,
    )

    # ── Top camera ────────────────────────────────────────────────────────────
    top_pos_nom, top_look_nom = _top_nominal(table_height)
    top_offset = np.array(perturbation.position_offset_for("top"), dtype=float)
    top_pos    = top_pos_nom + top_offset
    top_lookat = _apply_tilt(top_pos, top_pos_nom, top_look_nom,
                              perturbation.tilt_deg_for("top"))

    top_cam = scene.add_camera(
        res=IMG_RES, pos=tuple(top_pos), lookat=tuple(top_lookat), fov=65, GUI=False,
    )

    # ── Wrist camera (attached to gripper link) ────────────────────────────────
    wrist_cam = scene.add_camera(
        res=IMG_RES, pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50, GUI=False,
    )
    gripper_link      = so101.get_link("gripper")
    base_translation  = np.array([0.01, -0.145, -0.062])
    wrist_offset      = np.array(perturbation.position_offset_for("wrist"), dtype=float)
    final_translation = base_translation + wrist_offset

    wrist_tilt_deg = perturbation.tilt_deg_for("wrist")
    theta = np.radians(90)
    c, s  = np.cos(theta), np.sin(theta)
    rotation = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0,   c,   -s, 0.0],
        [0.0,   s,    c, 0.0],
        [0.0, 0.0,  0.0, 1.0],
    ])
    if wrist_tilt_deg != 0.0:
        rotation = _rotate_4x4_z(rotation, wrist_tilt_deg)
    offset_T          = rotation.copy()
    offset_T[:3, 3]   = final_translation
    wrist_cam.attach(gripper_link, offset_T)

    if perturbation.label != "nominal":
        print(f"[camera perturbation] {perturbation.label}")

    return {"context": context_cam, "wrist": wrist_cam, "top": top_cam}


# ══════════════════════════════════════════════════════════════════════════════
#  Witness cameras
# ══════════════════════════════════════════════════════════════════════════════

def attach_witness_cameras(scene, table_height: float) -> dict:
    """Context + top witness cameras (no robot handle needed)."""
    ctx_pos, ctx_lookat = _context_nominal(table_height)
    top_pos, top_lookat = _top_nominal(table_height)

    w_context = scene.add_camera(
        res=WITNESS_RES, pos=tuple(ctx_pos), lookat=tuple(ctx_lookat), fov=60, GUI=False,
    )
    w_top = scene.add_camera(
        res=WITNESS_RES, pos=tuple(top_pos), lookat=tuple(top_lookat), fov=65, GUI=False,
    )
    print(f"[witness cameras] context + top at {WITNESS_RES[1]}×{WITNESS_RES[0]} "
          f"(wrist witness requires attach_witness_cameras_with_robot)")
    return {"witness_context": w_context, "witness_top": w_top}


def attach_witness_cameras_with_robot(scene, so101, table_height: float) -> dict:
    """Full three-camera witness rig including an attached wrist camera."""
    ctx_pos, ctx_lookat = _context_nominal(table_height)
    top_pos, top_lookat = _top_nominal(table_height)

    w_context = scene.add_camera(
        res=WITNESS_RES, pos=tuple(ctx_pos), lookat=tuple(ctx_lookat), fov=60, GUI=False,
    )
    w_top = scene.add_camera(
        res=WITNESS_RES, pos=tuple(top_pos), lookat=tuple(top_lookat), fov=65, GUI=False,
    )
    w_wrist = scene.add_camera(
        res=WITNESS_RES, pos=(0.0, 0.0, 0.0), lookat=(1.0, 0.0, 0.0), fov=50, GUI=False,
    )
    gripper_link = so101.get_link("gripper")
    theta = np.radians(90)
    c, s  = np.cos(theta), np.sin(theta)
    rotation = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0,   c,   -s, 0.0],
        [0.0,   s,    c, 0.0],
        [0.0, 0.0,  0.0, 1.0],
    ])
    offset_T       = rotation.copy()
    offset_T[:3, 3] = np.array([0.01, -0.145, -0.062])
    w_wrist.attach(gripper_link, offset_T)

    print(f"[witness cameras] context + wrist + top at {WITNESS_RES[1]}×{WITNESS_RES[0]}")
    return {"witness_context": w_context, "witness_wrist": w_wrist, "witness_top": w_top}


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _apply_tilt(cam_pos, nominal_cam_pos, nominal_lookat, tilt_deg):
    if tilt_deg == 0.0:
        return nominal_lookat.copy()
    look_vec = nominal_lookat - nominal_cam_pos
    theta    = np.radians(tilt_deg)
    c, s     = np.cos(theta), np.sin(theta)
    rot_z    = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return cam_pos + rot_z @ look_vec


def _rotate_4x4_z(mat: np.ndarray, deg: float) -> np.ndarray:
    theta = np.radians(deg)
    c, s  = np.cos(theta), np.sin(theta)
    rz = np.array([
        [ c, -s, 0.0, 0.0],
        [ s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return rz @ mat


# ══════════════════════════════════════════════════════════════════════════════
#  Success / sub-goal checks  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def is_success(cube, target_zone, table_height, radius=0.05, min_height=0.005):
    cube_pos    = cube.get_pos()
    target_pos  = target_zone.get_pos()
    xy_dist     = torch.norm(cube_pos[:2] - target_pos[:2])
    on_target   = xy_dist < radius
    above_table = cube_pos[2] > (table_height + min_height)
    not_airborne = cube_pos[2] < (table_height + 0.05)
    return bool(on_target and above_table and not_airborne)


def check_sub_goals(so101, cube, target_zone, table_height,
                    radius=0.05, min_height=0.005, _latch: dict | None = None):
    cube_pos     = cube.get_pos()
    gripper_pos  = so101.get_link("gripper").get_pos()
    gripper_dist = torch.norm(gripper_pos - cube_pos).item()
    near_block   = gripper_dist < 0.05
    is_elevated  = cube_pos[2].item() > (table_height + 0.02)

    if _latch is not None:
        if is_elevated and near_block:
            _latch["lifted"] = True
        lifted = _latch["lifted"]
    else:
        lifted = bool(is_elevated and near_block)

    placed  = is_success(cube, target_zone, table_height, radius, min_height)
    partial = 0.2 * near_block + 0.4 * lifted + 1.0 * placed
    return {"near_block": near_block, "lifted": lifted, "placed": placed, "Sum": partial}


# ══════════════════════════════════════════════════════════════════════════════
#  Episode reset  (extended to reposition spheres)
# ══════════════════════════════════════════════════════════════════════════════

def reset_episode(
    scene,
    so101,
    cube,
    table_height: float,
    home_dofs=None,
    sphere_entities: list | None = None,
    sphere_configs:  list | None = None,
):
    """
    Reset the robot and cube to their home positions.

    Parameters
    ----------
    sphere_entities : list[gs.Entity] or None
        Genesis entity handles returned by build_environment().
    sphere_configs  : list[SphereConfig] or None
        Matching SphereConfig objects (same order as sphere_entities).
        When both are provided, each sphere is repositioned to its
        configured XY location so it lands on the table surface.
    """
    if home_dofs is None:
        home_dofs = np.zeros(so101.n_dofs)

    so101.set_dofs_position(home_dofs)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home_dofs)
    cube.set_pos(torch.tensor([0.25, 0.0, table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    # Reposition distractor spheres
    if sphere_entities and sphere_configs:
        for entity, cfg in zip(sphere_entities, sphere_configs):
            z = table_height + cfg.radius
            entity.set_pos(np.array([cfg.xy[0], cfg.xy[1], z]))
            entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(10):
        scene.step()

    return home_dofs


# ══════════════════════════════════════════════════════════════════════════════
#  Main (demonstration collection example)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Collect 200 demonstrations.  Edit scene_pert below to try any
    combination of cube color, position preset, and sphere distractors.

    Examples
    --------
    # Nominal (matches training distribution)
    scene_pert = ScenePerturbation()

    # Blue cube + two flanking spheres
    scene_pert = ScenePerturbation(
        cube_color="blue",
        spheres=SPHERE_PRESETS["two_flanking"],
    )

    # Yellow cube fixed at workspace_edge + high clutter
    scene_pert = ScenePerturbation(
        cube_color="yellow",
        cube_position_preset="workspace_edge",
        spheres=SPHERE_PRESETS["high_clutter"],
    )
    """
    from oracle_direct import collect_demonstrations
    from scene_params import ScenePerturbation, SPHERE_PRESETS, NOMINAL_SCENE

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path   = os.path.join(script_dir, "../so101_arm/so101_new_calib.xml")

    # ── Choose your perturbation here ────────────────────────────────────────
    scene_pert = NOMINAL_SCENE   # change to try other conditions

    gs.init(backend=gs.gpu, seed=42)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=16),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=100, tolerance=1e-9,
            constraint_timeconst=0.006,
            enable_self_collision=True, box_box_detection=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False, world_frame_size=0.5, show_cameras=False,
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
        cube_color=scene_pert.cube_color,
        spheres=scene_pert.spheres,
    )

    so101 = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, table_height))
    )

    training_cams = attach_cameras(scene, so101, table_height)
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

    def success_fn(c, tz, th):
        return is_success(c, tz, th, radius=0.05, min_height=0.005)

    def sub_goals_fn(robot, c, tz, th, latch=None):
        return check_sub_goals(robot, c, tz, th, radius=0.05, min_height=0.005,
                                _latch=latch)

    collect_demonstrations(
        so101=so101, scene=scene,
        cameras=training_cams, witness_cameras=witness_cams,
        cube=cube, target_zone=target_zone,
        sphere_entities=sphere_entities, sphere_configs=scene_pert.spheres,
        table_height=table_height,
        is_success_fn=success_fn, check_sub_goals_fn=sub_goals_fn,
        n_episodes=200,
        output_dir=f"demos/{scene_pert.label}/",
        fps=10,
        repo_id=f"local/genesis_pickplace_{scene_pert.label}",
        scene_perturbation=scene_pert,
    )


if __name__ == "__main__":
    main()