import os
import numpy as np
import genesis as gs
import torch
from oracle_direct import SO101Oracle, OracleConfig, collect_demonstrations
from scene_params import CUBE_COLORS, DEFAULT_CUBE_COLOR, IMG_RES


# ══════════════════════════════════════════════════════════════════════════════
#  Environment
# ══════════════════════════════════════════════════════════════════════════════

def build_environment(scene, table_height, cube_color: str = DEFAULT_CUBE_COLOR):
    table_width       = 1.0
    table_depth       = 1.0
    surface_thickness = 0.05
    leg_thickness     = 0.05

    # Tabletop
    scene.add_entity(
        gs.morphs.Box(
            size=(table_width, table_depth, surface_thickness),
            pos=(0.0, 0.0, table_height - surface_thickness / 2.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(friction=0.01, coup_friction=0.1, coup_softness=0.001),
    )

    # Legs
    for x in [-table_width / 2 + leg_thickness / 2, table_width / 2 - leg_thickness / 2]:
        for y in [-table_depth / 2 + leg_thickness / 2, table_depth / 2 - leg_thickness / 2]:
            scene.add_entity(
                gs.morphs.Box(
                    size=(leg_thickness, leg_thickness, table_height - surface_thickness),
                    pos=(x, y, (table_height - surface_thickness) / 2.0),
                    fixed=True,
                )
            )

    # Cube — color is parameterised (defaults to red)
    cube_rgb = CUBE_COLORS.get(cube_color, CUBE_COLORS[DEFAULT_CUBE_COLOR])["rgb"]
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

    # Target zone — visual only, no collision
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

    return cube, target_zone


# ══════════════════════════════════════════════════════════════════════════════
#  Cameras
# ══════════════════════════════════════════════════════════════════════════════

def attach_cameras(scene, so101, table_height,
                    context_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    context_tilt_deg: float = 0.0):
    """
    Two cameras that match the SmolVLA training distribution:

    context_cam  — fixed third-person view from the front-right of the table.
                   Sees the full workspace: arm, cube, and target zone.

    wrist_cam    — attached to the gripper link, offset to the SIDE so it
                   looks horizontally at the jaw tips and the object being
                   grasped.  The offset_T below places it ~8 cm to the right
                   of the gripper centreline and ~4 cm behind the jaw tips,
                   rotated so it looks forward (toward the cube).

    """

    # ── Context camera (fixed, world frame) ──────────────────────────────────

    # Nominal pose, unchanged from the original hardcoded values.
    context_pos_nominal    = np.array([0.5, -0.4, table_height + 0.55])
    context_lookat_nominal = np.array([0.2, 0.0,  table_height + 0.05])

    # context_offset translates the camera position. Default (0,0,0) leaves
    # pos exactly at the nominal value.
    context_pos = context_pos_nominal + np.array(context_offset, dtype=float)

   # context_tilt_deg rotates the LOOK DIRECTION (about world Z) from the
    # (possibly offset) camera position. Default 0.0 leaves the look target
    # at the nominal point, so pos==nominal + tilt==0 reproduces the
    # original camera exactly.
    if context_tilt_deg:
        look_vec = context_lookat_nominal - context_pos_nominal
        theta = np.radians(context_tilt_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot_z = np.array([[c, -s, 0.0],
                          [s,  c, 0.0],
                          [0.0, 0.0, 1.0]])
        look_vec = rot_z @ look_vec
        context_lookat = context_pos + look_vec
    else:
        context_lookat = context_lookat_nominal

    context_cam = scene.add_camera(
        res=IMG_RES,
        pos=tuple(context_pos),
        lookat=tuple(context_lookat),
        fov=60,
        GUI=False,
    )
    
    #! Camera top — fixed, world frame, not rendered yet but ready to use
    top_cam = scene.add_camera(
        res=IMG_RES,
        pos=(0.0, 0.0, table_height + 1.0),
        lookat=(0.0, 0.0, table_height + 0.1),
        fov=65,
        GUI=False,
    )

    # ── Wrist camera (attached to gripper link) ───────────────────────────────
    # We create the camera at a dummy world position; attach() overrides it.
    wrist_cam = scene.add_camera(
        res=IMG_RES,
        pos=(0.0, 0.0, 0.0),   # placeholder — will be overridden by attach()
        lookat=(1.0, 0.0, 0.0),
        fov=50,
        GUI=False,              # shows in the Genesis viewer window
    )

    gripper_link = so101.get_link("gripper")

    # offset_T: 4×4 transform of the camera in the gripper link's local frame.
    #
    # Layout of the SO-101 gripper (looking from above):
    #   +X  →  forward (along jaw opening direction)
    #   +Y  →  left of the arm
    #   +Z  →  up
    #
    # We place the camera:
    #   - 8 cm to the RIGHT  (+Y = -0.08 in local frame, since right = -Y)
    #   - 3 cm BEHIND the jaw tips  (-X = -0.03)
    #   - 2 cm ABOVE the jaw midpoint (+Z = +0.02)
    # Then rotate it 90° about the local Z axis so its optical axis points
    # forward (+X) and the "up" vector is local +Z.
    #
    # Rotation: camera looks in local +X direction.
    # In Genesis attach(), the transform rotates the camera's default
    # "looking toward -Z" into the desired orientation.
    # A 90° rotation about local Y maps -Z → +X.
    theta = np.radians(90)
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([
        [ 1.0, 0.0,  0.0, 0.0],
        [ 0.0,   c,   -s, 0.0],
        [ 0.0,   s,    c, 0.0],
        [ 0.0, 0.0,  0.0, 1.0],
    ])
    offset_T = rotation.copy()
    offset_T[:3, 3] = np.array([0.01, -0.145, -0.062])   # as it is fixed on the non moving gripper

    wrist_cam.attach(gripper_link, offset_T)

    return {"context": context_cam, "wrist": wrist_cam, "top": top_cam}


# ══════════════════════════════════════════════════════════════════════════════
#  Success / sub-goal checks
# ══════════════════════════════════════════════════════════════════════════════

def is_success(cube, target_zone, table_height, radius=0.05, min_height=0.005):
    cube_pos   = cube.get_pos()
    target_pos = target_zone.get_pos()
    xy_dist    = torch.norm(cube_pos[:2] - target_pos[:2])
    on_target  = xy_dist < radius
    above_table = cube_pos[2] > (table_height + min_height)
    not_airborne = cube_pos[2] < (table_height + 0.05)
    return bool(on_target and above_table and not_airborne)


def check_sub_goals(so101, cube, target_zone, table_height, radius=0.05, min_height=0.005, _latch: dict | None = None):
    """
    Evaluate per-step sub-goal completions.
 
    Parameters
    ----------
    _latch : dict or None
        Mutable dict {"lifted": bool} shared across all calls within one
        episode.  Pass the same dict object every step so that `lifted`
        latches True once the block is elevated and never resets to False
        mid-episode (which would otherwise happen the moment the gripper
        moves away from the block toward the target zone).
 
        Pass None (or omit) to get unlatched per-step behaviour — useful
        for single-step diagnostic calls outside an episode loop.
    """
    cube_pos    = cube.get_pos()
    gripper_pos = so101.get_link("gripper").get_pos()
    gripper_dist = torch.norm(gripper_pos - cube_pos).item()
    near_block  = gripper_dist < 0.05
    is_elevated = cube_pos[2].item() > (table_height + 0.02)

    # Latched lifted: once the block is off the table it counts for the
    # rest of the episode, even after the gripper moves to the target zone.
    if _latch is not None:
        if is_elevated and near_block:
            _latch["lifted"] = True
        lifted = _latch["lifted"]
    else:
        lifted = bool(is_elevated and near_block)

    placed      = is_success(cube, target_zone, table_height, radius, min_height)
    partial = 0.2 * near_block + 0.4 * lifted + 1.0 * placed
    return {"near_block": near_block, "lifted": lifted, "placed": placed, "Sum": partial}


# ══════════════════════════════════════════════════════════════════════════════
#  Episode reset
# ══════════════════════════════════════════════════════════════════════════════

def reset_episode(scene, so101, cube, table_height, home_dofs=None):
    if home_dofs is None:
        home_dofs = np.zeros(so101.n_dofs)
    so101.set_dofs_position(home_dofs)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home_dofs)
    cube.set_pos(torch.tensor([0.25, 0.0, table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    for _ in range(10):
        scene.step()
    return home_dofs


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path   = os.path.join(script_dir, "../so101_arm/so101_new_calib.xml")

    gs.init(backend=gs.gpu, seed=42)

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
            world_frame_size=0.5,
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

    # Cameras must be added before scene.build()
    cameras = attach_cameras(scene, so101, table_height)

    scene.build()

    # ── Joint gains ───────────────────────────────────────────────────────────
    arm_dofs = np.arange(5)
    so101.set_dofs_kp(np.array([4000, 4000, 3000, 2000, 2000]), dofs_idx_local=arm_dofs)
    so101.set_dofs_kv(np.array([400,  400,  300,  200,  200]),  dofs_idx_local=arm_dofs)

    gripper_dof = np.array([5])
    so101.set_dofs_kp(np.array([800.0]), dofs_idx_local=gripper_dof)
    so101.set_dofs_kv(np.array([80.0]),  dofs_idx_local=gripper_dof)
    so101.set_dofs_force_range(
        lower=np.array([-100.0]),
        upper=np.array([ 100.0]),
        dofs_idx_local=gripper_dof,
    )

    # Gripper friction (must be set after build)
    for link_name in ["gripper", "moving_jaw_so101_v1"]:
        so101.get_link(link_name).set_friction(5.0)

    # ── Wrappers with closed-over constants ───────────────────────────────────
    def success_fn(c, tz, th):
        return is_success(c, tz, th, radius=0.05, min_height=0.005)

    def sub_goals_fn(robot, c, tz, th, latch=None):
        return check_sub_goals(robot, c, tz, th, radius=0.05, min_height=0.005, _latch=latch)

    # ── Collect demonstrations ────────────────────────────────────────────────
    nominal_cameras = cameras 
    collect_demonstrations(
        so101=so101,
        scene=scene,
        cameras=nominal_cameras,
        cube=cube,
        target_zone=target_zone,
        table_height=table_height,
        is_success_fn=success_fn,
        check_sub_goals_fn=sub_goals_fn,
        n_episodes=200,
        output_dir="demos/lerobot/",
        fps=30,
        repo_id="local/genesis_pickplace",
    )


if __name__ == "__main__":
    main()