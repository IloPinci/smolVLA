import os
import numpy as np
import genesis as gs
import torch
from oracle import SO101Oracle, OracleConfig, collect_demonstrations



# * The building of the enviroment 
def build_environment(scene, table_height):
    # table variables
    table_width = 1.0
    table_depth = 1.0
    surface_thickness = 0.05
    leg_thickness = 0.05

    #? create the tabletop
    scene.add_entity(
        gs.morphs.Box(
            size=(table_width, table_depth, surface_thickness),
            pos=(0.0, 0.0, table_height - (surface_thickness / 2.0)),
            fixed=True
        ),
        material=gs.materials.Rigid(
            friction=0.01,
            coup_friction=0.1,
            coup_softness=0.001,
        ),
    )

    #? create the legs
    for x in [-table_width/2 + leg_thickness/2, table_width/2 - leg_thickness/2]:
        for y in [-table_depth/2 + leg_thickness/2, table_depth/2 - leg_thickness/2]:
            scene.add_entity(
                gs.morphs.Box(
                    size=(leg_thickness, leg_thickness, table_height - surface_thickness),
                    pos=(x, y, (table_height - surface_thickness) / 2.0),
                    fixed=True
                )
            )

    #! the cube (color)
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.03, 0.03, 0.03),
            pos=(0.3, 0.0, table_height + 0.015),
        ),
        material=gs.materials.Rigid(
            rho = 1000.0,
            friction=3.0,
            coup_friction=1.5,
            coup_softness=0.0001,
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 0.0, 0.0)
        ),
        #vis_mode = 'collision'
    )

    # ? the target space
    # FIX #6: target zone is purely visual — make it non-collidable so the
    # gripper/cube can't physically interact with it during release.
    target_zone = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.05,
            height=0.001,
            pos=(0.15, -0.15, table_height + 0.001),
            fixed=True,
            collision=False,   # ← visual marker only; no contact geometry
        ),
        surface=gs.surfaces.Rough(color=(0.0, 1.0, 0.0))
    )

    return cube, target_zone

# * Create the cameras
def attach_cameras(scene, table_height):
    
    #! front (position)
    cam_front = scene.add_camera(
        res=(512, 512),
        pos=(1.0, 0.0, table_height + 0.5),
        lookat=(0.0, 0.0, table_height + 0.1),
        fov=65,
        GUI=False   
    )

    #! top (position)
    cam_top = scene.add_camera(
        res=(512, 512),
        pos=(0.0, 0.0, table_height + 1.0),
        lookat=(0.0, 0.0, table_height + 0.1),
        fov=65,
        GUI=False
    )

    #! wrist
    cam_wrist = scene.add_camera(
        res=(512, 512),
        pos=(0.05, 0.0, table_height + 0.3),
        lookat=(0.0, 0.0, table_height),
        fov=90,
        GUI=False
    )
    
    return {"front": cam_front, "top": cam_top, "wrist": cam_wrist}

# * Check if the cube has gone to the target  
def is_sucess(cube, target_zone, min_height, table_height, radius):
    cube_pos = cube.get_pos()
    target_pos = target_zone.get_pos()

    xy_displacement = torch.norm(cube_pos[:2] - target_pos[:2])

    on_target    = xy_displacement < radius    
    above_table  = cube_pos[2] > (table_height + min_height)
    not_too_high = cube_pos[2] < (table_height + 0.05)

    return bool(above_table and on_target and not_too_high)

# * For each episode we reset everything to as it was
def reset_episode(scene, so101, cube, table_height, home_dofs=None):
    if home_dofs is None:
        home_dofs = np.zeros(so101.n_dofs)
    
    # FIX #7: teleport joints instantly, don't just set motor targets
    so101.set_dofs_position(home_dofs)
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))
    so101.control_dofs_position(home_dofs)

    cube.set_pos(torch.tensor([0.3, 0.0, table_height + 0.015]))
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(10):
        scene.step()

    return home_dofs

# * To get a better understanding in the case of failure
def check_sub_goals(so101, cube, target_zone, table_height, min_height, radius):
    """
    Returns a dict of sub-goal completions and a weighted score.
    Weights per plan: gripper-near-block=0.2, block-lifted=0.4, placed=1.0
    """
    cube_pos   = cube.get_pos()
    target_pos = target_zone.get_pos()

    #? Sub-goal 1: gripper within 5 cm of block
    gripper_link = so101.get_link("gripper")
    gripper_pos  = gripper_link.get_pos()
    gripper_dist = torch.norm(gripper_pos - cube_pos).item()
    near_block   = gripper_dist < 0.05

    #? Sub-goal 2: block lifted > 2 cm off table
    is_elevated = cube_pos[2].item() > (table_height + 0.02)
    lifted = bool(is_elevated and near_block)

    #? Sub-goal 3: block placed at target
    placed = is_sucess(cube, target_zone, min_height, table_height, radius)

    partial_sum = 0.0
    if near_block: partial_sum += 0.2
    if lifted:     partial_sum += 0.4
    if placed:     partial_sum += 1.0

    return {
        "near_block": near_block,
        "lifted":     lifted,
        "placed":     placed,
        "Sum":        partial_sum,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "../so101_arm/so101_new_calib.xml")

    gs.init(backend=gs.gpu, seed=42)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=16,
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=100,
            tolerance=1e-9,
            constraint_timeconst=0.006,
            enable_self_collision=True,
            box_box_detection=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.5,
            show_cameras=False,
            ambient_light=(0.25, 0.25, 0.25),
            lights=[
                {"type": "directional", "dir": (-0.5, -0.5, -1.0), "intensity": 8.0, "color": (1.0, 0.97, 0.90)},
                {"type": "directional", "dir": (0.5, 0.5, -0.5), "intensity": 2.5, "color": (0.6, 0.7, 1.0)},
            ],
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    
    table_height = 0.8
    radius = 0.05
    min_height = 0.005
    cube, target_zone = build_environment(scene, table_height)
    
    def oracle_success_wrapper(c, tz, th):
        return is_sucess(c, tz, min_height, th, radius)

    def oracle_sub_goals_wrapper(robot, c, tz, th):
        return check_sub_goals(robot, c, tz, th, min_height, radius)

    so101 = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path, 
            pos=(0.0, 0.0, table_height)
        ),
        #vis_mode = 'collision'
    )

    cameras = attach_cameras(scene, table_height)

    gripper_link = so101.get_link("gripper")
    offset_T = np.eye(4)
    offset_T[:3, 3] = np.array([0.0, 0.04, 0.1])
    cameras["wrist"].attach(gripper_link, offset_T)

    scene.build()

    # ── Gripper contact friction ───────────────────────────────────────────
    for link_name in ["gripper", "moving_jaw_so101_v1"]:
        link = so101.get_link(link_name)
        link.set_friction(5.0)

    # -- Arm joints (0-4): high stiffness for precise IK tracking
    arm_dofs = np.arange(5)
    so101.set_dofs_kp(np.array([4000, 4000, 3000, 2000, 2000]), dofs_idx_local=arm_dofs)
    so101.set_dofs_kv(np.array([400,  400,  300,  200,  200]),  dofs_idx_local=arm_dofs)
    
    # FIX #4: substantially higher kp/kv on the gripper DOF so the PD spring
    # produces enough squeeze force to resist gravity + inertia during the
    # carry phase. Was kp=200/kv=20 — far too weak for a loaded one-sided jaw.
    gripper_dof = np.array([5])
    so101.set_dofs_kp(np.array([800.0]),  dofs_idx_local=gripper_dof)   # was 200
    so101.set_dofs_kv(np.array([80.0]),   dofs_idx_local=gripper_dof)   # was 20
    so101.set_dofs_force_range(
        lower=np.array([-100.0]),   # was -50 — allow stronger squeeze
        upper=np.array([ 100.0]),
        dofs_idx_local=gripper_dof,
    )

    # Set friction on all gripper collision links (second pass after build)
    for link_name in ["gripper", "moving_jaw_so101_v1"]:
        link = so101.get_link(link_name)
        link.set_friction(5.0)   # FIX #4: raised from 3.0 to match first pass

    records = collect_demonstrations(
        so101        = so101,
        scene        = scene,
        cameras      = cameras,
        cube         = cube,
        target_zone  = target_zone,
        table_height = table_height,
        is_success_fn      = oracle_success_wrapper,
        check_sub_goals_fn = oracle_sub_goals_wrapper,
        n_episodes   = 150,
        output_dir   = "demos/baseline/",
    )
    

    n_dofs = so101.n_dofs
    n_episodes = 2
    max_steps = 200
    sum_metrics = []

    for ep in range(n_episodes):
        print(f"--- Starting Episode {ep} ---")
        reset_episode(scene, so101, cube, table_height)

        max_sum = 0.0
        achieved_goals = {"near_block": False, "lifted": False, "placed": False}

        for i in range(max_steps):
            target_pos = np.full(n_dofs, np.sin(i / 50.0))
            so101.control_dofs_position(target_pos)

            scene.step()

            cameras["wrist"].move_to_attach()

            status = check_sub_goals(so101, cube, target_zone, table_height, min_height, radius)
            
            max_sum = max(max_sum, status["Sum"])
            achieved_goals["near_block"] |= status["near_block"]
            achieved_goals["lifted"]     |= status["lifted"]
            achieved_goals["placed"]     |= status["placed"]

            if status["placed"]:
                print(f"Task Completed at episode {ep}, step {i}")
                break
            
            if i % 10 == 0:
                rgb_front, _, _, _ = cameras["front"].render()
                rgb_top, _, _, _ = cameras["top"].render()
                rgb_wrist, _, _, _ = cameras["wrist"].render()
                
        print(f"Episode {ep} finished.\n Max Sum: {max_sum:.2f} \n Goals: {achieved_goals}")

if __name__ == "__main__":
    main()