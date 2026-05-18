import os
import numpy as np
import genesis as gs
import torch
from oracle import SO101Oracle, OracleConfig, collect_demonstrations



# * The building of the enviroment 
def build_environment(scene,  table_height):
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
        )
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
            pos=(0.25, 0.0, table_height + 0.015)
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 0.0, 0.0)       # ! Here you can change the color
        )
    )

    # ? the target space
    target_zone = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.05,
            height=0.001,
            pos=(-0.15, -0.15, table_height + 0.001)
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
    
    return {"front": cam_front, "top": cam_top}

# * Check if the cube has gone to the targed  
def is_sucess(cube, target_zone, min_height, table_height, radius):
    cube_pos = cube.get_pos()
    target_pos = target_zone.get_pos()

    # the cube must be inside the radius of the target
    xy_displacement = torch.norm(cube_pos[:2] - target_pos[:2])

    on_target = xy_displacement < radius    
    above_table = cube_pos[2] > (table_height + min_height)     # it is not under the table
    not_too_high  = cube_pos[2] < (table_height + 0.05)         # not floating above the table

    return bool (above_table and on_target and not_too_high)

# * For each episode we reset eveything to as it was
def reset_episode(scene, so101, cube, table_height, home_dofs=None):
    # arm home pose: all joints at zero 
    if home_dofs is None:
        home_dofs = np.zeros(so101.n_dofs)
    
    # robot
    so101.set_dofs_position(home_dofs)                  # home position for the robot 
    so101.set_dofs_velocity(np.zeros(so101.n_dofs))     # delete any residual velocities
    so101.control_dofs_position(home_dofs)              # the motors targets are updated so the robot maintains the home position

    # cube
    cube.set_pos(torch.tensor([0.25, 0.0, table_height + 0.015]))   # original position
    cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))   # orient if tumbled

    # Let physics settle for 10 steps before starting the episode
    for x in range(10):
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
    lifted = bool(is_elevated and near_block)       # to avoid tumbling being counted as lifting

    #? Sub-goal 3: block placed at target (Pass the missing variables here)
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
    # path for the robot arm
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "../so101_arm/so101_new_calib.xml")

    gs.init(backend=gs.gpu, seed=42)

    scene = gs.Scene(
        #? the lighting level should be kept constants so it doesn't ruin the testing
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.5,
            show_cameras=False,     # Todo: True -> if we want to have the cameras shown in the video (also their FOV is shown)
            ambient_light=(0.25, 0.25, 0.25),
            lights=[
                {"type": "directional", "dir": (-0.5, -0.5, -1.0), "intensity": 8.0, "color": (1.0, 0.97, 0.90)},
                {"type": "directional", "dir": (0.5, 0.5, -0.5), "intensity": 2.5, "color": (0.6, 0.7, 1.0)},
            ],
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    
    # the enviroment setup
    table_height = 0.8
    radius = 0.05
    min_height = 0.005      # if more than it is not considered resting but lifted
    cube, target_zone = build_environment(scene, table_height)
    
    # 1. Define wrapper callbacks that inject the missing threshold variables
    def oracle_success_wrapper(c, tz, th):
        return is_sucess(c, tz, min_height, th, radius)

    def oracle_sub_goals_wrapper(robot, c, tz, th):
        return check_sub_goals(robot, c, tz, th, min_height, radius)



    # ? The robot
    so101 = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path, 
            pos=(0.0, 0.1, table_height)
        )
    )

    #! Cameras
    cameras = attach_cameras(scene, table_height)

    # build the scene
    scene.build()

    records = collect_demonstrations(
        so101        = so101,
        scene        = scene,
        cameras      = cameras,
        cube         = cube,
        target_zone  = target_zone,
        table_height = table_height,
        is_success_fn      = oracle_success_wrapper,        # your function
        check_sub_goals_fn = oracle_sub_goals_wrapper,   # your function
        n_episodes   = 150,
        output_dir   = "demos/baseline/",
    )
    

    n_dofs = so101.n_dofs

    n_episodes = 2
    max_steps = 200

    # the array that stores all the sums
    sum_metrics = []

    for ep in range(n_episodes):
        print(f"--- Starting Episode {ep} ---")
        reset_episode(scene, so101, cube, table_height)

        # we store the max for each episode
        max_sum = 0.0
        achieved_goals = {"near_block": False, "lifted": False, "placed": False}


        for i in range(max_steps):
            target_pos = np.full(n_dofs, np.sin(i / 50.0))
            so101.control_dofs_position(target_pos)

            scene.step()

            # we see the status that every step causes
            status = check_sub_goals(so101, cube, target_zone, table_height, min_height, radius)
            
            # update the statuses
            max_sum = max(max_sum, status["Sum"])
            achieved_goals["near_block"] |= status["near_block"]
            achieved_goals["lifted"]     |= status["lifted"]
            achieved_goals["placed"]     |= status["placed"]

            # if it was a success
            if status["placed"]:
                print(f"Task Completed at episode {ep}, step {i}")
                break
            
            # we render the cameras every 10 steps 
            if i % 10 == 0:
                rgb_front, _, _, _ = cameras["front"].render()
                rgb_top, _, _, _ = cameras["top"].render()
                
        # some feedback
        print(f"Episode {ep} finished.\n Max Sum: {max_sum:.2f} \n Goals: {achieved_goals}")

if __name__ == "__main__":
    main()