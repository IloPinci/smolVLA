import os
import numpy as np
import genesis as gs
from pynput import keyboard

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "../so101_arm/so101_new_calib.xml")

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

# the obot
so101 = scene.add_entity(
    gs.morphs.MJCF(
        file=xml_path,
        pos=(0.0, 0.0, 0.0)
    )
)

# Red target block
target = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.2, 0.1, 0.02),
    ),
    surface=gs.surfaces.Rough(color=(0.9, 0.2, 0.2)),
)

# Recording camera — wide angle to capture full robot + block movement
record_cam = scene.add_camera(
    res=(854, 480),
    pos=(0.25, 0.0, 0.8),      # Diagonal view from the side
    lookat=(0.2, 0.05, 0.02),   # Look at robot workspace center
    fov=55,                    # Wide FOV to capture full arm range
    GUI=False,
)


scene.build()

n_dofs = so101.n_dofs
target_pos = np.zeros(n_dofs)
step_size = 0.01

# Initialize key state tracking
active_keys = set()

def on_press(key):
    try:
        active_keys.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        active_keys.discard(key.char)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    if n_dofs >= 6:
        # Joint 0: Shoulder Pan
        if '1' in active_keys: target_pos[0] += step_size
        if '2' in active_keys: target_pos[0]  -= step_size

        # Joint 1: Shoulder Pitch
        if '3' in active_keys: target_pos[1]  += step_size
        if '4' in active_keys: target_pos[1]  -= step_size
        
        # Joint 2: Elbow Pitch
        if '5' in active_keys: target_pos[2]  += step_size
        if '6' in active_keys: target_pos[2]  -= step_size
            
        # Joint 3: Wrist Pitch
        if '7' in active_keys: target_pos[3]  += step_size
        if '8' in active_keys: target_pos[3]  -= step_size

        # Joint 4: Wrist Roll
        if '9' in active_keys: target_pos[4]  += step_size
        if '0' in active_keys: target_pos[4]  -= step_size

        # Joint 5: Gripper left
        if '-' in active_keys: target_pos[5]  += step_size
        if '=' in active_keys: target_pos[5]  -= step_size

        # Joint 6: Gripper right
        if '[' in active_keys: target_pos[6]  += step_size
        if ']' in active_keys: target_pos[6]  -= step_size

    so101.control_dofs_position(target_pos)
    scene.step()