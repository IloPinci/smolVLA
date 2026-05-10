import os
import numpy as np
import genesis as gs

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "../so101_arm/so101_new_calib.xml")

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

# Define table parameters
table_height = 0.8
table_width = 1.0
table_depth = 1.0

# Add table. Z-position is half the height to rest on the plane.
table = scene.add_entity(
    gs.morphs.Box(
        size=(table_width, table_depth, table_height),
        pos=(0.0, 0.0, table_height / 2.0)
    )
)

# Add SO101. Z-position is set to table_height to rest on top.
so101 = scene.add_entity(
    gs.morphs.MJCF(file=xml_path, 
                   pos=(0.0, 0.0, table_height))

)

scene.build()

n_dofs = so101.n_dofs

for i in range(1000):
    target_pos = np.zeros(n_dofs)
    
    if n_dofs > 0:
        target_pos = np.sin(i / 50.0) 
        
    so101.control_dofs_position(target_pos)
    
    scene.step()