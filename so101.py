import os
import numpy as np
import genesis as gs

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "../so101_arm/so101_new_calib.xml")

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

so101 = scene.add_entity(
    gs.morphs.MJCF(file=xml_path),
)

scene.build()

n_dofs = so101.n_dofs

for i in range(1000):
    target_pos = np.zeros(n_dofs)
    
    if n_dofs > 0:
        target_pos = np.sin(i / 50.0) 
        
    so101.control_dofs_position(target_pos)
    
    scene.step()