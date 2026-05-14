import os
import genesis as gs

gs.init(backend=gs.gpu, headless=True)

scene = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(show_world_frame=True),
)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

cam = scene.add_camera(
    res=(1280, 1024),   
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=40,
    GUI=False,
)

scene.build()

sim_dt = scene.dt          # simulation timestep
target_video_fps = 30
render_every = max(1, int(1.0 / (target_video_fps * sim_dt)))  # auto-calculate

print(f"Rendering every {render_every} steps to achieve {target_video_fps} FPS video")

cam.start_recording()

for i in range(1000):
    scene.step()
    if i % render_every == 0:
        cam.render()

# Save as mp4
cam.stop_recording(save_to_filename='/home/ter-ws-3/jt_cogar/videos/so101.mp4', fps=60)
print("Video saved to franka_sim.mp4")


