import numpy as np
import struct

BASE = "/home/snape/jt_cogar/so101_arm/assets"

def get_stl_bounds(filepath):
    with open(filepath, 'rb') as f:
        f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        vertices = []
        for _ in range(num_triangles):
            f.read(12)
            for _ in range(3):
                v = struct.unpack('<3f', f.read(12))
                vertices.append(v)
            f.read(2)

    verts = np.array(vertices)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    size = maxs - mins
    center = (mins + maxs) / 2.0
    half = size / 2.0

    print(f"  Full size (m):      x={size[0]:.4f}  y={size[1]:.4f}  z={size[2]:.4f}")
    print(f"  Half-extents (m):   x={half[0]:.4f}  y={half[1]:.4f}  z={half[2]:.4f}")
    print(f"  Mesh center (m):    x={center[0]:.4f}  y={center[1]:.4f}  z={center[2]:.4f}")
    return half, center

print("=== moving_jaw_so101_v1 (moving jaw) ===")
mj_half, mj_center = get_stl_bounds(f"{BASE}/moving_jaw_so101_v1.stl")

print("\n=== wrist_roll_follower_so101_v1 (static jaw body) ===")
wf_half, wf_center = get_stl_bounds(f"{BASE}/wrist_roll_follower_so101_v1.stl")