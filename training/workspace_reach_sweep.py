"""
workspace_reach_sweep.py
-------------------------
Empirically maps the SO-101's reachable workspace by sweeping a grid of
(x, y) positions and attempting IK at each one, at the heights the oracle
actually uses during grasp / lift / carry / place. This is more reliable
than guessing from link lengths because joint limits, the fixed pregrasp
orientation, and IK convergence all shrink the *usable* reach relative to
the raw kinematic reach.

For each grid point and each height:
  1. Run so101.inverse_kinematics() toward the target pos/quat.
  2. Temporarily apply the resulting qpos and read back the actual
     end-effector position (forward-kinematics check) -- IK solvers can
     return a "solution" that doesn't actually converge.
  3. Check the qpos against joint limits (a converged IK solution can
     still be physically unreachable if it ignores limits).
  4. Mark the point reachable only if position error is small AND all
     joints are within limits.
  5. Restore the robot to its home pose before the next sample.

Two views of the result are produced:

  (A) ASCII heatmap per height — quick visual grid, printed to terminal.

  (B) Radial reach map — for a set of angles around the robot base
      (0,0), walks outward from r_min to r_max and reports the closest
      and farthest reachable radius in that direction (i.e. the inner
      and outer boundary of the reachable annulus). This directly
      answers "how close can the cube spawn, and how far, before it's
      unreachable, from every direction around the base" — which is
      exactly what you need to validate a jitter range against.

Usage
-----
    python workspace_reach_sweep.py \
        --xml ../so101_arm/so101_new_calib.xml \
        --out results/workspace_reach.png

    # skip the grid sweep / PNG, just print the radial map fast
    python workspace_reach_sweep.py --no_grid --angle_step 15
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import genesis as gs

from scene_params import CUBE_POSITIONS

# ── grasp orientation used by the oracle (training/oracle_direct.py) ─────────
GRASP_QUAT = np.array([0.707107, 0.0, -0.707107, 0.0])

# ── heights the oracle actually visits (training/oracle_direct.py waypoints) ─
GRIPPER_LENGTH      = 0.06
PREGRASP_CLEARANCE  = 0.12
LIFT_HEIGHT         = 0.15
CARRY_CLEARANCE     = 0.15

POS_TOL = 1e-4
ROT_TOL = 1e-4
FK_ERROR_THRESH = 0.005   # 5 mm -- point counts as "reachable" below this


def build_minimal_scene(xml_path: str, table_height: float):
    """Just the robot -- no cube/cameras needed for a pure IK sweep."""
    gs.init(backend=gs.gpu, seed=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=16),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=100, tolerance=1e-9,
            constraint_timeconst=0.006,
            enable_self_collision=True, box_box_detection=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    so101 = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, table_height))
    )
    scene.build()
    return scene, so101


def get_joint_limits(so101):
    """Best-effort joint limit lookup -- different Genesis versions expose
    this slightly differently, so fall back to 'no limit check' if absent."""
    for attr in ("get_dofs_limit", "get_dof_limits", "get_dofs_limits"):
        if hasattr(so101, attr):
            try:
                lower, upper = getattr(so101, attr)()
                lower = lower.cpu().numpy() if torch.is_tensor(lower) else np.asarray(lower)
                upper = upper.cpu().numpy() if torch.is_tensor(upper) else np.asarray(upper)
                print(f"[info] joint limits found via so101.{attr}()")
                return lower, upper
            except Exception:
                continue
    print("[warn] Could not read joint limits from this Genesis build -- "
          "skipping the limit check (FK error check still applies).")
    return None, None


def check_point(so101, end_effector, home_qpos, target_xyz, quat,
                 lower=None, upper=None):
    """Return (reachable: bool, fk_error: float) for one (x, y, z) target."""
    try:
        qpos = so101.inverse_kinematics(
            link=end_effector, pos=np.asarray(target_xyz), quat=quat,
            pos_tol=POS_TOL, rot_tol=ROT_TOL,
        )
    except Exception:
        return False, float("nan")

    qpos_np = qpos.cpu().numpy() if torch.is_tensor(qpos) else np.asarray(qpos)

    within_limits = True
    if lower is not None and upper is not None:
        within_limits = bool(np.all(qpos_np >= lower - 1e-6) and
                              np.all(qpos_np <= upper + 1e-6))

    # Forward-kinematics check: apply qpos, read back actual EE position.
    so101.set_dofs_position(qpos)
    actual_pos = end_effector.get_pos()
    actual_pos = actual_pos.cpu().numpy() if torch.is_tensor(actual_pos) else np.asarray(actual_pos)
    fk_error = float(np.linalg.norm(actual_pos - np.asarray(target_xyz)))

    # restore home pose so each sample starts from the same configuration
    so101.set_dofs_position(home_qpos)

    reachable = within_limits and (fk_error < FK_ERROR_THRESH)
    return reachable, fk_error


def sweep_grid(so101, end_effector, home_qpos, z, xs, ys, lower, upper):
    grid = np.zeros((len(xs), len(ys)), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ok, _ = check_point(
                so101, end_effector, home_qpos,
                target_xyz=(x, y, z), quat=GRASP_QUAT,
                lower=lower, upper=upper,
            )
            grid[i, j] = ok
    return grid


def sweep_radial(so101, end_effector, home_qpos, z, center, angles_deg,
                  r_min, r_max, r_step, lower, upper):
    """
    For each angle, walk r outward from r_min to r_max and find the
    contiguous reachable interval [r_inner, r_outer]. Returns a list of
    dicts: {angle_deg, r_inner, r_outer, n_gaps}.
    n_gaps counts any reachable->unreachable->reachable transitions
    inside the scanned range (should normally be 0; >0 means the
    workspace has a hole at that angle, e.g. a singularity).
    """
    cx, cy = center
    radii = np.arange(r_min, r_max + 1e-9, r_step)
    results = []
    for angle in angles_deg:
        theta = np.radians(angle)
        reach_flags = []
        for r in radii:
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            ok, _ = check_point(
                so101, end_effector, home_qpos,
                target_xyz=(x, y, z), quat=GRASP_QUAT,
                lower=lower, upper=upper,
            )
            reach_flags.append(ok)

        reach_flags = np.array(reach_flags)
        idx_true = np.where(reach_flags)[0]
        if len(idx_true) == 0:
            results.append({"angle_deg": angle, "r_inner": None,
                             "r_outer": None, "n_gaps": 0})
            continue

        r_inner = radii[idx_true[0]]
        r_outer = radii[idx_true[-1]]

        # count gaps: transitions from True to False before the last True
        segment = reach_flags[idx_true[0]:idx_true[-1] + 1]
        n_gaps = int(np.sum((segment[:-1] == True) & (segment[1:] == False)))

        results.append({"angle_deg": angle, "r_inner": float(r_inner),
                         "r_outer": float(r_outer), "n_gaps": n_gaps})
    return results


def print_ascii_grid(grid, xs, ys, label, char_reach="#", char_block="."):
    """Print a compact ASCII heatmap to stdout. Rows = y (top=+y), cols = x."""
    print(f"\n── ASCII grid: {label} ──────────────────────────────")
    print(f"   x: {xs[0]:.3f} .. {xs[-1]:.3f}   y: {ys[0]:.3f} .. {ys[-1]:.3f}")
    print(f"   '{char_reach}' = reachable   '{char_block}' = not reachable\n")
    # print rows from max y (top) to min y (bottom) so it reads like a map
    for j in range(len(ys) - 1, -1, -1):
        row = "".join(char_reach if grid[i, j] else char_block
                       for i in range(len(xs)))
        print(f"  y={ys[j]:+.3f}  {row}")
    footer = "".join(str(i % 10) for i in range(len(xs)))
    print(f"            {footer}   (col index, x increases →)")


def print_radial_table(results, label):
    print(f"\n── Radial reach map: {label} ──────────────────────────")
    print(f"  {'angle':>6}  {'r_inner':>9}  {'r_outer':>9}  {'span':>7}  {'gaps':>5}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*5}")
    for r in results:
        if r["r_inner"] is None:
            print(f"  {r['angle_deg']:6.0f}  {'--':>9}  {'--':>9}  {'--':>7}  {'--':>5}   (unreachable at all radii)")
        else:
            span = r["r_outer"] - r["r_inner"]
            gap_flag = "  <-- HOLE" if r["n_gaps"] > 0 else ""
            print(f"  {r['angle_deg']:6.0f}  {r['r_inner']:9.3f}  {r['r_outer']:9.3f}  "
                  f"{span:7.3f}  {r['n_gaps']:5d}{gap_flag}")

    valid = [r for r in results if r["r_inner"] is not None]
    if valid:
        worst_inner = max(valid, key=lambda r: r["r_inner"])   # largest "closest" = most restrictive near boundary
        best_inner  = min(valid, key=lambda r: r["r_inner"])
        worst_outer = min(valid, key=lambda r: r["r_outer"])   # smallest "farthest" = most restrictive far boundary
        best_outer  = max(valid, key=lambda r: r["r_outer"])
        print(f"\n  Tightest inner bound : {worst_inner['r_inner']:.3f} m "
              f"at angle {worst_inner['angle_deg']:.0f}°  "
              f"(safe min radius must be >= this, for ALL directions)")
        print(f"  Loosest  inner bound : {best_inner['r_inner']:.3f} m "
              f"at angle {best_inner['angle_deg']:.0f}°")
        print(f"  Tightest outer bound : {worst_outer['r_outer']:.3f} m "
              f"at angle {worst_outer['angle_deg']:.0f}°  "
              f"(safe max radius must be <= this, for ALL directions)")
        print(f"  Loosest  outer bound : {best_outer['r_outer']:.3f} m "
              f"at angle {best_outer['angle_deg']:.0f}°")
        print(f"\n  => A radius range that is reachable from EVERY direction "
              f"scanned: [{worst_inner['r_inner']:.3f}, {worst_outer['r_outer']:.3f}] m")
    else:
        print("\n  [warn] no reachable points found in this scan range at all.")


def draw_jitter_rect(ax, center, half_extent, **kwargs):
    cx, cy = center
    hx, hy = half_extent
    rect = mpatches.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                               fill=False, linewidth=1.8, **kwargs)
    ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="so101_arm/so101_new_calib.xml")
    parser.add_argument("--table_height", type=float, default=0.8)
    parser.add_argument("--out", default="results/workspace_reach.png")

    # grid sweep (ASCII + PNG)
    parser.add_argument("--no_grid", action="store_true",
                        help="skip the 2D grid sweep / PNG, only do the radial scan")
    parser.add_argument("--nx", type=int, default=35)
    parser.add_argument("--ny", type=int, default=35)
    parser.add_argument("--x_min", type=float, default=0.02)
    parser.add_argument("--x_max", type=float, default=0.42)
    parser.add_argument("--y_min", type=float, default=-0.28)
    parser.add_argument("--y_max", type=float, default=0.28)

    # radial sweep
    parser.add_argument("--no_radial", action="store_true")
    parser.add_argument("--angle_step", type=float, default=10.0,
                        help="degrees between radial scan directions")
    parser.add_argument("--angle_min", type=float, default=-90.0)
    parser.add_argument("--angle_max", type=float, default=90.0,
                        help="SO-101 base typically only reaches the +x half-plane; "
                             "widen to -180/180 if you need the full circle")
    parser.add_argument("--r_min", type=float, default=0.03)
    parser.add_argument("--r_max", type=float, default=0.45)
    parser.add_argument("--r_step", type=float, default=0.01)
    parser.add_argument("--radial_center", nargs=2, type=float, default=[0.0, 0.0],
                        help="origin for the radial scan, default = robot base")

    # current jitter settings -- keep in sync with ScenePerturbation.resolve_spawn
    parser.add_argument("--cube_center",   nargs=2, type=float, default=[0.24, 0.00])
    parser.add_argument("--cube_half",     nargs=2, type=float, default=[0.04, 0.04])
    parser.add_argument("--target_center", nargs=2, type=float, default=[0.15, -0.15])
    parser.add_argument("--target_half",   nargs=2, type=float, default=[0.05, 0.05])
    args = parser.parse_args()

    xml_path = os.path.abspath(args.xml)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scene, so101 = build_minimal_scene(xml_path, args.table_height)
    end_effector = so101.get_link("moving_jaw_so101_v1")
    home_qpos_raw = so101.get_dofs_position()
    home_qpos = home_qpos_raw.clone() if torch.is_tensor(home_qpos_raw) else np.array(home_qpos_raw)

    lower, upper = get_joint_limits(so101)

    th = args.table_height
    heights = {
        "Pregrasp (wp0)": th + PREGRASP_CLEARANCE + GRIPPER_LENGTH,
        "Grasp (wp1)":    th + 0.01 + GRIPPER_LENGTH,
        "Lift (wp2)":     th + LIFT_HEIGHT + GRIPPER_LENGTH,
        "Place (wp4)":    th + 0.04 + GRIPPER_LENGTH,
    }

    print("=" * 70)
    print("SO-101 WORKSPACE REACH SWEEP")
    print("=" * 70)
    for label, z in heights.items():
        print(f"  {label:<16} z = {z:.3f} m")
    print()

    # ── 2D grid sweep (ASCII + optional PNG) ─────────────────────────────────
    all_grids = {}
    if not args.no_grid:
        xs = np.linspace(args.x_min, args.x_max, args.nx)
        ys = np.linspace(args.y_min, args.y_max, args.ny)
        print(f"Grid sweep: {args.nx}x{args.ny} points x {len(heights)} heights "
              f"= {args.nx * args.ny * len(heights)} IK calls\n")

        for label, z in heights.items():
            grid = sweep_grid(so101, end_effector, home_qpos, z, xs, ys, lower, upper)
            all_grids[label] = grid
            pct = 100.0 * grid.mean()
            print(f"[{label}] reachable: {pct:.1f}% of grid")
            print_ascii_grid(grid, xs, ys, label)

        # PNG too, in case you can copy it out to view later
        fig, axes = plt.subplots(1, len(heights), figsize=(5 * len(heights), 5),
                                  sharey=True)
        if len(heights) == 1:
            axes = [axes]
        for ax, (label, z) in zip(axes, heights.items()):
            grid = all_grids[label]
            ax.imshow(grid.T.astype(float), origin="lower",
                      extent=[xs[0], xs[-1], ys[0], ys[-1]],
                      cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", alpha=0.85)
            ax.set_title(f"{label}\n{100*grid.mean():.0f}% reachable")
            ax.set_xlabel("x (m)")
            for name, preset in CUBE_POSITIONS.items():
                cx, cy = preset["cube_xy"]
                ax.plot(cx, cy, marker="x", color="black", markersize=6)
                ax.annotate(name, (cx, cy), fontsize=6, color="black",
                            xytext=(2, 2), textcoords="offset points")
            draw_jitter_rect(ax, args.cube_center, args.cube_half, edgecolor="blue")
            draw_jitter_rect(ax, args.target_center, args.target_half, edgecolor="purple")
        axes[0].set_ylabel("y (m)")
        handles = [
            mpatches.Patch(edgecolor="blue", facecolor="none", label="cube jitter range"),
            mpatches.Patch(edgecolor="purple", facecolor="none", label="target jitter range"),
            plt.Line2D([0], [0], marker="x", color="black", linestyle="None",
                       label="CUBE_POSITIONS presets"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, -0.03))
        fig.suptitle("SO-101 IK Reachability -- by waypoint height", fontsize=13)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n[ok] PNG also saved -> {out_path} (optional, for later viewing)")

    # ── Radial scan: closest / farthest reachable radius per direction ──────
    if not args.no_radial:
        angles = np.arange(args.angle_min, args.angle_max + 1e-9, args.angle_step)
        print(f"\n\nRadial sweep: {len(angles)} angles x "
              f"{len(np.arange(args.r_min, args.r_max + 1e-9, args.r_step))} radii "
              f"x {len(heights)} heights\n")

        radial_results = {}
        for label, z in heights.items():
            res = sweep_radial(so101, end_effector, home_qpos, z,
                               center=tuple(args.radial_center),
                               angles_deg=angles,
                               r_min=args.r_min, r_max=args.r_max, r_step=args.r_step,
                               lower=lower, upper=upper)
            radial_results[label] = res
            print_radial_table(res, label)

        # ── Combined "safe at every height" envelope ─────────────────────────
        print("\n" + "=" * 70)
        print("COMBINED RADIUS ENVELOPE (safe at every height checked)")
        print("=" * 70)
        global_inner = max(
            max(r["r_inner"] for r in res if r["r_inner"] is not None)
            for res in radial_results.values()
        )
        global_outer = min(
            min(r["r_outer"] for r in res if r["r_outer"] is not None)
            for res in radial_results.values()
        )
        print(f"  Safe radius range from base {tuple(args.radial_center)}: "
              f"[{global_inner:.3f}, {global_outer:.3f}] m")
        print(f"  (smaller than this = arm can't reach down/over the cube; "
              f"larger than this = out of reach at some waypoint height)")

        # check current jitter rectangles against this envelope
        print(f"\n  Checking your jitter ranges against this envelope:")
        for name, center, half in [
            ("cube",   args.cube_center,   args.cube_half),
            ("target", args.target_center, args.target_half),
        ]:
            cx, cy = center
            hx, hy = half
            corners = [(cx + dx * hx, cy + dy * hy) for dx in (-1, 1) for dy in (-1, 1)]
            bx, by = args.radial_center
            radii = [np.hypot(x - bx, y - by) for x, y in corners]
            r_lo, r_hi = min(radii), max(radii)
            ok = (r_lo >= global_inner) and (r_hi <= global_outer)
            status = "OK -- inside safe envelope" if ok else "WARNING -- outside safe envelope"
            print(f"    {name:<8} corner radii: [{r_lo:.3f}, {r_hi:.3f}] m  -> {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()