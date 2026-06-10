"""
verify_dataset.py
-----------------
Run from the project root:
    python verify_dataset.py --data_dir demos/baseline/

Checks every HDF5 episode for schema, shape, value sanity, and prints a
summary.  Saves a CSV of per-episode stats and renders a grid of 5 random
episodes (first / middle / last frame of context + wrist) to verify.png.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np

# ── optional matplotlib (only for the frame grid) ────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — frame grid will be skipped.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Expected schema
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_DATASETS = {
    "data/observation.images.context": {"dtype": "uint8",   "ndim": 4},  # [T,224,224,3]
    "data/observation.images.wrist":   {"dtype": "uint8",   "ndim": 4},
    "data/observation.state":          {"dtype": "float32", "ndim": 2},  # [T,6]
    "data/action":                     {"dtype": "float32", "ndim": 2},  # [T,7]
    "meta/success":                    {"dtype": None,      "ndim": 0},  # scalar
    "meta/episode_id":                 {"dtype": None,      "ndim": 0},
    "meta/language_instruction":       {"dtype": None,      "ndim": 0},
}

IMG_SHAPE   = (224, 224, 3)
STATE_DIM   = 6
ACTION_DIM  = 6
MIN_FRAMES  = 20   # any episode shorter than this is probably corrupt


# ══════════════════════════════════════════════════════════════════════════════
#  Per-episode check
# ══════════════════════════════════════════════════════════════════════════════

def check_episode(path: Path) -> dict:
    """Return a dict with pass/fail flags and stats for one HDF5 file."""
    result = {
        "file": path.name,
        "ok": True,
        "errors": [],
        "warnings": [],
        "T": None,
        "success": None,
        "state_min": None,
        "state_max": None,
        "action_max_delta": None,
    }

    try:
        with h5py.File(path, "r") as f:

            # ── Schema check ─────────────────────────────────────────────────
            for key, spec in EXPECTED_DATASETS.items():
                if key not in f:
                    result["errors"].append(f"missing dataset: {key}")
                    result["ok"] = False
                    continue
                ds = f[key]
                if spec["ndim"] is not None and ds.ndim != spec["ndim"]:
                    result["errors"].append(
                        f"{key}: expected ndim={spec['ndim']}, got {ds.ndim}"
                    )
                    result["ok"] = False
                if spec["dtype"] and not np.issubdtype(ds.dtype, np.dtype(spec["dtype"]).type):
                    result["errors"].append(
                        f"{key}: expected dtype~{spec['dtype']}, got {ds.dtype}"
                    )
                    result["ok"] = False

            if not result["ok"]:
                return result   # don't proceed if schema is broken

            # ── Shape checks ─────────────────────────────────────────────────
            T = f["data/action"].shape[0]
            result["T"] = T

            if T < MIN_FRAMES:
                result["errors"].append(f"only {T} frames — episode too short (min {MIN_FRAMES})")
                result["ok"] = False

            ctx_shape = f["data/observation.images.context"].shape[1:]  # H,W,C
            if ctx_shape != IMG_SHAPE:
                result["errors"].append(f"context image shape {ctx_shape} != {IMG_SHAPE}")
                result["ok"] = False

            wrist_shape = f["data/observation.images.wrist"].shape[1:]
            if wrist_shape != IMG_SHAPE:
                result["errors"].append(f"wrist image shape {wrist_shape} != {IMG_SHAPE}")
                result["ok"] = False

            state_dim = f["data/observation.state"].shape[1]
            if state_dim != STATE_DIM:
                result["errors"].append(f"state dim {state_dim} != {STATE_DIM}")
                result["ok"] = False

            action_dim = f["data/action"].shape[1]
            if action_dim != ACTION_DIM:
                result["errors"].append(f"action dim {action_dim} != {ACTION_DIM}")
                result["ok"] = False

            # ── Temporal alignment ────────────────────────────────────────────
            for key in [
                "data/observation.images.context",
                "data/observation.images.wrist",
                "data/observation.state",
            ]:
                if f[key].shape[0] != T:
                    result["errors"].append(
                        f"{key} has {f[key].shape[0]} frames but action has {T}"
                    )
                    result["ok"] = False

            # ── Value sanity ──────────────────────────────────────────────────
            state = f["data/observation.state"][:]
            action = f["data/action"][:]

            result["state_min"] = float(state.min())
            result["state_max"] = float(state.max())
            result["action_max_delta"] = float(np.abs(action[:, :5]).max())

            # Joint angles should be within ±3π (generous bound — catches NaN/Inf)
            if not np.isfinite(state).all():
                result["errors"].append("state contains NaN or Inf")
                result["ok"] = False
            elif np.abs(state).max() > 3 * np.pi:
                result["warnings"].append(
                    f"state max abs = {np.abs(state).max():.3f} rad — unusually large"
                )

            if not np.isfinite(action).all():
                result["errors"].append("action contains NaN or Inf")
                result["ok"] = False

            # Large single-step deltas suggest a discontinuous trajectory
            if result["action_max_delta"] > 0.5:
                result["warnings"].append(
                    f"max joint delta = {result['action_max_delta']:.4f} rad — may be too large"
                )

            # Images should not be all-black (rendering failure)
            ctx_mean = float(f["data/observation.images.context"][:].mean())
            if ctx_mean < 5.0:
                result["errors"].append(
                    f"context images look all-black (mean pixel = {ctx_mean:.1f})"
                )
                result["ok"] = False

            wrist_mean = float(f["data/observation.images.wrist"][:].mean())
            if wrist_mean < 5.0:
                result["errors"].append(
                    f"wrist images look all-black (mean pixel = {wrist_mean:.1f})"
                )
                result["ok"] = False

            # meta
            result["success"] = bool(f["meta/success"][()])

    except Exception as e:
        result["errors"].append(f"could not open file: {e}")
        result["ok"] = False

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Frame grid visualisation
# ══════════════════════════════════════════════════════════════════════════════

def render_frame_grid(paths: list[Path], out_path: Path, n_episodes: int = 5):
    """
    For each sampled episode render 3 context frames (first / mid / last)
    and 3 wrist frames in a grid.  Saves to out_path.
    """
    if not HAS_MPL:
        return

    sample = random.sample(paths, min(n_episodes, len(paths)))
    n = len(sample)
    fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))
    if n == 1:
        axes = axes[None, :]   # keep 2-D indexing

    fig.suptitle("Dataset frame sample — context (left 3) · wrist (right 3)", fontsize=11)

    for row, path in enumerate(sample):
        with h5py.File(path, "r") as f:
            ctx   = f["data/observation.images.context"][:]
            wrist = f["data/observation.images.wrist"][:]
            T     = ctx.shape[0]

        indices = [0, T // 2, T - 1]
        labels  = ["t=0", f"t={T//2}", f"t={T-1}"]

        for col, (idx, lbl) in enumerate(zip(indices, labels)):
            axes[row, col].imshow(ctx[idx])
            axes[row, col].set_title(f"ctx {lbl}", fontsize=7)
            axes[row, col].axis("off")

            axes[row, col + 3].imshow(wrist[idx])
            axes[row, col + 3].set_title(f"wrist {lbl}", fontsize=7)
            axes[row, col + 3].axis("off")

        axes[row, 0].set_ylabel(path.stem, fontsize=7, rotation=0, labelpad=60, va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ok] frame grid saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="demos/baseline/", help="HDF5 episode directory")
    parser.add_argument("--out_csv",   default="verify_results.csv")
    parser.add_argument("--out_img",   default="verify_frames.png")
    parser.add_argument("--n_preview", type=int, default=5, help="episodes to render in frame grid")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files    = sorted(data_dir.glob("episode_*.hdf5"))

    if not files:
        print(f"[error] no episode_*.hdf5 files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} episode files in {data_dir}\n")

    # ── meta.json ────────────────────────────────────────────────────────────
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print("── meta.json ──────────────────────────────────────────")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()
    else:
        print("[warn] meta.json not found\n")

    # ── Per-episode checks ───────────────────────────────────────────────────
    results   = []
    n_ok      = 0
    n_warn    = 0
    n_fail    = 0

    for path in files:
        r = check_episode(path)
        results.append(r)
        if r["ok"] and not r["warnings"]:
            n_ok += 1
        elif r["ok"] and r["warnings"]:
            n_warn += 1
            print(f"[warn]  {r['file']}  T={r['T']}")
            for w in r["warnings"]:
                print(f"        ⚠  {w}")
        else:
            n_fail += 1
            print(f"[FAIL]  {r['file']}")
            for e in r["errors"]:
                print(f"        ✗  {e}")
            for w in r["warnings"]:
                print(f"        ⚠  {w}")

    # ── Summary ──────────────────────────────────────────────────────────────
    all_T     = [r["T"]     for r in results if r["T"]     is not None]
    all_delta = [r["action_max_delta"] for r in results if r["action_max_delta"] is not None]
    n_success = sum(1 for r in results if r["success"])

    print(f"\n── Summary ────────────────────────────────────────────")
    print(f"  Total episodes checked : {len(files)}")
    print(f"  ✓  clean               : {n_ok}")
    print(f"  ⚠  warnings only       : {n_warn}")
    print(f"  ✗  failed checks       : {n_fail}")
    print(f"  Episodes marked success: {n_success} / {len(files)}")
    if all_T:
        print(f"  Frames per episode     : min={min(all_T)}  max={max(all_T)}  mean={np.mean(all_T):.0f}")
    if all_delta:
        print(f"  Max joint delta (rad)  : min={min(all_delta):.4f}  max={max(all_delta):.4f}  mean={np.mean(all_delta):.4f}")

    # ── CSV export ───────────────────────────────────────────────────────────
    import csv
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "ok", "T", "success",
            "state_min", "state_max", "action_max_delta",
            "errors", "warnings",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                **r,
                "errors":   "; ".join(r["errors"]),
                "warnings": "; ".join(r["warnings"]),
            })
    print(f"\n[ok] per-episode CSV saved → {args.out_csv}")

    # ── Failure gate ─────────────────────────────────────────────────────────
    if n_fail > 0:
        print(f"\n[GATE FAIL] {n_fail} episodes have errors. Fix or discard them before fine-tuning.")
        sys.exit(1)
    elif n_success / len(files) < 0.85:
        print(f"\n[GATE WARN] success rate {n_success/len(files):.1%} < 85% — "
              f"consider collecting more demos before fine-tuning.")
    else:
        print(f"\n[GATE PASS] Dataset looks healthy. Proceed to fine-tuning.")

    # ── Frame grid ───────────────────────────────────────────────────────────
    ok_files = [Path(args.data_dir) / r["file"] for r in results if r["ok"]]
    render_frame_grid(ok_files, Path(args.out_img), n_episodes=args.n_preview)


if __name__ == "__main__":
    main()