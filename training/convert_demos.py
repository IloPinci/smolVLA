"""
convert_demos.py
----------------
Converts the raw HDF5 episodes produced by oracle.py into the LeRobot v2.1
dataset format that lerobot/scripts/train.py expects.

LeRobot v2.1 layout:
    <output_dir>/
        data/chunk-000/
            episode_000000.parquet   ← per-step tabular data (state, action, timestamps)
            episode_000001.parquet
            ...
        videos/chunk-000/
            observation.images.camera1/episode_000000.mp4
            observation.images.camera2/episode_000000.mp4
            observation.images.camera3/episode_000000.mp4
            ...
        meta/
            info.json       ← dataset-level metadata (features, fps, splits …)
            episodes.jsonl  ← one line per episode (index, length, task)
            tasks.jsonl     ← task list

Usage:
    python convert_demos.py \
        --hdf5_dir  demos/baseline/ \
        --out_dir   demos/lerobot/ \
        --fps       30 \
        --val_frac  0.2 \
        --repo_id   local/genesis_pickplace

Requirements: h5py, pandas, pyarrow, imageio[ffmpeg], numpy
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd

# ── Camera keys (must match oracle.py KEY_REMAP) ─────────────────────────────
CAMERA_KEYS = ["camera1", "camera2", "camera3"]
STATE_DIM   = 6
ACTION_DIM  = 6
CHUNK_SIZE  = 1000   # episodes per chunk folder (standard LeRobot convention)


# ══════════════════════════════════════════════════════════════════════════════
#  HDF5 → episode data
# ══════════════════════════════════════════════════════════════════════════════

def load_episode(path: Path) -> dict | None:
    """
    Load one HDF5 episode and return a dict with numpy arrays.
    Returns None if the file is corrupt or has the wrong schema.
    """
    try:
        with h5py.File(path, "r") as f:
            frames = {}
            for key in CAMERA_KEYS:
                ds_key = f"data/observation.images.{key}"
                if ds_key not in f:
                    print(f"  [skip] {path.name}: missing {ds_key}")
                    return None
                frames[key] = f[ds_key][:]          # [T, H, W, 3] uint8

            state  = f["data/observation.state"][:]  # [T, 6] float32
            action = f["data/action"][:]              # [T, 6] float32
            T      = action.shape[0]

            if state.shape[0] != T or any(frames[k].shape[0] != T for k in CAMERA_KEYS):
                print(f"  [skip] {path.name}: temporal misalignment")
                return None

            lang = f["meta/language_instruction"][()].decode("utf-8")
            success = bool(f["meta/success"][()])

        return {
            "frames":  frames,   # dict[str, ndarray]
            "state":   state,
            "action":  action,
            "T":       T,
            "lang":    lang,
            "success": success,
        }
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Write one episode
# ══════════════════════════════════════════════════════════════════════════════

def write_episode(ep_data: dict, ep_idx: int, out_dir: Path, fps: int, task_idx: int = 0):
    """
    Write one episode's Parquet file and per-camera MP4s.
    """
    T    = ep_data["T"]
    chunk_str = f"chunk-{ep_idx // CHUNK_SIZE:03d}"
    ep_str    = f"episode_{ep_idx:06d}"

    # ── Parquet (tabular data) ────────────────────────────────────────────────
    parquet_dir = out_dir / "data" / chunk_str
    parquet_dir.mkdir(parents=True, exist_ok=True)

    timestamps  = np.arange(T, dtype=np.float32) / fps   # seconds
    frame_idxs  = np.arange(T, dtype=np.int64)

    rows = {
        "timestamp":    timestamps,
        "frame_index":  frame_idxs,
        "episode_index": np.full(T, ep_idx, dtype=np.int64),
        "task_index":   np.full(T, task_idx, dtype=np.int64),
        "index":        frame_idxs,   # global step index (filled in meta later)
    }

    # State columns
    for i in range(STATE_DIM):
        rows[f"observation.state.{i}"] = ep_data["state"][:, i]

    # Action columns
    for i in range(ACTION_DIM):
        rows[f"action.{i}"] = ep_data["action"][:, i]

    # Pack state and action as list-columns (LeRobot v2.1 expects array columns)
    df = pd.DataFrame({
        "timestamp":     timestamps,
        "frame_index":   frame_idxs,
        "episode_index": np.full(T, ep_idx, dtype=np.int64),
        "task_index":    np.full(T, task_idx, dtype=np.int64),
        "index":         frame_idxs,
        "observation.state": list(ep_data["state"]),    # list of float32 arrays
        "action":             list(ep_data["action"]),   # list of float32 arrays
        "next.done":     np.concatenate([np.zeros(T - 1, dtype=bool), [True]]),
    })
    df.to_parquet(parquet_dir / f"{ep_str}.parquet", index=False)

    # ── Videos ───────────────────────────────────────────────────────────────
    for cam_key in CAMERA_KEYS:
        obs_key = f"observation.images.{cam_key}"
        vid_dir = out_dir / "videos" / chunk_str / obs_key
        vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path = vid_dir / f"{ep_str}.mp4"

        frames = ep_data["frames"][cam_key]   # [T, H, W, 3] uint8
        iio.imwrite(
            str(vid_path),
            frames,
            fps=fps,
            codec="libx264",
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Meta files
# ══════════════════════════════════════════════════════════════════════════════

def write_meta(out_dir: Path, episodes_meta: list[dict], fps: int,
               lang: str, repo_id: str, val_frac: float):
    """
    Write meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl.
    """
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    n_eps   = len(episodes_meta)
    n_val   = max(1, int(math.floor(n_eps * val_frac)))
    n_train = n_eps - n_val
    total_frames = sum(e["length"] for e in episodes_meta)

    # ── tasks.jsonl ──────────────────────────────────────────────────────────
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": lang}) + "\n")

    # ── episodes.jsonl ───────────────────────────────────────────────────────
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")

    # ── info.json ────────────────────────────────────────────────────────────
    H, W = 256, 256

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": [f"motor_{i}" for i in range(STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [f"motor_{i}" for i in range(ACTION_DIM)],
        },
        "timestamp":     {"dtype": "float32", "shape": [1]},
        "frame_index":   {"dtype": "int64",   "shape": [1]},
        "episode_index": {"dtype": "int64",   "shape": [1]},
        "task_index":    {"dtype": "int64",   "shape": [1]},
        "index":         {"dtype": "int64",   "shape": [1]},
        "next.done":     {"dtype": "bool",    "shape": [1]},
    }

    for cam_key in CAMERA_KEYS:
        features[f"observation.images.{cam_key}"] = {
            "dtype":   "video",
            "shape":   [H, W, 3],
            "names":   ["height", "width", "channel"],
            "video_info": {
                "video.fps":              float(fps),
                "video.codec":            "h264",
                "video.pix_fmt":          "yuv420p",
                "video.is_depth_map":     False,
                "has_audio":              False,
            },
        }

    info = {
        "codebase_version":  "v2.1",
        "robot_type":        "so101",
        "total_episodes":    n_eps,
        "total_frames":      total_frames,
        "total_tasks":       1,
        "total_videos":      n_eps * len(CAMERA_KEYS),
        "total_chunks":      math.ceil(n_eps / CHUNK_SIZE),
        "chunks_size":       CHUNK_SIZE,
        "fps":               fps,
        "splits": {
            "train": f"0:{n_train}",
            "val":   f"{n_train}:{n_eps}",
        },
        "data_path":         "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features":          features,
        "repo_id":           repo_id,
        "tasks":             [lang],
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n── Dataset meta written ──────────────────────────────────")
    print(f"  Total episodes : {n_eps}")
    print(f"  Train / val    : {n_train} / {n_val}")
    print(f"  Total frames   : {total_frames}")
    print(f"  Output         : {out_dir}")

    return n_train, n_val


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir", default="demos/baseline/",
                        help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--out_dir",  default="demos/lerobot/",
                        help="Output directory for LeRobot v2.1 dataset")
    parser.add_argument("--fps",      type=int, default=30,
                        help="Frame rate for video encoding")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of episodes to reserve as validation")
    parser.add_argument("--repo_id",  default="local/genesis_pickplace",
                        help="Dataset identifier written into info.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete out_dir if it already exists")
    args = parser.parse_args()

    hdf5_dir = Path(args.hdf5_dir)
    out_dir  = Path(args.out_dir)

    # ── Sanity check ─────────────────────────────────────────────────────────
    hdf5_files = sorted(hdf5_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        print(f"[error] No episode_*.hdf5 files found in {hdf5_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files in {hdf5_dir}")

    if out_dir.exists():
        if args.overwrite:
            print(f"[info] Removing existing {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[error] {out_dir} already exists. Use --overwrite to replace it.")
            return

    # ── Convert episodes ──────────────────────────────────────────────────────
    episodes_meta = []
    lang          = None
    ep_idx        = 0

    for hdf5_path in hdf5_files:
        print(f"  Converting {hdf5_path.name} → episode {ep_idx:06d} …", end=" ")

        ep_data = load_episode(hdf5_path)
        if ep_data is None:
            continue

        if lang is None:
            lang = ep_data["lang"]

        write_episode(ep_data, ep_idx, out_dir, args.fps)

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks":         [ep_data["lang"]],
            "length":        ep_data["T"],
        })

        print(f"T={ep_data['T']}  success={ep_data['success']}")
        ep_idx += 1

    if ep_idx == 0:
        print("[error] No episodes were converted successfully.")
        return

    # ── Write meta ────────────────────────────────────────────────────────────
    n_train, n_val = write_meta(
        out_dir, episodes_meta, args.fps,
        lang or "Pick up the red block and place it on the green target.",
        args.repo_id, args.val_frac,
    )

    print(f"\n[ok] Conversion complete.")
    print(f"     Pass to finetune.py with:  --dataset_dir {out_dir}")
    print(f"     Train split: episodes 0–{n_train - 1}")
    print(f"     Val   split: episodes {n_train}–{ep_idx - 1}")


if __name__ == "__main__":
    main()