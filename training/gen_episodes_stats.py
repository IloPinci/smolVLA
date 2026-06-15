"""
gen_episodes_stats.py
----------------------
Generates the missing meta/episodes_stats.jsonl for a v2.1 dataset written
by oracle_direct.py / convert_demos.py, so convert_dataset_v21_to_v30 can run.

Usage:
    python gen_episodes_stats.py --root demos/lerobot
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import jsonlines
import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import get_feature_stats


def compute_episode_stats(ep_idx, parquet_path, root, features):
    df = pd.read_parquet(parquet_path)
    ep_stats = {}

    for key, spec in features.items():
        dtype = spec["dtype"]
        if dtype in ("string", "language"):
            continue

        if dtype == "video":
            vid_path = root / "videos" / "chunk-000" / key / f"episode_{ep_idx:06d}.mp4"
            frames = iio.imread(str(vid_path))            # [T, H, W, 3] uint8
            T = frames.shape[0]
            idxs = np.linspace(0, T - 1, min(T, 32)).astype(int)
            arr = frames[idxs].transpose(0, 3, 1, 2).astype(np.float64)  # [N,3,H,W]
            stats = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
            stats = {
                k: (v if k == "count" else np.squeeze(v / 255.0, axis=0))
                for k, v in stats.items()
            }
        else:
            if key not in df.columns:
                continue
            arr = np.stack(df[key].to_numpy()).astype(np.float64)
            axis = 0
            keepdims = arr.ndim == 1
            stats = get_feature_stats(arr, axis=axis, keepdims=keepdims)

        ep_stats[key] = {k: np.asarray(v).tolist() for k, v in stats.items()}

    return ep_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="demos/lerobot")
    args = parser.parse_args()

    root = Path(args.root)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    features = info["features"]

    episode_files = sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))
    print(f"Found {len(episode_files)} episodes")

    out_path = root / "meta" / "episodes_stats.jsonl"
    with jsonlines.open(out_path, "w") as writer:
        for ep_idx, pq in enumerate(episode_files):
            stats = compute_episode_stats(ep_idx, pq, root, features)
            writer.write({"episode_index": ep_idx, "stats": stats})
            print(f"  [{ep_idx + 1}/{len(episode_files)}] done", end="\r")

    print(f"\n[ok] wrote {len(episode_files)} entries -> {out_path}")


if __name__ == "__main__":
    main()