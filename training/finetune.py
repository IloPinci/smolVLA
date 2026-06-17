"""
finetune.py
-----------
Fine-tunes SmolVLA on a LeRobot v3.0 dataset produced by oracle_direct.py.

Usage:
    python finetune.py \
        --dataset_dir  demos/lerobot/ \
        --repo_id      local/genesis_pickplace \
        --out_dir      outputs/train/smolvla_genesis/ \
        --steps        20000 \
        --batch_size   64

What this script does:
  1. Validates that dataset_dir contains a v3.0 info.json (codebase_version v3.0).
  2. Builds and streams the lerobot-train command (lerobot 0.5.x CLI).
  3. Saves finetune_run.json with full run metadata.

LeRobot 0.5.x training CLI:
    lerobot-train \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=<repo_id> \
        --dataset.root=<local_path>  ← avoids Hub download for local datasets \
        --batch_size=64 \
        --steps=20000 \
        --output_dir=<out_dir> \
        --policy.device=cuda

Key flags:
    --dataset.root   tells the trainer to use local files instead of downloading
                     from the Hub; set to the same path as --dataset_dir.
    --policy.path    the base checkpoint; fine-tuned weights override the head.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(dataset_dir: Path):
    """Check that dataset_dir is a valid LeRobot v3.0 dataset."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"[error] {info_path} not found.")
        print("        Make sure oracle_direct.py ran successfully and called finalize().")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "unknown")
    n_eps   = info.get("total_episodes", 0)
    fps     = info.get("fps", "?")

    if not version.startswith("v3"):
        print(f"[warn] Dataset version is '{version}', expected 'v3.x'.")
        print("       If you have a v2.1 dataset, run:")
        print("         python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=...")

    print(f"[ok] Dataset v{version}: {n_eps} episodes at {fps} fps")
    cam_keys = [k for k in info.get("features", {}) if "images" in k]
    print(f"     Cameras : {cam_keys}")
    print(f"     Task    : {info.get('tasks', ['?'])[0]}")

    if n_eps < 50:
        print(f"\n[warn] Only {n_eps} episodes — SmolVLA recommends ≥50 for reliable performance.")
        print("       Consider collecting more before fine-tuning.")

    return info


def find_last_checkpoint(out_dir: Path) -> Path | None:
    """Return the pretrained_model dir of the last saved checkpoint."""
    last = out_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    ckpt_dirs = sorted((out_dir / "checkpoints").glob("*/pretrained_model"))
    return ckpt_dirs[-1] if ckpt_dirs else None


def build_command(args, dataset_dir: Path) -> list[str]:
    """
    Build the lerobot-train command for lerobot 0.5.x.

    The CLI uses argparse-style --key value flags (NOT draccus key=value).
    --dataset.root tells the trainer to load from disk, not the Hub.
    """
    device = "cpu" if args.cpu else "cuda"

    cmd = [
        "lerobot-train",
        f"--policy.path=lerobot/smolvla_base",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_dir}",      # ← local v3.0 root, skips Hub download
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--output_dir={args.out_dir}",
        f"--job_name=smolvla_genesis_finetune",
        f"--policy.device={device}",
        f"--save_freq={args.save_freq}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
    ]

    if args.wandb and args.wandb_project:
        cmd.append(f"--wandb.project={args.wandb_project}")

    if args.resume:
        cmd.append("--resume=true")

    return cmd


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLA on a local LeRobot v3.0 dataset."
    )
    parser.add_argument("--dataset_dir",   default="demos/lerobot/",
                        help="Local LeRobot v3.0 dataset (output of oracle_direct.py)")
    parser.add_argument("--repo_id",       default="local/genesis_pickplace",
                        help="Must match the repo_id used during collection")
    parser.add_argument("--out_dir",       default="outputs/train/smolvla_genesis/",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--steps",         type=int, default=20000)
    parser.add_argument("--batch_size",    type=int, default=64,
                        help="64 fits on an A100; reduce to 16–32 for smaller GPUs")
    parser.add_argument("--save_freq",     type=int, default=5000)
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="smolvla_genesis")
    parser.add_argument("--cpu",           action="store_true",
                        help="Force CPU — very slow, for debugging only")
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir     = Path(args.out_dir)

    # ── Validate ──────────────────────────────────────────────────────────────
    info = validate_dataset(dataset_dir)

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = build_command(args, dataset_dir)

    print(f"\n── Training command ──────────────────────────────────────")
    print("  " + " \\\n    ".join(cmd))
    print(f"─────────────────────────────────────────────────────────\n")

    # ── Write pending run metadata ────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "started_at":      datetime.now().isoformat(),
        "dataset_dir":     str(dataset_dir),
        "repo_id":         args.repo_id,
        "out_dir":         str(out_dir.resolve()),
        "steps":           args.steps,
        "batch_size":      args.batch_size,
        "save_freq":       args.save_freq,
        "base_checkpoint": "lerobot/smolvla_base",
        "n_episodes":      info.get("total_episodes"),
        "fps":             info.get("fps"),
        "dataset_version": info.get("codebase_version"),
        "wandb":           args.wandb,
        "command":         " ".join(cmd),
        "status":          "running",
        "finished_at":     None,
        "return_code":     None,
        "last_checkpoint": None,
    }
    pending_path = out_dir / "finetune_run_pending.json"
    with open(pending_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── Launch ────────────────────────────────────────────────────────────────
    print("[info] Launching lerobot-train …\n")
    t0   = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)

    proc.wait()
    elapsed = time.time() - t0

    # ── Save log ──────────────────────────────────────────────────────────────
    log_path = out_dir / "train_log.txt"
    with open(log_path, "w") as f:
        f.writelines(log_lines)
    print(f"\n[ok] Training log → {log_path}")

    # ── Finalise metadata ─────────────────────────────────────────────────────
    final_path = out_dir / "finetune_run.json"
    pending_path.rename(final_path)
    last_ckpt = find_last_checkpoint(out_dir)

    run_meta.update({
        "finished_at":     datetime.now().isoformat(),
        "return_code":     proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "status":          "ok" if proc.returncode == 0 else "failed",
        "last_checkpoint": str(last_ckpt) if last_ckpt else None,
    })
    with open(final_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    status = "SUCCESS ✓" if proc.returncode == 0 else "FAILED ✗"
    print(f"\n── Fine-tuning summary ───────────────────────────────────")
    print(f"  Status          : {status}")
    print(f"  Elapsed         : {elapsed / 60:.1f} min")
    print(f"  Last checkpoint : {last_ckpt}")
    print(f"  Run metadata    : {final_path}")
    print(f"─────────────────────────────────────────────────────────")

    if proc.returncode != 0:
        print(f"\n[error] Training failed (exit code {proc.returncode}).")
        print(f"        Full log: {log_path}")
        sys.exit(proc.returncode)

    print(f"\n[next] Evaluate the fine-tuned model:")
    print(f"       python zeroshot_eval.py --checkpoint {last_ckpt}")


if __name__ == "__main__":
    main()