"""
finetune.py
-----------
Launches SmolVLA fine-tuning against the LeRobot v2.1 dataset produced by
convert_demos.py, then patches the output config.json so n_action_steps=50
(required for chunked inference), and writes a finetune_run.json log.

Usage:
    python finetune.py \
        --dataset_dir  demos/lerobot/ \
        --out_dir      outputs/train/smolvla_genesis/ \
        --steps        20000 \
        --batch_size   16 \
        --wandb

The script:
  1. Validates that the dataset directory looks like a LeRobot v2.1 dataset.
  2. Builds the lerobot train command and streams its output live.
  3. After training exits (success or failure), patches n_action_steps in
     every config.json found under the output directory.
  4. Writes finetune_run.json with the run metadata for analyze_finetune.py.

Requirements: lerobot (installed from source with smolvla extras)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(dataset_dir: Path):
    """Light sanity check that the directory looks like a LeRobot v2.1 dataset."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"[error] {info_path} not found.")
        print(f"        Run convert_demos.py first to create the LeRobot dataset.")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    n_eps = info.get("total_episodes", 0)
    fps   = info.get("fps", "?")
    print(f"[ok] Dataset validated: {n_eps} episodes at {fps} fps")
    print(f"     Cameras: {[k for k in info.get('features', {}) if 'images' in k]}")
    return info


def find_lerobot_train():
    """
    Verify lerobot_train is importable and return its module path.
    lerobot 0.5.x uses draccus (dataclass-based config) — overrides must be
    passed as positional `key=value` args, NOT `--key value`.
    """
    result = subprocess.run(
        ["python", "-c", "import lerobot.scripts.lerobot_train"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return "lerobot.scripts.lerobot_train"

    print("[error] Cannot import lerobot.scripts.lerobot_train.")
    print("        Make sure lerobot is installed:  pip install -e '.[smolvla]'")
    sys.exit(1)


def patch_n_action_steps(out_dir: Path, n: int = 50):
    """
    Find every config.json under out_dir and set n_action_steps = n.
    LeRobot saves config.json with n_action_steps=1 by default; the correct
    value for SmolVLA chunked inference is 50.
    """
    configs = list(out_dir.rglob("config.json"))
    if not configs:
        print("[warn] No config.json found under output dir — skipping patch.")
        return

    patched = 0
    for cfg_path in configs:
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)

            if cfg.get("n_action_steps") != n:
                cfg["n_action_steps"] = n
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                patched += 1
                print(f"  [patch] n_action_steps → {n}  in  {cfg_path}")
        except Exception as e:
            print(f"  [warn] Could not patch {cfg_path}: {e}")

    if patched == 0:
        print(f"[info] n_action_steps already = {n} in all config.json files.")
    else:
        print(f"[ok] Patched {patched} config.json file(s).")


def find_last_checkpoint(out_dir: Path) -> Path | None:
    """Return the path to the pretrained_model dir of the last checkpoint."""
    last = out_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last

    # Fall back: find highest-numbered checkpoint
    ckpt_dirs = sorted((out_dir / "checkpoints").glob("*/pretrained_model"))
    return ckpt_dirs[-1] if ckpt_dirs else None


# ══════════════════════════════════════════════════════════════════════════════
#  Build training command
# ══════════════════════════════════════════════════════════════════════════════

def build_command(args) -> list[str]:
    """
    Build the draccus/lerobot_train command for lerobot 0.5.x.

    IMPORTANT: lerobot 0.5.x uses draccus (dataclass-based config), NOT argparse.
    Overrides are passed as positional `key=value` arguments — NO leading `--`.
    Using `--key value` (argparse style) will cause draccus to silently ignore
    the overrides or raise an unrecognised-argument error.

    Key mapping (verified against lerobot 0.5.x TrainPipelineConfig):
      policy.pretrained_path  ← PreTrainedConfig.pretrained_path
      policy.type             ← which policy class to instantiate
      dataset.repo_id         ← DatasetConfig.repo_id
      dataset.root            ← DatasetConfig.root  (local path, no Hub download)
      output_dir              ← TrainPipelineConfig.output_dir
      steps                   ← TrainPipelineConfig.steps
      batch_size              ← TrainPipelineConfig.batch_size
      save_freq               ← TrainPipelineConfig.save_freq
      eval_freq               ← TrainPipelineConfig.eval_freq
      policy.device           ← PreTrainedConfig.device
      wandb.enable            ← WandBConfig.enable
      wandb.project           ← WandBConfig.project
      job_name                ← TrainPipelineConfig.job_name
      resume                  ← TrainPipelineConfig.resume
    """
    dataset_dir = Path(args.dataset_dir).resolve()
    device      = "cpu" if args.cpu else "cuda"

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.pretrained_path=lerobot/smolvla_base",
        f"--policy.type=smolvla",
        f"--policy.push_to_hub=false",
        f"--dataset.repo_id=local/genesis_pickplace",
        f"--dataset.root={dataset_dir}",
        f"--output_dir={args.out_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--save_freq={args.save_freq}",
        f"--policy.device={device}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
        f"--job_name=smolvla_genesis_finetune",
        f"--resume={'true' if args.resume else 'false'}",
    ]

    if args.wandb and args.wandb_project:
        cmd.append(f"--wandb.project={args.wandb_project}")

    if args.eval_freq:
        cmd.append(f"--eval_freq={args.eval_freq}")

    return cmd


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   default="demos/lerobot/",
                        help="LeRobot v2.1 dataset directory (output of convert_demos.py)")
    parser.add_argument("--out_dir",       default="outputs/train/smolvla_genesis/",
                        help="Where to save checkpoints and logs")
    parser.add_argument("--steps",         type=int, default=20000,
                        help="Total training steps")
    parser.add_argument("--batch_size",    type=int, default=16,
                        help="Batch size (reduce if OOM: 8GB→8, 16GB→16, 24GB→32)")
    parser.add_argument("--save_freq",     type=int, default=5000,
                        help="Save a checkpoint every N steps")
    parser.add_argument("--eval_freq",     type=int, default=None,
                        help="Evaluate every N steps (omit to skip sim eval during training)")
    parser.add_argument("--wandb",         action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="smolvla_genesis",
                        help="W&B project name")
    parser.add_argument("--cpu",           action="store_true",
                        help="Force CPU (for debugging only — very slow)")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume training if output_dir already exists")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir     = Path(args.out_dir)

    # ── Validate dataset ──────────────────────────────────────────────────────
    info = validate_dataset(dataset_dir)

    # ── Find train script ─────────────────────────────────────────────────────
    train_script = find_lerobot_train()
    print(f"[ok] Using train script: {train_script}")

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = build_command(args)

    print(f"\n── Training command ──────────────────────────────────────")
    print("  " + " \\\n    ".join(cmd))
    print(f"─────────────────────────────────────────────────────────\n")

    # ── Write run metadata (before launch, so it exists even if training crashes)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "started_at":      datetime.now().isoformat(),
        "dataset_dir":     str(dataset_dir.resolve()),
        "out_dir":         str(out_dir.resolve()),
        "steps":           args.steps,
        "batch_size":      args.batch_size,
        "save_freq":       args.save_freq,
        "base_checkpoint": "lerobot/smolvla_base",
        "n_episodes":      info.get("total_episodes"),
        "fps":             info.get("fps"),
        "wandb":           args.wandb,
        "command":         " ".join(cmd),
        "status":          "running",
        "finished_at":     None,
        "return_code":     None,
        "last_checkpoint": None,
        "n_action_steps_patched": False,
    }
    run_meta_path = out_dir.parent / "finetune_run_pending.json"
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── Launch training ───────────────────────────────────────────────────────
    print(f"[info] Launching training … output streaming below.\n")
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output live
    log_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)

    proc.wait()
    elapsed = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    final_meta_path = out_dir / "finetune_run.json"
    run_meta_path.rename(final_meta_path)
    run_meta_path = final_meta_path

    # ── Save raw log ──────────────────────────────────────────────────────────
    log_path = out_dir / "train_log.txt"
    with open(log_path, "w") as f:
        f.writelines(log_lines)
    print(f"\n[ok] Full training log saved → {log_path}")

    # ── Patch n_action_steps ──────────────────────────────────────────────────
    print(f"\n── Patching config.json (n_action_steps → 50) ────────────")
    patch_n_action_steps(out_dir)
    patched = True

    # ── Find last checkpoint path ─────────────────────────────────────────────
    last_ckpt = find_last_checkpoint(out_dir)

    # ── Update run metadata ───────────────────────────────────────────────────
    run_meta.update({
        "finished_at":            datetime.now().isoformat(),
        "return_code":            proc.returncode,
        "elapsed_seconds":        round(elapsed, 1),
        "status":                 "ok" if proc.returncode == 0 else "failed",
        "last_checkpoint":        str(last_ckpt) if last_ckpt else None,
        "n_action_steps_patched": patched,
    })
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Fine-tuning summary ───────────────────────────────────")
    print(f"  Status          : {'SUCCESS ✓' if proc.returncode == 0 else 'FAILED ✗'}")
    print(f"  Elapsed         : {elapsed / 60:.1f} min")
    print(f"  Last checkpoint : {last_ckpt}")
    print(f"  Run metadata    : {run_meta_path}")
    print(f"─────────────────────────────────────────────────────────")

    if proc.returncode != 0:
        print(f"\n[error] Training exited with code {proc.returncode}.")
        print(f"        Check {log_path} for the full error output.")
        sys.exit(proc.returncode)

    print(f"\n[next] Run the fine-tuned policy evaluation:")
    print(f"       python finetuned_eval.py --checkpoint {last_ckpt}")
    print(f"\n[next] Analyze training curves:")
    print(f"       python analyze_finetune.py --run_dir {out_dir}")


if __name__ == "__main__":
    main()