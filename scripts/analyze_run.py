"""
Analyze a completed VAGEN training run from console logs and JSONL rollout files.

Usage:
    python scripts/analyze_run.py \
        --log /home/eiu4164/projects/VAGEN/logs/vagen_grpo_sokoban_3b_5722481.out \
        --rollout_dir /projects/p33224/vagen_runs/rollout/sokoban_grpo_sglang_disk_3b \
        --val_dir /projects/p33224/vagen_runs/validation/sokoban_grpo_sglang_disk_3b \
        [--plot]        # save plots as PNG (requires matplotlib)
        [--samples N]   # print N example trajectories from final validation (default: 3)
"""

import argparse
import json
import re
import os
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Parse .out log → extract per-step metrics
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "critic/score/mean",
    "actor/entropy",
    "custom_metrics/train/reward_variance",
    "response_length/mean",
    "num_turns/mean",
    "actor/grad_norm",
    "actor/lr",
    # validation (only present at test_freq steps)
    "val-core/sokoban/reward/mean@1",
    "val-aux/sokoban/traj_success/mean@1",
    "custom_metrics/val/reward_variance",
]


def parse_log(log_path: str) -> dict[str, list]:
    """Extract per-step metrics from the .out log."""
    steps = []
    metrics: dict[str, list] = defaultdict(list)

    step_pattern = re.compile(r"step:(\d+) - (.+)")
    kv_pattern = re.compile(r"([\w/\-@]+):(np\.float64\(|np\.int32\()?([-\d.e+]+)\)?")

    with open(log_path) as f:
        for line in f:
            m = step_pattern.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kv_str = m.group(2)
            row = {}
            for km in kv_pattern.finditer(kv_str):
                row[km.group(1)] = float(km.group(3))
            if row:
                steps.append(step)
                for k in METRIC_KEYS:
                    metrics[k].append(row.get(k, None))

    metrics["step"] = steps
    return dict(metrics)


# ---------------------------------------------------------------------------
# 2. Plot training curves
# ---------------------------------------------------------------------------

def plot_curves(metrics: dict, out_dir: str):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    steps = metrics["step"]

    panels = [
        ("critic/score/mean",                    "Train reward (score/mean)"),
        ("actor/entropy",                         "Policy entropy"),
        ("custom_metrics/train/reward_variance",  "Within-group reward variance"),
        ("response_length/mean",                  "Response length (mean tokens)"),
        ("num_turns/mean",                        "Avg turns per episode"),
        ("actor/grad_norm",                       "Gradient norm"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, panels):
        vals = metrics.get(key, [])
        clean = [(s, v) for s, v in zip(steps, vals) if v is not None]
        if clean:
            xs, ys = zip(*clean)
            ax.plot(xs, ys)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    print(f"Saved training curves → {path}")

    # Validation metrics (sparse)
    val_keys = [
        ("val-core/sokoban/reward/mean@1",       "Val reward"),
        ("val-aux/sokoban/traj_success/mean@1",  "Val traj success rate"),
    ]
    val_clean = {}
    for k, _ in val_keys:
        val_clean[k] = [(s, v) for s, v in zip(steps, metrics.get(k, [])) if v is not None]

    if any(val_clean.values()):
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (k, title) in zip(axes2, val_keys):
            if val_clean[k]:
                xs, ys = zip(*val_clean[k])
                ax.plot(xs, ys, marker="o")
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path2 = os.path.join(out_dir, "val_curves.png")
        plt.savefig(path2, dpi=120)
        print(f"Saved validation curves → {path2}")


# ---------------------------------------------------------------------------
# 3. Print summary statistics from metrics
# ---------------------------------------------------------------------------

def print_summary(metrics: dict):
    steps = metrics["step"]
    if not steps:
        print("No steps found in log.")
        return

    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY  ({len(steps)} steps, {steps[0]}→{steps[-1]})")
    print(f"{'='*60}")

    def _summarize(key, label):
        vals = [v for v in metrics.get(key, []) if v is not None]
        if not vals:
            return
        first, last = vals[0], vals[-1]
        peak = max(vals)
        print(f"  {label:<40}  first={first:.4f}  last={last:.4f}  peak={peak:.4f}")

    _summarize("critic/score/mean",                   "Train reward (mean)")
    _summarize("actor/entropy",                        "Policy entropy")
    _summarize("custom_metrics/train/reward_variance", "Reward variance (train)")
    _summarize("response_length/mean",                 "Response length (mean)")
    _summarize("num_turns/mean",                       "Avg turns / episode")
    _summarize("actor/grad_norm",                      "Grad norm")

    # Validation
    val_reward = [(s, v) for s, v in zip(steps, metrics.get("val-core/sokoban/reward/mean@1", [])) if v is not None]
    val_succ   = [(s, v) for s, v in zip(steps, metrics.get("val-aux/sokoban/traj_success/mean@1", [])) if v is not None]
    if val_reward:
        print(f"\n  Validation checkpoints:")
        for (s, r), (_, sc) in zip(val_reward, val_succ):
            print(f"    step {s:>4}  reward={r:.4f}  traj_success={sc:.4f}")


# ---------------------------------------------------------------------------
# 4. Inspect JSONL trajectories
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_samples(entries: list[dict], n: int, step_label: str):
    print(f"\n{'='*60}")
    print(f"SAMPLE TRAJECTORIES  [{step_label}]  (showing {min(n, len(entries))}/{len(entries)})")
    print(f"{'='*60}")

    scores = [e["score"] for e in entries if "score" in e]
    if scores:
        import statistics
        print(f"Score stats:  mean={statistics.mean(scores):.4f}  "
              f"median={statistics.median(scores):.4f}  "
              f"min={min(scores):.4f}  max={max(scores):.4f}")

    # Sort descending by score; show top/bottom mix
    sorted_entries = sorted(entries, key=lambda e: e.get("score", 0), reverse=True)
    sample_idx = []
    if len(sorted_entries) >= n:
        half = n // 2
        sample_idx = list(range(half)) + list(range(len(sorted_entries) - (n - half), len(sorted_entries)))
    else:
        sample_idx = list(range(len(sorted_entries)))

    for rank, idx in enumerate(sample_idx):
        e = sorted_entries[idx]
        score = e.get("score", "?")
        gt = e.get("gts", "?")
        output = e.get("output", "")
        label = "TOP" if rank < len(sample_idx) // 2 else "BOTTOM"
        print(f"\n--- [{label}] score={score}  gt={gt} ---")
        # Truncate long outputs
        if len(output) > 800:
            print(output[:400] + "\n  [...truncated...]\n" + output[-200:])
        else:
            print(output)


def score_distribution(entries: list[dict]):
    from collections import Counter
    buckets = Counter()
    for e in entries:
        s = e.get("score", None)
        if s is not None:
            bucket = round(s * 10) / 10  # round to 0.1
            buckets[bucket] += 1
    print("\n  Score distribution (rounded to 0.1):")
    for k in sorted(buckets):
        bar = "#" * buckets[k]
        print(f"    {k:.1f}  {bar}  ({buckets[k]})")


# ---------------------------------------------------------------------------
# 5. Compare early vs late rollout reward
# ---------------------------------------------------------------------------

def compare_early_late(rollout_dir: str, n_sample: int = 5):
    rollout_path = Path(rollout_dir)
    jsonl_files = sorted(rollout_path.glob("*.jsonl"), key=lambda p: int(p.stem))
    if len(jsonl_files) < 2:
        return

    early_files = jsonl_files[:n_sample]
    late_files  = jsonl_files[-n_sample:]

    def mean_score(files):
        scores = []
        for f in files:
            for e in load_jsonl(str(f)):
                if "score" in e:
                    scores.append(e["score"])
        return sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'='*60}")
    print("EARLY vs LATE ROLLOUT REWARD")
    print(f"{'='*60}")
    print(f"  Early steps ({[f.stem for f in early_files]}):  mean reward = {mean_score(early_files):.4f}")
    print(f"  Late  steps ({[f.stem for f in late_files]}):  mean reward = {mean_score(late_files):.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help=".out log file path")
    parser.add_argument("--rollout_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--plot", action="store_true", help="Save PNG plots")
    parser.add_argument("--plot_dir", default="./analysis_plots")
    parser.add_argument("--samples", type=int, default=3, help="Validation trajectories to print")
    args = parser.parse_args()

    # 1. Parse log
    print(f"Parsing log: {args.log}")
    metrics = parse_log(args.log)
    print_summary(metrics)

    # 2. Plots
    if args.plot:
        plot_curves(metrics, args.plot_dir)

    # 3. Early vs late rollout
    if args.rollout_dir and Path(args.rollout_dir).exists():
        compare_early_late(args.rollout_dir)

    # 4. Validation JSONL samples
    if args.val_dir:
        val_path = Path(args.val_dir)
        # Find latest validation file
        val_files = sorted(val_path.glob("*.jsonl"), key=lambda p: int(p.stem))
        if val_files:
            latest = val_files[-1]
            print(f"\nLoading validation file: {latest}")
            entries = load_jsonl(str(latest))
            score_distribution(entries)
            print_samples(entries, n=args.samples, step_label=f"step {latest.stem}")


if __name__ == "__main__":
    main()
