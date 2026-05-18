"""
Quick local verification for partial_obs_radius in Sokoban.

Runs in TEXT mode (no GPU needed). For each seed:
  - Shows the full 6x6 grid alongside the 3x3 partial view
  - Verifies the player (P) is always in the center cell
  - Takes a few random steps and checks the view follows the player

Also optionally saves vision mode crops to disk.

Usage:
    python diagnose_pomdp.py
    python diagnose_pomdp.py --seeds 10000 10001 10002 --vision --save_dir ./pomdp_test
"""

import asyncio
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from vagen.envs.sokoban.sokoban_env import Sokoban


BASE_CFG = dict(
    dim_room=(6, 6),
    max_steps=100,
    num_boxes=1,
    max_actions_per_step=3,
    action_sep=",",
    use_example_in_sys_prompt=False,
    prompt_format="wm",
)

STEP_SEQUENCE = ["up", "right", "down", "left", "up"]


def divider(label: str = "") -> str:
    return f"\n{'─' * 50}  {label}\n" if label else "\n" + "─" * 50


def side_by_side(left_lines: list[str], right_lines: list[str], left_title: str, right_title: str) -> str:
    max_left_w = max(len(l) for l in [left_title] + left_lines)
    gap = 6
    out = []
    out.append(left_title.ljust(max_left_w) + " " * gap + right_title)
    n = max(len(left_lines), len(right_lines))
    for i in range(n):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        out.append(l.ljust(max_left_w) + " " * gap + r)
    return "\n".join(out)


def check_player_centered(obs_str: str) -> tuple[bool, str]:
    """Check that P is in the center cell of the 3x3 grid."""
    lines = [l for l in obs_str.strip().split("\n") if l.strip()]
    if len(lines) != 3:
        return False, f"Expected 3 rows, got {len(lines)}"
    center_row = lines[1]
    cells = [c.strip() for c in center_row.split() if c.strip()]
    if len(cells) != 3:
        return False, f"Expected 3 cells in center row, got: {center_row!r}"
    center_cell = cells[1]
    ok = center_cell in ("P", "S")
    return ok, f"center cell = '{center_cell}'"


async def run_seed_text(seed: int, verbose: bool = True) -> dict:
    """Run one seed in text mode, full obs vs partial obs."""
    env_full = Sokoban({**BASE_CFG, "render_mode": "text", "partial_obs_radius": 0})
    env_part = Sokoban({**BASE_CFG, "render_mode": "text", "partial_obs_radius": 1})

    obs_f, _ = await env_full.reset(seed=seed)
    obs_p, _ = await env_part.reset(seed=seed)

    results = []

    if verbose:
        print(divider(f"Seed {seed} — initial"))

    for step_i in range(len(STEP_SEQUENCE) + 1):
        full_grid = obs_f["obs_str"].strip()
        part_grid = obs_p["obs_str"].strip()

        # Extract just the grid lines (skip template text)
        full_lines = [l for l in full_grid.split("\n") if any(c in l for c in "#_OXPSsv√")]
        part_lines = [l for l in part_grid.split("\n") if any(c in l for c in "#_OXPSsv√")]

        player_ok, note = check_player_centered("\n".join(part_lines))
        results.append({"step": step_i, "player_centered": player_ok, "note": note})

        status = "OK" if player_ok else "FAIL"
        if verbose:
            label = f"Step {step_i} [{status}] {note}"
            print(side_by_side(full_lines, part_lines, "Full 6×6", "Partial 3×3"))
            print(f"  → {label}")

        if step_i < len(STEP_SEQUENCE):
            action = STEP_SEQUENCE[step_i]
            # Build a minimal valid response string
            action_str = f"<observation>test</observation><think>moving</think><answer>{action}</answer><prediction>test</prediction>"
            obs_f, _, done_f, _ = await env_full.step(action_str)
            obs_p, _, done_p, _ = await env_part.step(action_str)
            if verbose:
                print(f"\n  → action: {action}")
            if done_f or done_p:
                if verbose:
                    print("  → episode done early")
                break

    await env_full.close()
    await env_part.close()
    return {"seed": seed, "steps": results, "all_centered": all(r["player_centered"] for r in results)}


async def run_seed_vision(seed: int, save_dir: str) -> None:
    """Save full and partial vision crops for visual inspection."""
    env_full = Sokoban({**BASE_CFG, "render_mode": "vision", "render_scale": 4, "partial_obs_radius": 0})
    env_part = Sokoban({**BASE_CFG, "render_mode": "vision", "render_scale": 4, "partial_obs_radius": 1})

    obs_f, _ = await env_full.reset(seed=seed)
    obs_p, _ = await env_part.reset(seed=seed)

    seed_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    placeholder = env_full.config.image_placeholder
    img_full = obs_f["multi_modal_input"][placeholder][0]
    img_part = obs_p["multi_modal_input"][placeholder][0]

    img_full.save(os.path.join(seed_dir, "step00_full.png"))
    img_part.save(os.path.join(seed_dir, "step00_partial.png"))

    for step_i, action in enumerate(STEP_SEQUENCE, start=1):
        action_str = f"<observation>test</observation><think>moving</think><answer>{action}</answer><prediction>test</prediction>"
        obs_f, _, done_f, _ = await env_full.step(action_str)
        obs_p, _, done_p, _ = await env_part.step(action_str)

        img_full = obs_f["multi_modal_input"][placeholder][0]
        img_part = obs_p["multi_modal_input"][placeholder][0]
        img_full.save(os.path.join(seed_dir, f"step{step_i:02d}_full.png"))
        img_part.save(os.path.join(seed_dir, f"step{step_i:02d}_partial.png"))

        if done_f or done_p:
            break

    await env_full.close()
    await env_part.close()
    print(f"  Vision images saved → {seed_dir}/")


async def main(seeds: list[int], vision: bool, save_dir: str) -> None:
    print("=" * 60)
    print("  POMDP Partial Observation Diagnostic")
    print("=" * 60)
    print(f"  Seeds: {seeds}")
    print(f"  Mode:  text (always) {'+ vision' if vision else ''}")
    print()

    # ── System prompt check ──────────────────────────────────────
    env_check = Sokoban({**BASE_CFG, "render_mode": "text", "partial_obs_radius": 1})
    sp = env_check.get_system_prompt()
    has_pomdp_note = "limited visibility" in sp
    print(f"System prompt contains POMDP description: {'YES' if has_pomdp_note else 'NO ← MISSING'}")
    print()
    await env_check.close()

    # ── Text mode per-seed ───────────────────────────────────────
    all_ok = True
    for seed in seeds:
        result = await run_seed_text(seed, verbose=True)
        ok = result["all_centered"]
        all_ok = all_ok and ok
        print(f"\nSeed {seed}: {'PASS' if ok else 'FAIL'} "
              f"({sum(r['player_centered'] for r in result['steps'])}/{len(result['steps'])} steps centered)")

    # ── Vision mode ───────────────────────────────────────────────
    if vision:
        print(divider("Vision mode"))
        os.makedirs(save_dir, exist_ok=True)
        for seed in seeds:
            await run_seed_vision(seed, save_dir)

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Result: {'ALL PASS' if all_ok else 'SOME FAILED — check output above'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001, 10002])
    parser.add_argument("--vision", action="store_true", help="Also save vision mode PNG crops")
    parser.add_argument("--save_dir", default="./pomdp_test", help="Directory for vision images")
    args = parser.parse_args()

    asyncio.run(main(args.seeds, args.vision, args.save_dir))
