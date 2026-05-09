#!/usr/bin/env python
"""Evaluate simple fixed-action policies on Crafter validation seeds."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _expand_seeds(seed_spec: Any, n_envs: int, base_seed: int = 0) -> list[int]:
    """Expand the simple seed directives used by VAGEN env yaml files."""
    if isinstance(seed_spec, int):
        seed_spec = [seed_spec]
    seed_spec = [int(x) for x in seed_spec]

    rng = random.Random(base_seed)
    if len(seed_spec) == 1:
        return [seed_spec[0] + i for i in range(n_envs)]
    if len(seed_spec) == 2:
        lo, hi = seed_spec
        return [rng.randint(lo, hi) for _ in range(n_envs)]
    if len(seed_spec) == 3:
        lo, hi, limit = seed_spec
        if limit == 1:
            population = list(range(lo, hi + 1))
            if len(population) < n_envs:
                raise ValueError("seed range is smaller than n_envs")
            if len(population) == n_envs:
                return population
            return rng.sample(population, n_envs)

        counts: dict[int, int] = {}
        seeds: list[int] = []
        while len(seeds) < n_envs:
            candidate = rng.randint(lo, hi)
            if counts.get(candidate, 0) >= limit:
                continue
            counts[candidate] = counts.get(candidate, 0) + 1
            seeds.append(candidate)
        return seeds
    raise ValueError(f"Unsupported seed directive: {seed_spec}")


def _load_crafter_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    envs = cfg.get("envs") or []
    for spec in envs:
        if spec.get("name") == "Crafter":
            return spec
    raise ValueError(f"No Crafter spec found in {path}")


def _format_action(action: str) -> str:
    return f"<think>Fixed policy baseline.</think><answer>{action}</answer>"


def _debug_semantic(info: dict[str, Any], turn: int) -> None:
    """Print semantic map diagnostics for coordinate system verification."""
    from vagen.envs.crafter.utils.utils import MATERIAL_NAMES, OBJECT_NAMES
    sem = info.get("semantic")
    pos = info.get("player_pos")
    if sem is None or pos is None:
        print(f"  [turn {turn}] semantic or player_pos missing")
        return

    def tile_name(tid: int) -> str:
        return MATERIAL_NAMES.get(tid) or OBJECT_NAMES.get(tid, f"id={tid}")

    px, py = int(pos[0]), int(pos[1])
    h, w = sem.shape
    print(f"  [turn {turn}] semantic shape={sem.shape}  player_pos={list(pos)}  => px={px} py={py}")
    print(f"    tile at [py,px]=semantic[{py},{px}] => {tile_name(int(sem[py, px]))}  (expect: player=12)")
    for label, dy, dx in [("North(move_up)", -1, 0), ("South(move_down)", +1, 0),
                           ("West(move_left)", 0, -1), ("East(move_right)", 0, +1)]:
        ny, nx = py + dy, px + dx
        if 0 <= ny < h and 0 <= nx < w:
            print(f"    {label}: semantic[{ny},{nx}] => {tile_name(int(sem[ny, nx]))}")


async def _run_one(seed: int, env_config: dict[str, Any], max_turns: int, actions: list[str],
                   debug_turns: int = 3) -> dict[str, Any]:
    from vagen.envs.crafter.crafter_env import CrafterEnv

    env = CrafterEnv(env_config=env_config)
    total_reward = 0.0
    action_trace: list[str] = []
    reward_trace: list[float] = []
    none_turns = 0
    last_info: dict[str, Any] = {}

    try:
        await env.reset(seed=seed)
        for turn in range(max_turns):
            action = actions[turn % len(actions)]
            _obs, reward, done, info = await env.step(_format_action(action))
            last_info = info
            total_reward += float(reward)
            reward_trace.append(float(reward))
            parsed_actions = info.get("actions", [])
            if parsed_actions:
                action_trace.extend(str(a) for a in parsed_actions)
            else:
                none_turns += 1
            # Print semantic diagnostics for the first few turns of the first seed only
            if debug_turns > 0 and turn < debug_turns:
                print(f"\n--- seed={seed} action={action} reward={reward:.2f} ---")
                _debug_semantic(info, turn)
            if done:
                break
    finally:
        await env.close()

    unlocked_raw = str(last_info.get("achievements_unlocked", ""))
    unlocked = [x for x in unlocked_raw.split("|") if x]
    if not unlocked and hasattr(env, "_unlocked"):
        unlocked = sorted(getattr(env, "_unlocked"))
    num_achievements = int(last_info.get("num_achievements", len(unlocked)) or 0)

    return {
        "seed": seed,
        "total_reward": total_reward,
        "success": num_achievements > 0,
        "num_achievements": num_achievements,
        "achievements": unlocked,
        "actions": action_trace,
        "action_counts": dict(Counter(action_trace)),
        "none_turns": none_turns,
        "turns": len(reward_trace),
        "rewards": reward_trace,
    }


async def _run_all(
    seeds: list[int],
    env_config: dict[str, Any],
    max_turns: int,
    actions: list[str],
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(seed: int, is_first: bool) -> dict[str, Any]:
        async with semaphore:
            # Only print semantic diagnostics for the first seed
            return await _run_one(seed, env_config, max_turns, actions,
                                  debug_turns=3 if is_first else 0)

    return await asyncio.gather(*(guarded(seed, i == 0) for i, seed in enumerate(seeds)))


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    achievement_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for row in results:
        achievement_counts.update(row["achievements"])
        action_counts.update(row["actions"])

    return {
        "n": len(results),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in results),
        "mean_reward": mean(float(row["total_reward"]) for row in results),
        "mean_num_achievements": mean(int(row["num_achievements"]) for row in results),
        "achievement_counts": dict(achievement_counts.most_common()),
        "action_counts": dict(action_counts.most_common()),
        "mean_turns": mean(int(row["turns"]) for row in results),
        "none_turns_total": sum(int(row["none_turns"]) for row in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-yaml",
        type=Path,
        default=Path("examples/train/crafter/val_crafter_vision.yaml"),
        help="Crafter val yaml, relative to cwd or absolute.",
    )
    parser.add_argument("--actions", default="do", help="Comma-separated fixed action cycle, e.g. do or do,move_left.")
    parser.add_argument("--max-envs", type=int, default=None, help="Limit number of validation seeds.")
    parser.add_argument("--max-turns", type=int, default=None, help="Override max turns from yaml.")
    parser.add_argument("--base-seed", type=int, default=0, help="Seed used for expanding sampled seed directives.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--render-mode",
        choices=["text", "vision", "yaml"],
        default="text",
        help="text is fastest and keeps env dynamics/rewards the same.",
    )
    parser.add_argument("--jsonl", type=Path, default=None, help="Optional per-episode output path.")
    args = parser.parse_args()

    spec = _load_crafter_spec(args.val_yaml)
    n_envs = int(spec.get("n_envs", 0))
    seeds = _expand_seeds(spec.get("seed", [0]), n_envs=n_envs, base_seed=args.base_seed)
    if args.max_envs is not None:
        seeds = seeds[: args.max_envs]

    env_config = dict(spec.get("config") or {})
    if args.render_mode != "yaml":
        env_config["render_mode"] = args.render_mode
    max_turns = int(args.max_turns or spec.get("max_turns", 15))
    actions = [x.strip() for x in args.actions.split(",") if x.strip()]
    if not actions:
        raise ValueError("--actions must contain at least one action")

    results = asyncio.run(
        _run_all(
            seeds=seeds,
            env_config=env_config,
            max_turns=max_turns,
            actions=actions,
            concurrency=max(1, int(args.concurrency)),
        )
    )
    summary = _summarize(results)

    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"policy": actions, "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
