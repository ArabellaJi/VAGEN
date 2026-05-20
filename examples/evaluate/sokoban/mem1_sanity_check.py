"""
MEM1-style sanity check for text-mode POMDP Sokoban.

Tests whether a base LLM can maintain a compact memory state (no full history)
across a POMDP Sokoban episode -- the core premise of MEM1 applied to spatial tasks.

Each step the model receives:  [system_prompt] + [memory_state_t] + [obs_t]
Each step the model outputs:   <think>...</think><answer>action</answer><memory>...</memory>

What to look for in output:
  [GOOD] <memory> tag present every step
  [GOOD] map entries grow over time as new cells are explored
  [GOOD] box/target status changes from "unknown" when they appear in the 3x3 window
  [GOOD] pos updates correctly after each action
  [BAD]  model copies previous memory verbatim without updating
  [BAD]  model hallucinates cells it hasn't seen yet
  [BAD]  pos drifts (wrong coordinate after a sequence of moves)

Usage (SGLang/vLLM server already running):
    cd VAGEN
    python examples/evaluate/sokoban/mem1_sanity_check.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --base-url http://localhost:30000/v1 \\
        --episodes 5

Usage (OpenAI API):
    OPENAI_API_KEY=sk-... python examples/evaluate/sokoban/mem1_sanity_check.py \\
        --model gpt-4o-mini --api-key $OPENAI_API_KEY \\
        --base-url https://api.openai.com/v1 --episodes 3
"""

import asyncio
import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parents[3]))

from openai import AsyncOpenAI
from vagen.envs.sokoban.sokoban_env import Sokoban

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Sokoban solver with limited visibility (3×3 window centered on you).

SYMBOLS:  # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
ACTIONS:  up | down | left | right

COORDINATE SYSTEM:
- Your starting position is (0, 0).
- Row increases downward: "up" → row-1, "down" → row+1.
- Column increases rightward: "left" → col-1, "right" → col+1.
- Your 3×3 window shows cells (row-1,col-1) through (row+1,col+1), with you (P) at the center.

GOAL: Push the box (X) onto the target (O). You cannot pull boxes.
SUCCESS: When you see the symbol √ in the observation, the box is on the target — task complete, stop moving.

─── REQUIRED OUTPUT FORMAT ───────────────────────────────────────────────────
<think>
your reasoning
</think>
<answer>action</answer>
<memory>
pos=(r,c) step=N
map: (r1,c1)=CELL (r2,c2)=CELL ...
box: (r,c)        ← omit this line entirely if box not yet seen
target: (r,c)     ← omit this line entirely if target not yet seen
</memory>
──────────────────────────────────────────────────────────────────────────────

MEMORY UPDATE RULES (apply every step):
1. Read your CURRENT pos=(r,c) from the previous memory.
2. Map the 3×3 observation to absolute coordinates using CURRENT pos:
     top-left=(r-1,c-1)  top=(r-1,c)  top-right=(r-1,c+1)
     left=(r,c-1)         [center=you, skip]  right=(r,c+1)
     bot-left=(r+1,c-1)  bot=(r+1,c)  bot-right=(r+1,c+1)
3. Add ALL visible cells to the map. Keep ALL previously known entries.
4. If you see X or √ in the observation, write "box: (absolute_r,absolute_c)" using the mapping above.
   If you see O or √ or S in the observation, write "target: (absolute_r,absolute_c)" using the mapping above.
   Only write box/target lines if you have seen them. Do NOT write them if unknown.
5. Apply your chosen action to get the NEXT pos and write it:
     pos=(r,c) + up → pos=(r-1,c) | down → pos=(r+1,c) | left → pos=(r,c-1) | right → pos=(r,c+1)
"""

EMPTY_MEMORY = """\
pos=(0,0) step=0
map: (not yet observed)
box=unknown
target=unknown"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict:
    think  = re.search(r"<think>(.*?)</think>",   text, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    memory = re.search(r"<memory>(.*?)</memory>", text, re.DOTALL)
    return {
        "think":      (think.group(1).strip()  if think  else ""),
        "answer":     (answer.group(1).strip() if answer else ""),
        "memory":     (memory.group(1).strip() if memory else EMPTY_MEMORY),
        "has_memory": memory is not None,
    }


def count_map_entries(memory_str: str) -> int:
    m = re.search(r"map:(.*?)(?:\n|$)", memory_str)
    if not m:
        return 0
    line = m.group(1)
    return len(re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", line))


def box_known(memory_str: str) -> bool:
    return bool(re.search(r"box=\(-?\d+\s*,\s*-?\d+\s*\)", memory_str))


def target_known(memory_str: str) -> bool:
    return bool(re.search(r"target=\(-?\d+\s*,\s*-?\d+\s*\)", memory_str))


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step: int
    obs: str
    action: str
    think: str
    memory: str
    has_memory_tag: bool
    map_entries: int
    box_located: bool
    target_located: bool
    reward: float


async def run_episode(
    client: AsyncOpenAI,
    model: str,
    seed: int,
    max_steps: int = 40,
    partial_obs_radius: int = 1,
    verbose: bool = True,
) -> dict:
    env = Sokoban({
        "dim_room": (6, 6),
        "max_steps": max_steps,
        "num_boxes": 1,
        "render_mode": "text",
        "max_actions_per_step": 1,
        "action_sep": ",",
        "prompt_format": "free_think",   # env only needs <answer>
        "partial_obs_radius": partial_obs_radius,
        "use_example_in_sys_prompt": False,
    })

    obs_dict, _ = await env.reset(seed=seed)
    obs_str = obs_dict["obs_str"]

    memory = EMPTY_MEMORY
    total_reward = 0.0
    success = False
    history: List[StepRecord] = []

    sep = "─" * 60

    if verbose:
        print(f"\n{sep}\nEpisode  seed={seed}\n{sep}")
        print(f"[Initial Observation]\n{obs_str}")

    done = False
    step = 0
    while not done and step < max_steps:
        # Build single-turn prompt: system + one user message (memory + obs)
        user_msg = (
            f"[Step {step} — Current Observation]\n"
            f"{obs_str}\n\n"
            f"[Your current memory state]\n"
            f"{memory}\n\n"
            "Update your memory and choose your next action."
        )
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
        ]

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content
        parsed = parse_response(raw)

        action_str = parsed["answer"].lower().strip()
        memory = parsed["memory"]

        rec = StepRecord(
            step=step,
            obs=obs_str,
            action=action_str,
            think=parsed["think"],
            memory=memory,
            has_memory_tag=parsed["has_memory"],
            map_entries=count_map_entries(memory),
            box_located=box_known(memory),
            target_located=target_known(memory),
            reward=0.0,
        )

        # Wrap action in the format the env expects (free_think = <answer>ACTION</answer>)
        env_action_str = f"<think>_</think><answer>{action_str}</answer>"
        obs_dict, reward, done, info = await env.step(env_action_str)
        obs_str = obs_dict["obs_str"]
        rec.reward = reward
        total_reward += reward
        success = info.get("success", False)
        history.append(rec)

        if verbose:
            tag_ok = "[+mem]" if rec.has_memory_tag else "[!mem]"
            box_ok = "box=found" if rec.box_located else "box=?"
            tgt_ok = "tgt=found" if rec.target_located else "tgt=?"
            print(
                f"\nStep {step:2d}  act={action_str:<6}  "
                f"map_cells={rec.map_entries:2d}  {box_ok}  {tgt_ok}  {tag_ok}"
                + (f"  *** reward={reward:.1f} ***" if reward != 0 else "")
            )
            print(f"  Think: {parsed['think'][:100]}")
            print(f"  Memory:\n" + "\n".join("    " + l for l in memory.splitlines()))

        step += 1

    await env.close()

    memory_tag_rate = sum(r.has_memory_tag for r in history) / max(len(history), 1)
    max_map_cells   = max((r.map_entries for r in history), default=0)
    ever_found_box  = any(r.box_located   for r in history)
    ever_found_tgt  = any(r.target_located for r in history)

    return {
        "seed":             seed,
        "steps":            step,
        "success":          success,
        "total_reward":     total_reward,
        "memory_tag_rate":  memory_tag_rate,
        "max_map_cells":    max_map_cells,
        "ever_found_box":   ever_found_box,
        "ever_found_target": ever_found_tgt,
        "history":          history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="MEM1 zero-shot sanity check — POMDP Sokoban")
    parser.add_argument("--model",              default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-url",           default="http://localhost:30000/v1")
    parser.add_argument("--api-key",            default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--episodes",           type=int, default=3)
    parser.add_argument("--seed-start",         type=int, default=42)
    parser.add_argument("--max-steps",          type=int, default=40)
    parser.add_argument("--partial-obs-radius", type=int, default=1,
                        help="0=full board (sanity baseline), 1=3x3 window (POMDP)")
    parser.add_argument("--quiet",              action="store_true",
                        help="Suppress per-step output; show only summary")
    args = parser.parse_args()

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    results = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        result = await run_episode(
            client, args.model, seed,
            max_steps=args.max_steps,
            partial_obs_radius=args.partial_obs_radius,
            verbose=not args.quiet,
        )
        results.append(result)

    # Summary
    n = len(results)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"SUMMARY  model={args.model}  partial_obs_radius={args.partial_obs_radius}")
    print(sep)
    print(f"  Episodes:         {n}")
    print(f"  Success rate:     {sum(r['success'] for r in results)}/{n}")
    print(f"  Avg steps:        {sum(r['steps'] for r in results)/n:.1f}")
    print(f"  Avg reward:       {sum(r['total_reward'] for r in results)/n:.3f}")
    print()
    print(f"  Memory tag rate:  {sum(r['memory_tag_rate'] for r in results)/n:.1%}  (want ~100%)")
    print(f"  Avg max map cells:{sum(r['max_map_cells'] for r in results)/n:.1f}  (want growth over steps)")
    print(f"  Box located:      {sum(r['ever_found_box'] for r in results)}/{n}  (episodes where box pos recorded)")
    print(f"  Target located:   {sum(r['ever_found_target'] for r in results)}/{n}  (episodes where tgt pos recorded)")
    print()
    print("Manual inspection checklist:")
    print("  [ ] pos updates match actions (count moves, verify final offset)")
    print("  [ ] map grows step-by-step (no sudden resets)")
    print("  [ ] box/target coords appear in memory when visible in 3x3 window")
    print("  [ ] model uses memory content in its <think> reasoning")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
