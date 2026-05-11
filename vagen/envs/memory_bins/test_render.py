"""
Local rendering test for MemoryBinsEnv.
Saves images for key visual states and a thumbnail comparison.
Run: python -m vagen.envs.memory_bins.test_render
"""
import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from PIL import Image
from vagen.envs.memory_bins.memory_bins_env import MemoryBinsEnv

SAVE_DIR = os.path.join(os.path.dirname(__file__), "test_renders")
THUMB_SCALE = 0.25


def save(img: Image.Image, name: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name)
    img.save(path)
    # thumbnail version alongside
    w, h = img.size
    thumb = img.resize((int(w * THUMB_SCALE), int(h * THUMB_SCALE)), Image.BILINEAR)
    thumb.save(path.replace(".png", "_thumb.png"))
    print(f"  saved {path}  ({img.size[0]}x{img.size[1]}) + thumb ({thumb.size[0]}x{thumb.size[1]})")


def get_img(obs):
    return obs["multi_modal_input"]["<image>"][0]


async def main():
    env = MemoryBinsEnv({"n_bins": 5, "n_instructions": 3})
    obs, _ = await env.reset(seed=42)
    # seed 42: bins (1,1)=blue (1,5)=red (3,3)=cyan (5,1)=green (5,5)=lime
    # instructions: red, lime, cyan
    print("instructions:", env.instructions)
    print("bin_contents:", {str(k): v for k, v in env.bin_contents.items()})
    print()

    async def go(a):
        o, rew, done, inf = await env.step(f"<think>.</think><answer>{a}</answer>")
        return o, rew, done, inf

    # 1. Initial state (all bins closed)
    print("1. Initial state — all bins closed")
    save(get_img(obs), "01_initial.png")

    # 2. Navigate to (0,5), open red bin at (1,5)
    print("2. Open correct bin (red)")
    for _ in range(5): await go("move_right")
    obs_open, _, _, _ = await go("open_bin")
    save(get_img(obs_open), "02_open_red_bin.png")

    # 3. State after bin closes — red bin now hidden empty
    print("3. After bin closes — hidden empty (looks closed)")
    obs_closed, _, _, _ = await go("move_left")
    save(get_img(obs_closed), "03_after_red_collected.png")

    # 4. Re-open the now-empty red bin to see empty indicator
    print("4. Re-open empty red bin — should show X")
    await go("move_right")                              # back to (0,5)
    obs_empty, _, _, _ = await go("open_bin")
    save(get_img(obs_empty), "04_reopen_empty_bin.png")

    # 5. Navigate to lime bin (5,5), open it
    print("5. Open lime bin")
    for _ in range(2): await go("move_right")           # (0,7)
    for _ in range(5): await go("move_down")            # (5,7)
    await go("move_left")                               # (5,6)
    obs_lime, _, _, _ = await go("open_bin")
    save(get_img(obs_lime), "05_open_lime_bin.png")

    # 6. Navigate to cyan bin (3,3), open it
    print("6. Open cyan bin")
    for _ in range(2): await go("move_up")              # (3,6)
    for _ in range(2): await go("move_left")            # (3,4)
    obs_cyan, _, done, inf = await go("open_bin")
    save(get_img(obs_cyan), "06_open_cyan_bin_done.png")
    print(f"   completed={inf['completed']}  done={done}")

    print(f"\nAll images saved to: {SAVE_DIR}")
    print("Check *_thumb.png files to verify colors are distinguishable at 25% scale.")


if __name__ == "__main__":
    asyncio.run(main())
