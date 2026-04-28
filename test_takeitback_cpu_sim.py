"""
Test whether TakeItBack-v0 can run with sim_backend="cpu" + render_backend="gpu".

Goal: verify that Ray CPU workers can use SAPIEN without conflicting with SGLang's
CUDA context. The key is sim_backend="cpu" avoids initialising a CUDA context for
physics, while Vulkan rendering (render_backend="gpu") does NOT share CUDA context
with PyTorch/SGLang.

Run on a GPU node (for Vulkan rendering):
  srun --gres=gpu:h100:1 --cpus-per-task=4 --mem=32G --pty bash
  conda activate vagen_noflash
  python test_takeitback_cpu_sim.py

Expected output:
  [1] import OK
  [2] env created OK  (no sapien.Device("cuda") crash)
  [3] reset OK  obs keys: ...
  [4] image shape: (H, W, 3)  saved to /tmp/takeitback_frame.png
  [5] step OK  reward: ...
  PASS: sim_backend=cpu works for TakeItBack-v0
"""

import sys
import traceback

# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------
try:
    import mikasa_robo_suite  # noqa: F401
    print("[1] import mikasa_robo_suite OK")
except Exception as e:
    print(f"[1] FAIL import: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Create env with sim_backend="cpu"
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    import numpy as np

    # Mirror PrimitiveSkill's approach:
    #   sim_backend="cpu"   → physics on CPU, no CUDA context in this process
    #   render_backend="gpu" → Vulkan rendering, independent of CUDA
    env = gym.make(
        "TakeItBack-v0",
        obs_mode="rgb",          # request RGB images
        sim_backend="cpu",
        render_backend="gpu",
        num_envs=1,
    )
    print("[2] env created OK")
except Exception as e:
    print(f"[2] FAIL create env: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Reset
# ---------------------------------------------------------------------------
try:
    obs, info = env.reset(seed=42)
    print(f"[3] reset OK  obs type: {type(obs)}")
    if hasattr(obs, "keys"):
        print(f"         obs keys: {list(obs.keys())}")
    elif isinstance(obs, dict):
        print(f"         obs keys: {list(obs.keys())}")
    else:
        print(f"         obs shape: {getattr(obs, 'shape', '?')}")
except Exception as e:
    print(f"[3] FAIL reset: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Render image
# ---------------------------------------------------------------------------
try:
    frame = env.render()  # returns np.ndarray (H, W, 3) for render_mode="rgb_array"
    if frame is None:
        # Some ManiSkill envs return None from render() when obs_mode=rgb;
        # try extracting image from obs dict directly.
        if isinstance(obs, dict) and "rgb" in obs:
            import torch
            frame_t = obs["rgb"]
            if isinstance(frame_t, torch.Tensor):
                frame = frame_t[0].cpu().numpy()  # (H, W, 3)
            else:
                frame = np.array(frame_t[0])
        elif hasattr(obs, "rgb"):
            frame = np.array(obs.rgb[0].cpu())

    if frame is not None:
        print(f"[4] image shape: {frame.shape}  dtype: {frame.dtype}")
        from PIL import Image
        img = Image.fromarray(frame.astype(np.uint8))
        img.save("/tmp/takeitback_frame.png")
        print("     saved to /tmp/takeitback_frame.png")
    else:
        print("[4] WARNING: render() returned None and no rgb in obs — check obs_mode")
except Exception as e:
    print(f"[4] FAIL render: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 5. Step with a zero action
# ---------------------------------------------------------------------------
try:
    action_space = env.action_space
    action = action_space.sample() * 0.0   # zero action
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"[5] step OK  reward: {reward}  terminated: {terminated}  truncated: {truncated}")
except Exception as e:
    print(f"[5] FAIL step: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 6. Check: did we accidentally create a CUDA context?
# ---------------------------------------------------------------------------
try:
    import torch
    if torch.cuda.is_initialized():
        print("[6] NOTE: CUDA context WAS initialised in this process")
        print("         (Vulkan renderer may have triggered it — check if it conflicts)")
    else:
        print("[6] CUDA context NOT initialised in this process — sim_backend=cpu is clean")
except Exception as e:
    print(f"[6] could not check CUDA context: {e}")

env.close()
print("\nPASS: sim_backend=cpu test complete for TakeItBack-v0")
