"""
Test TakeItBack-v0 with sim_backend="cpu":
  - Read images from obs["sensor_data"] (not env.render())
  - Diagnose whether CUDA context comes from SAPIEN or just torch import
  - Check if render_backend="cpu" avoids CUDA initialisation entirely

Run on a GPU node:
  srun --gres=gpu:h100:1 --cpus-per-task=4 --mem=32G --pty bash
  conda activate vagen_noflash
  cd /home/eiu4164/projects/VAGEN
  python test_takeitback_cpu_sim.py
"""

import sys
import traceback
import numpy as np

# Check CUDA state BEFORE any imports
import torch
cuda_before_import = torch.cuda.is_initialized()
print(f"[pre] CUDA initialised before env import: {cuda_before_import}")

# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------
try:
    import mikasa_robo_suite  # noqa: F401
    import gymnasium as gym
    print("[1] import OK")
except Exception as e:
    print(f"[1] FAIL import: {e}")
    sys.exit(1)

cuda_after_import = torch.cuda.is_initialized()
print(f"[1b] CUDA initialised after mikasa import: {cuda_after_import}")

# ---------------------------------------------------------------------------
# 2. Create env — try render_backend="cpu" first to avoid Vulkan/CUDA init
# ---------------------------------------------------------------------------
env = None
for render_backend in ["cpu", "gpu"]:
    try:
        cuda_before_make = torch.cuda.is_initialized()
        env = gym.make(
            "TakeItBack-v0",
            obs_mode="rgb",
            sim_backend="cpu",
            render_backend=render_backend,
            render_mode="rgb_array",   # needed for env.render()
            num_envs=1,
        )
        cuda_after_make = torch.cuda.is_initialized()
        print(f"[2] env created OK  render_backend={render_backend}")
        print(f"    CUDA before make: {cuda_before_make}  after make: {cuda_after_make}")
        if cuda_after_make and not cuda_before_make:
            print(f"    --> render_backend={render_backend} triggered CUDA init")
        break
    except Exception as e:
        print(f"[2] FAIL render_backend={render_backend}: {e}")
        env = None

if env is None:
    print("[2] FAIL: could not create env with either render_backend")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Reset and inspect obs
# ---------------------------------------------------------------------------
try:
    obs, info = env.reset(seed=42)
    print(f"[3] reset OK  obs keys: {list(obs.keys())}")

    # Inspect sensor_data structure
    if "sensor_data" in obs:
        sd = obs["sensor_data"]
        print(f"    sensor_data type: {type(sd)}")
        if hasattr(sd, "keys"):
            print(f"    sensor_data keys: {list(sd.keys())}")
            for cam_name, cam_data in sd.items():
                print(f"      cam '{cam_name}' keys: {list(cam_data.keys()) if hasattr(cam_data, 'keys') else type(cam_data)}")
                if hasattr(cam_data, "keys") and "rgb" in cam_data:
                    rgb = cam_data["rgb"]
                    print(f"        rgb shape: {rgb.shape}  dtype: {rgb.dtype}")
except Exception as e:
    print(f"[3] FAIL reset: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Extract and save image from sensor_data
# ---------------------------------------------------------------------------
try:
    from PIL import Image

    frame = None
    if "sensor_data" in obs:
        sd = obs["sensor_data"]
        # Try first camera's rgb
        for cam_name, cam_data in sd.items():
            if hasattr(cam_data, "keys") and "rgb" in cam_data:
                rgb_t = cam_data["rgb"]   # shape: (num_envs, H, W, 3) tensor
                if hasattr(rgb_t, "cpu"):
                    frame = rgb_t[0].cpu().numpy().astype(np.uint8)
                else:
                    frame = np.array(rgb_t[0]).astype(np.uint8)
                print(f"[4] image from sensor_data['{cam_name}']['rgb']  shape: {frame.shape}")
                break

    if frame is not None:
        img = Image.fromarray(frame)
        img.save("/tmp/takeitback_frame.png")
        print("    saved to /tmp/takeitback_frame.png")
    else:
        # Fallback: try env.render()
        frame = env.render()
        if frame is not None:
            if hasattr(frame, "cpu"):
                frame = frame[0].cpu().numpy().astype(np.uint8)
            print(f"[4] image from env.render()  shape: {frame.shape}")
            Image.fromarray(frame).save("/tmp/takeitback_frame.png")
            print("    saved to /tmp/takeitback_frame.png")
        else:
            print("[4] WARNING: no image found in obs or render()")
except Exception as e:
    print(f"[4] FAIL image extraction: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 5. Step
# ---------------------------------------------------------------------------
try:
    action = env.action_space.sample() * 0.0
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"[5] step OK  reward: {reward}  terminated: {terminated}")
    if "info" in info2:
        print(f"    info keys: {list(info2.keys())}")
except Exception as e:
    print(f"[5] FAIL step: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 6. Final CUDA check
# ---------------------------------------------------------------------------
cuda_final = torch.cuda.is_initialized()
print(f"\n[6] CUDA context initialised at end: {cuda_final}")
if cuda_final:
    try:
        mem = torch.cuda.memory_allocated() / 1e6
        print(f"    GPU memory allocated by this process: {mem:.1f} MB")
        print("    If < 100 MB, this is lightweight init (Vulkan interop) and")
        print("    likely will NOT conflict with SGLang's 40% GPU_MEMORY_UTIL.")
    except Exception:
        pass
else:
    print("    Clean: no CUDA context. sim_backend=cpu fully avoids GPU conflict.")

env.close()
print("\nDone.")
