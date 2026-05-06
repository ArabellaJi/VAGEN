import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

import crafter

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.crafter.utils.prompt import (
    system_prompt,
    format_prompt,
    init_observation_template,
    action_template,
)
from vagen.envs.crafter.utils.utils import (
    ACTION_LOOKUP,
    format_status_text,
    numpy_to_pil,
    parse_response,
)


@dataclass
class CrafterEnvConfig:
    render_mode: str = "vision"           # "vision" or "text"
    render_scale: int = 8                 # upscale: 64 × 8 = 512 px
    area: Tuple[int, int] = (64, 64)      # world size in tiles
    length: int = 10000                   # internal crafter episode length limit
    max_actions_per_step: int = 1         # LLM actions per turn
    action_sep: str = ","
    image_placeholder: str = "<image>"
    prompt_format: str = "free_think"     # "free_think" or "wm"
    achievement_reward: float = 1.0       # reward per new achievement unlocked
    format_reward: float = 0.0            # bonus when format is correct and action valid
    format_penalty: float = 0.0           # penalty when no valid action parsed
    use_example_in_sys_prompt: bool = True


class CrafterEnv(GymImageEnv):
    """
    Crafter environment wrapped for Vagen's multi-turn VLM training loop.

    Key design decisions:
    - A new crafter.Env is created on each reset() to support per-episode seeding.
    - Reward = achievement_reward per newly unlocked achievement (sparse, up to +22 total).
    - "success" in info is True once any achievement is unlocked in the episode.
    - Status text (health/food/drink/energy + inventory + achievements) is always
      prepended to the observation so the VLM can read game state without parsing pixels.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = CrafterEnvConfig(**env_config)
        self.env: Optional[crafter.Env] = None
        self.total_reward: float = 0.0
        self.valid_actions: List[str] = []
        self._unlocked: Set[str] = set()
        self._last_info: Dict[str, Any] = {}
        self._last_obs_array: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # GymImageEnv abstract methods
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self.env is not None:
            try:
                await asyncio.to_thread(self.env.close)
            except Exception:
                pass

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        def _make_and_reset():
            env = crafter.Env(area=self.config.area, length=self.config.length, seed=seed)
            obs = env.reset()
            return env, obs

        self.env, obs_array = await asyncio.to_thread(_make_and_reset)
        self.total_reward = 0.0
        self.valid_actions = []
        self._unlocked = set()
        self._last_info = {}
        self._last_obs_array = obs_array

        return await self._render_async(obs_array, init_obs=True), {}

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": self._get_system_prompt()}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            response=action_str,
            prompt_format=self.config.prompt_format,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        self.valid_actions = []
        info.update(parsed)

        action_list: List[str] = parsed.get("actions", [])
        metrics = {
            "turn_metrics": {
                "action_is_valid": bool(action_list) and parsed.get("format_correct", False),
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
                "num_achievements": len(self._unlocked),
            },
        }

        obs_array = self._last_obs_array

        for action in action_list:
            if action not in ACTION_LOOKUP:
                metrics["turn_metrics"]["action_is_valid"] = False
                break

            action_int = ACTION_LOOKUP[action]
            obs_array, _, step_done, step_info = await asyncio.to_thread(
                self.env.step, action_int
            )
            self._last_info = step_info
            self._last_obs_array = obs_array
            self.valid_actions.append(action)
            metrics["turn_metrics"]["action_is_effective"] = True

            # Reward new achievements
            for ach_name, unlocked in step_info.get("achievements", {}).items():
                if unlocked and ach_name not in self._unlocked:
                    self._unlocked.add(ach_name)
                    reward += self.config.achievement_reward

            if step_done:
                done = True
                break

        if self.valid_actions:
            reward += self.config.format_reward
        else:
            reward += self.config.format_penalty

        metrics["traj_metrics"]["num_achievements"] = len(self._unlocked)
        info["metrics"] = metrics
        info["success"] = len(self._unlocked) > 0
        self.total_reward += reward

        return await self._render_async(obs_array, init_obs=False), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        fmt = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            prompt_format=self.config.prompt_format,
            add_example=self.config.use_example_in_sys_prompt,
        )
        return system_prompt() + "\n\n" + fmt

    async def _render_async(
        self, obs_array: Optional[np.ndarray], init_obs: bool
    ) -> Dict[str, Any]:
        status_text = format_status_text(self._last_info, self._unlocked)
        multi_modal_input = None

        if self.config.render_mode == "vision":
            if obs_array is None:
                obs_array = await asyncio.to_thread(self.env.render)
            pil_img = numpy_to_pil(obs_array)
            if self.config.render_scale > 1:
                w, h = pil_img.size
                pil_img = pil_img.resize(
                    (w * self.config.render_scale, h * self.config.render_scale),
                    Image.NEAREST,
                )
            img_str = self.config.image_placeholder
            multi_modal_input = {self.config.image_placeholder: [pil_img]}
        else:
            # Text mode: embed status directly as the "image"
            img_str = status_text
            status_text = ""

        if init_obs:
            obs_str = init_observation_template(img_str, status_text)
        else:
            obs_str = action_template(self.valid_actions, img_str, status_text)

        obs: Dict[str, Any] = {"obs_str": obs_str}
        if multi_modal_input is not None:
            obs["multi_modal_input"] = multi_modal_input
        return obs


# ------------------------------------------------------------------
# Quick local test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import os

    async def _test(render_mode: str = "vision", save_path: str = "./crafter_test"):
        cfg = {
            "render_mode": render_mode,
            "prompt_format": "free_think",
            "render_scale": 4,
        }
        env = CrafterEnv(cfg)

        print("=== System Prompt ===")
        sp = await env.system_prompt()
        print(sp["obs_str"])
        print("\n" + "=" * 60 + "\n")

        obs, _ = await env.reset(seed=42)
        print("=== Initial Observation ===")
        print(obs["obs_str"])

        os.makedirs(save_path, exist_ok=True)
        if "multi_modal_input" in obs:
            obs["multi_modal_input"][env.config.image_placeholder][0].save(
                os.path.join(save_path, "step_0.png")
            )

        for step in range(1, 6):
            try:
                raw = input(f"\nStep {step} — action: ")
            except EOFError:
                raw = "move_right"
            action_input = f"<think>Testing.</think><answer>{raw}</answer>"
            obs, reward, done, info = await env.step(action_input)
            print(f"reward={reward:.2f}  done={done}  achievements={len(env._unlocked)}")
            print(obs["obs_str"])
            if "multi_modal_input" in obs:
                obs["multi_modal_input"][env.config.image_placeholder][0].save(
                    os.path.join(save_path, f"step_{step}.png")
                )
            if done:
                break

        print(f"\nTotal reward: {env.total_reward:.2f}")
        await env.close()

    asyncio.run(_test())
