"""
ShellGame Vagen environment (GymImageEnv async interface).

Wraps MIKASA-Robo ShellGame[Touch/Push/Pick]-v0 as a 2-turn memory task:
  Turn 1 (reset obs):  Cups raised — LLM observes ball position.
  Turn 2 (step 1 obs): Cups lowered — LLM must recall & choose correct cup.

This tests whether the LLM can retain key information across context turns
(concat mode) vs. having no access to it (non-concat mode).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.mikasa_robo.utils.parse import parse_response
from vagen.envs.mikasa_robo.utils.prompt import (
    CUP_NAMES,
    initial_obs_template,
    result_obs_template,
    step_obs_template,
    system_prompt,
)

# Distance between adjacent cups along y-axis (metres), from MIKASA-Robo source.
_MIN_DIST = 0.2

# Slices into the flat 40-D state obs (batch dim already squeezed).
# Layout: qpos[0:9] | qvel[9:18] | tcp_pose[18:25] | obj_pose[25:32]
#         | ball_pose[32:39] | oracle_info[39]
_SL_TARGET_CUP_POS = slice(25, 28)  # xyz of the cup that hides the ball
_SL_BALL_POS = slice(32, 35)        # xyz of the red ball
_IDX_ORACLE = 39                     # cup index (0=left, 1=center, 2=right)


@dataclass
class ShellGameEnvConfig:
    env_id: str = "ShellGameTouch-v0"   # ShellGameTouch-v0 | ShellGamePush-v0 | ShellGamePick-v0
    obs_mode: str = "state"
    format_reward: float = 0.02          # reward for correct <answer> format
    success_reward: float = 1.0          # reward for identifying the correct cup
    prompt_format: str = "free_think"    # kept for API compatibility


class ShellGameEnv(GymImageEnv):
    """2-turn Vagen wrapper around MIKASA-Robo ShellGame environments."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        valid = {f.name for f in fields(ShellGameEnvConfig)}
        self.cfg = ShellGameEnvConfig(**{k: v for k, v in env_config.items() if k in valid})
        self._gym_env = None
        self._oracle: int = -1            # correct cup index (0/1/2)
        self._cup_positions: Dict[str, float] = {}  # name -> y coordinate
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # GymImageEnv abstract interface
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._gym_env is not None:
            await asyncio.to_thread(self._gym_env.close)
            self._gym_env = None

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import gymnasium as gym
        import mikasa_robo_suite  # noqa: F401 — registers envs as side effect

        if self._gym_env is not None:
            await asyncio.to_thread(self._gym_env.close)

        self._gym_env = await asyncio.to_thread(
            gym.make, self.cfg.env_id, obs_mode=self.cfg.obs_mode
        )
        obs, _info = await asyncio.to_thread(self._gym_env.reset, seed=seed)

        # obs: Tensor[1, 40] — squeeze the batch dimension
        obs_t = obs[0]

        self._oracle = int(obs_t[_IDX_ORACLE].item())  # 0=left, 1=center, 2=right
        ball_y = obs_t[_SL_BALL_POS][1].item()

        # The obs only stores the TARGET cup's pose. Reconstruct all 3 cup y-positions:
        # cups are equally spaced by _MIN_DIST; oracle tells us which slot is the target.
        target_cup_y = obs_t[_SL_TARGET_CUP_POS][1].item()
        y_offsets = [-_MIN_DIST, 0.0, _MIN_DIST]          # left=0, center=1, right=2
        center_y = target_cup_y - y_offsets[self._oracle]  # back-compute the center cup y
        self._cup_positions = {
            "left":   center_y - _MIN_DIST,
            "center": center_y,
            "right":  center_y + _MIN_DIST,
        }

        self._step_count = 0
        obs_str = initial_obs_template(self._cup_positions, ball_y)
        return {"obs_str": obs_str}, {}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._step_count += 1
        parsed = parse_response(action_str)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if self._step_count == 1:
            # Turn 1: LLM observed the scene — cups now lower. Ask for the choice.
            obs_str = step_obs_template()
            info["metrics"] = {
                "turn_metrics": {
                    "action_is_valid": True,
                    "action_is_effective": False,
                },
                "traj_metrics": {"success": False},
            }

        else:
            # Turn 2: evaluate the LLM's cup choice.
            cup_choice = parsed.get("cup_choice")
            correct_cup = CUP_NAMES[self._oracle]
            success = cup_choice == correct_cup

            if parsed.get("format_correct", False):
                reward += self.cfg.format_reward
            if success:
                reward += self.cfg.success_reward

            done = True
            info["success"] = success
            info["metrics"] = {
                "turn_metrics": {
                    "action_is_valid": cup_choice is not None,
                    "action_is_effective": success,
                },
                "traj_metrics": {"success": success},
            }
            obs_str = result_obs_template(cup_choice, correct_cup, success)

        return {"obs_str": obs_str}, reward, done, info
