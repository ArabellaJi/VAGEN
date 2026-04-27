"""
ShellGame Vagen environment (GymImageEnv async interface).

Pure-Python implementation that replicates MIKASA-Robo ShellGameTouch-v0
episode randomisation WITHOUT importing SAPIEN/ManiSkill, so it runs safely
inside Ray CPU workers that share a GPU node with SGLang.

Episode randomisation (from MIKASA-Robo shell_game_touch.py source):
  oracle  = rng.choice([0, 1, 2])          # which cup hides the ball
  center  = rng.uniform(-0.1, 0.1, (2,))   # table-centre (x, y)
  cup_y   = center_y + {left:-0.2, center:0, right:+0.2}
  ball_y  = cup_y[oracle]

2-turn memory task:
  Turn 1 (reset obs):  Cups raised — LLM observes all cup/ball positions.
  Turn 2 (step 1 obs): Cups lowered — LLM must recall & choose correct cup.

concat mode  → LLM sees turn-1 obs in context → can recall → succeeds
window1 mode → LLM only sees turn-2 obs (no ball info) → random ~33%
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

import numpy as np

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.mikasa_robo.utils.parse import parse_response
from vagen.envs.mikasa_robo.utils.prompt import (
    CUP_NAMES,
    initial_obs_template,
    result_obs_template,
    step_obs_template,
    system_prompt,
)

_MIN_DIST = 0.2          # metres between adjacent cups (from MIKASA-Robo source)
_BALL_RADIUS = 0.02      # metres


@dataclass
class ShellGameEnvConfig:
    env_id: str = "ShellGameTouch-v0"   # kept for logging / future extension
    format_reward: float = 0.02
    success_reward: float = 1.0
    prompt_format: str = "free_think"   # kept for API compatibility


class ShellGameEnv(GymImageEnv):
    """2-turn Vagen shell-game environment (no SAPIEN / no GPU required)."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        valid = {f.name for f in fields(ShellGameEnvConfig)}
        self.cfg = ShellGameEnvConfig(**{k: v for k, v in env_config.items() if k in valid})
        self._oracle: int = -1
        self._cup_positions: Dict[str, float] = {}
        self._ball_y: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # GymImageEnv abstract interface
    # ------------------------------------------------------------------

    async def close(self) -> None:
        pass  # no resources to release

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rng = np.random.default_rng(seed)

        # Replicate MIKASA-Robo _initialize_episode randomisation exactly.
        self._oracle = int(rng.choice([0, 1, 2]))          # 0=left, 1=center, 2=right
        center_y = float(rng.uniform(-0.1, 0.1))

        y_offsets = [-_MIN_DIST, 0.0, _MIN_DIST]
        self._cup_positions = {
            "left":   center_y - _MIN_DIST,
            "center": center_y,
            "right":  center_y + _MIN_DIST,
        }
        self._ball_y = center_y + y_offsets[self._oracle]

        self._step_count = 0
        obs_str = initial_obs_template(self._cup_positions, self._ball_y)
        return {"obs_str": obs_str}, {}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._step_count += 1
        parsed = parse_response(action_str)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if self._step_count == 1:
            # Turn 1: acknowledge observation, present the hidden-cups challenge.
            obs_str = step_obs_template()
            info["metrics"] = {
                "turn_metrics": {
                    "action_is_valid": True,
                    "action_is_effective": False,
                },
                "traj_metrics": {"success": False},
            }

        else:
            # Turn 2: evaluate cup choice.
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
