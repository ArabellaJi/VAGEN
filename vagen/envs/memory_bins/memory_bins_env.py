"""
MemoryBins-v0  (Design A)
=========================
Agent explores N opaque bins and retrieves objects per sequential instructions.
Finding the target cube (opening the correct bin) is rewarded immediately —
no explicit delivery step required.

Key design choices
------------------
- open_bin on the target bin → reward + advance to next instruction.
- Bins close at the start of every step (one-step visibility window).
- After a cube is collected, that bin looks IDENTICAL to any other closed bin
  from the outside.  The empty indicator is only shown when the agent opens
  the bin again — testing whether the agent remembers which bins are depleted.
- Object pool has 8 colors; n_bins=5 by default so elimination reasoning
  is impossible.
- Instructions are sampled without replacement from the placed colors so
  every instruction is achievable exactly once.
- cell_size=64 → 512×512 rendering (~335 visual tokens for Qwen2.5-VL).
"""

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from vagen.envs.gym_image_env import GymImageEnv

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red":    (220, 60,  60),
    "blue":   (60,  110, 220),
    "green":  (50,  180, 50),
    "purple": (160, 60,  220),
    "orange": (220, 130, 40),
    "cyan":   (40,  200, 200),
    "pink":   (220, 90,  160),
    "lime":   (140, 220, 50),
}
ALL_COLORS = list(OBJECT_COLORS.keys())

_BG         = (40,  40,  40)
_GRID_LINE  = (65,  65,  65)
_AGENT      = (255, 220, 0)
_BIN_CLOSED = (105, 105, 105)
_BIN_OPEN   = (165, 165, 165)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_GRID_SIZE = 8
_AGENT_START: Tuple[int, int] = (0, 0)

_FIXED_BIN_POSITIONS: List[Tuple[int, int]] = [
    (1, 1), (1, 5), (3, 3), (5, 1), (5, 5),
    (2, 6), (6, 2), (7, 4),
]

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
_ACTION_NAMES = [
    "move_up", "move_down", "move_left", "move_right", "open_bin",
]
_MOVE_DELTA: Dict[str, Tuple[int, int]] = {
    "move_up":    (-1,  0),
    "move_down":  ( 1,  0),
    "move_left":  ( 0, -1),
    "move_right": ( 0,  1),
}
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MemoryBinsConfig:
    n_bins: int = 5
    n_instructions: int = 3
    max_steps: int = 100
    bin_layout: str = "fixed"      # "fixed" | "random"
    cell_size: int = 64            # pixels per cell; 8*64 = 512x512
    image_placeholder: str = "<image>"
    step_penalty: float = 0.1
    bin_open_penalty: float = 0.1
    task_reward: float = 10.0      # reward for opening the correct bin


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MemoryBinsEnv(GymImageEnv):

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = MemoryBinsConfig(**env_config)
        if self.config.n_bins > len(_FIXED_BIN_POSITIONS):
            raise ValueError(
                f"n_bins={self.config.n_bins} exceeds available fixed positions "
                f"({len(_FIXED_BIN_POSITIONS)})."
            )
        if self.config.n_bins > len(ALL_COLORS):
            raise ValueError("n_bins must be <= number of object colors (8).")
        if self.config.n_instructions > self.config.n_bins:
            raise ValueError(
                "n_instructions must be <= n_bins so every instruction is achievable."
            )

        self.rng: random.Random = random.Random()
        self.bin_positions: List[Tuple[int, int]] = []
        self.bin_contents: Dict[Tuple[int, int], Optional[str]] = {}
        self.bin_open: Dict[Tuple[int, int], bool] = {}
        self.pending_collect: set = set()   # bins whose content clears next step
        self.instructions: List[str] = []
        self.current_idx: int = 0
        self.agent_pos: Tuple[int, int] = _AGENT_START
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # GymImageEnv interface
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    async def close(self) -> None:
        pass

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": _SYSTEM_PROMPT}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.rng = random.Random(seed)

        if self.config.bin_layout == "fixed":
            self.bin_positions = list(_FIXED_BIN_POSITIONS[: self.config.n_bins])
        else:
            candidates = [
                (r, c)
                for r in range(_GRID_SIZE)
                for c in range(_GRID_SIZE)
                if (r, c) != _AGENT_START
            ]
            self.rng.shuffle(candidates)
            self.bin_positions = candidates[: self.config.n_bins]

        placed_colors = self.rng.sample(ALL_COLORS, self.config.n_bins)
        self.bin_contents = {
            pos: color
            for pos, color in zip(self.bin_positions, placed_colors)
        }
        self.bin_open = {pos: False for pos in self.bin_positions}
        self.pending_collect = set()
        self.instructions = self.rng.sample(placed_colors, self.config.n_instructions)

        self.current_idx = 0
        self.agent_pos = _AGENT_START
        self.step_count = 0

        return self._make_obs(init=True), self._get_info()

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Clear bins collected last step (content was kept visible for one render).
        for pos in self.pending_collect:
            self.bin_contents[pos] = None
        self.pending_collect.clear()

        # Bins opened last step are now closed again.
        for pos in self.bin_positions:
            self.bin_open[pos] = False

        action = self._parse_action(action_str)
        reward = -self.config.step_penalty
        action_valid = action in _ACTION_NAMES
        found_target = False

        if action in _MOVE_DELTA:
            dr, dc = _MOVE_DELTA[action]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if self._passable(nr, nc):
                self.agent_pos = (nr, nc)

        elif action == "open_bin":
            target_bin = self._find_adjacent_bin()
            if target_bin is not None:
                reward -= self.config.bin_open_penalty
                self.bin_open[target_bin] = True
                # Reward if this bin contains the current target.
                if (
                    self.current_idx < len(self.instructions)
                    and self.bin_contents[target_bin] == self.instructions[self.current_idx]
                ):
                    reward += self.config.task_reward
                    self.pending_collect.add(target_bin)  # clear content next step
                    self.current_idx += 1
                    found_target = True

        self.step_count += 1
        all_done = self.current_idx >= len(self.instructions)
        done = all_done or self.step_count >= self.config.max_steps

        info = self._get_info()
        info["success"] = all_done
        info["metrics"] = {
            "traj_metrics": {
                "success": all_done,
                "completed": self.current_idx,
            },
            "turn_metrics": {
                "action_is_valid": action_valid,
                "found_target": found_target,
            },
        }
        return self._make_obs(init=False), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_action(self, action_str: str) -> str:
        m = _ANSWER_RE.search(action_str)
        if m:
            candidate = m.group(1).strip().lower()
            if candidate in _ACTION_NAMES:
                return candidate
        for name in _ACTION_NAMES:
            if name in action_str.lower():
                return name
        return ""

    def _passable(self, r: int, c: int) -> bool:
        if not (0 <= r < _GRID_SIZE and 0 <= c < _GRID_SIZE):
            return False
        return (r, c) not in self.bin_positions

    def _find_adjacent_bin(self) -> Optional[Tuple[int, int]]:
        """Return the first orthogonally adjacent bin (empty or not), or None."""
        r, c = self.agent_pos
        for pos in self.bin_positions:
            if abs(pos[0] - r) + abs(pos[1] - c) == 1:
                return pos
        return None

    def _get_info(self) -> Dict[str, Any]:
        idx = self.current_idx
        return {
            "instruction": (
                self.instructions[idx] if idx < len(self.instructions) else "done"
            ),
            "completed": idx,
            "agent_pos": self.agent_pos,
            "step": self.step_count,
            "_bin_contents": dict(self.bin_contents),
        }

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _make_obs(self, init: bool) -> Dict[str, Any]:
        idx = self.current_idx
        if idx < len(self.instructions):
            instr_text = (
                f"Find the {self.instructions[idx]} cube.  "
                f"[{idx + 1}/{len(self.instructions)}]"
            )
        else:
            instr_text = "All instructions completed."

        step_text = f"Step: {self.step_count}/{self.config.max_steps}"
        ph = self.config.image_placeholder

        obs_str = (
            f"Instruction: {instr_text}\n"
            f"{step_text}\n\n"
            f"{ph}"
        )
        return {
            "obs_str": obs_str,
            "multi_modal_input": {ph: [self._render_image()]},
        }

    def _render_image(self) -> Image.Image:
        cs = self.config.cell_size
        size = _GRID_SIZE * cs
        img = Image.new("RGB", (size, size), _BG)
        draw = ImageDraw.Draw(img)

        # Grid lines
        for i in range(_GRID_SIZE + 1):
            draw.line([(i * cs, 0), (i * cs, size)], fill=_GRID_LINE, width=1)
            draw.line([(0, i * cs), (size, i * cs)], fill=_GRID_LINE, width=1)

        # Bins
        for pos in self.bin_positions:
            content = self.bin_contents[pos]
            is_open = self.bin_open[pos]

            if not is_open:
                # Closed: looks identical whether it has content or not.
                self._fill_cell(draw, pos, _BIN_CLOSED, cs)
            elif content is not None:
                # Open with content: show object color as inner square.
                self._fill_cell(draw, pos, _BIN_OPEN, cs)
                pr, pc = pos
                inner = cs // 3
                draw.rectangle(
                    [pc * cs + inner, pr * cs + inner,
                     pc * cs + inner * 2, pr * cs + inner * 2],
                    fill=OBJECT_COLORS[content],
                )
            else:
                # Open but empty: agent just opened a depleted bin.
                self._fill_cell(draw, pos, _BIN_OPEN, cs)
                pr, pc = pos
                pad = cs // 5
                x0, y0 = pc * cs + pad, pr * cs + pad
                x1, y1 = (pc + 1) * cs - pad, (pr + 1) * cs - pad
                draw.line([(x0, y0), (x1, y1)], fill=_GRID_LINE, width=3)
                draw.line([(x0, y1), (x1, y0)], fill=_GRID_LINE, width=3)

        # Agent
        r, c = self.agent_pos
        margin = cs // 6
        draw.rectangle(
            [c * cs + margin, r * cs + margin,
             (c + 1) * cs - margin, (r + 1) * cs - margin],
            fill=_AGENT,
        )

        return img

    @staticmethod
    def _fill_cell(
        draw: ImageDraw.ImageDraw,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        cs: int,
    ) -> None:
        r, c = pos
        margin = cs // 8
        draw.rectangle(
            [c * cs + margin, r * cs + margin,
             (c + 1) * cs - margin, (r + 1) * cs - margin],
            fill=color,
        )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an agent in an 8x8 grid world. Your task is to find specific colored \
cubes hidden inside bins by following sequential instructions.

VISUAL LEGEND:
  Yellow square               — you (the agent)
  Dark gray square            — bin (closed; contents unknown)
  Light gray + colored square — bin you just opened (showing its cube color)
  Light gray + X pattern      — bin you just opened, but it is already empty

IMPORTANT: closed bins all look the same whether they contain a cube or not. \
You must remember which bins you have already searched.

ACTIONS (respond with exactly one per turn):
  move_up    : move one cell upward
  move_down  : move one cell downward
  move_left  : move one cell leftward
  move_right : move one cell rightward
  open_bin   : open an adjacent bin to reveal its contents; if it contains \
your current target cube you immediately succeed and move to the next instruction

RULES:
  - You cannot walk through bins.
  - Bins close again at the start of the next step.
  - Opening bins costs extra penalty — use your memory to avoid re-opening \
bins you have already inspected or depleted.

FORMAT — always respond like this:
<think>your reasoning here</think>
<answer>action_name</answer>"""
