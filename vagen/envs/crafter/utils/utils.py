import re
from typing import Dict, List, Set, Any
from PIL import Image
import numpy as np


# All 17 Crafter actions in index order
ACTIONS = [
    "noop",
    "move_left", "move_right", "move_up", "move_down",
    "do",
    "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
]

ACTION_LOOKUP: Dict[str, int] = {name: i for i, name in enumerate(ACTIONS)}

# Survival stats shown in status text
SURVIVAL_STATS = ("health", "food", "drink", "energy")


def format_status_text(info: Dict[str, Any], unlocked: Set[str]) -> str:
    """Build a compact status string from the Crafter info dict."""
    if not info:
        return ""

    inv = info.get("inventory", {})

    stats = " | ".join(
        f"{k.capitalize()}:{inv.get(k, '?')}/9" for k in SURVIVAL_STATS
    )

    items = {
        k: v for k, v in inv.items()
        if k not in SURVIVAL_STATS and isinstance(v, int) and v > 0
    }
    items_str = " ".join(f"{k}:{v}" for k, v in items.items()) if items else "none"

    n = len(unlocked)
    ach_str = ", ".join(sorted(unlocked)) if unlocked else "none"

    return f"{stats}\nInventory: {items_str}\nAchievements ({n}/22): {ach_str}"


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None

    if not match:
        think_content = action_content = ""
        actions: List[str] = []
    else:
        think_content = match.group(1).strip()
        action_content = match.group(2).strip()
        actions = [a.strip().lower() for a in action_content.split(action_sep) if a.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)

    return {
        "llm_raw_response": response,
        "llm_response": f"<think>{think_content}</think><answer>{action_content}</answer>",
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_wm(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    pattern = (
        r"<observation>(.*?)</observation>\s*"
        r"<think>(.*?)</think>\s*"
        r"<answer>(.*?)</answer>\s*"
        r"<prediction>(.*?)</prediction>"
    )
    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None

    if not match:
        observation_content = think_content = action_content = prediction_content = ""
        actions: List[str] = []
    else:
        observation_content = match.group(1).strip()
        think_content = match.group(2).strip()
        action_content = match.group(3).strip()
        prediction_content = match.group(4).strip()
        actions = [a.strip().lower() for a in action_content.split(action_sep) if a.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)

    return {
        "llm_raw_response": response,
        "llm_response": (
            f"<observation>{observation_content}</observation>"
            f"<think>{think_content}</think>"
            f"<answer>{action_content}</answer>"
            f"<prediction>{prediction_content}</prediction>"
        ),
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": think_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = ",",
    max_actions: int = 1,
) -> Dict:
    if prompt_format == "free_think":
        return parse_free_think(response, action_sep, max_actions)
    elif prompt_format == "wm":
        return parse_wm(response, action_sep, max_actions)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), mode="RGB")
