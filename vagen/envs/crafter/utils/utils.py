import re
from typing import Dict, List, Set, Any, Optional
from PIL import Image
import numpy as np


# Crafter tile type IDs (from crafter data.yaml)
MATERIAL_NAMES: Dict[int, str] = {
    0: "water",
    1: "grass",
    2: "stone",
    3: "path",
    4: "sand",
    5: "tree",
    6: "lava",
    7: "coal",
    8: "iron",
    9: "diamond",
    10: "table",
    11: "furnace",
}
OBJECT_NAMES: Dict[int, str] = {
    12: "player",
    13: "cow",
    14: "zombie",
    15: "skeleton",
    16: "arrow",
    17: "plant",
}

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


def format_nearby_tiles(info: Dict[str, Any], facing: Optional[str] = None) -> str:
    """Return a text description of the 4 tiles adjacent to the player.

    Uses info["semantic"] (2D numpy array, indexed [y, x]) and
    info["player_pos"] ([x, y] in Crafter convention).
    Returns empty string if either field is missing.
    """
    # Tiles where "do" is meaningful (collect/mine/drink/eat/attack)
    DO_USEFUL = {
        "tree":     "collect wood",
        "stone":    "mine stone (needs wood_pickaxe)",
        "coal":     "mine coal (needs wood_pickaxe)",
        "iron":     "mine iron (needs stone_pickaxe)",
        "diamond":  "mine diamond (needs iron_pickaxe)",
        "water":    "drink water",
        "cow":      "attack/eat",
        "plant":    "eat plant",
    }
    # Tiles that are dangerous or impassable with a note
    NOTES = {
        "lava":     "DANGER: do not enter",
        "table":    "stand here to craft tools",
        "furnace":  "stand here to smelt iron",
    }

    semantic = info.get("semantic")
    player_pos = info.get("player_pos")
    if semantic is None or player_pos is None:
        return ""

    try:
        px, py = int(player_pos[0]), int(player_pos[1])
        h, w = semantic.shape

        # (label, dy, dx, move_action)
        # Assumes semantic[y,x], player_pos=[x,y], move_up=y-1, move_right=x+1
        directions = [
            ("North", -1,  0, "move_up"),
            ("South", +1,  0, "move_down"),
            ("West",   0, -1, "move_left"),
            ("East",   0, +1, "move_right"),
        ]

        facing_line = f"Current facing: {facing}" if facing else "Current facing: unknown (no move yet)"

        parts = []
        for label, dy, dx, move_action in directions:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w:
                tile_id = int(semantic[ny, nx])
                name = MATERIAL_NAMES.get(tile_id) or OBJECT_NAMES.get(tile_id, f"tile{tile_id}")
            else:
                name = "void"

            already_facing = (facing == label)
            if name in DO_USEFUL:
                if already_facing:
                    hint = f"already facing this — use do to {DO_USEFUL[name]}"
                else:
                    hint = f"{move_action} to face it, then do to {DO_USEFUL[name]}"
            elif name in NOTES:
                hint = NOTES[name]
            else:
                hint = f"{move_action} to move/explore"

            parts.append(f"{label}={name}({hint})")

        return facing_line + "\nNearby tiles: " + ", ".join(parts)
    except Exception:
        return ""


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
