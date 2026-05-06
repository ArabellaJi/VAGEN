from vagen.envs.crafter.utils.utils import ACTIONS


def system_prompt() -> str:
    action_list = ", ".join(ACTIONS)
    return f"""You are playing Crafter, an open-world survival game viewed from above.

Goal: Survive and unlock achievements by exploring, collecting resources, crafting tools, and defeating enemies.

Survival stats (range 0–9, higher is better):
  Health: decreases when attacked; reaching 0 ends the game
  Food:   decreases over time; eat cows or crops to restore
  Drink:  decreases over time; stand next to water and use "do" to restore
  Energy: decreases when moving; use "sleep" to restore

Available actions (use exact names):
  {action_list}

Action guide:
  move_left/right/up/down — move one tile; also sets your facing direction
  do        — collect resource, attack enemy, or drink water in facing direction
  sleep     — rest in place (restores energy; use when energy < 3)
  place_*   — place an item from your inventory onto the ground
  make_*    — craft a tool or weapon (requires a workbench nearby and materials)
  noop      — do nothing

Progression tips:
  1. Collect wood: face a tree, use "do"
  2. Place a table: place_table (needs 2 wood)
  3. Craft a pickaxe: stand next to table, make_wood_pickaxe (needs 2 wood)
  4. Mine stone/coal/iron with pickaxe: face the block, use "do"
  5. Build a furnace to smelt iron: place_furnace (needs 4 stone)
  6. Upgrade tools and weapons for stronger mining and combat"""


def init_observation_template(img_str: str, status_text: str) -> str:
    if status_text:
        return f"""[Initial Observation]:
{status_text}
{img_str}
Decide your next action."""
    return f"""[Initial Observation]:
{img_str}
Decide your next action."""


def action_template(valid_actions: list, img_str: str, status_text: str) -> str:
    action_str = valid_actions if valid_actions else "none"
    if status_text:
        return f"""Executed action: {action_str}
{status_text}
{img_str}
Decide your next action."""
    return f"""Executed action: {action_str}
{img_str}
Decide your next action."""


def format_prompt(
    max_actions_per_step: int,
    action_sep: str,
    prompt_format: str = "free_think",
    add_example: bool = True,
) -> str:
    if prompt_format == "free_think":
        return _free_think_format(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return _wm_format(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def _free_think_format(max_actions_per_step: int, action_sep: str, add_example: bool) -> str:
    base = f"""You may take up to {max_actions_per_step} action(s) per turn, separated by "{action_sep}".
Respond in this format:
<think>...</think><answer>...</answer>"""

    if not add_example:
        return base

    example = """

Example:
<think>My energy is 2/9 which is critically low. I should sleep to recover before I can move around again.</think>
<answer>sleep</answer>"""
    return base + example


def _wm_format(max_actions_per_step: int, action_sep: str, add_example: bool) -> str:
    base = f"""You may take up to {max_actions_per_step} action(s) per turn, separated by "{action_sep}".
Respond in this exact format:
<observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>

<observation>: Describe your current survival stats, what you see nearby (terrain, resources, enemies), and your inventory.
<think>: Reason about what goal to pursue and why this action moves you toward it.
<answer>: The action(s) to execute (exact action names only).
<prediction>: Describe what you expect to change after the action (position, stats, inventory, environment)."""

    if not add_example:
        return base

    example = """

Example:
<observation>Health:9/9 Food:8/9 Water:9/9 Energy:9/9. I see a tree one tile to my right and open grass ahead. Inventory is empty. No achievements yet.</observation>
<think>I need wood first to craft any tools. The tree is to my right, so I should move right to face it and collect wood with "do" next turn.</think>
<answer>move_right</answer>
<prediction>I will move one tile right and now face the tree. Energy will decrease slightly. My position changes but inventory stays the same.</prediction>"""
    return base + example
