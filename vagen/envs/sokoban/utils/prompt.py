def system_prompt(partial_obs_radius: int = 0):
    """Return the system prompt for Sokoban solver"""
    base = """You are a Sokoban solver.
Sokoban Quick Guide
Goal: Push all boxes onto targets.
Symbols (If image is provided there are no symbols):
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
Rules:
1. Push boxes (can't pull).
2. Avoid walls.
Actions you can take: Left, Down, Right, Up."""
    if partial_obs_radius > 0:
        window = 2 * partial_obs_radius + 1
        base += f"""

Important: You have limited visibility. Each observation shows only a {window}×{window} grid centered on your current position (P). The full map extends beyond this window. The box and target may be outside your current view — you must explore by moving to find them. Use your history of past observations to remember where you have seen objects."""
    return base

def init_observation_template(img_str):
    """Template for initial observation"""
    return f"""[Initial Observation]:
{img_str}
Decide your next action(s)."""

def action_template(valid_action, img_str):
    """Template for action feedback"""
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{img_str}
Decide your next action(s)."""


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think", partial_obs_radius: int = 0):
    """Generate format prompt based on the specified format"""
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return wm_format_prompt(max_actions_per_step, action_sep, add_example, partial_obs_radius)
    elif prompt_format == "free_wm":
        return free_wm_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "mem1":
        return mem1_format_prompt(add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")

def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_think format"""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your reasoning, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        examples = f"""
Example 1:
<think>The box is one step below me, and the target is two steps below me. I should go down to reach the box and then push it down to the target.</think>
<answer>Down</answer>

Example 2:
<think>The box is to the right of me, and the target is further to the right. I need to move right to get behind the box and push it toward the target.</think>
<answer>Right</answer>

Example 3:
<think>The box is above me, and the target is above the box. I should move up to reach the box and then push it upward to the target.</think>
<answer>Up</answer>
"""
        return base_prompt + "\n" + examples

    return base_prompt



def wm_format_prompt(max_actions_per_step, action_sep, add_example=True, partial_obs_radius: int = 0):
    """Generate format prompt for wm_new format with explicit row/column distinction"""
    if partial_obs_radius > 0:
        obs_rules = """Rules for <observation> and <prediction>:
- If the box or target is visible in your current view, describe its position relative to the player using EXACTLY:
  - ONE vertical term: `above`, `below`, or `same row`
  - ONE horizontal term: `left`, `right`, or `same column`
  - Pattern: "X is <vertical> and <horizontal> of the player"
- If the box or target is NOT visible in your current view, write "not visible" for that object.
- Do NOT use the word `same` alone."""
    else:
        obs_rules = """Rules for <observation> and <prediction>:
- You must strictly describe the relative position of the `target` and any visible `box` objects **relative to the player**.
- For each object, you MUST include:
  - exactly ONE vertical relationship: `above`, `below`, or `same row`
  - exactly ONE horizontal relationship: `left`, `right`, or `same column`
- Use ONLY the terms: `above`, `below`, `same row`, `left`, `right`, `same column`.
- Always use the phrasing pattern:
  "X is <vertical> and <horizontal> of the player".
- Do NOT use the word `same` alone.
- Do not include any extra information."""

    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>.

{obs_rules}

Rules for <answer>:
- Output 1 to {max_actions_per_step} action(s).
- Valid actions are: Up, Down, Left, Right.
- Separate multiple actions with `{action_sep}`.
"""

    if add_example:
        examples = f"""
Example 1:
<observation>The box is below and right of the player, and the target is below and right of the player</observation>
<think>I should move right to align my column with the box and the target</think>
<answer>Right</answer>
<prediction>The box will be below and same column of the player, and the target will be below and same column of the player</prediction>

Example 2:
<observation>The box is above and left of the player, and the target is above and same column of the player</observation>
<think>I should move up to align my row with the box and reach the target's row position</think>
<answer>Up</answer>
<prediction>The box will be same row and left of the player, and the target will be same row and same column of the player</prediction>

Example 3:
<observation>The box is same row and right of the player, and the target is same row and left of the player</observation>
<think>I should move right to push the box right while keeping the target on my left</think>
<answer>Right</answer>
<prediction>The box will be same row and right of the player, and the target will be same row and left of the player</prediction>
"""
        return base_prompt + "\n" + examples

    return base_prompt


def mem1_format_prompt(add_example=True):
    """Format prompt for MEM1 mode: single action per step + compact memory state."""
    base_prompt = """\
You see a 3×3 grid centered on you each step. Use your memory state to remember \
where the box and target are across steps.

POSITION TRACKING:
- You start at offset (0,0). Each move shifts your offset by 1:
    up→(r-1,c)  down→(r+1,c)  left→(r,c-1)  right→(r,c+1)
- Window deltas from your position:
    (-1,-1) (-1,0) (-1,+1)
    (0,-1)  [YOU]  (0,+1)
    (+1,-1) (+1,0) (+1,+1)
- Absolute offset of a visible object = your_offset + window_delta.

SUCCESS: When you see √ in the observation, the task is complete — choose any action \
and write your final memory.

Required output format (exactly this structure):
<think>reasoning</think>
<answer>action</answer>
<memory>
me=(r,c) step=N
box=(r,c)
target=(r,c)
</memory>

Rules:
- Output exactly ONE action: up | down | left | right
- Write "box=(r,c)" only if you saw X or √ this step or in a prior step.
- Write "target=(r,c)" only if you saw O, √, or S this step or in a prior step.
- Update me=(r,c) to your NEXT position after applying your chosen action.
- If box/target not yet seen, omit those lines entirely."""

    if add_example:
        base_prompt += """

Example (box visible to the right, target not yet seen, starting position):
<think>Box is to my right at window offset (0,+1). My pos is (0,0), so box=(0,1). No target seen yet. I'll move right to push it.</think>
<answer>right</answer>
<memory>
me=(0,1) step=1
box=(0,1)
</memory>

Example (box known from memory, target now visible above):
<think>Box is at (2,1) from memory. I see O above me at (-1,0), my pos is (3,1), so target=(2,1). Box and target are at the same offset — already solved!</think>
<answer>up</answer>
<memory>
me=(2,1) step=7
box=(2,1)
target=(2,1)
</memory>"""

    return base_prompt


def free_wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_wm format: observation, answer, prediction with free reasoning between tags."""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your response must be in the format of:
<observation>...</observation> your reasoning <answer>...</answer> your reasoning <prediction>...</prediction>.

You may include free-form reasoning text between the tags.

Rules for <observation> and <prediction>:
- You must strictly describe the relative position of the `target` and any visible `box` objects **relative to the player**.
- For each object, you MUST include:
  - exactly ONE vertical relationship: `above`, `below`, or `same row`
  - exactly ONE horizontal relationship: `left`, `right`, or `same column`
- Use ONLY the terms: `above`, `below`, `same row`, `left`, `right`, `same column`.
- Always use the phrasing pattern:
  "X is <vertical> and <horizontal> of the player".
- Do NOT use the word `same` alone.
- Do not include any extra information.

Rules for <answer>:
- Output 1 to {max_actions_per_step} action(s).
- Valid actions are: Up, Down, Left, Right.
- Separate multiple actions with `{action_sep}`.
"""

    if add_example:
        examples = f"""
Example 1:
<observation>The box is below and right of the player, and the target is below and right of the player</observation>
I should move right to align my column with the box and the target.
<answer>Right</answer>
After moving right, the box and target should still be below me but now in the same column.
<prediction>The box will be below and same column of the player, and the target will be below and same column of the player</prediction>

Example 2:
<observation>The box is above and left of the player, and the target is above and same column of the player</observation>
I should move up to align my row with the box and reach the target's row position.
<answer>Up</answer>
After moving up, the box should be on the same row to my left, and the target on the same row in the same column.
<prediction>The box will be same row and left of the player, and the target will be same row and same column of the player</prediction>

Example 3:
<observation>The box is same row and right of the player, and the target is same row and left of the player</observation>
I should move right to push the box right while keeping the target on my left.
<answer>Right</answer>
After pushing right, the box stays to my right and the target stays to my left.
<prediction>The box will be same row and right of the player, and the target will be same row and left of the player</prediction>
"""
        return base_prompt + "\n" + examples

    return base_prompt