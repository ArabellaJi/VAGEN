CUP_NAMES = ["left", "center", "right"]


def system_prompt() -> str:
    return (
        "You are a robot arm participating in a shell game.\n"
        "Three cups are arranged in a row on a table: left, center, and right.\n"
        "A red ball is placed under one cup. The cups will be briefly raised to reveal the ball, "
        "then lowered to hide it.\n\n"
        "Your task has two phases:\n"
        "  Phase 1 (Observe): The cups are raised. Observe which cup is directly above the red ball "
        "and remember it. Respond with your observation.\n"
        "  Phase 2 (Choose):  The cups are lowered. Based on what you remember, "
        "identify which cup hides the ball.\n\n"
        "Always format your final cup choice as:\n"
        "<think>your reasoning</think>\n"
        "<answer>left</answer>  OR  <answer>center</answer>  OR  <answer>right</answer>"
    )


def initial_obs_template(cup_positions: dict, ball_y: float) -> str:
    left_y = cup_positions["left"]
    center_y = cup_positions["center"]
    right_y = cup_positions["right"]
    return (
        "[Phase 1: Observe — Cups Raised]\n"
        "The three cups have been lifted, revealing the red ball underneath.\n\n"
        "Cup positions (y-coordinate along the table):\n"
        f"  Left cup:   {left_y:+.3f} m\n"
        f"  Center cup: {center_y:+.3f} m\n"
        f"  Right cup:  {right_y:+.3f} m\n\n"
        f"Red ball position: {ball_y:+.3f} m\n\n"
        "The cups will be lowered shortly. Remember which cup is directly above the ball.\n"
        "Describe what you observe."
    )


def step_obs_template() -> str:
    return (
        "[Phase 2: Choose — Cups Lowered]\n"
        "The cups have been lowered. The red ball is no longer visible.\n\n"
        "Based on what you observed in Phase 1, which cup is the ball under?\n"
        "Respond with:\n"
        "<think>your reasoning</think>\n"
        "<answer>left</answer>  OR  <answer>center</answer>  OR  <answer>right</answer>"
    )


def result_obs_template(chosen: str, correct: str, success: bool) -> str:
    if success:
        return f"[Result] Correct! The ball was under the {correct} cup."
    chosen_str = chosen if chosen else "nothing valid"
    return f"[Result] Incorrect. You chose '{chosen_str}', but the ball was under the {correct} cup."
