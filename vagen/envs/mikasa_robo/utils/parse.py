import re
from typing import Dict, Any

_CUP_PATTERN = re.compile(r"<answer>\s*(left|center|right)\s*</answer>", re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_response(response: str) -> Dict[str, Any]:
    """Parse LLM response for ShellGame.

    Returns:
        dict with keys:
            cup_choice (str | None): "left", "center", or "right"
            format_correct (bool): True if <answer> tag with valid cup name found
    """
    match = _CUP_PATTERN.search(response)
    if match:
        return {
            "cup_choice": match.group(1).lower(),
            "format_correct": True,
        }
    return {
        "cup_choice": None,
        "format_correct": False,
    }
