#!/usr/bin/env python
"""Analyze logged Crafter validation media tables.

The script works on wandb media-table JSON exports such as
media_table_val_generations_99_*.table.json. It summarizes score,
achievement text visible in the logged trajectory, and executed actions.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
EXEC_RE = re.compile(r"Executed action: ([^\n\r]+)")
ACH_RE = re.compile(r"Achievements \((\d+)/22\): ([^\n\r]+)")


def _clean_answer(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _parse_exec(raw: str) -> list[str]:
    raw = raw.strip()
    if raw == "none":
        return []
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return [raw]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _longest_streak(actions: list[str]) -> tuple[str, int]:
    best_action = ""
    best_len = 0
    cur_action = ""
    cur_len = 0
    for action in actions:
        if action == cur_action:
            cur_len += 1
        else:
            cur_action = action
            cur_len = 1
        if cur_len > best_len:
            best_action = cur_action
            best_len = cur_len
    return best_action, best_len


def _achievement_names(output: str) -> tuple[int, Counter[str]]:
    max_count = 0
    names: Counter[str] = Counter()
    for count_s, raw_names in ACH_RE.findall(output):
        max_count = max(max_count, int(count_s))
        if raw_names.strip() == "none":
            continue
        for name in raw_names.split(","):
            name = name.strip()
            if name:
                names[name] += 1
    return max_count, names


def _iter_samples(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    columns = obj["columns"]
    data = obj["data"]
    col_idx = {name: i for i, name in enumerate(columns)}
    for row in data:
        step = row[col_idx["step"]]
        for sample_i in range(1, 6):
            output_key = f"output_{sample_i}"
            score_key = f"score_{sample_i}"
            if output_key not in col_idx or score_key not in col_idx:
                continue
            output = row[col_idx[output_key]]
            score = float(row[col_idx[score_key]])
            answers = [_clean_answer(x) for x in ANSWER_RE.findall(output)]
            exec_turns = [_parse_exec(x) for x in EXEC_RE.findall(output)]
            exec_actions = [a for turn in exec_turns for a in turn]
            none_turns = sum(1 for turn in exec_turns if not turn)
            max_ach, ach_names = _achievement_names(output)
            yield {
                "path": path,
                "step": step,
                "sample_i": sample_i,
                "score": score,
                "output": output,
                "answers": answers,
                "exec_turns": exec_turns,
                "exec_actions": exec_actions,
                "none_turns": none_turns,
                "max_ach_text": max_ach,
                "ach_names_text": ach_names,
            }


def _print_summary(samples: list[dict], score_min: float) -> None:
    by_step: dict[int, list[dict]] = {}
    for sample in samples:
        by_step.setdefault(int(sample["step"]), []).append(sample)

    print("Per-step summary")
    print("step\tn\tmean_score\tscore>=min\tmean_text_ach\tvisible_achievement_names")
    for step in sorted(by_step):
        rows = by_step[step]
        ach_names: Counter[str] = Counter()
        for row in rows:
            ach_names.update(row["ach_names_text"])
        mean_score = mean(row["score"] for row in rows)
        mean_text_ach = mean(row["max_ach_text"] for row in rows)
        n_good = sum(1 for row in rows if row["score"] >= score_min)
        print(
            f"{step}\t{len(rows)}\t{mean_score:.3f}\t{n_good}/{len(rows)}"
            f"\t{mean_text_ach:.3f}\t{dict(ach_names.most_common(8))}"
        )

    all_names: Counter[str] = Counter()
    all_scores = []
    for sample in samples:
        all_names.update(sample["ach_names_text"])
        all_scores.append(sample["score"])
    print()
    print(f"Overall mean score: {mean(all_scores):.3f}")
    print(f"Overall visible achievement names: {dict(all_names.most_common(20))}")
    print(
        "Note: visible achievement text can miss achievements earned on the final action, "
        "because the final post-action observation is not always logged."
    )


def _print_good_trajectories(samples: list[dict], score_min: float, limit: int) -> None:
    good = [sample for sample in samples if sample["score"] >= score_min]
    good.sort(key=lambda x: (int(x["step"]), int(x["sample_i"])))

    print()
    print(f"Trajectories with score >= {score_min:g}: {len(good)}")
    for sample in good[:limit]:
        actions = sample["exec_actions"]
        action_counts = Counter(actions)
        do_ratio = (action_counts["do"] / len(actions)) if actions else 0.0
        streak_action, streak_len = _longest_streak(actions)
        first_answers = sample["answers"][:8]
        visible_names = dict(sample["ach_names_text"].most_common())
        print()
        print(
            f"{sample['path'].name} step={sample['step']} sample={sample['sample_i']} "
            f"score={sample['score']:.1f} visible_ach={sample['max_ach_text']}"
        )
        print(f"actions={actions}")
        print(
            f"action_counts={dict(action_counts)} do_ratio={do_ratio:.2f} "
            f"longest_streak={streak_action}:{streak_len} none_turns={sample['none_turns']}"
        )
        print(f"visible_achievement_names={visible_names}")
        print(f"first_answers={first_answers}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", nargs="+", type=Path)
    parser.add_argument("--score-min", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    samples: list[dict] = []
    for path in args.tables:
        samples.extend(_iter_samples(path))

    if not samples:
        raise SystemExit("No samples found.")

    _print_summary(samples, args.score_min)
    _print_good_trajectories(samples, args.score_min, args.limit)


if __name__ == "__main__":
    main()
