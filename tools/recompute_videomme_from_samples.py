#!/usr/bin/env python3
"""Recompute VideoMME sample accuracy from a samples JSONL file.

This is useful when a model returns a long rationale with the final option
letter at the end and the original filter did not extract it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHOICES = {"A", "B", "C", "D"}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value)
    return str(value)


def extract_option(text: str) -> str:
    text = as_text(text).strip()
    if not text:
        return ""

    tail = text[-4000:]
    patterns = [
        r"(?:final\s+answer|answer|option|choice|therefore|so)\s*(?:is|:)?\s*[\(\[\{]*\s*([A-D])\s*[\)\]\}.]?\s*$",
        r"答案\s*(?:是|:)?\s*([A-D])\s*$",
        r"正确答案\s*(?:是|:)?\s*([A-D])\s*$",
        r"([A-D])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, tail, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: choose the last standalone option letter in the tail.
    matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", tail, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def sample_text(sample: dict[str, Any]) -> str:
    if "resps" in sample:
        return as_text(sample["resps"])
    return as_text(sample.get("filtered_resps"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path, help="Path to *_samples_videomme.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Path for corrected JSONL")
    args = parser.parse_args()

    output = args.output or args.samples.with_name(args.samples.stem + "_recomputed.jsonl")

    total = 0
    valid = 0
    correct_old = 0
    correct_new = 0
    changed = 0
    fixed_correct = 0
    bad_lines: list[tuple[int, str]] = []
    missing_new = 0

    with args.samples.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for lineno, line in enumerate(src, 1):
            total += 1
            if not line.strip():
                bad_lines.append((lineno, "blank line"))
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_lines.append((lineno, f"JSONDecodeError: {exc}"))
                continue

            valid += 1
            target = as_text(sample.get("target")).strip().upper()
            old_pred = as_text(sample.get("filtered_resps")).strip().upper()
            new_pred = extract_option(sample_text(sample))

            if old_pred == target:
                correct_old += 1
            if new_pred == target:
                correct_new += 1
            if new_pred != old_pred:
                changed += 1
            if old_pred != target and new_pred == target:
                fixed_correct += 1
            if new_pred not in CHOICES:
                missing_new += 1

            sample["recomputed_pred_answer"] = new_pred
            sample["recomputed_score"] = 1.0 if new_pred == target else 0.0
            if isinstance(sample.get("videomme_perception_score"), dict):
                sample["videomme_perception_score_recomputed"] = dict(sample["videomme_perception_score"])
                sample["videomme_perception_score_recomputed"]["pred_answer"] = new_pred
                sample["videomme_perception_score_recomputed"]["score"] = sample["recomputed_score"]

            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"input: {args.samples}")
    print(f"output: {output}")
    print(f"total lines: {total}")
    print(f"valid samples: {valid}")
    print(f"bad lines skipped: {len(bad_lines)}")
    if bad_lines:
        print("bad line examples:")
        for lineno, err in bad_lines[:10]:
            print(f"  line {lineno}: {err}")
    print(f"old correct: {correct_old}/{valid} = {correct_old / valid:.6f}")
    print(f"new correct: {correct_new}/{valid} = {correct_new / valid:.6f}")
    print(f"changed predictions: {changed}")
    print(f"old wrong -> new correct: {fixed_correct}")
    print(f"new missing/non-ABCD: {missing_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
