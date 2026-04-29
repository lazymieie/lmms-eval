#!/usr/bin/env python3
"""Evaluate VideoSeek run outputs against VideoMME ground truth.

The script scans logs/videoseek_runs/videomme_doc{doc_id}_idx{idx}/, selects one
successful run per doc, extracts the final option letter from prediction.json
or trajectory.json, and recomputes aggregate accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("/gemini/space/gjx/lmms-eval/logs/videoseek_runs")
DEFAULT_SAMPLES_GLOB = "/gemini/space/gjx/lmms-eval/logs/qwen35_397b_videomme_api/*/*samples_videomme.jsonl"
CHOICES = {"A", "B", "C", "D"}


def parse_parent(parent: Path) -> tuple[int, int] | None:
    match = re.fullmatch(r"videomme_doc(\d+)_idx(\d+)", parent.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def has_result(run_dir: Path) -> bool:
    return (run_dir / "prediction.json").is_file() and (run_dir / "trajectory.json").is_file()


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

    matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", tail, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sample_doc_map(paths: list[Path]) -> dict[int, dict[str, Any]]:
    docs: dict[int, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = sample.get("doc_id")
                metric = sample.get("videomme_perception_score") or {}
                video_id = metric.get("videoID")
                target = sample.get("target") or metric.get("answer")
                if isinstance(doc_id, int) and video_id and target:
                    docs[doc_id] = {
                        "videoID": video_id,
                        "target": str(target).strip().upper(),
                        "input": sample.get("input", ""),
                        "question_id": metric.get("question_id"),
                        "duration": metric.get("duration"),
                        "category": metric.get("category"),
                        "sub_category": metric.get("sub_category"),
                        "task_category": metric.get("task_category"),
                    }
    return docs


def discover_sample_paths(samples_jsonl: list[Path] | None, samples_glob: str | None) -> list[Path]:
    paths = list(samples_jsonl or [])
    if not samples_glob:
        return paths
    if samples_glob.startswith("/"):
        paths.extend(sorted(Path("/").glob(samples_glob.lstrip("/"))))
    else:
        paths.extend(sorted(Path().glob(samples_glob)))
    return paths


def select_run(parent: Path, mode: str) -> Path | None:
    children = [child for child in parent.iterdir() if child.is_dir()]
    if not children:
        return None
    children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if mode == "latest_success":
        for child in children:
            if has_result(child):
                return child
        return None
    if mode == "latest_any":
        return children[0]
    raise ValueError(f"unsupported selection mode: {mode}")


def evaluate_one(doc_id: int, idx: int, parent: Path, run_dir: Path | None, docs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    doc = docs.get(doc_id, {})
    record: dict[str, Any] = {
        "doc_id": doc_id,
        "idx": idx,
        "parent_dir": str(parent),
        "run_dir": str(run_dir) if run_dir else None,
        "videoID": doc.get("videoID"),
        "target": doc.get("target", ""),
        "question_id": doc.get("question_id"),
        "duration": doc.get("duration"),
        "category": doc.get("category"),
        "sub_category": doc.get("sub_category"),
        "task_category": doc.get("task_category"),
        "has_result": False,
        "pred_text": "",
        "pred_answer": "",
        "score": 0.0,
    }
    if run_dir is None or not has_result(run_dir):
        return record

    prediction = load_json(run_dir / "prediction.json")
    trajectory = load_json(run_dir / "trajectory.json")
    pred_text = as_text(prediction.get("prediction")) or as_text(trajectory.get("final_answer"))
    pred_answer = extract_option(pred_text)

    record.update(
        {
            "has_result": True,
            "pred_text": pred_text,
            "pred_answer": pred_answer,
            "score": 1.0 if pred_answer and pred_answer == record["target"] else 0.0,
            "finish_reason": trajectory.get("finish_reason"),
            "total_steps": trajectory.get("total_steps"),
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--samples-jsonl",
        type=Path,
        action="append",
        default=None,
        help="Existing lmms-eval samples JSONL to use for doc_id -> answer mapping. Can be repeated.",
    )
    parser.add_argument("--samples-glob", default=DEFAULT_SAMPLES_GLOB)
    parser.add_argument("--select-run", choices=["latest_success", "latest_any"], default="latest_success")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-id", type=int, action="append", default=None)
    parser.add_argument("--output-jsonl", type=Path, default=Path("logs/videoseek_videomme_eval.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("logs/videoseek_videomme_eval_summary.json"))
    args = parser.parse_args()

    sample_paths = discover_sample_paths(args.samples_jsonl, args.samples_glob)
    docs = load_sample_doc_map(sample_paths)
    if not docs:
        raise SystemExit("No doc mapping loaded from samples JSONL. Pass --samples-jsonl explicitly.")

    rows: list[tuple[int, int, Path]] = []
    for parent in sorted(args.runs_root.glob("videomme_doc*_idx*")):
        parsed = parse_parent(parent)
        if parsed is None or not parent.is_dir():
            continue
        doc_id, idx = parsed
        if doc_id not in docs:
            continue
        rows.append((doc_id, idx, parent))

    if args.doc_id is not None:
        wanted = set(args.doc_id)
        rows = [row for row in rows if row[0] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for doc_id, idx, parent in rows:
        run_dir = select_run(parent, args.select_run)
        results.append(evaluate_one(doc_id, idx, parent, run_dir, docs))

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_docs = len(results)
    answered_docs = sum(1 for record in results if record["has_result"])
    missing_docs = total_docs - answered_docs
    valid_preds = sum(1 for record in results if record["pred_answer"] in CHOICES)
    correct = sum(int(record["score"]) for record in results)
    accuracy_total = (correct / total_docs) if total_docs else 0.0
    accuracy_answered = (correct / answered_docs) if answered_docs else 0.0
    accuracy_valid = (correct / valid_preds) if valid_preds else 0.0

    summary = {
        "runs_root": str(args.runs_root),
        "select_run": args.select_run,
        "total_docs": total_docs,
        "answered_docs": answered_docs,
        "missing_docs": missing_docs,
        "valid_preds": valid_preds,
        "correct": correct,
        "accuracy_total": accuracy_total,
        "accuracy_answered": accuracy_answered,
        "accuracy_valid": accuracy_valid,
        "output_jsonl": str(args.output_jsonl),
    }
    with args.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"output_jsonl: {args.output_jsonl}")
    print(f"summary_json: {args.summary_json}")
    print(f"total_docs: {total_docs}")
    print(f"answered_docs: {answered_docs}")
    print(f"missing_docs: {missing_docs}")
    print(f"valid_preds: {valid_preds}")
    print(f"correct: {correct}")
    print(f"accuracy_total: {accuracy_total:.6f}")
    print(f"accuracy_answered: {accuracy_answered:.6f}")
    print(f"accuracy_valid: {accuracy_valid:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
