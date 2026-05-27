#!/usr/bin/env python3
"""Compare VideoSeek VideoMME results against a baseline samples JSONL.

The script aligns records by ``doc_id`` and writes:

* ``comparison.jsonl``: one row per aligned sample
* ``changed.jsonl``: samples where correctness differs
* ``improved.jsonl``: baseline wrong, VideoSeek correct
* ``regressed.jsonl``: baseline correct, VideoSeek wrong
* ``summary.json``: aggregate counts and rates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHOICES = {"A", "B", "C", "D"}
DEFAULT_VIDEOSEEK_SCORES = "videomme_scores.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value)
    return str(value)


def resolve_videoseek_scores(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / DEFAULT_VIDEOSEEK_SCORES
    if not path.is_file():
        raise FileNotFoundError(f"VideoSeek score JSONL not found: {path}")
    return path


def by_doc_id(records: list[dict[str, Any]], source: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if "doc_id" not in record:
            raise KeyError(f"{source} record has no doc_id: {record}")
        doc_id = int(record["doc_id"])
        if doc_id in indexed:
            raise ValueError(f"{source} has duplicate doc_id={doc_id}")
        indexed[doc_id] = record
    return indexed


def baseline_score(record: dict[str, Any]) -> dict[str, Any]:
    metric = record.get("videomme_perception_score")
    if not isinstance(metric, dict):
        metric = {}
    pred_answer = as_text(metric.get("pred_answer") or record.get("filtered_resps")).strip().upper()
    answer = as_text(metric.get("answer") or record.get("target")).strip().upper()
    raw_score = metric.get("score")
    if raw_score is None:
        raw_score = 1.0 if pred_answer == answer and pred_answer in CHOICES else 0.0
    return {
        "pred_answer": pred_answer,
        "answer": answer,
        "score": float(raw_score),
        "correct": float(raw_score) == 1.0,
        "question_id": metric.get("question_id"),
        "videoID": metric.get("videoID"),
        "duration": metric.get("duration"),
        "category": metric.get("category"),
        "sub_category": metric.get("sub_category"),
        "task_category": metric.get("task_category"),
    }


def videoseek_score(record: dict[str, Any]) -> dict[str, Any]:
    score = float(record.get("score", 0.0) or 0.0)
    sample_dir = Path(record["sample_dir"]).expanduser().resolve() if record.get("sample_dir") else None
    return {
        "pred_answer": as_text(record.get("pred_answer")).strip().upper(),
        "raw_prediction": as_text(record.get("raw_prediction")),
        "answer": as_text(record.get("answer")).strip().upper(),
        "score": score,
        "correct": score == 1.0,
        "question_id": record.get("question_id"),
        "videoID": record.get("videoID"),
        "duration": record.get("duration"),
        "category": record.get("category"),
        "sub_category": record.get("sub_category"),
        "task_category": record.get("task_category"),
        "sample_dir": str(sample_dir) if sample_dir else None,
        "trajectory_path": str(sample_dir / "trajectory.json") if sample_dir else None,
        "prediction_path": str(sample_dir / "prediction.json") if sample_dir else None,
        "metrics_path": str(sample_dir / "metrics.json") if sample_dir else None,
        "prediction_error": record.get("prediction_error"),
        "finish_reason": record.get("finish_reason"),
        "is_valid_choice": record.get("is_valid_choice"),
    }


def summarize(comparison: list[dict[str, Any]], missing_baseline: list[int], missing_videoseek: list[int]) -> dict[str, Any]:
    total = len(comparison)
    baseline_correct = sum(1 for item in comparison if item["baseline_correct"])
    videoseek_correct = sum(1 for item in comparison if item["videoseek_correct"])
    both_correct = sum(1 for item in comparison if item["baseline_correct"] and item["videoseek_correct"])
    both_wrong = sum(1 for item in comparison if not item["baseline_correct"] and not item["videoseek_correct"])
    improved = sum(1 for item in comparison if item["change_type"] == "improved")
    regressed = sum(1 for item in comparison if item["change_type"] == "regressed")
    changed_correctness = improved + regressed
    same_correctness = total - changed_correctness

    def pct(count: int) -> float:
        return 100.0 * count / total if total else 0.0

    return {
        "aligned_samples": total,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": pct(baseline_correct),
        "videoseek_correct": videoseek_correct,
        "videoseek_accuracy": pct(videoseek_correct),
        "delta_correct": videoseek_correct - baseline_correct,
        "delta_accuracy": pct(videoseek_correct) - pct(baseline_correct),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "same_correctness": same_correctness,
        "changed_correctness": changed_correctness,
        "improved": improved,
        "regressed": regressed,
        "missing_in_baseline": missing_baseline,
        "missing_in_videoseek": missing_videoseek,
    }


def build_comparison(videoseek_records: list[dict[str, Any]], baseline_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    videoseek_by_doc = by_doc_id(videoseek_records, "VideoSeek")
    baseline_by_doc = by_doc_id(baseline_records, "baseline")
    common_doc_ids = sorted(set(videoseek_by_doc) & set(baseline_by_doc))
    missing_baseline = sorted(set(videoseek_by_doc) - set(baseline_by_doc))
    missing_videoseek = sorted(set(baseline_by_doc) - set(videoseek_by_doc))

    comparison: list[dict[str, Any]] = []
    for doc_id in common_doc_ids:
        vs = videoseek_score(videoseek_by_doc[doc_id])
        base = baseline_score(baseline_by_doc[doc_id])
        if vs["correct"] and not base["correct"]:
            change_type = "improved"
        elif base["correct"] and not vs["correct"]:
            change_type = "regressed"
        elif vs["correct"] and base["correct"]:
            change_type = "both_correct"
        else:
            change_type = "both_wrong"

        comparison.append(
            {
                "doc_id": doc_id,
                "question_id": vs["question_id"] or base["question_id"],
                "videoID": vs["videoID"] or base["videoID"],
                "duration": vs["duration"] or base["duration"],
                "category": vs["category"] or base["category"],
                "sub_category": vs["sub_category"] or base["sub_category"],
                "task_category": vs["task_category"] or base["task_category"],
                "answer": vs["answer"] or base["answer"],
                "baseline_pred_answer": base["pred_answer"],
                "baseline_score": base["score"],
                "baseline_correct": base["correct"],
                "videoseek_pred_answer": vs["pred_answer"],
                "videoseek_raw_prediction": vs["raw_prediction"],
                "videoseek_score": vs["score"],
                "videoseek_correct": vs["correct"],
                "change_type": change_type,
                "trajectory_path": vs["trajectory_path"],
                "sample_dir": vs["sample_dir"],
                "prediction_path": vs["prediction_path"],
                "metrics_path": vs["metrics_path"],
                "videoseek_finish_reason": vs["finish_reason"],
                "videoseek_prediction_error": vs["prediction_error"],
                "input": baseline_by_doc[doc_id].get("input"),
            }
        )
    return comparison, missing_baseline, missing_videoseek


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoseek", required=True, type=Path, help="VideoSeek run dir or videomme_scores.jsonl")
    parser.add_argument("--baseline-samples", required=True, type=Path, help="Baseline *_samples_videomme_long.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for comparison outputs")
    args = parser.parse_args()

    videoseek_scores = resolve_videoseek_scores(args.videoseek)
    baseline_samples = args.baseline_samples.expanduser().resolve()
    if not baseline_samples.is_file():
        raise FileNotFoundError(f"Baseline samples JSONL not found: {baseline_samples}")

    comparison, missing_baseline, missing_videoseek = build_comparison(load_jsonl(videoseek_scores), load_jsonl(baseline_samples))
    changed = [item for item in comparison if item["change_type"] in {"improved", "regressed"}]
    improved = [item for item in comparison if item["change_type"] == "improved"]
    regressed = [item for item in comparison if item["change_type"] == "regressed"]

    output_dir = args.output_dir.expanduser().resolve()
    write_jsonl(output_dir / "comparison.jsonl", comparison)
    write_jsonl(output_dir / "changed.jsonl", changed)
    write_jsonl(output_dir / "improved.jsonl", improved)
    write_jsonl(output_dir / "regressed.jsonl", regressed)

    summary = summarize(comparison, missing_baseline, missing_videoseek)
    summary["videoseek_scores"] = str(videoseek_scores)
    summary["baseline_samples"] = str(baseline_samples)
    summary["output_dir"] = str(output_dir)
    write_json(output_dir / "summary.json", summary)

    print(f"aligned_samples: {summary['aligned_samples']}")
    print(f"baseline_correct: {summary['baseline_correct']} ({summary['baseline_accuracy']:.2f})")
    print(f"videoseek_correct: {summary['videoseek_correct']} ({summary['videoseek_accuracy']:.2f})")
    print(f"delta_correct: {summary['delta_correct']}")
    print(f"improved: {summary['improved']}")
    print(f"regressed: {summary['regressed']}")
    print(f"changed_correctness: {summary['changed_correctness']}")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
