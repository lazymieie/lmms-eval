#!/usr/bin/env python3
"""Score native VideoSeek VideoMME-v2 run directories.

This script targets the native VideoSeek artifact layout produced by
`lmms_eval.models.simple.videoseek`, for example:

    run_20260513_xxx/
      run_manifest.json
      run_summary.json
      videomme_v2_doc0_idx0/
        prediction.json
        trajectory.json
        metrics.json

It reads the saved per-sample predictions, looks up the corresponding
VideoMME-v2 ground truth via task/doc_id, and writes a score report.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SAMPLE_DIR_PATTERN = re.compile(r"(?P<task>.+)_doc(?P<doc_id>\d+)_idx(?P<idx>\d+)$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_sample_dir_name(path: Path) -> dict[str, Any] | None:
    match = SAMPLE_DIR_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return {
        "task": match.group("task"),
        "doc_id": int(match.group("doc_id")),
        "index": int(match.group("idx")),
        "sample_dir": str(path),
    }


def discover_samples(run_dir: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}

    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.is_file():
        payload = load_json(run_summary_path)
        samples = payload.get("samples", [])
        for sample in samples:
            normalized = {
                "task": str(sample["task"]),
                "doc_id": int(sample["doc_id"]),
                "index": int(sample["index"]),
                "sample_dir": str(sample["sample_dir"]),
            }
            by_key[(normalized["task"], normalized["doc_id"], normalized["index"])] = normalized

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_sample_dir_name(child)
        if parsed is not None:
            by_key[(parsed["task"], parsed["doc_id"], parsed["index"])] = parsed
    return sorted(by_key.values(), key=lambda item: (item["task"], item["doc_id"], item["index"]))


def load_task_docs(task_names: list[str], split: str) -> dict[str, Any]:
    from lmms_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager("WARNING")
    task_dict = get_task_dict(task_names, task_manager=task_manager)
    docs_by_task: dict[str, Any] = {}
    for task_name, task in task_dict.items():
        if split == "test" and task.has_test_docs():
            docs_by_task[task_name] = task.test_docs()
        elif split == "validation" and task.has_validation_docs():
            docs_by_task[task_name] = task.validation_docs()
        elif task.has_test_docs():
            docs_by_task[task_name] = task.test_docs()
        elif task.has_validation_docs():
            docs_by_task[task_name] = task.validation_docs()
        else:
            raise ValueError(f"Task {task_name} has neither test nor validation docs")
    return docs_by_task


def read_prediction(sample_dir: Path) -> tuple[str, str | None]:
    prediction_path = sample_dir / "prediction.json"
    if not prediction_path.is_file():
        return "", "missing prediction.json"
    payload = load_json(prediction_path)
    return str(payload.get("prediction", "") or ""), payload.get("error")


def evaluate_sample(sample: dict[str, Any], docs_by_task: dict[str, Any]) -> dict[str, Any]:
    from lmms_eval.tasks.videomme_v2.utils import videomme_v2_process_results

    sample_dir = Path(sample["sample_dir"])
    task_name = sample["task"]
    doc_id = int(sample["doc_id"])
    doc = docs_by_task[task_name][doc_id]

    raw_prediction, prediction_error = read_prediction(sample_dir)
    metric = videomme_v2_process_results(doc, [raw_prediction])["videomme_v2_score"]
    trajectory_path = sample_dir / "trajectory.json"
    finish_reason = None
    if trajectory_path.is_file():
        trajectory = load_json(trajectory_path)
        finish_reason = trajectory.get("finish_reason")

    return {
        "task": task_name,
        "doc_id": doc_id,
        "index": int(sample["index"]),
        "sample_dir": str(sample_dir),
        "question_id": metric["question_id"],
        "video_id": metric["video_id"],
        "group_type": metric["group_type"],
        "group_structure": metric["group_structure"],
        "level": metric.get("level"),
        "second_head": metric.get("second_head"),
        "third_head": metric.get("third_head"),
        "answer": metric["answer"],
        "raw_prediction": raw_prediction,
        "pred_answer": metric["pred_answer"],
        "score": metric["score"],
        "prediction_error": prediction_error,
        "finish_reason": finish_reason,
        "is_valid_choice": metric["pred_answer"] in {"A", "B", "C", "D", "E", "F", "G", "H"},
    }


def summarize(records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    from lmms_eval.tasks.videomme_v2.utils import (
        videomme_v2_aggregate_level_1,
        videomme_v2_aggregate_level_2,
        videomme_v2_aggregate_level_3,
        videomme_v2_aggregate_logic,
        videomme_v2_aggregate_relevance,
        videomme_v2_aggregate_results,
    )

    metrics = [
        {
            "video_id": record["video_id"],
            "question_id": record["question_id"],
            "group_type": record["group_type"],
            "group_structure": record["group_structure"],
            "level": record["level"],
            "second_head": record["second_head"],
            "third_head": record["third_head"],
            "pred_answer": record["pred_answer"],
            "answer": record["answer"],
            "score": record["score"],
        }
        for record in records
    ]
    invalid_predictions = sum(1 for record in records if not record["is_valid_choice"])
    correct = sum(int(record["score"]) for record in records)
    return {
        "run_dir": str(run_dir),
        "num_samples": len(records),
        "correct": correct,
        "invalid_predictions": invalid_predictions,
        "overall_score": videomme_v2_aggregate_results(metrics) if metrics else 0.0,
        "relevance_score": videomme_v2_aggregate_relevance(metrics) if metrics else 0.0,
        "logic_score": videomme_v2_aggregate_logic(metrics) if metrics else 0.0,
        "level_1_score": videomme_v2_aggregate_level_1(metrics) if metrics else 0.0,
        "level_2_score": videomme_v2_aggregate_level_2(metrics) if metrics else 0.0,
        "level_3_score": videomme_v2_aggregate_level_3(metrics) if metrics else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Path to one VideoSeek run_<timestamp> directory")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Per-sample score report path")
    parser.add_argument("--summary-json", type=Path, default=None, help="Aggregate score report path")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    samples = discover_samples(run_dir)
    if not samples:
        raise SystemExit(f"No sample directories found under: {run_dir}")

    task_names = sorted({sample["task"] for sample in samples})
    docs_by_task = load_task_docs(task_names, split=args.split)
    records = [evaluate_sample(sample, docs_by_task) for sample in samples]
    records.sort(key=lambda item: (item["task"], item["doc_id"], item["index"]))

    default_prefix = run_dir / "videomme_v2_scores"
    output_jsonl = args.output_jsonl or default_prefix.with_suffix(".jsonl")
    summary_json = args.summary_json or default_prefix.with_name(default_prefix.name + "_summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(records, run_dir)
    summary["output_jsonl"] = str(output_jsonl)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"run_dir: {run_dir}")
    print(f"samples: {summary['num_samples']}")
    print(f"correct: {summary['correct']}")
    print(f"invalid_predictions: {summary['invalid_predictions']}")
    print(f"overall_score: {summary['overall_score']:.4f}")
    print(f"relevance_score: {summary['relevance_score']:.4f}")
    print(f"logic_score: {summary['logic_score']:.4f}")
    print(f"level_1_score: {summary['level_1_score']:.4f}")
    print(f"level_2_score: {summary['level_2_score']:.4f}")
    print(f"level_3_score: {summary['level_3_score']:.4f}")
    print(f"output_jsonl: {output_jsonl}")
    print(f"summary_json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
