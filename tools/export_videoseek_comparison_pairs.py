#!/usr/bin/env python3
"""Export per-task baseline/VideoSeek JSONL pairs for manual analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def by_doc_id(records: list[dict[str, Any]], source: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        doc_id = int(record["doc_id"])
        if doc_id in indexed:
            raise ValueError(f"{source} has duplicate doc_id={doc_id}")
        indexed[doc_id] = record
    return indexed


def safe_name(value: Any) -> str:
    text = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def first_generation_info(sample: dict[str, Any]) -> dict[str, Any]:
    generation_info = sample.get("generation_info") or []
    if isinstance(generation_info, list) and generation_info:
        first = generation_info[0]
        if isinstance(first, dict):
            return first
    return {}


def baseline_pair_record(sample: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    metric = sample.get("videomme_perception_score") or {}
    generation_info = first_generation_info(sample)
    return {
        "record_type": "baseline_without_videoseek",
        "doc_id": sample.get("doc_id"),
        "question_id": comparison.get("question_id") or metric.get("question_id"),
        "videoID": comparison.get("videoID") or metric.get("videoID"),
        "duration": comparison.get("duration") or metric.get("duration"),
        "category": comparison.get("category") or metric.get("category"),
        "sub_category": comparison.get("sub_category") or metric.get("sub_category"),
        "task_category": comparison.get("task_category") or metric.get("task_category"),
        "input": sample.get("input"),
        "answer": metric.get("answer") or sample.get("target"),
        "pred_answer": metric.get("pred_answer") or sample.get("filtered_resps"),
        "raw_response": sample.get("resps"),
        "filtered_response": sample.get("filtered_resps"),
        "score": metric.get("score"),
        "correct": comparison.get("baseline_correct"),
        "reason": generation_info.get("reasoning"),
        "finish_reason": generation_info.get("finish_reason"),
        "frames_used": generation_info.get("frames_used"),
        "token_counts": sample.get("token_counts"),
        "doc_hash": sample.get("doc_hash"),
    }


def videoseek_pair_record(comparison: dict[str, Any]) -> dict[str, Any]:
    trajectory_path = Path(comparison["trajectory_path"])
    trajectory = load_json(trajectory_path)
    return {
        "record_type": "with_videoseek",
        "doc_id": comparison.get("doc_id"),
        "question_id": comparison.get("question_id"),
        "videoID": comparison.get("videoID"),
        "duration": comparison.get("duration"),
        "category": comparison.get("category"),
        "sub_category": comparison.get("sub_category"),
        "task_category": comparison.get("task_category"),
        "input": comparison.get("input"),
        "answer": comparison.get("answer"),
        "pred_answer": comparison.get("videoseek_pred_answer"),
        "raw_prediction": comparison.get("videoseek_raw_prediction"),
        "score": comparison.get("videoseek_score"),
        "correct": comparison.get("videoseek_correct"),
        "trajectory_path": str(trajectory_path),
        "sample_dir": comparison.get("sample_dir"),
        "prediction_path": comparison.get("prediction_path"),
        "metrics_path": comparison.get("metrics_path"),
        "finish_reason": comparison.get("videoseek_finish_reason"),
        "prediction_error": comparison.get("videoseek_prediction_error"),
        "trajectory": trajectory,
    }


def export_pairs(comparison_path: Path, baseline_samples_path: Path, output_dir: Path) -> dict[str, Any]:
    comparisons = load_jsonl(comparison_path)
    baseline_by_doc = by_doc_id(load_jsonl(baseline_samples_path), "baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for comparison in comparisons:
        doc_id = int(comparison["doc_id"])
        if doc_id not in baseline_by_doc:
            raise KeyError(f"baseline sample missing doc_id={doc_id}")
        question_id = safe_name(comparison.get("question_id"))
        output_path = output_dir / f"doc{doc_id:04d}_{question_id}.jsonl"
        pair = [
            baseline_pair_record(baseline_by_doc[doc_id], comparison),
            videoseek_pair_record(comparison),
        ]
        with output_path.open("w", encoding="utf-8") as handle:
            for record in pair:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written.append(str(output_path))

    manifest = {
        "comparison_path": str(comparison_path),
        "baseline_samples_path": str(baseline_samples_path),
        "output_dir": str(output_dir),
        "num_tasks": len(written),
        "files": written,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-jsonl", required=True, type=Path, help="improved.jsonl or regressed.jsonl")
    parser.add_argument("--baseline-samples", required=True, type=Path, help="Baseline *_samples_videomme_long.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for one-file-per-task JSONL pairs")
    args = parser.parse_args()

    manifest = export_pairs(
        args.comparison_jsonl.expanduser().resolve(),
        args.baseline_samples.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(f"num_tasks: {manifest['num_tasks']}")
    print(f"output_dir: {manifest['output_dir']}")
    print(f"manifest: {Path(manifest['output_dir']) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
