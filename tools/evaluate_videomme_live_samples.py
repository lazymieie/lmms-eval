#!/usr/bin/env python3
"""Score VideoMME live sample directories written before evaluator completed.

This targets the ``run_*_live_samples`` layout written by
``lmms_eval.evaluator._write_live_sample_file``:

    run_20260528_xxx_live_samples/
      run_manifest.json
      videomme_long_wo_subtitle_doc0_rank000/
        sample.json
        raw_response.txt
        filtered_response.txt

or alternative response-only layouts such as:

    run_20260528_xxx_live_samples/
      videomme_long_doc897_rank000/
        response_000897.json
        output.txt
        reasoning.txt

It reads per-sample JSON payloads and recomputes VideoMME scores from the saved
task/doc_id plus prediction text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.tasks.videomme.utils import videomme_aggregate_results, videomme_process_results


SAMPLE_DIR_PATTERN = re.compile(r"(?P<task>.+)_doc(?P<doc_id>\d+)_rank(?P<rank>\d+)$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(as_text(item) for item in value)
    return str(value)


def parse_sample_dir_name(path: Path) -> dict[str, Any] | None:
    match = SAMPLE_DIR_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return {
        "task": match.group("task"),
        "doc_id": int(match.group("doc_id")),
        "rank": int(match.group("rank")),
    }


def pick_sample_file(sample_dir: Path, filter_key: str | None) -> Path | None:
    if filter_key:
        candidate = sample_dir / f"sample_{filter_key}.json"
        if candidate.is_file():
            return candidate
    default_candidate = sample_dir / "sample.json"
    if default_candidate.is_file():
        return default_candidate
    candidates = sorted(sample_dir.glob("sample*.json"))
    if candidates:
        return candidates[0]
    response_candidates = sorted(sample_dir.glob("response_*.json"))
    if response_candidates:
        return response_candidates[0]
    generic_candidates = sorted(sample_dir.glob("*.json"))
    if generic_candidates:
        return generic_candidates[0]
    return None


def discover_sample_files(live_samples_dir: Path, filter_key: str | None) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    for child in sorted(live_samples_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_sample_dir_name(child)
        if parsed is None:
            continue
        sample_file = pick_sample_file(child, filter_key)
        if sample_file is None:
            continue
        discovered.append(
            {
                **parsed,
                "sample_dir": str(child),
                "sample_file": str(sample_file),
            }
        )
    return discovered


def choose_prediction_text(sample: dict[str, Any], sample_dir: Path) -> str:
    filtered = sample.get("filtered_resps")
    filtered_text = as_text(filtered).strip()
    if filtered_text:
        return filtered_text
    response_text = as_text(sample.get("response")).strip()
    if response_text:
        return response_text
    resps_text = as_text(sample.get("resps")).strip()
    if resps_text:
        return resps_text
    output_path = sample_dir / "output.txt"
    if output_path.is_file():
        return output_path.read_text(encoding="utf-8").strip()
    return ""


def load_task_docs(task_names: list[str], split: str) -> dict[str, Any]:
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


def evaluate_sample(sample_info: dict[str, Any], docs_by_task: dict[str, Any]) -> dict[str, Any]:
    sample_path = Path(sample_info["sample_file"])
    sample_dir = Path(sample_info["sample_dir"])
    payload = load_json(sample_path)
    task_name = str(payload.get("task_name") or sample_info["task"])
    doc_id = int(payload.get("doc_id", sample_info["doc_id"]))
    doc = docs_by_task[task_name][doc_id]
    prediction_text = choose_prediction_text(payload, sample_dir)
    metric = videomme_process_results(doc, [prediction_text])["videomme_perception_score"]
    success = payload.get("success")
    generation_info = payload.get("generation_info")
    finish_reason = generation_info.get("finish_reason") if isinstance(generation_info, dict) else None

    return {
        "task": task_name,
        "doc_id": doc_id,
        "rank": int(payload.get("rank", sample_info["rank"])),
        "request_index": payload.get("request_index"),
        "sample_dir": str(sample_dir),
        "sample_file": str(sample_path),
        "question_id": metric["question_id"],
        "videoID": metric["videoID"],
        "duration": metric["duration"],
        "category": metric["category"],
        "sub_category": metric["sub_category"],
        "task_category": metric["task_category"],
        "answer": metric["answer"],
        "prediction_text": prediction_text,
        "pred_answer": metric["pred_answer"],
        "score": float(metric["score"]),
        "success": success,
        "error": payload.get("error"),
        "finish_reason": finish_reason,
        "token_counts": payload.get("token_counts"),
        "written_at": payload.get("written_at"),
    }


def summarize(records: list[dict[str, Any]], live_samples_dir: Path) -> dict[str, Any]:
    metrics = [
        {
            "question_id": record["question_id"],
            "duration": record["duration"],
            "category": record["category"],
            "sub_category": record["sub_category"],
            "task_category": record["task_category"],
            "pred_answer": record["pred_answer"],
            "answer": record["answer"],
            "score": record["score"],
            "videoID": record["videoID"],
        }
        for record in records
    ]
    overall = videomme_aggregate_results(metrics) if metrics else 0.0
    correct = sum(int(record["score"]) for record in records)
    invalid_predictions = sum(1 for record in records if record["pred_answer"] not in {"A", "B", "C", "D"})
    return {
        "live_samples_dir": str(live_samples_dir),
        "num_samples": len(records),
        "correct": correct,
        "invalid_predictions": invalid_predictions,
        "overall_score": overall,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("live_samples_dir", type=Path, help="Path to one run_*_live_samples directory")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--filter-key", default=None, help="Prefer sample_<filter-key>.json over sample.json")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Per-sample score report path")
    parser.add_argument("--summary-json", type=Path, default=None, help="Aggregate score report path")
    args = parser.parse_args()

    live_samples_dir = args.live_samples_dir.expanduser().resolve()
    if not live_samples_dir.is_dir():
        raise SystemExit(f"Live samples directory not found: {live_samples_dir}")

    samples = discover_sample_files(live_samples_dir, args.filter_key)
    if not samples:
        raise SystemExit(f"No sample/result json files found under: {live_samples_dir}")

    task_names = sorted({sample["task"] for sample in samples})
    docs_by_task = load_task_docs(task_names, split=args.split)
    records = [evaluate_sample(sample_info, docs_by_task) for sample_info in samples]
    records.sort(key=lambda item: (item["task"], item["doc_id"], item["rank"]))

    default_prefix = live_samples_dir / "videomme_live_scores"
    output_jsonl = args.output_jsonl or default_prefix.with_suffix(".jsonl")
    summary_json = args.summary_json or default_prefix.with_name(default_prefix.name + "_summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(records, live_samples_dir)
    summary["output_jsonl"] = str(output_jsonl)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"live_samples_dir: {live_samples_dir}")
    print(f"samples: {summary['num_samples']}")
    print(f"correct: {summary['correct']}")
    print(f"invalid_predictions: {summary['invalid_predictions']}")
    print(f"overall_score: {summary['overall_score']:.4f}")
    print(f"output_jsonl: {output_jsonl}")
    print(f"summary_json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
