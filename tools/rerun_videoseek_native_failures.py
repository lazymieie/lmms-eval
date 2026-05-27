#!/usr/bin/env python3
"""Rerun missing or invalid native VideoSeek VideoMME samples in-place.

Targets the native artifact layout produced by `lmms_eval.models.simple.videoseek`.
It identifies:
 - missing sample directories for expected doc_ids
 - existing sample directories without prediction.json
 - existing samples whose prediction does not extract to A/B/C/D

Each target is rerun using the same subprocess-isolated VideoSeek worker and the
existing run directory is updated in place.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger as eval_logger

from lmms_eval.models.simple.videoseek import _run_request_in_subprocess, _safe_int, _write_failure_artifacts
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.tasks.videomme.utils import extract_characters_regex as extract_videomme_answer
from lmms_eval.tasks.videomme_v2.utils import extract_characters_regex as extract_videomme_v2_answer


def valid_choices_for_task(task_name: str | None) -> set[str]:
    normalized = str(task_name or "")
    if normalized.startswith("videomme_v2"):
        return {"A", "B", "C", "D", "E", "F", "G", "H"}
    return {"A", "B", "C", "D"}


def extract_prediction_answer(raw_prediction: str, task_name: str | None = None) -> str:
    normalized = str(task_name or "")
    if normalized.startswith("videomme_v2"):
        return extract_videomme_v2_answer(raw_prediction)
    return extract_videomme_answer(raw_prediction)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_sample_dir_name(path: Path) -> dict[str, Any] | None:
    import re

    match = re.fullmatch(r"(?P<task>.+)_doc(?P<doc_id>\d+)_idx(?P<idx>\d+)$", path.name)
    if match is None:
        return None
    return {
        "task": match.group("task"),
        "doc_id": int(match.group("doc_id")),
        "index": int(match.group("idx")),
        "sample_dir": str(path),
    }


def discover_existing_samples(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_sample_dir_name(child)
        if parsed is not None:
            records[(parsed["task"], parsed["doc_id"])] = parsed
    return records


def load_task_docs(task_names: list[str], split: str) -> dict[str, Any]:
    task_manager = TaskManager("WARNING")
    task_dict = get_task_dict(task_names, task_manager=task_manager)
    loaded: dict[str, Any] = {}
    for task_name, task in task_dict.items():
        if split == "test" and task.has_test_docs():
            docs = task.test_docs()
        elif split == "validation" and task.has_validation_docs():
            docs = task.validation_docs()
        elif task.has_test_docs():
            docs = task.test_docs()
        elif task.has_validation_docs():
            docs = task.validation_docs()
        else:
            raise ValueError(f"Task {task_name} has neither test nor validation docs")
        loaded[task_name] = {"task": task, "docs": docs}
    return loaded


def classify_existing_sample(sample_dir: Path, task_name: str | None = None) -> tuple[str, str]:
    prediction_path = sample_dir / "prediction.json"
    if not prediction_path.is_file():
        return "missing_prediction_file", "prediction.json is missing"
    try:
        payload = load_json(prediction_path)
    except Exception as exc:
        return "invalid_prediction_json", str(exc)
    raw_prediction = str(payload.get("prediction", "") or "")
    pred_answer = extract_prediction_answer(raw_prediction, task_name=task_name)
    if pred_answer not in valid_choices_for_task(task_name):
        return "invalid_prediction", raw_prediction[:200]
    return "ok", pred_answer


def build_targets(
    run_dir: Path,
    loaded_tasks: dict[str, Any],
    existing: dict[tuple[str, int], dict[str, Any]],
    limit: int | None = None,
    doc_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for task_name, bundle in loaded_tasks.items():
        docs = bundle["docs"]
        total_docs = len(docs)
        max_docs = min(total_docs, limit) if limit is not None else total_docs
        for doc_id in range(max_docs):
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            key = (task_name, doc_id)
            sample = existing.get(key)
            if sample is None:
                targets.append(
                    {
                        "task": task_name,
                        "doc_id": doc_id,
                        "index": doc_id,
                        "sample_dir": str(run_dir / f"{task_name}_doc{doc_id}_idx{doc_id}"),
                        "reason": "missing_sample_dir",
                        "reason_detail": "",
                    }
                )
                continue

            status, detail = classify_existing_sample(Path(sample["sample_dir"]), task_name=sample["task"])
            if status != "ok":
                targets.append({**sample, "reason": status, "reason_detail": detail})
    return targets


def build_agent_config(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name or manifest.get("model_name", "openai/qwen3.5-4b"),
        "api_base": args.api_base or manifest.get("api_base", "http://127.0.0.1:8000/v1"),
        "api_key": args.api_key,
        "api_version": args.api_version,
        "reasoning_effort": args.reasoning_effort if args.reasoning_effort is not None else manifest.get("reasoning_effort", "medium"),
        "seed": args.seed,
        "temperature": args.temperature if args.temperature is not None else manifest.get("temperature", 0.0),
        "max_tokens": args.max_tokens if args.max_tokens is not None else manifest.get("max_tokens", 4096),
        "max_steps": args.max_steps if args.max_steps is not None else manifest.get("max_steps", 6),
        "timeout": args.timeout if args.timeout is not None else manifest.get("timeout", 1800),
        "decision_timeout": (
            args.decision_timeout if args.decision_timeout is not None else manifest.get("decision_timeout")
        ),
        "tool_timeout": args.tool_timeout if args.tool_timeout is not None else manifest.get("tool_timeout"),
        "final_answer_timeout": (
            args.final_answer_timeout if args.final_answer_timeout is not None else manifest.get("final_answer_timeout")
        ),
        "decision_api_retry_attempts": (
            args.decision_api_retry_attempts
            if args.decision_api_retry_attempts is not None
            else manifest.get("decision_api_retry_attempts")
        ),
        "tool_api_retry_attempts": (
            args.tool_api_retry_attempts
            if args.tool_api_retry_attempts is not None
            else manifest.get("tool_api_retry_attempts")
        ),
        "final_answer_api_retry_attempts": (
            args.final_answer_api_retry_attempts
            if args.final_answer_api_retry_attempts is not None
            else manifest.get("final_answer_api_retry_attempts")
        ),
    }


def backup_existing_artifacts(sample_dir: Path) -> None:
    backup_dir = sample_dir / "_backup_before_rerun"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_patterns = [
        "prediction.json",
        "trajectory.json",
        "metrics.json",
        "timeout_snapshot.json",
        "timeout_traceback.txt",
        "timeout_snapshot_error.txt",
        "decision_error_debug.jsonl",
        "decision_error_step*.json",
        "decision_empty_actions_step*.json",
    ]
    copied_paths: set[Path] = set()
    for pattern in artifact_patterns:
        for source in sample_dir.glob(pattern):
            if not source.is_file() or source in copied_paths:
                continue
            shutil.copy2(source, backup_dir / f"{timestamp}_{source.name}")
            copied_paths.add(source)


def cleanup_stale_artifacts(sample_dir: Path) -> None:
    artifact_patterns = [
        "prediction.json",
        "trajectory.json",
        "metrics.json",
        "timeout_snapshot.json",
        "timeout_traceback.txt",
        "timeout_snapshot_error.txt",
        "decision_error_debug.jsonl",
        "decision_error_step*.json",
        "decision_empty_actions_step*.json",
    ]
    removed_paths: set[Path] = set()
    for pattern in artifact_patterns:
        for target in sample_dir.glob(pattern):
            if not target.is_file() or target in removed_paths:
                continue
            target.unlink(missing_ok=True)
            removed_paths.add(target)


def rebuild_run_summary(run_dir: Path) -> None:
    sample_results: list[dict[str, Any]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_sample_dir_name(child)
        if parsed is None:
            continue
        prediction = ""
        prediction_path = child / "prediction.json"
        if prediction_path.is_file():
            try:
                prediction = str(load_json(prediction_path).get("prediction", "") or "")
            except Exception:
                prediction = ""
        usage_summary = {}
        frame_summary = {}
        metrics_path = child / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics = load_json(metrics_path)
                usage_summary = metrics.get("usage_summary", {})
                frame_summary = metrics.get("frame_summary", {})
            except Exception:
                usage_summary = {}
                frame_summary = {}
        sample_results.append(
            {
                "index": parsed["index"],
                "task": parsed["task"],
                "doc_id": parsed["doc_id"],
                "sample_dir": str(child),
                "prediction": prediction,
                "usage_summary": usage_summary,
                "frame_summary": frame_summary,
            }
        )

    aggregate = {
        "num_samples": len(sample_results),
        "total_prompt_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_prompt_tokens", 0)) for item in sample_results),
        "total_completion_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_completion_tokens", 0)) for item in sample_results),
        "total_reasoning_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_reasoning_tokens", 0)) for item in sample_results),
        "total_visible_output_tokens": sum(_safe_int(item.get("usage_summary", {}).get("total_visible_output_tokens", 0)) for item in sample_results),
        "max_prompt_tokens_single_call": max((_safe_int(item.get("usage_summary", {}).get("max_prompt_tokens_single_call", 0)) for item in sample_results), default=0),
        "total_frames_sampled": sum(_safe_int(item.get("frame_summary", {}).get("total_frames_sampled", 0)) for item in sample_results),
        "total_image_inputs": sum(_safe_int(item.get("frame_summary", {}).get("total_image_inputs", 0)) for item in sample_results),
    }
    payload = {"run_dir": str(run_dir), "aggregate": aggregate, "samples": sorted(sample_results, key=lambda item: item["index"])}
    (run_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rerun_one(target: dict[str, Any], loaded_tasks: dict[str, Any], run_dir: Path, agent_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    task_name = target["task"]
    doc_id = int(target["doc_id"])
    task = loaded_tasks[task_name]["task"]
    doc = loaded_tasks[task_name]["docs"][doc_id]
    context = task.doc_to_text(doc)
    visuals = task.doc_to_visual(doc)
    if not isinstance(visuals, list):
        visuals = [visuals]
    if not visuals or not isinstance(visuals[0], str):
        raise ValueError(f"VideoSeek expects a local video path, got: {visuals}")

    sample_dir = Path(target["sample_dir"])
    if sample_dir.exists():
        backup_existing_artifacts(sample_dir)
        cleanup_stale_artifacts(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    video_path = visuals[0]
    sample_request = {
        "video_path": video_path,
        "question": context,
        "output_dir": str(sample_dir),
        "config": agent_config,
        "verbose": args.verbose,
        "extract_answer": not args.no_extract_answer,
    }

    print(f"[start] task={task_name} doc_id={doc_id} idx={target['index']} reason={target['reason']}", flush=True)
    try:
        result = _run_request_in_subprocess(sample_request, timeout=int(agent_config["timeout"]))
        if result.get("error"):
            raise RuntimeError(result["error"])
        prediction = str(result.get("prediction", "") or "")
        record = {
            "task": task_name,
            "doc_id": doc_id,
            "index": int(target["index"]),
            "sample_dir": str(sample_dir),
            "status": "success" if prediction.strip() else "empty_prediction",
            "prediction": prediction,
            "reason": target["reason"],
        }
        print(
            f"[done] task={task_name} doc_id={doc_id} idx={target['index']} "
            f"status={record['status']} prediction={prediction[:32]!r}",
            flush=True,
        )
        return record
    except Exception as exc:
        error = str(exc).replace("\n", " ")[:500]
        _write_failure_artifacts(sample_dir, context, error)
        record = {
            "task": task_name,
            "doc_id": doc_id,
            "index": int(target["index"]),
            "sample_dir": str(sample_dir),
            "status": "failed",
            "prediction": "",
            "reason": target["reason"],
            "error": error,
        }
        print(
            f"[done] task={task_name} doc_id={doc_id} idx={target['index']} "
            f"status=failed error={error[:160]}",
            flush=True,
        )
        return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Path to one native VideoSeek run_<timestamp> directory")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--limit", type=int, default=None, help="Only inspect/rerun the first N docs per task")
    parser.add_argument("--doc-id", type=int, action="append", default=None, help="Only rerun specific doc ids; can be repeated")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default="any")
    parser.add_argument("--api-version", default="")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--decision-timeout", type=int, default=None)
    parser.add_argument("--tool-timeout", type=int, default=None)
    parser.add_argument("--final-answer-timeout", type=int, default=None)
    parser.add_argument("--decision-api-retry-attempts", type=int, default=None)
    parser.add_argument("--tool-api-retry-attempts", type=int, default=None)
    parser.add_argument("--final-answer-api-retry-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="Number of samples to rerun concurrently")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-extract-answer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-jsonl", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"run_manifest.json not found under: {run_dir}")
    manifest = load_json(manifest_path)

    existing = discover_existing_samples(run_dir)
    task_names = sorted({sample["task"] for sample in existing.values()}) if existing else []
    if not task_names:
        raise SystemExit(f"No native VideoSeek sample directories found under: {run_dir}")

    loaded_tasks = load_task_docs(task_names, split=args.split)
    targets = build_targets(
        run_dir,
        loaded_tasks,
        existing,
        limit=args.limit,
        doc_ids=set(args.doc_id) if args.doc_id else None,
    )

    agent_config = build_agent_config(manifest, args)

    print(f"run_dir: {run_dir}")
    print(f"tasks: {', '.join(task_names)}")
    print(f"existing_samples: {len(existing)}")
    print(f"rerun_targets: {len(targets)}")
    print(f"workers: {max(1, args.workers)}")
    print(
        "effective_agent_config: "
        + json.dumps(
            {
                "model_name": agent_config.get("model_name"),
                "api_base": agent_config.get("api_base"),
                "max_steps": agent_config.get("max_steps"),
                "max_tokens": agent_config.get("max_tokens"),
                "timeout": agent_config.get("timeout"),
                "decision_timeout": agent_config.get("decision_timeout"),
                "tool_timeout": agent_config.get("tool_timeout"),
                "final_answer_timeout": agent_config.get("final_answer_timeout"),
                "decision_api_retry_attempts": agent_config.get("decision_api_retry_attempts"),
                "tool_api_retry_attempts": agent_config.get("tool_api_retry_attempts"),
                "final_answer_api_retry_attempts": agent_config.get("final_answer_api_retry_attempts"),
            },
            ensure_ascii=False,
        )
    )
    for target in targets[:20]:
        print(f"  task={target['task']} doc_id={target['doc_id']} idx={target['index']} reason={target['reason']}")
    if args.dry_run:
        return 0
    report: list[dict[str, Any]] = []
    max_workers = max(1, int(args.workers))
    if max_workers == 1:
        for completed, target in enumerate(targets, start=1):
            report.append(rerun_one(target, loaded_tasks, run_dir, agent_config, args))
            print(f"[progress] completed={completed}/{len(targets)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(rerun_one, target, loaded_tasks, run_dir, agent_config, args): target
                for target in targets
            }
            for completed, future in enumerate(as_completed(future_map), start=1):
                target = future_map[future]
                try:
                    report.append(future.result())
                except Exception as exc:
                    sample_dir = Path(target["sample_dir"])
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    error = str(exc).replace("\n", " ")[:500]
                    _write_failure_artifacts(sample_dir, f"doc_id={target['doc_id']}", error)
                    report.append(
                        {
                            "task": target["task"],
                            "doc_id": int(target["doc_id"]),
                            "index": int(target["index"]),
                            "sample_dir": str(sample_dir),
                            "status": "failed",
                            "prediction": "",
                            "reason": target["reason"],
                            "error": error,
                        }
                    )
                print(f"[progress] completed={completed}/{len(targets)}", flush=True)

    rebuild_run_summary(run_dir)

    report_jsonl = args.report_jsonl or (run_dir / "rerun_report.jsonl")
    with report_jsonl.open("w", encoding="utf-8") as handle:
        for item in report:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    success = sum(1 for item in report if item["status"] == "success")
    failed = sum(1 for item in report if item["status"] == "failed")
    empty = sum(1 for item in report if item["status"] == "empty_prediction")
    print(f"success: {success}")
    print(f"empty_prediction: {empty}")
    print(f"failed: {failed}")
    print(f"report_jsonl: {report_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
