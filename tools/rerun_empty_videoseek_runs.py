#!/usr/bin/env python3
"""Rerun VideoSeek for samples whose latest run directory is empty.

The script scans logs/videoseek_runs/videomme_doc{doc_id}_idx{idx}/, finds
parents whose newest child run has no prediction.json/trajectory.json, rebuilds
the VideoMME prompt from the local HF dataset, and launches the VideoSeek CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_PROMPT_SUFFIX = "\nAnswer with the option's letter from the given choices directly."
DEFAULT_SAMPLES_GLOB = "/gemini/space/gjx/lmms-eval/logs/qwen35_397b_videomme_api/*/*samples_videomme.jsonl"


def parse_parent(parent: Path) -> tuple[int, int] | None:
    match = re.fullmatch(r"videomme_doc(\d+)_idx(\d+)", parent.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def has_result(run_dir: Path) -> bool:
    return (run_dir / "prediction.json").is_file() and (run_dir / "trajectory.json").is_file()


def latest_child(parent: Path) -> Path | None:
    children = [p for p in parent.iterdir() if p.is_dir()]
    if not children:
        return None
    return max(children, key=lambda p: p.stat().st_mtime)


def any_success_after(parent: Path, timestamp: float) -> bool:
    for child in parent.iterdir():
        if child.is_dir() and child.stat().st_mtime >= timestamp and has_result(child):
            return True
    return False


def discover_targets(root: Path, only_latest_empty: bool = True) -> list[tuple[int, int, Path]]:
    targets: list[tuple[int, int, Path]] = []
    for parent in sorted(root.glob("videomme_doc*_idx*")):
        parsed = parse_parent(parent)
        if parsed is None or not parent.is_dir():
            continue
        doc_id, idx = parsed
        child = latest_child(parent)
        if child is None:
            targets.append((doc_id, idx, parent))
            continue
        if has_result(child):
            continue
        if only_latest_empty and any_success_after(parent, child.stat().st_mtime):
            continue
        targets.append((doc_id, idx, parent))
    return targets


def build_prompt(doc: dict) -> str:
    if "__input" in doc:
        return str(doc["__input"])
    options = "\n".join(str(opt) for opt in doc["options"])
    return (
        "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"{doc['question']}\n"
        f"{options}"
        f"{DEFAULT_PROMPT_SUFFIX}"
    )


def video_path_for(doc: dict, dataset_root: Path) -> Path:
    stem = doc["videoID"]
    for suffix in (".mp4", ".MP4", ".mkv"):
        path = dataset_root / "data" / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No video file found for videoID={stem} under {dataset_root / 'data'}")


def subtitle_path_for(doc: dict, dataset_root: Path) -> Path | None:
    path = dataset_root / "subtitle" / f"{doc['videoID']}.srt"
    return path if path.exists() else None


def run_one(args: argparse.Namespace, doc: dict, doc_id: int, idx: int, output_dir: Path) -> dict:
    cmd = [
        sys.executable,
        "-c",
        "import litellm; litellm.drop_params=True; from videoseek.cli import main; raise SystemExit(main())",
        "--video_path",
        str(video_path_for(doc, args.dataset_root)),
        "--user_query",
        build_prompt(doc),
        "--output_dir",
        str(output_dir),
        "--model_name",
        args.model_name,
        "--api_base",
        "OPENAI_API_BASE",
        "--api_key",
        "OPENAI_API_KEY",
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--max_steps",
        str(args.max_steps),
        "--action_parse_mode",
        args.action_parse_mode,
    ]
    subtitle_path = subtitle_path_for(doc, args.dataset_root)
    if subtitle_path is not None:
        cmd.extend(["--subtitle_path", str(subtitle_path)])
    if args.verbose:
        cmd.append("--verbose")
    if args.debug_tool_responses:
        cmd.append("--debug_tool_responses")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.videoseek_root}:{env.get('PYTHONPATH', '')}"
    env["OPENAI_API_BASE"] = args.api_base
    env["OPENAI_API_KEY"] = args.api_key

    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(args.videoseek_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
        check=False,
    )

    latest = latest_child(output_dir)
    success = result.returncode == 0 and latest is not None and has_result(latest) and latest.stat().st_mtime >= started
    return {
        "doc_id": doc_id,
        "idx": idx,
        "videoID": doc["videoID"],
        "target": doc.get("answer") or doc.get("__target"),
        "success": success,
        "returncode": result.returncode,
        "output_dir": str(output_dir),
        "latest_run_dir": str(latest) if latest else None,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def load_sample_doc_map(paths: list[Path]) -> dict[int, dict]:
    docs: dict[int, dict] = {}
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
                prompt = sample.get("input")
                target = sample.get("target") or metric.get("answer")
                if isinstance(doc_id, int) and video_id and prompt:
                    docs[doc_id] = {
                        "videoID": video_id,
                        "__input": prompt,
                        "__target": target,
                        "answer": target,
                    }
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("/gemini/space/gjx/lmms-eval/logs/videoseek_runs"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/gemini/space/zyf/datasets/lmms-lab/Video-MME/videomme"))
    parser.add_argument(
        "--samples-jsonl",
        type=Path,
        action="append",
        default=None,
        help="Existing lmms-eval samples JSONL to use for doc_id -> prompt/videoID mapping. Can be repeated.",
    )
    parser.add_argument("--samples-glob", default=DEFAULT_SAMPLES_GLOB)
    parser.add_argument("--videoseek-root", type=Path, default=Path("/gemini/space/gjx/videoseek"))
    parser.add_argument("--api-base", default="http://10.233.45.74:5590/v1")
    parser.add_argument("--api-key", default="any")
    parser.add_argument("--model-name", default="openai/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--action-parse-mode", default="tool_call", choices=["tool_call", "text_action", "auto"])
    parser.add_argument("--max-tokens", type=int, default=102400)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--doc-id", type=int, action="append", default=None, help="Rerun only specific doc id; can be repeated.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug-tool-responses", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("logs/videoseek_empty_rerun_report.jsonl"))
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/gemini/space/zyf")

    targets = discover_targets(args.runs_root)
    if args.doc_id is not None:
        wanted = set(args.doc_id)
        targets = [target for target in targets if target[0] in wanted]
    if args.limit is not None:
        targets = targets[: args.limit]

    print(f"targets: {len(targets)}")
    for doc_id, idx, parent in targets[:20]:
        print(f"  doc_id={doc_id} idx={idx} output_dir={parent}")
    if args.dry_run:
        return 0

    sample_paths = list(args.samples_jsonl or [])
    if args.samples_glob:
        sample_paths.extend(sorted(Path().glob(args.samples_glob) if not args.samples_glob.startswith("/") else Path("/").glob(args.samples_glob.lstrip("/"))))
    sample_doc_map = load_sample_doc_map(sample_paths)
    missing_doc_ids = [doc_id for doc_id, _, _ in targets if doc_id not in sample_doc_map]
    dataset = None
    if missing_doc_ids:
        print(f"sample mapping missing {len(missing_doc_ids)} docs; falling back to dataset indexing for those docs")
        from datasets import load_dataset

        dataset = load_dataset(
            str(args.dataset_root),
            split="test",
            token=True,
            cache_dir=str(args.dataset_root),
        )

    def get_doc(doc_id: int) -> dict:
        if doc_id in sample_doc_map:
            return sample_doc_map[doc_id]
        if dataset is None:
            raise KeyError(f"doc_id {doc_id} not found in sample mapping and dataset is not loaded")
        return dataset[doc_id]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with args.report.open("a", encoding="utf-8") as report:
        workers = max(1, int(args.workers))
        if workers == 1:
            for n, (doc_id, idx, parent) in enumerate(targets, 1):
                print(f"[{n}/{len(targets)}] rerun doc_id={doc_id} idx={idx}")
                record = run_one(args, get_doc(doc_id), doc_id, idx, parent)
                ok += int(record["success"])
                print(f"  success={record['success']} latest={record['latest_run_dir']}")
                report.write(json.dumps(record, ensure_ascii=False) + "\n")
                report.flush()
        else:
            print(f"running with workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_one, args, get_doc(doc_id), doc_id, idx, parent): (n, doc_id, idx)
                    for n, (doc_id, idx, parent) in enumerate(targets, 1)
                }
                for future in as_completed(futures):
                    n, doc_id, idx = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "doc_id": doc_id,
                            "idx": idx,
                            "success": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ok += int(record["success"])
                    print(f"[{n}/{len(targets)}] doc_id={doc_id} idx={idx} success={record['success']} latest={record.get('latest_run_dir')}")
                    report.write(json.dumps(record, ensure_ascii=False) + "\n")
                    report.flush()
    print(f"done: {ok}/{len(targets)} succeeded")
    print(f"report: {args.report}")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
