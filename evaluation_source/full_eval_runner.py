#!/usr/bin/env python3
"""Reproducible full NQ+HotpotQA clean/AERIS evaluation.

This runner is intentionally separate from the historical 100-question runner.
It reads the official Search-R1 test split (3,610 NQ + 7,405 HotpotQA), writes
append-only checkpoints, and never modifies the existing benchmark/results files.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=["nq", "hotpotqa"], default=["nq", "hotpotqa"])
    p.add_argument("--data-root", default="/root/full_eval_data")
    p.add_argument("--output-root", default="/root/full_eval_results")
    p.add_argument("--config", default="/root/autodl-tmp/aeris_v4/configs/default.json")
    p.add_argument("--target-budget", type=int, default=4)
    p.add_argument("--retrieval-budget", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--clean-only", action="store_true")
    p.add_argument("--attack-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_items(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"]
    out = []
    for item in items:
        answers = item.get("answers") or [item.get("answer", "")]
        out.append(
            {
                "id": str(item["id"]),
                "question": str(item["question"]),
                "answers": [str(x) for x in answers if str(x).strip()],
                "source_id": str(item.get("source_id", item["id"])),
            }
        )
    return out


def compact_clean(item: Dict[str, Any], result: Any, exact: float, f1: float) -> Dict[str, Any]:
    trace = result.trace or {}
    return {
        "example_id": item["id"],
        "source_id": item["source_id"],
        "question": item["question"],
        "gold_answers": item["answers"],
        "clean_prediction": result.prediction,
        "clean_em": float(exact),
        "clean_f1": float(f1),
        "target_status": result.status,
        "latency_s": float(result.latency_s),
        "turns": len(trace.get("turns", [])),
        "num_search": len(trace.get("searches", [])),
    }


def make_environment(clean_eval: Any, aeris_server: Any, source_digest: str) -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "aeris_source_sha256": source_digest,
        "model_path": clean_eval.MODEL_PATH,
        "retriever_path": clean_eval.RETRIEVER_MODEL,
        "index_path": clean_eval.INDEX_PATH,
        "index_size_bytes": os.path.getsize(clean_eval.INDEX_PATH),
        "weights_mutated": False,
    }


def main() -> int:
    args = parse_args()
    # Import the established AERIS implementation only after parsing.  This
    # keeps this full runner independent of the historical 100-item loader.
    sys.path.insert(0, "/root/autodl-tmp/aeris_v4")
    sys.path.insert(0, "/root/autodl-tmp/aeris_v4/src")
    sys.path.insert(0, "/root/autodl-tmp")
    import server_aeris_runner as aeris_server  # type: ignore
    import server_build_eval as clean_eval  # type: ignore
    from aeris.config import AttackConfig  # type: ignore
    from aeris.io import append_record, completed_ids, consolidate_records, write_successes_csv, write_summary  # type: ignore
    from aeris.metrics import exact_match, token_f1  # type: ignore
    from aeris.search import AerisSearch  # type: ignore
    from aeris.types import Example  # type: ignore

    output_root = Path(args.output_root)
    data_root = Path(args.data_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = AttackConfig.from_json(args.config)
    config.target_budget = args.target_budget
    config.retrieval_budget = args.retrieval_budget
    config.seed = args.seed
    config.validate()

    # Load the index, retriever, and target model once for both stages.
    index, e5_tokenizer, e5_model = clean_eval.build_index()
    offsets = np.load(clean_eval.OFFSET_PATH)
    retriever = aeris_server.SharedE5Retriever(index, e5_tokenizer, e5_model, offsets)
    tokenizer = aeris_server.AutoTokenizer.from_pretrained(clean_eval.MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = aeris_server.AutoModelForCausalLM.from_pretrained(
        clean_eval.MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    oracle = aeris_server.InForageOracle(tokenizer, model, retriever)
    search = AerisSearch(retriever, oracle, config, semantic_similarity=retriever.similarity)
    environment = make_environment(clean_eval, aeris_server, aeris_server.source_digest())

    try:
        for dataset in args.datasets:
            items = load_items(data_root / f"benchmark_{dataset}.json")
            if args.limit is not None:
                items = items[: args.limit]
            out = output_root / dataset
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "run_config.json", {**asdict(config), "arguments": vars(args), "item_count": len(items)})
            write_json(out / "environment.json", environment)
            clean_path = out / "clean_records.jsonl"
            clean_rows = read_jsonl(clean_path)
            clean_done = {str(row["example_id"]) for row in clean_rows}
            if not args.attack_only:
                started_dataset = time.monotonic()
                for ordinal, item in enumerate(items, 1):
                    if item["id"] in clean_done:
                        continue
                    started = time.monotonic()
                    result = oracle.evaluate(item["question"], item["id"])
                    em = exact_match(result.prediction, item["answers"]) if result.status == "ok" else 0.0
                    f1 = token_f1(result.prediction, item["answers"]) if result.status == "ok" else 0.0
                    row = compact_clean(item, result, em, f1)
                    append_jsonl(clean_path, row)
                    clean_done.add(item["id"])
                    print(
                        f"[clean] {dataset} {ordinal}/{len(items)} id={item['id']} "
                        f"em={int(em)} status={result.status} elapsed={time.monotonic()-started:.1f}s",
                        flush=True,
                    )
                    if ordinal % 25 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                clean_rows = read_jsonl(clean_path)
                positives = [row for row in clean_rows if row["clean_em"] == 1.0 and row["target_status"] == "ok"]
                write_json(out / "positive_examples.jsonl", "") if False else None
                pos_path = out / "positive_examples.jsonl"
                pos_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in positives), encoding="utf-8")
                write_json(
                    out / "clean_summary.json",
                    {
                        "dataset": dataset,
                        "total": len(clean_rows),
                        "expected_total": len(items),
                        "clean_correct": len(positives),
                        "clean_accuracy": len(positives) / max(1, len(clean_rows)),
                        "status_counts": {s: sum(1 for r in clean_rows if r["target_status"] == s) for s in sorted({r["target_status"] for r in clean_rows})},
                        "elapsed_s": time.monotonic() - started_dataset,
                    },
                )
                print(f"[clean-summary] {dataset} total={len(clean_rows)} clean_correct={len(positives)}", flush=True)
            else:
                clean_rows = read_jsonl(clean_path)

            if args.clean_only:
                continue
            # Attack only clean-correct examples.  AERIS records use the
            # existing append/sidecar implementation for crash-safe resume.
            positives = [row for row in clean_rows if row["clean_em"] == 1.0 and row["target_status"] == "ok"]
            records_path = out / "records.jsonl"
            done = completed_ids(records_path) if args.resume else set()
            if records_path.exists() and not args.resume:
                raise RuntimeError(f"{records_path} exists; use --resume")
            print(f"[attack] dataset={dataset} positives={len(positives)} done={len(done)}", flush=True)
            for ordinal, row in enumerate(positives, 1):
                if row["example_id"] in done:
                    continue
                example = Example(
                    example_id=row["example_id"],
                    question=row["question"],
                    gold_answers=row["gold_answers"],
                    dataset=dataset,
                    clean_prediction=row["clean_prediction"],
                    clean_em=float(row["clean_em"]),
                    clean_f1=float(row["clean_f1"]),
                    metadata={"clean_turns": row["turns"], "clean_num_search": row["num_search"], "clean_latency_s": row["latency_s"]},
                )
                started = time.monotonic()
                record = search.attack(example)
                append_record(records_path, record)
                done.add(row["example_id"])
                print(
                    f"[attack] {dataset} {ordinal}/{len(positives)} id={row['example_id']} "
                    f"success={record.success} r={record.retrieval_calls} t={record.target_calls} "
                    f"elapsed={time.monotonic()-started:.1f}s ops={record.best.operators}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
            consolidate_records(records_path)
            summary = write_summary(records_path, out / "summary.json")
            summary["mode"] = "server_full_official_split_end_to_end"
            summary["clean_correct_source"] = str(out / "positive_examples.jsonl")
            write_json(out / "summary.json", summary)
            write_successes_csv(records_path, out / "successes.csv")
            print(json.dumps({"dataset": dataset, **summary}, ensure_ascii=False), flush=True)
    finally:
        retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

