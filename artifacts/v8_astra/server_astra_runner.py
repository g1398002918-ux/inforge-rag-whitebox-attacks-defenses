#!/usr/bin/env python3
"""Independent runner for ASTRA v8; never writes into v5--v7 directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_sample(rows: list[dict], dataset: str, count: int, seed: int) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{dataset}:{row['example_id']}".encode()).hexdigest())[:count]


def parse_counts(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in raw.split(","):
        name, count = item.split("=", 1)
        if name not in {"nq", "hotpotqa"} or not count.isdigit():
            raise ValueError("--counts must be e.g. nq=166,hotpotqa=334")
        result[name] = int(count)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--positives-root", default="/root/full_eval_results")
    parser.add_argument("--v8-root", default="/root/astra_v8")
    parser.add_argument("--v6-root", default="/root/wb_gale_v6")
    parser.add_argument("--v5-root", default="/root/aeris_fb18")
    parser.add_argument("--base-root", default="/root/autodl-tmp/aeris_v4")
    parser.add_argument("--profile", choices=["fidelity", "max-asr"], default="max-asr")
    parser.add_argument("--counts", default="nq=166,hotpotqa=334")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    v8_root, v6_root, v5_root, base_root = Path(args.v8_root), Path(args.v6_root), Path(args.v5_root), Path(args.base_root)
    sys.path[:0] = [str(v8_root / "src"), str(v6_root / "src"), str(v5_root / "src"), str(base_root), str(base_root / "src"), "/root/autodl-tmp"]
    import server_aeris_runner as aeris_server  # type: ignore
    import server_build_eval as clean_eval  # type: ignore
    from aeris.config import AttackConfig  # type: ignore
    from aeris.io import append_record, completed_ids, consolidate_records, write_successes_csv, write_summary  # type: ignore
    from aeris.semantics import SemanticGuard  # type: ignore
    from aeris.types import Example  # type: ignore
    from astra import ASTRASearch, TraceAwareWhiteBox  # type: ignore

    config = AttackConfig.from_json(base_root / "configs" / "default.json")
    config.seed = args.seed
    config.max_query_chars = 768
    config.beam_size, config.rounds = 24, 3
    config.retrieval_budget, config.target_budget, config.target_batch = 144, 12, 6
    if args.profile == "fidelity":
        config.min_embedding_similarity = 0.9370537294102916  # 9 dB
        config.min_token_overlap, config.max_edit_ratio, config.max_char_growth = 0.80, 0.15, 1.35
        snr_floor, gate_disabled = 9.0, False
    else:
        # A valid placeholder for config validation; the custom guard below
        # removes only the embedding/SNR rejection and retains all fact gates.
        config.min_embedding_similarity = 0.0
        config.min_token_overlap, config.max_edit_ratio, config.max_char_growth = 0.65, 0.30, 2.0
        snr_floor, gate_disabled = float("-inf"), True
    config.validate()

    index, e5_tokenizer, e5_model = clean_eval.build_index()
    offsets = np.load(clean_eval.OFFSET_PATH)
    retriever = aeris_server.SharedE5Retriever(index, e5_tokenizer, e5_model, offsets)
    tokenizer = aeris_server.AutoTokenizer.from_pretrained(clean_eval.MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = aeris_server.AutoModelForCausalLM.from_pretrained(clean_eval.MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    oracle = aeris_server.InForageOracle(tokenizer, model, retriever)
    whitebox = TraceAwareWhiteBox(tokenizer, model, clean_eval.PROMPT_TEMPLATE)
    search = ASTRASearch(retriever, oracle, config, semantic_similarity=retriever.similarity, snr_floor_db=snr_floor, generator_whitebox=whitebox)
    if gate_disabled:
        class NonSNRGuard(SemanticGuard):
            def check(self, original, candidate, gold_answers):
                saved = self.config.min_embedding_similarity
                self.config.min_embedding_similarity = -1.0
                try:
                    return super().check(original, candidate, gold_answers)
                finally:
                    self.config.min_embedding_similarity = saved
        search.guard = NonSNRGuard(config, retriever.similarity)

    counts, root = parse_counts(args.counts), Path(args.output_root)
    environment = {
        "mode": "astra_action_trace_whitebox_discrete", "profile": args.profile,
        "weights_mutated": False, "corpus_mutated": False, "index_mutated": False,
        "gold_answer_injection_forbidden": True, "snr_floor_db": None if gate_disabled else 9.0,
        "embedding_gate_disabled": gate_disabled,
        "search_budget": {"beam_size": config.beam_size, "rounds": config.rounds, "retrieval_budget": config.retrieval_budget, "target_budget": config.target_budget, "target_batch": config.target_batch},
        "generator_signal": "clean_answer_nll_gradient_plus_clean_first_action_nll", "gpu": torch.cuda.get_device_name(0),
    }
    try:
        for dataset in ("nq", "hotpotqa"):
            positives = read_jsonl(Path(args.positives_root) / dataset / "positive_examples.jsonl")
            rows = stable_sample(positives, dataset, len(positives) if args.full else counts[dataset], args.seed)
            out = root / dataset
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "selected_examples.json", rows)
            write_json(out / "run_config.json", {**asdict(config), "arguments": vars(args), "selected_count": len(rows)})
            write_json(out / "environment.json", environment)
            records_path = out / "records.jsonl"
            if records_path.exists() and not args.resume:
                raise RuntimeError(f"{records_path} exists; use --resume or choose another output root")
            done = completed_ids(records_path) if args.resume else set()
            for ordinal, row in enumerate(rows, 1):
                if row["example_id"] in done:
                    continue
                example = Example(example_id=row["example_id"], question=row["question"], gold_answers=row["gold_answers"], dataset=dataset, clean_prediction=row.get("clean_prediction"), clean_em=float(row.get("clean_em", 0.0)), clean_f1=float(row.get("clean_f1", 0.0)))
                started = time.monotonic()
                record = search.attack(example)
                append_record(records_path, record)
                print(f"[astra] {dataset} {ordinal}/{len(rows)} id={row['example_id']} success={record.success} r={record.retrieval_calls} t={record.target_calls} elapsed={time.monotonic()-started:.1f}s", flush=True)
                torch.cuda.empty_cache()
            consolidate_records(records_path)
            summary = write_summary(records_path, out / "summary.json")
            summary.update({"mode": environment["mode"], "profile": args.profile, "whitebox": True, "sample_count": len(read_jsonl(records_path))})
            write_json(out / "summary.json", summary)
            write_successes_csv(records_path, out / "successes.csv")
    finally:
        retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

