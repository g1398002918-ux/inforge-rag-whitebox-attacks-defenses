#!/usr/bin/env python3
"""Read-only target-model trajectory-SNR audit for completed ASTRA records.

T-SNR is a *model-space* end-to-end behavior metric, not an acoustic SNR and
not the e5 input-semantic SNR.  For each clean/attacked pair, we serialize the
observed trajectory (question, structured actions, retrieved evidence and final
answer), mean-pool the frozen target generator's final hidden layer, L2-normalize
it, and compute

  T-SNR = -10 log10(||z_clean - z_attack||_2^2),  ||z||_2 = 1.

The metric is accompanied by cosine similarity and the underlying energy.  No
weights, corpus, index, prompts, answers, or existing attack records are
modified.  It deliberately excludes free-form chain-of-thought text and gold
answers; only structured, observable trajectory fields are used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def observed_trace(record: dict, key: str) -> str:
    """Canonical observable trajectory; excludes raw thought strings and golds."""
    trace = record.get("best", {}).get("target_trace", {}).get(key, {}) or {}
    lines = [f"QUESTION: {record.get('question', '')}"]
    for turn in trace.get("turns", []):
        action, content = turn.get("action"), turn.get("content")
        if action and content:
            lines.append(f"ACTION[{action}]: {content}")
    for step, search in enumerate(trace.get("searches", []), 1):
        lines.append(f"RETRIEVAL_QUERY[{step}]: {search.get('query', '')}")
        docs = search.get("docs", [])[:3]
        evidence = " | ".join(
            f"{doc.get('doc_id','')}::{doc.get('title','')}::{float(doc.get('score', 0.0)):.5f}"
            for doc in docs
        )
        lines.append(f"TOP3_EVIDENCE[{step}]: {evidence}")
    return "\n".join(lines)


@torch.inference_mode()
def representation(model, tokenizer, text: str, max_tokens: int):
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens, add_special_tokens=False)
    device = model.device
    input_ids = batch["input_ids"].to(device)
    mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
    output = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    last = output.hidden_states[-1][0].float()
    weights = mask[0].unsqueeze(-1).float()
    pooled = (last * weights).sum(dim=0) / weights.sum().clamp_min(1.0)
    return torch.nn.functional.normalize(pooled, dim=0), int(mask.sum().item())


def summarize(rows: list[dict]) -> dict:
    def values(key, finite=True):
        raw = [r[key] for r in rows if r.get(key) is not None]
        return [x for x in raw if math.isfinite(x)] if finite else raw
    def mean(xs): return sum(xs) / len(xs) if xs else None
    def median(xs):
        if not xs: return None
        xs = sorted(xs); n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return {
        "count": len(rows),
        "finite_t_snr_count": len(values("trajectory_snr_db")),
        "infinite_t_snr_count": sum(math.isinf(r.get("trajectory_snr_db", float("nan"))) for r in rows),
        "mean_t_snr_db": mean(values("trajectory_snr_db")),
        "median_t_snr_db": median(values("trajectory_snr_db")),
        "mean_trajectory_cosine": mean(values("trajectory_cosine")),
        "median_trajectory_cosine": median(values("trajectory_cosine")),
        "mean_noise_energy": mean(values("trajectory_noise_energy")),
        "median_noise_energy": median(values("trajectory_noise_energy")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", default="/root/astra_v8_max500_20261010")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-root", default="/root/autodl-tmp/aeris_v4")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="process at most this many rows; 0 means all")
    args = parser.parse_args()
    base_root, records_root, output_root = Path(args.base_root), Path(args.records_root), Path(args.output_root)
    sys.path[:0] = [str(base_root), str(base_root / "src"), "/root/autodl-tmp"]
    import server_aeris_runner as aeris_server  # type: ignore
    import server_build_eval as clean_eval  # type: ignore

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "trajectory_snr.jsonl"
    done = {json.loads(line)["key"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()} if args.resume and out_path.exists() else set()
    tokenizer = aeris_server.AutoTokenizer.from_pretrained(clean_eval.MODEL_PATH, trust_remote_code=True)
    model = aeris_server.AutoModelForCausalLM.from_pretrained(clean_eval.MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    total = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for dataset in ("nq", "hotpotqa"):
            for record in read_jsonl(records_root / dataset / "records.jsonl"):
                if args.limit and total >= args.limit:
                    break
                key = f"{dataset}:{record['example_id']}"
                if key in done:
                    continue
                clean = observed_trace(record, "clean_action_probe")
                attacked = observed_trace(record, "oracle")
                z_clean, clean_tokens = representation(model, tokenizer, clean, args.max_tokens)
                z_attack, attack_tokens = representation(model, tokenizer, attacked, args.max_tokens)
                cosine = float(torch.dot(z_clean, z_attack).clamp(-1, 1).cpu())
                noise = max(0.0, 2.0 * (1.0 - cosine))
                snr = float("inf") if noise <= 1e-12 else -10.0 * math.log10(noise)
                row = {
                    "key": key, "dataset": dataset, "example_id": record["example_id"], "success": bool(record.get("success")),
                    "trajectory_cosine": cosine, "trajectory_noise_energy": noise, "trajectory_snr_db": snr,
                    "clean_trace_tokens": clean_tokens, "attacked_trace_tokens": attack_tokens,
                    "input_embedding_similarity": record.get("best", {}).get("embedding_similarity"),
                    "retrieval_drift": record.get("best", {}).get("retrieval_drift"),
                }
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                total += 1
                print(f"[t-snr] {key} success={row['success']} snr={snr:.3f} n={noise:.6g}", flush=True)
                torch.cuda.empty_cache()
            if args.limit and total >= args.limit:
                break
    rows = read_jsonl(out_path)
    summary = {
        "metric": "target_model_trajectory_snr_db",
        "formula": "-10 log10(||z_clean-z_attack||_2^2), with unit-normalized mean-pooled final hidden states",
        "trajectory": "question + structured action/content + Top-3 retrieved doc id/title/score + final answer; excludes raw thought and gold answers",
        "model": "frozen target generator; output_hidden_states=True; no gradients",
        "max_tokens": args.max_tokens,
        "weights_mutated": False, "corpus_mutated": False, "index_mutated": False,
        "all": summarize(rows),
        "successful_attacks": summarize([r for r in rows if r["success"]]),
        "by_dataset": {d: summarize([r for r in rows if r["dataset"] == d]) for d in ("nq", "hotpotqa")},
        "status_counts": dict(Counter(r["dataset"] for r in rows)),
    }
    write_json(output_root / "trajectory_snr_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

