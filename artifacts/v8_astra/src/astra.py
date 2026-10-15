"""ASTRA: action-trace-aware white-box discrete attack.

The target model stays frozen.  Gradients only select text locations; all
submitted inputs are discrete question strings.
"""
from __future__ import annotations

import random
import re
from typing import Iterable, Sequence

import torch

from aeris.metrics import exact_match, token_f1
from aeris.types import AttackRecord, Candidate, Example
from fb18_search import micro_mutations, semantic_snr_db
from whitebox_gale import GeneratorWhiteBox, WhiteBoxGALESearch


class TraceAwareWhiteBox(GeneratorWhiteBox):
    """Adds likelihood scoring for a genuine clean agent action trace."""

    def _completion_batch(self, question: str, completion: str):
        prompt = self._prompt(question)
        prefix = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        full = self.tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False)
        labels = full["input_ids"].clone()
        labels[:, : int(prefix["input_ids"].shape[1])] = -100
        return full, labels

    @torch.inference_mode()
    def reference_completion_nll(self, question: str, completion: str) -> float:
        encoded, labels = self._completion_batch(question, completion)
        output = self.model(
            input_ids=encoded["input_ids"].to(self.model.device),
            attention_mask=encoded.get("attention_mask", None).to(self.model.device)
            if encoded.get("attention_mask") is not None else None,
            labels=labels.to(self.model.device), use_cache=False,
        )
        return float(output.loss.detach().float().cpu())


def multi_boundary_mutations(
    question: str, spans: Sequence[tuple[int, int, float]]
) -> Iterable[tuple[str, str]]:
    """Use several internal positions at gradient-salient English words."""
    words = list(re.finditer(r"[A-Za-z]{6,}", question))
    yielded: set[str] = set()
    for start, end, _ in spans:
        for word in words:
            if not (word.start() <= start < word.end() or word.start() < end <= word.end()):
                continue
            width = len(word.group(0))
            for fraction, label in ((1 / 3, "third"), (1 / 2, "middle"), (2 / 3, "two_thirds")):
                position = word.start() + max(1, min(width - 1, round(width * fraction)))
                text = question[:position] + "\u200b" + question[position:]
                if text not in yielded:
                    yielded.add(text)
                    yield f"gradient_boundary_{label}", text
            break


def salient_anchor_echoes(question: str, spans: Sequence[tuple[int, int, float]]) -> Iterable[tuple[str, str]]:
    """Repeat existing salient anchors; no new entities or factual claims."""
    words = list(re.finditer(r"[A-Za-z][A-Za-z'-]{2,}", question))
    yielded: set[str] = set()
    for start, end, _ in spans:
        for word in words:
            if word.start() <= start < word.end() or word.start() < end <= word.end():
                anchor = word.group(0)
                text = f"{question.rstrip('?？')} — {anchor}?"
                if text not in yielded:
                    yielded.add(text)
                    yield "salient_anchor_echo", text
                break


def lexical_similarity(left: str, right: str) -> float:
    a = set(re.findall(r"[A-Za-z0-9]+", left.lower()))
    b = set(re.findall(r"[A-Za-z0-9]+", right.lower()))
    return len(a & b) / max(1, len(a | b))


class ASTRASearch(WhiteBoxGALESearch):
    """White-box answer/action suppression with diverse discrete selection."""

    def __init__(self, *args, action_weight: float = 0.35, diversity_weight: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_weight = action_weight
        self.diversity_weight = diversity_weight

    def _candidate_texts(self, parent, original_question, salient_spans, rng):
        if parent.depth == 0:
            yield from multi_boundary_mutations(original_question, salient_spans)
            yield from salient_anchor_echoes(original_question, salient_spans)
        yield from micro_mutations(parent.text, rng)

    def _select_diverse(self, ranked: list[Candidate], limit: int) -> list[Candidate]:
        remaining = list(ranked)
        selected: list[Candidate] = []
        while remaining and len(selected) < limit:
            choice = max(
                remaining,
                key=lambda candidate: candidate.proxy_score - self.diversity_weight * max(
                    (lexical_similarity(candidate.text, prior.text) for prior in selected), default=0.0
                ),
            )
            selected.append(choice)
            remaining.remove(choice)
        return selected

    def attack(self, example: Example) -> AttackRecord:
        rng = random.Random(f"astra:{self.config.seed}:{example.dataset}:{example.example_id}")
        clean_prediction, clean_em, clean_f1, target_calls = self._clean_result(example)
        original = Candidate(
            text=example.question, parent_text="", operators=[], depth=0,
            attacked_prediction=clean_prediction, attacked_em=clean_em, attacked_f1=clean_f1,
            target_status="ok" if clean_prediction else None, target_score=0.0,
        )
        if clean_em < 1.0 or not clean_prediction:
            return AttackRecord(
                example_id=example.example_id, dataset=example.dataset, question=example.question,
                gold_answers=example.gold_answers, clean_prediction=clean_prediction,
                clean_em=clean_em, clean_f1=clean_f1, eligible=False, success=False,
                best=original, retrieval_calls=0, target_calls=target_calls,
                seed=self.config.seed, failure_kind="clean_incorrect_or_unavailable",
            )

        # This extra clean call is measurement-only and is not charged against
        # the adversarial target budget.  Its first raw action is self-reference,
        # never an attacker-selected target string.
        clean_probe = self.target.evaluate(example.question, example.example_id)
        action_raw = ""
        if clean_probe.status == "ok":
            turns = clean_probe.trace.get("turns", [])
            if turns:
                action_raw = str(turns[0].get("raw", ""))[:2000]
        signal = self.generator_whitebox.signal(example.question, clean_prediction)
        clean_action_nll = (
            self.generator_whitebox.reference_completion_nll(example.question, action_raw)
            if action_raw else None
        )
        clean_docs = self.retriever.retrieve(example.question, self.config.retrieval_topk)
        retrieval_calls = 1
        generated, evaluated = {example.question}, {example.question}
        all_candidates: list[Candidate] = [original]
        beam: list[Candidate] = [original]
        successful: list[Candidate] = []

        for round_index in range(self.config.rounds):
            scored: list[Candidate] = []
            parents = beam if round_index else [original]
            for parent in parents:
                for operator, text in self._candidate_texts(parent, example.question, signal.salient_spans, rng):
                    if text in generated or retrieval_calls >= self.config.retrieval_budget:
                        continue
                    generated.add(text)
                    decision = self.guard.check(example.question, text, example.gold_answers)
                    if not decision.ok or semantic_snr_db(decision.embedding_similarity) < self.snr_floor_db:
                        continue
                    candidate = Candidate(
                        text=text, parent_text=parent.text, operators=[*parent.operators, operator],
                        depth=parent.depth + 1, semantic_ok=True, semantic_reason="astra_passed_guard",
                        token_overlap=decision.overlap, embedding_similarity=decision.embedding_similarity,
                        edit_ratio=decision.edit,
                    )
                    boundary_gain = self._retrieval_features(clean_docs, candidate, example)
                    retrieval_calls += 1
                    answer_nll = self.generator_whitebox.reference_nll(text, clean_prediction)
                    answer_delta = answer_nll - signal.reference_nll
                    action_nll = None
                    action_delta = 0.0
                    if clean_action_nll is not None:
                        action_nll = self.generator_whitebox.reference_completion_nll(text, action_raw)
                        action_delta = action_nll - clean_action_nll
                    score = answer_delta + self.action_weight * action_delta + 0.20 * boundary_gain
                    candidate.proxy_score = score
                    candidate.target_trace = {"whitebox": {
                        "reference_answer": clean_prediction,
                        "clean_answer_nll": signal.reference_nll,
                        "candidate_answer_nll": answer_nll,
                        "answer_nll_delta": answer_delta,
                        "clean_action_nll": clean_action_nll,
                        "candidate_action_nll": action_nll,
                        "action_nll_delta": action_delta,
                        "action_weight": self.action_weight,
                        "boundary_gain": boundary_gain,
                        "salient_spans": [list(span) for span in signal.salient_spans],
                    }, "clean_action_probe": clean_probe.trace}
                    scored.append(candidate)
                    all_candidates.append(candidate)
            if not scored:
                break
            scored.sort(key=lambda item: (item.proxy_score, -item.edit_ratio), reverse=True)
            beam = self._select_diverse(scored, self.config.beam_size)
            allowance = min(self.config.target_batch, self.config.target_budget - target_calls)
            for candidate in beam:
                if allowance <= 0 or candidate.text in evaluated:
                    continue
                result = self.target.evaluate(candidate.text, example.example_id)
                target_calls += 1
                allowance -= 1
                evaluated.add(candidate.text)
                candidate.attacked_prediction = result.prediction
                candidate.target_status = result.status
                candidate.target_trace["oracle"] = result.trace
                if result.status == "ok":
                    candidate.attacked_em = exact_match(result.prediction, example.gold_answers)
                    candidate.attacked_f1 = token_f1(result.prediction, example.gold_answers)
                candidate.target_score = self._target_score(candidate, clean_f1)
                if self._is_success(candidate):
                    successful.append(candidate)
            if successful or target_calls >= self.config.target_budget:
                break

        if successful:
            best = min(successful, key=lambda item: (item.edit_ratio, len(item.operators), -item.proxy_score))
            failure_kind = ""
        else:
            candidates = [item for item in all_candidates if item is not original]
            best = max(candidates or [original], key=lambda item: (item.proxy_score, -item.edit_ratio))
            failure_kind = "astra_budget_exhausted_no_success"
        return AttackRecord(
            example_id=example.example_id, dataset=example.dataset, question=example.question,
            gold_answers=example.gold_answers, clean_prediction=clean_prediction,
            clean_em=clean_em, clean_f1=clean_f1, eligible=True, success=bool(successful),
            best=best, retrieval_calls=retrieval_calls, target_calls=target_calls,
            seed=self.config.seed, failure_kind=failure_kind,
        )

