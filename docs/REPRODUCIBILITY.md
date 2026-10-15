# Reproducibility guide

The final attack artifact is the V8 ASTRA implementation under `artifacts/v8_astra/`.
It is a white-box query-space attack: the target model and retriever are inspected for
their tokenization, retrieval scores, and generation interfaces, but no target weights
are modified. Run the attack only on systems and datasets for which you have
authorization.

## Inputs

Provide the InForage checkpoint, the E5 retriever, the local Wikipedia corpus and its
index, and the NQ/HotpotQA evaluation files. The original absolute paths used on the
experiment server are recorded in the run metadata, but are not required; pass local
paths through the runner arguments or environment variables.

## Evaluation definitions

- A successful attack is counted only when the clean query is correct and the attacked
  end-to-end answer is incorrect under the same evaluator.
- The final full evaluation contains 1,343 NQ and 2,713 HotpotQA questions (4,056 total).
- T-SNR is computed from the normalized E5 query embeddings of the clean and edited
  queries. Numerical saturation is reported as `Infinity` when the embedding distance
  is at or below the implementation tolerance.

## Expected final snapshot

The recorded full run has 451/1,343 successful NQ attacks, 1,138/2,713 successful
HotpotQA attacks, and 1,589/4,056 overall (39.1765%). Re-running can differ slightly
with hardware, library, tokenizer, or decoding changes; preserve the run configuration
and evaluator version alongside new results.

## Reproducibility checklist

1. Pin the Python and model-library versions used by the target environment.
2. Verify corpus/index/checkpoint checksums before starting.
3. Run the clean evaluator and retain its JSON/CSV output.
4. Run `server_astra_runner.py` on the clean-correct subset.
5. Run the end-to-end evaluator and compute the summary from raw records.
6. Do not commit checkpoints, credentials, private server paths, or raw multi-gigabyte
   corpora to this public repository.

