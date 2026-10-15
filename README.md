# InForage RAG White-Box Attacks and Defenses

Research code and experiment artifacts for white-box adversarial evaluation of the InForage retrieval-augmented generation (RAG) system.

## Scope

This repository preserves the final V8 ASTRA attack implementation and the associated 500-question and 4,056-question end-to-end evaluations. The attack operates on the input query and leaves target-model weights unchanged.

- Attack implementation: `artifacts/v8_astra/src/astra.py`
- Method specification: `artifacts/v8_astra/METHOD.md`
- 500-question report: `artifacts/v8_astra/report_500.html`
- Full 4,056-question report: `artifacts/v8_astra/report_full4056.html`
- Reproducibility scripts: `artifacts/v8_astra/server_astra_runner.py`, `server_e2e_trajectory_snr.py`

## Final evaluation snapshot

- NQ: 451 / 1,343 successful attacks (33.5815%)
- HotpotQA: 1,138 / 2,713 successful attacks (41.9462%)
- Overall: 1,589 / 4,056 successful attacks (39.1765%)

The repository also records the 500-question ASTRA result and the trajectory-SNR analysis used in the final report.

## Repository layout

- `artifacts/v8_astra/`: final ASTRA source, reports, analyses, and run metadata.
- `inforge_reference/`: lightweight InForage/InfoSeek reference implementation and research source.
- `docs/`: artifact and reproducibility notes.

## Large artifacts intentionally excluded

GitHub is not used as a model/data registry. The following local artifacts are intentionally excluded and documented in `docs/ARTIFACT_MANIFEST.md`: model checkpoints, the Python virtual environment, the 18.5GB corpus, FAISS/index arrays, and raw per-question result archives. They must be supplied separately when reproducing the experiment.

The earlier local attack-version directories (11 through 17) were removed as requested; V8 ASTRA is the retained attack version.

## Reproducibility

See `docs/REPRODUCIBILITY.md` and the method document for the command-line interfaces, evaluation definitions, SNR/T-SNR calculation, and expected input paths.

## Citation

This repository contains experimental extensions around the InForage/InfoSeek codebase. Please cite the upstream project and the relevant attack report when using these artifacts.
