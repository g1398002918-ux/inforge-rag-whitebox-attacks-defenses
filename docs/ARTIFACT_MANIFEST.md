# Artifact manifest

This public repository contains the code and compact reports needed to understand the
experiment. Large or sensitive runtime artifacts remain outside GitHub and must be
mounted separately.

| Local artifact | Approximate size | Repository status |
| --- | ---: | --- |
| `InfoSeek/checkpoints/` | 12.67 GiB | Excluded; provide checkpoint separately |
| `InfoSeek/venv/` | 4.77 GiB | Excluded; recreate from environment specification |
| `InfoSeek/data/` | 0.94 GiB | Excluded; provide corpus separately |
| `InfoSeek/index/` | 0.57 GiB | Excluded; provide index separately |
| `_full_eval_source/` raw benchmark/data files | up to hundreds of MiB | Excluded when they contain question records or generated outputs |
| V8 raw result directories | variable, thousands of records | Excluded; compact summaries and reports are included |

The retained source tree is under `inforge_reference/`; the final attack and reports are
under `artifacts/v8_astra/`. This separation keeps the repository cloneable and avoids
publishing model weights, datasets, credentials, or per-question private traces.

