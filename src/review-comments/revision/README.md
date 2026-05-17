# IJIES Revision Workspace — paper 20262542 (comment-264)

This folder is **self-contained**: all new code, manuscript patches, and
generated artefacts for the IJIES "Accept after Major Revision" cycle live
here, isolated from the existing `src/` tree.

```
revision/
├── code/                     ← new Python modules, importable as ``revision.code.<m>``
│   ├── common.py             ← shared paths, CV iterator, table writer
│   ├── leakage_ablation.py   ← R2.4 — Table 8
│   ├── fairness.py           ← R2.5 — Table 10 + Fig. 13
│   ├── causal_psm.py         ← R2.2 — Table 9
│   ├── sota_baselines.py     ← R1.4 / R2.1 — Table 7
│   ├── replication_tables.py ← R2.3 — Tables A1–A4
│   ├── feature_count_audit.py← R1.5 / R1.6 / R1.7 — corrected Table 2 + taxonomy
│   └── run_all.py            ← orchestrator (calls every stage above)
├── manuscript_patches/       ← red-font text snippets, one per reviewer comment
├── output/
│   ├── tables/               ← .md + .csv produced by the modules
│   └── figures/              ← .png figures produced by the modules
├── run_all.cmd               ← Windows wrapper
└── run_all.sh                ← POSIX wrapper
```

## How to run

From the project root:

```bash
# Windows
recent-review-comments\revision\run_all.cmd

# POSIX
recent-review-comments/revision/run_all.sh

# Or directly
python recent-review-comments/revision/code/run_all.py
```

`run_all.py` runs **seven** stages in order, isolates failures (a missing
optional dependency such as LightGBM disables only the affected table), and
prints a final OK/FAILED summary:

| # | Stage | Output |
| --- | --- | --- |
| 1 | `replication_tables` | Tables A1–A4 (.md + .csv) |
| 2 | `feature_audit` | Corrected Table 2 + taxonomy + per-feature trace |
| 3 | `leakage_ablation` | Table 8 |
| 4 | `fairness` | Table 10 + Fig. 13 |
| 5 | `causal_psm` | Table 9 |
| 6 | `sota_comparison` | Table 7 |
| 7 | `fill_patches` | Filled-in patches under `manuscript_patches/filled/` plus two `.docx` files (combined patches + response letter) with all inserted text in **red font** |

The `.docx` outputs land at:

```
revision/output/Revised_Manuscript_Patches_Combined.docx
revision/output/Response_Letter_20262542.docx
```

Both have inserted/modified text in red, satisfying the editor's instruction
"Please highlight modifications and additions inside the paper by red font."

## Pre-requisites

The revision modules **read** from `src/config.py`, `src/data/loader.py`,
`src/data/preprocessor.py`, and the cached arrays produced by the existing
pipeline (`results/preprocessed/X.npy`, `y_*.npy`, `feature_names.json`,
`encoders.pkl`). They never modify the existing `src/` tree.

If the cached arrays are missing the orchestrator will invoke
`src.data.preprocessor.preprocess_pipeline()` on first run.

Optional dependencies for individual stages:

| Stage | Optional dependency |
| --- | --- |
| `sota_baselines` (Abukader recipe) | `lightgbm` |
| `sota_baselines` (Kalita recipe) | `torch` |
| `fairness` (calibration figure) | `matplotlib` |
| `fill_patches` (.docx export) | `python-docx` |

If any of these is missing the affected row reports
*"library unavailable"* / *"figure skipped"* and the rest of the pipeline
continues uninterrupted.

## Output → manuscript mapping

Every entry in `manuscript_patches/` is keyed to one or more files in
`output/tables/`. Replace each `« fill »` placeholder in a patch with the
corresponding cell from the matching CSV before pasting the snippet into the
Word document (in red font, per the editor's instruction).

| Patch | Source CSV |
| --- | --- |
| `06_section_3_10_leakage_controls.md` | `output/tables/leakage_ablation.csv` |
| `08_section_5_4_sota_comparison.md` | `output/tables/sota_comparison.csv` |
| `09_section_5_5_causal_psm.md` | `output/tables/causal_psm.csv` |
| `10_section_5_6_fairness.md` | `output/tables/fairness_subgroups.csv` |
| `13_appendix_a_replication.md` | `output/tables/tableA{1,2,3,4}_*.csv` |
| `14_table2_corrected.md` | `output/tables/feature_audit_table2.csv` |
| `01_abstract.md` | `output/tables/feature_audit_summary.json` (use `final_feature_matrix_size`) |
| `05_section_3_9_recommendation_taxonomy.md` | `output/tables/feature_audit_taxonomy.csv` |
