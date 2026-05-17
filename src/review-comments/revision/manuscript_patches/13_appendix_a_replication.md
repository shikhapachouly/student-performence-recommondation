# Patch 13 — New Appendix A Replication Package (R2.3)

Insert as a new Appendix A immediately before the References section.

## Appendix A. Replication Package

The complete pipeline used in this paper is publicly released at

> https://github.com/shikhapachouly/student-recommondation

under the MIT licence. The repository includes preprocessed data tensors, training/CV scripts, the SHAP-based recommendation engine, and the revision-cycle modules described in this appendix. The dataset is available for legitimate academic research on signed data-use agreement.

The four tables that follow are produced verbatim by `python -m revision.code.replication_tables` and are committed under `recent-review-comments/revision/output/tables/`.

* **Table A1** — Optuna search space per model
  *(`tableA1_search_spaces.md`)*
* **Table A2** — Best hyperparameters per (model × objective)
  *(`tableA2_best_hyperparams.md`)*
* **Table A3** — Preprocessing mappings (ordinal encodings, binary columns, low-variance drops, target features)
  *(`tableA3_preprocessing.md`)*
* **Table A4** — Runtime / CV / SMOTE / SHAP configuration and software versions
  *(`tableA4_runtime.md`)*
* **Table A5** — Per-feature classification trace (raw column → Table-2 category → recommendation tier)
  *(`feature_audit_full.md`)*
* **Table A6** — Full per-subgroup fairness audit
  *(`fairness_subgroups.md`)*

### Reproduction in three commands

```bash
git clone https://github.com/shikhapachouly/student-recommondation.git
pip install -r requirements.txt
python -m src.pipeline --stage all       # produces all paper tables and figures
python -m revision.code.run_all          # produces all revision tables and figures
```

Random seeds: `numpy = 42`, `random = 42`, `torch.manual_seed = 42`. CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. SMOTE: `k_neighbors = 5`, `random_state = 42`, applied **only** within each training fold.
