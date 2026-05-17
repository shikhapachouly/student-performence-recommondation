# Response to Reviewers — Paper ID 20262542

**Paper Title:** Explainable AI for Student Success: A Multi-Objective Framework with SHAP-Based Personalized Recommendations
**Journal:** International Journal of Intelligent Engineering and Systems (IJIES)
**Decision:** Accept after Major Revision

---

Dear Editor and Reviewers,

We sincerely thank the editor and both reviewers for their detailed and constructive feedback. Each comment has been addressed in the revised manuscript. **All modifications in the manuscript are highlighted in red font.** Page and line numbers below refer to the revised version. The artefacts referenced in this letter are available in the public revision package at `recent-review-comments/revision/` (folder layout: `code/`, `manuscript_patches/`, `output/tables/`, `output/figures/`).

A reproducible orchestrator script is provided:

```bash
python recent-review-comments/revision/code/run_all.py
```

This regenerates every table cited below from the same dataset and the same 5-fold stratified CV protocol used in the original manuscript.

---

## Reviewer 1

### R1.1 — Drawbacks of conventional techniques and positioning of this work (§2)

> *"In Sect. 2, the drawbacks (Not limitation = research depth) of each conventional technique should be described. The authors should emphasize the difference with other methods to clarify the position of this work further."*

**Response.** Section 2 has been substantially expanded. For each cited technique we now state (i) what the method does, (ii) its specific *drawback* in the context of multi-objective student-success modelling, and (iii) how our work resolves that drawback. See `manuscript_patches/02_section2_drawbacks.md` (red-font additions to §2.1, §2.2, §2.3) and the closing positioning paragraph at the end of §2.4.

### R1.2 — Notation list

> *"To help readers' understanding, the authors should add a notation list."*

**Response.** A new **Table 3a — Notation** is inserted at the end of §3 (before §4) listing every symbol used across Eqs. (1)–(19): \(N\), \(d\), \(x_i\), \(g_i\), \(b_i\), \(t_i\), \(p_i\), \(C_k\), \(w_c\), \(\phi_j\), \(\lambda\), \(\gamma\), \(F_{\text{mod}}\), \(F_{\text{ctx}}\), \(F_{\text{imm}}\), \(K\), TPE, ATT, FNR. See `manuscript_patches/03_notation_table.md`.

### R1.3 — Figure quality (Figs 3, 5, 7, etc.)

> *"In figures, letters are small and blurry."*

**Response.** All figures have been regenerated at 300 DPI with minimum 10 pt sans-serif labels. Wide figures (Figs 1, 3, 5–7, 9, 10) now use one-column page-width layout per the editor's instruction. Embedded `(a)/(b)/(c)` sub-labels have been moved into the captions. The new figures live under `recent-review-comments/revision/output/figures/` and the corresponding paper-side outputs in `results/paper/figures/`.

### R1.4 — Comparison with state-of-the-art (2024–2025)

> *"…no comparison data with state-of-the-art methods is shown… In-depth comparison and discussion with state-of-the-art methods proposed in 2024–2025 are necessary."*

**Response.** A new §5.4 *"Comparison with Recent State-of-the-Art XAI-EDM Methods"* presents head-to-head F1-macro scores on **our** dataset under the **identical** 5-fold stratified CV protocol (SMOTE on training folds only, seed 42). Three representative 2025 recipes are reimplemented:

| Recipe | Source code | Reference |
| --- | --- | --- |
| Metaheuristic-tuned LightGBM + SHAP | `revision/code/sota_baselines.py::_make_lightgbm` | Abukader et al. 2025 [16] |
| Stacking ensemble + SHAP | `revision/code/sota_baselines.py::_make_stacking` | Liu et al. 2025 [15] |
| Bi-LSTM + SHAP | `revision/code/sota_baselines.py::_BiLSTMClassifier` | Kalita et al. 2025 [13] |

Results populate **Table 7** in `manuscript_patches/08_section_5_4_sota_comparison.md` from `revision/output/tables/sota_comparison.csv`. Following Reviewer 2's reframing (R2.1) we now describe the contribution as the *unified multi-objective + automated recommendation* design rather than as a claim of architectural superiority.

### R1.5 / R1.6 / R1.7 — Feature-count inconsistencies

> *"50+ features (Abstract) vs 43 (§4.1) vs Table 2 sums to 40."*
> *"Table 2 row counts do not equal the stated total."*
> *"16 + 6 + 19 = 41 ≠ 43 in the recommendation taxonomy."*

**Response.** All three inconsistencies have been **corrected** in the revised manuscript, and the numerical reconciliation has been automated to prevent recurrence. The script `revision/code/feature_count_audit.py` reads the canonical `feature_names.json` produced by `src/data/preprocessor.py` and emits an audit table whose row counts match the live feature matrix **exactly**. The Abstract, Section 3.3, Section 4.1, Table 2, and the recommendation taxonomy have all been re-stated using the audited values, and they now agree on a single, consistent count of **42 model-input features**.

**Evidence (full reconciliation).** Out of the 52 raw survey columns:

$$
52 - \underbrace{4}_{\text{IDs}} - \underbrace{7}_{\text{target-derived}} - \underbrace{4}_{\text{low-variance}} + \underbrace{5}_{\text{engineered}} = 42 \text{ model-input features.}
$$

Source: `revision/output/tables/feature_audit_summary.json`, field `final_feature_matrix_size = 42`.

Specifically, the corrections are:

* **Abstract** (`manuscript_patches/01_abstract.md`) — corrected from the previous "50+ features" to "**52 raw survey features that, after preprocessing and engineering, yield 42 model-input features**", with the value populated directly from `feature_audit_summary.json` (`final_feature_matrix_size`).
* **§4.1** — corrected from the previous "43" to **42**, matching the audited count.
* **Table 2** (`manuscript_patches/14_table2_corrected.md`) — corrected so that the per-category row counts now sum to **42** exactly (previously summed to 40). Counts are populated from `feature_audit_table2.csv`, with engineered features broken out as a separate row.
* **Recommendation taxonomy, §3.9** (`manuscript_patches/05_section_3_9_recommendation_taxonomy.md`) — corrected from the previous "16 + 6 + 19 = 41 ≠ 43" using the live counts from `MODIFIABLE_FEATURES`, `CONTEXTUAL_FEATURES`, and `IMMUTABLE_FEATURES` in `src/config.py`, plus the 5 engineered indicators and the 7 leakage-excluded target columns. The three taxonomy partitions plus the engineered and excluded columns now enumerate all 52 raw columns with no overlaps or gaps. The full per-column trace is in `feature_audit_full.md` (Table A5 of Appendix A).

### R1.8 / R1.9 — `3.800d7` and `6.600d7` artefacts

**Response.** Word/Equation-Editor export artefacts. Replaced with the correct numerical ratios:

* §5.5: "academic_index … is **approximately 3.82× more influential**" (was `3.800d7`).
* §5.5 Discussion: "workshop participation has a SHAP impact **approximately 6.62× larger**" (was `6.600d7`).

See `manuscript_patches/11_section_5_5_text_corrections.md`.

### R1.10 — Transformer F1 range (0.41–0.58 vs Table 4 0.423–0.584)

**Response.** Corrected to **"0.42–0.58"** consistently across §5.2 prose. See `manuscript_patches/15_table4_consistency.md`.

---

## Reviewer 2

### R2.1 — No reproducible comparison with 2024+ XAI-based methods on a common protocol

**Response.** Addressed jointly with R1.4 in new §5.4. Implementations are released in `revision/code/sota_baselines.py`. Superiority claims are softened throughout: the contribution emphasised is the unified multi-objective + recommendation pipeline rather than raw F1.

### R2.2 — SHAP is associational, not causal

> *"The recommendation engine treats SHAP attribution as if it can support intervention prioritization, but SHAP only explains model predictions and does not establish causal effects."*

**Response.** Two changes:

1. **Reframing.** Recommendations are described as *hypothesis-generating decision support for academic advisors* throughout the Abstract, §1.4, §3.9, §6, and the recommendation prose around Table 6. See `manuscript_patches/01_abstract.md` and `manuscript_patches/12_section_6_threats_validity_addendum.md`.
2. **New §5.5 Causal Plausibility via Propensity-Score Matching.** For the top-three modifiable features (workshop participation, peer-group quality, stress coping), we estimate the Average Treatment Effect on the Treated (ATT) on `current_cgpa` via 1:1 nearest-neighbour matching on logit-propensity (caliper = 0.2 SD) and report a 95 % bootstrap CI. Implementation: `revision/code/causal_psm.py`. Numbers: `revision/output/tables/causal_psm.csv` → **Table 9**. The text is explicit that PSM is observational and not a substitute for randomised intervention evidence; this aligns the manuscript with the reviewer's stated standard.

A new bullet in §5.7 (Threats to Validity) and a revised §6 future-work paragraph commit to cluster-randomised intervention trials and longitudinal causal-effect estimation.

### R2.3 — Reproducibility insufficient

> *"…omits the complete hyperparameter search spaces, final best parameters per objective, preprocessing mappings, feature-generation code, SMOTE parameters, fold indices, software versions…"*

**Response.** A new **Appendix A — Replication Package** has been added (`manuscript_patches/13_appendix_a_replication.md`) containing four tables auto-generated by `revision/code/replication_tables.py`:

| Appendix table | Generator | Output file |
| --- | --- | --- |
| **Table A1** Optuna search spaces | `write_search_spaces` | `tableA1_search_spaces.{md,csv}` |
| **Table A2** Best hyperparameters per (model × objective) | `write_best_hyperparams` | `tableA2_best_hyperparams.{md,csv}` |
| **Table A3** Preprocessing mappings | `write_preprocessing_mappings` | `tableA3_preprocessing.{md,csv}` |
| **Table A4** Runtime / CV / SMOTE / software versions | `write_runtime_config` | `tableA4_runtime.{md,csv}` |

The complete pipeline, including this revision package, is publicly released at <https://github.com/shikhapachouly/student-recommondation> under the MIT licence. Reproduction is reduced to three commands (`git clone`, `pip install -r requirements.txt`, `python -m src.pipeline --stage all` followed by `python recent-review-comments/revision/code/run_all.py`).

### R2.4 — Target-leakage concern with `academic_index`

> *"If CGPA or strongly label-defining variables remain in predictors for CGPA category or at-risk prediction, the task may become partially circular."*

**Response.** This concern is fully addressed. The leakage controls were already partly enforced in code; they are now explicit in the manuscript:

1. **`current_cgpa` is excluded from `academic_index`.** Only `tenth_percentage` and `twelth_percentage` are aggregated. See `src/data/feature_engineering.py` lines 36–50 and the inline comment in `src/config.py` (`TARGET_FEATURES`).
2. **Target-defining columns are dropped from the feature matrix** for every objective: `current_cgpa`, `backlog_status`, `internship_completion`, `academic_projects_completed_yn`, `internship_payment_status`, `placement_status`, `higher_education_interest_yn`. Enforced in `src/data/preprocessor.py::encode_features`.
3. **`engagement_score`** uses only `extracurricular_support`, `workshop_seminar_participation_yn`, `community_service_yn` — internships and projects are excluded.
4. **New §3.10 Leakage Controls** (`manuscript_patches/06_section_3_10_leakage_controls.md`) lists every excluded raw column per objective.
5. **Table 8 Leakage-free ablation** reports F1-macro for each objective with and without prior academic features (`tenth_percentage`, `twelth_percentage`, `academic_index`). Implementation: `revision/code/leakage_ablation.py`. Output: `revision/output/tables/leakage_ablation.csv`. The drop confirms the engineered `academic_index` carries genuine prior-academic signal rather than a residual echo of `current_cgpa`.

### R2.5 — Fairness and subgroup analysis missing

> *"…the paper does not evaluate subgroup performance, calibration, disparate error rates."*

**Response.** New **§5.6 Fairness and Subgroup Analysis** (`manuscript_patches/10_section_5_6_fairness.md`) reports per-subgroup F1-macro, FNR for the positive class, Brier score, and the equalised-odds gap across `gender`, `residence_type`, and `family_income_bracket`, for all three objectives. A calibration plot for the at-risk task by gender is **Fig. 13** (`revision/output/figures/fairness_calibration.png`). The full per-subgroup table appears as **Table A6** in the appendix (`fairness_subgroups.{md,csv}`). The §5.6 discussion warns that the appearance of `gender` among top SHAP features reflects regional dataset correlations, not causal influence, and recommends post-hoc reweighting plus restriction of recommendations to `MODIFIABLE_FEATURES` only.

### R2.6 — DL training fairness (insufficient detail)

> *"…the manuscript does not provide enough detail on optimization, early stopping, batch size, epochs, learning-rate schedules, categorical embeddings, or regularization for TabNet, FT-Transformer, and SAINT."*

**Response.** A new **§4.3 Deep-Learning Training Configuration** (`manuscript_patches/07_section_4_3_dl_config.md`) lists optimiser, scheduler, batch size, epoch budget, early-stopping patience, in-fold validation split, categorical embedding scheme, dropout, gradient clipping, sparsity penalty, depth, heads, and dimension per architecture. Crucially, the **Optuna search budget is identical to that of XGBoost** (50 trials × 3 inner folds = 150 inner-CV evaluations), so the DL gap reported in Table 4 cannot be attributed to under-tuning. The text reframes the result as regime-specific to small-N educational survey data, not as a categorical claim about tabular DL.

---

## Editor's Comments

Each item from the editor's format checklist is tracked in `manuscript_patches/17_formatting_checklist.md`. Highlights:

* Body 11 pt, tables 10 pt, equations 11 pt italic — applied throughout.
* All wide figures one-column; no two figures side-by-side; no `(a)/(b)/(c)` embedded in image; no borders; no rotated sheets; no figure split across pages.
* All revisions in **red font**.
* References reformatted to IJIES sample style with first-name initials, vol/no/pp/year/doi.
* **Conflicts of Interest** and **Author Contributions** present in the IJIES wording (`manuscript_patches/16_conflicts_and_authorship.md`).
* Manuscript length now ≥ 8 pages with the new §3.10, §4.3, §5.4, §5.5, §5.6, and Appendix A.

---

## Summary of Major Additions in the Revision

| Section | Type | Reviewer comment | Source |
| --- | --- | --- | --- |
| Notation table (end of §3) | New table | R1.2 | `manuscript_patches/03_notation_table.md` |
| §3.4 leakage clarification | Edit | R1.5, R2.4 | `manuscript_patches/04_section_3_4_feature_eng_clarification.md` |
| §3.9 taxonomy reconciliation | Edit | R1.7 | `manuscript_patches/05_section_3_9_recommendation_taxonomy.md` |
| §3.10 Leakage controls + Table 8 | New section | R2.4 | `manuscript_patches/06_section_3_10_leakage_controls.md` + `revision/code/leakage_ablation.py` |
| §4.3 DL training configuration | New section | R2.6 | `manuscript_patches/07_section_4_3_dl_config.md` |
| §5.4 SOTA comparison + Table 7 | New section | R1.4, R2.1 | `manuscript_patches/08_section_5_4_sota_comparison.md` + `revision/code/sota_baselines.py` |
| §5.5 Causal plausibility (PSM) + Table 9 | New section | R2.2 | `manuscript_patches/09_section_5_5_causal_psm.md` + `revision/code/causal_psm.py` |
| §5.6 Fairness audit + Table 10 + Fig. 13 | New section | R2.5 | `manuscript_patches/10_section_5_6_fairness.md` + `revision/code/fairness.py` |
| §5.7 Threats addendum and §6 reframing | Edit | R2.2 | `manuscript_patches/12_section_6_threats_validity_addendum.md` |
| Numerical/text corrections (3.82×, 6.62×, 0.42–0.58) | Edit | R1.8, R1.9, R1.10 | `manuscript_patches/11_section_5_5_text_corrections.md`, `manuscript_patches/15_table4_consistency.md` |
| Corrected Table 2 + new Table A5 | Edit + new | R1.5, R1.6 | `manuscript_patches/14_table2_corrected.md` + `revision/code/feature_count_audit.py` |
| Appendix A Replication package (Tables A1–A4) | New appendix | R2.3 | `manuscript_patches/13_appendix_a_replication.md` + `revision/code/replication_tables.py` |
| Conflicts of Interest + Author Contributions | New | Editor mandate | `manuscript_patches/16_conflicts_and_authorship.md` |
| Formatting compliance (red-font, 11 pt body, 10 pt tables, etc.) | Edit | Editor mandate | `manuscript_patches/17_formatting_checklist.md` |

We thank the reviewers and editor again for their constructive feedback. The revisions have substantially strengthened the manuscript's methodological rigor, transparency, and positioning relative to the state of the art.

Sincerely,
Shikha Pachouly and D.S. Bormane
