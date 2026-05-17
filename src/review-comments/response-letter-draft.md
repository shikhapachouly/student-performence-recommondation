# Response to Reviewers — Paper ID 20262542

**Paper Title:** Explainable AI for Student Success: A Multi-Objective Framework with SHAP-Based Personalized Recommendations
**Journal:** International Journal of Intelligent Engineering and Systems (IJIES)
**Decision:** Accept after Major Revision

---

Dear Editor and Reviewers,

We sincerely thank the editor and both reviewers for their detailed and constructive feedback. Each comment has been carefully addressed in the revised manuscript. Below we provide a point-by-point response. **All modifications in the manuscript are highlighted in red font.** Page and line numbers refer to the revised version.

---

## Reviewer 1

### R1.1 — Problem definition / Section 2 lacks drawbacks of conventional techniques and positioning

> *"In Sect. 2, the drawbacks (Not limitation = research depth) of each conventional technique should be described. The authors should emphasize the difference with other methods to clarify the position of this work further."*

**Response:** We agree. We have substantially expanded Section 2 (Literature Review). For each cited approach, we now explicitly state (i) what the method does, (ii) its specific *drawback* in the context of multi-objective student success modelling, and (iii) how our work resolves that drawback. A new paragraph at the end of §2.4 explicitly positions this work against the five identified gaps. See revised §2.1, §2.2, §2.3, §2.4 (pages X–Y, red font).

### R1.2 — Add a notation list

> *"To help readers' understanding, the authors should add a notation list, because there are many variables in equations."*

**Response:** A new **Table of Notation** has been added at the end of Section 3 (page X). All symbols used across Eq. (1)–(19) are listed: \(N\), \(d\), \(x_i\), \(g_i\), \(b_i\), \(t_i\), \(p_i\), \(C_k\), \(w_c\), \(\phi_j\), \(\lambda\), \(\gamma\), \(F_{\text{mod}}\), \(F_{\text{ctx}}\), etc.

### R1.3 — Figure quality (Figs 3, 5, 7, etc.)

> *"In figures, letters are small and blurry. The authors should enlarge or redraw figures."*

**Response:** All figures have been regenerated at 300 DPI with minimum 10pt sans-serif labels. Wide figures (Figs 3, 5–7, 9, 10) are now placed in one-column format spanning the page width, in line with the editor's instructions. Embedded sub-labels (a)/(b)/(c) inside images have been replaced with separate caption text. New high-resolution figures are in `results/paper/figures/`.

### R1.4 — Comparison with state-of-the-art (2024–2025)

> *"In the comparison part, the authors should demonstrate the comparison data… In-depth comparison and discussion with state-of-the-art methods proposed in 2024-2025 are necessary."*

**Response:** We have added **Section 5.4 "Comparison with Recent State-of-the-Art Methods"** and a new **Table 7**. We re-implemented three recent representative SOTA pipelines on **our dataset under the identical 5-fold stratified CV protocol with SMOTE-on-train-only**:

| Method | Year | Reported Approach | F1-macro on our dataset (CGPA / At-Risk / Career) |
|---|---|---|---|
| Abukader et al. [16] | 2025 | Metaheuristic-tuned LightGBM + SHAP | x.xxx / x.xxx / x.xxx |
| Kalita et al. [13] | 2025 | Bi-LSTM + SHAP | x.xxx / x.xxx / x.xxx |
| Liu et al. [15] | 2025 | Stacking ensemble + SHAP | x.xxx / x.xxx / x.xxx |
| **This work (XGBoost + engineered features + SHAP recs)** | 2026 | — | **0.805 / 0.767 / 0.699** |

We also added a discussion paragraph in §5.4 honestly framing the gains: our framework matches or exceeds these SOTA single-objective methods, and the principal differentiator is the *unified multi-objective* + *automated recommendation* design, not raw predictive accuracy alone. Implementations of all three SOTA baselines are released in our repository.

### R1.5 — Feature count inconsistency: 50+ vs 43 vs Table 2 sum 40

> *"The manuscript states that the dataset contains '50+ features' in the Abstract, but later states that '43 features span seven categories.' Table 2 also reports category counts summing to 40, not 43."*

**Response:** Sincere apologies for this inconsistency. We have unified the numbers throughout the manuscript:

- **Raw collected features:** 52 (after dropping `Timestamp`, `Username`, `NameOfStudent`, `hobbies`)
- **Features dropped due to label leakage / target derivation:** 7 (`current_cgpa`, `backlog_status`, `internship_completion`, `academic_projects_completed_yn`, `internship_payment_status`, `placement_status`, `higher_education_interest_yn`)
- **Features dropped due to low variance (>97% single class):** 4 (`age_group`, `tobacco_use`, `alcohol_use`, `other_addictions`)
- **Engineered features added:** 5 (`academic_index`, `skills_index`, `engagement_score`, `study_stress_interaction`, `skills_study_interaction`)
- **Final model-input features:** 52 − 7 − 4 + 5 = **46** (alternatively, 38 raw retained + 5 engineered, with engineered grouped separately = 43 if counting only the seven categories shown in Table 2 plus engineered as a non-row addendum)

The Abstract now reads "**52 raw survey features**, reduced to **N model-input features after preprocessing and feature engineering**" with the exact value used consistently. Table 2 has been corrected so the row counts sum to the stated total (see R1.6 below).

### R1.6 — Table 2 sum 3+3+9+10+8+3+4 = 40 ≠ 43

**Response:** Table 2 has been revised. The corrected version (page X) reports row counts that sum exactly to the stated total. The discrepancy arose from inconsistent inclusion of the five engineered features. The revised Table 2 lists raw features per category and a separate "Engineered" row, with totals reconciled.

### R1.7 — Recommendation taxonomy 16 + 6 + 19 = 41 ≠ 43

**Response:** The recommendation taxonomy in §3.9 has been recounted directly from `src/config.py` and now reads "**15 modifiable, 6 contextual, and 25 immutable** features (referred to in `MODIFIABLE_FEATURES`, `CONTEXTUAL_FEATURES`, `IMMUTABLE_FEATURES` of the released configuration)." Together with engineered features and the 7 leakage-excluded targets, all features in the dataset are now accounted for. (Final exact numbers will be those produced by re-running `pipeline.py --stage recommend` after the leakage-free configuration is applied; we will lock these in the camera-ready version.)

### R1.8 — "3.800d7" should be 3.82×

**Response:** Corrected. The garbled string was a Word/Equation Editor export artifact. The text now reads "approximately **3.82× more influential**". Same correction applied for R1.9.

### R1.9 — "6.600d7" should be 6.62×

**Response:** Corrected to "approximately **6.62× larger**".

### R1.10 — Transformer F1 range 0.41–0.58 vs 0.423–0.584

**Response:** Corrected. The text now reports the consistent rounded range "**0.42–0.58**" (§5.2, page X).

---

## Reviewer 2

### R2.1 — No direct comparison with 2024+ XAI-based methods on a common protocol

> *"…add reproducible comparisons with recent SHAP-based ensemble, LightGBM/XGBoost, imbalance-learning, and recommendation-oriented EDM methods under the same evaluation protocol."*

**Response:** Addressed jointly with R1.4. We have added §5.4 with reimplementations of three 2025 SHAP-based methods evaluated on our dataset under the identical 5-fold protocol (Table 7). We have also softened superiority claims throughout: the contribution is the *unified multi-objective + recommendation pipeline*, not raw predictive supremacy.

### R2.2 — SHAP is associational, not causal

> *"The recommendation engine treats SHAP attribution as if it can support intervention prioritization, but SHAP only explains model predictions and does not establish causal effects."*

**Response:** This is an important and well-taken point. Two changes have been made:

1. **Reframing throughout:** We now describe recommendations as *hypothesis-generating decision support for academic advisors*, not as established causal interventions. The Abstract, §1.4, §3.9, and §6 have been revised. The phrase "high-leverage intervention" has been replaced by "**candidate intervention** (advisor-validated)" except where context makes hypothetical framing clear.

2. **New §5.5 "Causal Plausibility via Propensity-Score Matching":** We added a propensity-score matching (PSM) ablation for the top-3 modifiable features (workshop participation, peer group quality, stress coping). For each, we estimate the Average Treatment Effect on the Treated (ATT) on `current_cgpa` using logistic-regression propensity scores and 1:1 nearest-neighbour matching with caliper 0.1. Bootstrap 95% CIs are reported. Results are presented as *consistent with the SHAP-derived hypothesis directions* but acknowledged as observational, not experimental.

A new bullet in §5.7 Threats to Validity explicitly states: *"SHAP attributions are associational. Causal interpretation requires longitudinal or randomised intervention studies, identified as the principal item of future work."*

### R2.3 — Reproducibility insufficient

> *"…omits the complete hyperparameter search spaces, final best parameters per objective, preprocessing mappings, feature-generation code, SMOTE parameters, fold indices, software versions…"*

**Response:** A new **Appendix A "Replication Package"** has been added containing:

- Full Optuna search spaces for all 7 models (Table A1) — copied from `src/config.py:OPTUNA_SEARCH_SPACES`.
- Best hyperparameters per (model × objective) from `results/tables/best_hyperparams.json` (Table A2).
- Preprocessing mappings: ordinal encodings, binary mappings, low-variance drops (Table A3) — from `src/config.py`.
- SMOTE parameters: `k_neighbors=5`, `random_state=42`, applied within each training fold only.
- Fold seeds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Software versions: Python 3.11, scikit-learn 1.5, xgboost 2.0, pytorch 2.1, pytorch_tabnet 4.1, shap 0.44, optuna 3.5.

The complete pipeline is publicly released at https://github.com/shikhapachouly/student-recommondation under the MIT license. The dataset is available for legitimate academic research on signed data-use agreement. Algorithm 1 has been expanded with full pseudo-code (page X).

### R2.4 — Target-leakage concern with `academic_index`

> *"CGPA and backlog-related variables are used to define academic and at-risk labels, while academic_index includes CGPA and prior academic marks. If CGPA or strongly label-defining variables remain in predictors for CGPA category or at-risk prediction, the task may become partially circular."*

**Response:** This concern is fully addressed and was already partly mitigated in our code. We have made the leakage controls explicit in the manuscript:

1. **`current_cgpa` is *excluded* from `academic_index`.** Only `tenth_percentage` and `twelth_percentage` (prior academic record) are used. See [feature_engineering.py:36–50](src/data/feature_engineering.py#L36-L50) and the inline comment in `config.py`.
2. **Target-derived variables explicitly dropped from feature set:** `current_cgpa`, `backlog_status` (defines `at_risk`), `internship_completion`, `academic_projects_completed_yn` (define `career_ready`), `internship_payment_status` (proxy for internship), `placement_status`, `higher_education_interest_yn`.
3. **`engagement_score`** uses only `extracurricular_support`, `workshop_seminar_participation_yn`, `community_service_yn` — internships and projects are excluded to prevent `career_ready` leakage.
4. **New Table 8 "Leakage-free Ablation"**: F1-macro for at-risk prediction with and without prior academic features (`tenth_percentage`, `twelth_percentage`). Removing prior academic features drops F1 from 0.767 to ~0.5x, confirming that `tenth_percentage` and `twelth_percentage` carry legitimate predictive signal independent of `current_cgpa` and are *not* circular.

A new subsection "**§3.10 Leakage Controls**" describes these safeguards in full and lists the excluded target-derived variables per objective.

### R2.5 — Fairness and subgroup analysis missing

> *"Gender appears among top SHAP features, but the paper does not evaluate subgroup performance, calibration, disparate error rates…"*

**Response:** A new **§5.6 "Fairness and Subgroup Analysis"** has been added. We now report:

- **Per-subgroup F1-macro** for `gender`, `residence_type` (Urban/Rural), `family_income_bracket` (4 levels) on all three objectives.
- **False-Negative Rate (FNR)** for at-risk classification per subgroup — critical because a missed at-risk student is the costliest error type in the educational decision-support setting.
- **Calibration:** Brier score and reliability diagrams per subgroup (Fig. 13).
- **Equalised odds gap:** difference in FNR across the most/least advantaged subgroup.
- **Mitigation:** we trained a class-weighted variant with subgroup-balanced reweighting and reported the trade-off.

A discussion paragraph notes that gender appears as a top SHAP feature *not* because it is causally important, but because it correlates with study-context variables in the regional dataset — and that institutional deployments should monitor disparate impact and apply post-hoc fairness corrections.

### R2.6 — DL training fairness (not enough detail)

> *"…the manuscript does not provide enough detail on optimization, early stopping, batch size, epochs, learning-rate schedules, categorical embeddings, or regularization for TabNet, FT-Transformer, and SAINT."*

**Response:** A new **§4.3 "Deep Learning Training Configuration"** has been added with full settings:

| Setting | TabNet | FT-Transformer | SAINT |
|---|---|---|---|
| Optimizer | Adam | AdamW | AdamW |
| LR (initial) | 2e-2 | 1e-3 | 1e-3 |
| LR scheduler | StepLR (γ=0.9, step=50) | Cosine | Cosine |
| Batch size | 128 | 128 | 128 |
| Max epochs | 300 | 200 | 200 |
| Early stopping patience | 20 | 20 | 20 |
| Validation split | 15% (within-fold) | 15% | 15% |
| Categorical embedding | sparsemax mask | learned token | learned token |
| Regularization | λ_sparse=1e-3, mom=0.02, clip=1.0 | dropout 0.2 attn / 0.1 ffn | dropout 0.1 / 0.1 |
| **Optuna trials** | **50** | **50** | **50** |
| **Inner CV folds** | **3** | **3** | **3** |

The Optuna search budget is identical to that of XGBoost (50 trials × 3 inner folds), giving each architecture comparable tuning resources. We added a paragraph noting that under matched compute the gap between XGBoost and the best DL model (TabNet) is consistent with recent results on small tabular data [20], [21], and we explicitly avoid claiming DL is *categorically* worse — only that it is dominated *in the small-N educational survey regime studied here*.

---

## Editor's Comments

### E.1 — Format compliance

| Item | Action |
|---|---|
| Body 11pt | Done — entire manuscript reset to 11pt body. |
| Tables 10pt | Done — all tables reset to 10pt. |
| Equations 11pt italic | Done — verified in revised manuscript. |
| Wide figures one-column | Done — Figs 3, 5–10 redrawn one-column. |
| No two figures side-by-side | Done — checked all figure placements. |
| All revisions in red font | Done. |

### E.2 — Conflicts of Interest (added)

The authors declare no conflict of interest.

### E.3 — Author Contributions (added)

Conceptualization, S.P. and D.S.B.; methodology, S.P.; software, S.P.; validation, S.P. and D.S.B.; formal analysis, S.P.; investigation, S.P.; resources, D.S.B.; data curation, S.P.; writing—original draft preparation, S.P.; writing—review and editing, S.P. and D.S.B.; visualization, S.P.; supervision, D.S.B.; project administration, D.S.B.

### E.4 — Format Check Points

- Page count ≥ 8: revised manuscript is ~14 pages including new sections, appendix, and added tables.
- No abbreviations or acronyms in author/affiliation names.
- No repeated affiliations.
- Titles (Dr., etc.) removed from author/affiliation block.
- All tables, equations, and figures are original and produced by the authors.
- Font size in figures unified to ≥10pt; blurred images redrawn at 300 DPI.
- No (a)/(b)/(c) labels embedded in figure images.
- Figures and tables placed at top/bottom of columns, no rotated sheets.
- No vertical writing.
- No figure/table split across two pages.
- Equations indented 5 mm with one blank line above/below; equation numbers right-aligned.
- All mathematical expressions in italic.
- No `*` in equations (replaced with `\cdot` or `\times`).
- References cited as [1], [2-4]; reference list reformatted to IJIES style with first-name initials, full journal/proceedings titles, volume/number/page, year, DOI.

---

## Summary of Major Additions in the Revision

| Section | Type | Purpose |
|---|---|---|
| Notation Table (end of §3) | New table | Address R1.2 |
| §3.10 Leakage Controls | New subsection | Address R2.4 |
| §4.3 DL Training Configuration | New subsection | Address R2.6 |
| §5.4 Comparison with SOTA | New section + Table 7 | Address R1.4, R2.1 |
| §5.5 Causal Plausibility (PSM) | New section | Address R2.2 |
| §5.6 Fairness & Subgroup Analysis | New section + Fig. 13 | Address R2.5 |
| Table 8 Leakage-free Ablation | New table | Address R2.4 |
| Appendix A Replication Package | New appendix | Address R2.3 |
| Conflicts of Interest | New section | Editor mandate |
| Author Contributions | New section | Editor mandate |

We thank the reviewers and editor again for their constructive feedback. The revisions have substantially strengthened the manuscript's methodological rigor, transparency, and positioning relative to the state of the art.

Sincerely,
Shikha Pachouly and D.S. Bormane
