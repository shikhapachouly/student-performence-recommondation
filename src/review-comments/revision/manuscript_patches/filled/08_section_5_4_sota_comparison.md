# Patch 08 — New §5.4 Comparison with 2024–2025 SOTA (R1.4, R2.1)

Insert as §5.4 (renumber subsequent subsections).

## 5.4 Comparison with Recent State-of-the-Art XAI-EDM Methods

Reviewers correctly noted that Table 1 lists recent 2024–2025 SHAP-based methods but the experimental section evaluates only generic baselines and tabular architectures. To put our claims on a common footing, we re-implemented three representative recent recipes and ran them on **our** dataset under our **identical** 5-fold stratified CV protocol (SMOTE on training folds only, seed 42). All implementations are released as part of the replication package (`recent-review-comments/revision/code/sota_baselines.py`).

**Recipes evaluated.**

* **Abukader et al. 2025 [16]** — metaheuristic-tuned LightGBM with class-balanced loss and SHAP explanation.
* **Liu et al. 2025 [15]** — stacking ensemble (Random Forest + Gradient Boosting + XGBoost base learners; logistic-regression meta-learner) with SHAP.
* **Kalita et al. 2025 [13]** — bidirectional LSTM treating the feature vector as a length-*d* sequence (the adaptation Kalita et al. apply to cross-sectional inputs), with SHAP via KernelExplainer.

**Table 7. SOTA comparison on our dataset (F1-macro, 5-fold stratified CV).** Numbers are populated from `revision/output/tables/sota_comparison.csv`.

| Method | CGPA F1-macro | At-Risk F1-macro | Career F1-macro |
| --- | --- | --- | --- |
| Abukader 2025 [16] — LightGBM + SHAP | 0.455 ± 0.014 | 0.637 ± 0.029 | 0.692 ± 0.020 |
| Liu 2025 [15] — Stacking ensemble + SHAP | 0.448 ± 0.026 | 0.612 ± 0.028 | 0.640 ± 0.025 |
| Kalita 2025 [13] — Bi-LSTM + SHAP | 0.308 ± 0.038 | 0.559 ± 0.042 | 0.518 ± 0.048 |
| **This work — XGBoost + engineered features + SHAP recs** | 0.805 ± 0.017 | 0.767 ± 0.040 | 0.699 ± 0.024 |


**Discussion.** Where this work attains a higher F1-macro the gap is consistent with the **engineered composite features**, not with the choice of classifier (XGBoost on raw features alone falls inside the SOTA band — see Table 4). We therefore frame our contribution as the *unified multi-objective formulation* and *automated SHAP-to-recommendation translation* rather than a claim of architectural superiority. Where SOTA matches our numbers, the differentiator is the recommendation-engine output (Table 6) which none of the cited works produce.
