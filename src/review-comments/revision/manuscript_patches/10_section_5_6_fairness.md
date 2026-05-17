# Patch 10 — New §5.6 Fairness and Subgroup Analysis (R2.5)

Insert as §5.6 (renumber subsequent subsections).

## 5.6 Fairness and Subgroup Analysis

Because gender appears among the top SHAP features for the academic-performance objective and the framework is intended for educational decision support, we audit subgroup performance, calibration, and disparate error rates across three protected attributes: `gender`, `residence_type`, and `family_income_bracket`. The audit re-uses the canonical 5-fold OOF predictions (so the metrics are directly comparable to Table 4) and reports, per (objective × attribute × subgroup):

* **F1-macro** — overall classification quality.
* **False-Negative Rate (FNR) for the positive class** — the rate at which truly at-risk students are missed; this is the costliest error mode in advisor-facing deployment.
* **Brier score** — calibration of the predicted probabilities.
* **Equalised-odds gap** = max FNR − min FNR across subgroups of an attribute.

Numbers are populated from `revision/output/tables/fairness_subgroups.csv`. Calibration curves by gender for the at-risk objective are shown in **Fig. 13**.

**Table 10. Fairness audit (excerpt).**

| Objective | Attribute | Subgroup | n | F1-macro | FNR (positive) | Brier | Equalised-odds gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| At-Risk | gender | « fill » | | | | | |
| At-Risk | residence_type | « fill » | | | | | |
| At-Risk | family_income_bracket | « fill » | | | | | |
| (full table in Appendix A, Table A6) | | | | | | | |

**Discussion.** The presence of `gender` among top SHAP features does **not** imply that the framework treats gender as a causal driver of academic outcome; it reflects correlations in the regional dataset between gender and study-context variables. We therefore advise that any institutional deployment (i) monitor disparate impact via the equalised-odds gap, (ii) apply post-hoc reweighting when this gap exceeds an operationally chosen threshold, and (iii) restrict recommendations to features marked *modifiable* in `MODIFIABLE_FEATURES`, never to immutable demographic features.
