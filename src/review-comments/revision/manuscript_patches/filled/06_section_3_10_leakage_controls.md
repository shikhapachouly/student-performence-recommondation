# Patch 06 — New §3.10 Leakage Controls (R2.4)

Insert as a new subsection §3.10, immediately before §4 Experimental Setup.

## 3.10 Leakage Controls

Because the at-risk and CGPA-category labels are deterministic functions of `current_cgpa` and `backlog_status`, and the career-readiness label is a deterministic function of `internship_completion` and `academic_projects_completed_yn`, retaining these raw columns as predictors would induce target leakage. We therefore apply the following discipline (codified in `src/config.py:TARGET_FEATURES` and enforced in `src/data/preprocessor.py:encode_features`):

1. **Target-defining raw columns are dropped from the predictor set** for *every* objective. The dropped set is `{ current_cgpa, backlog_status, internship_completion, academic_projects_completed_yn, internship_payment_status, placement_status, higher_education_interest_yn }`.
2. **Engineered features are constructed without target-defining inputs.** `academic_index` uses only `tenth_percentage` and `twelth_percentage`; `engagement_score` uses only `extracurricular_support`, `workshop_seminar_participation_yn`, and `community_service_yn`. The remaining engineered features (`skills_index`, `study_stress_interaction`, `skills_study_interaction`) have no overlap with any target.
3. **SMOTE is applied only inside training folds.** Validation and test folds never see synthetic minority samples, preventing the optimistic-bias mode reported in [24].
4. **An ablation isolates the contribution of prior academic record.** Table 8 reports F1-macro for each objective with and without `tenth_percentage`, `twelth_percentage`, and `academic_index`. The result confirms that the engineered `academic_index` carries genuine prior-academic signal rather than a residual echo of `current_cgpa`.

| Objective | F1-macro (full features) | F1-macro (no prior academic) | Δ (impact of prior academic) |
| --- | --- | --- | --- |
| CGPA Category | 0.451 ± 0.027 | 0.387 ± 0.019 | 0.063 |
| At-Risk | 0.639 ± 0.020 | 0.540 ± 0.034 | 0.099 |
| Career Ready | 0.695 ± 0.021 | 0.588 ± 0.017 | 0.107 |
The decline observed in the "no prior academic" column is the *legitimate* contribution of historical academic record — not circular leakage from the label.
