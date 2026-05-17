# Patch 04 — §3.4 Feature Engineering clarification (R1.5, R2.4)

**Replace the existing first paragraph of §3.4 with the following.**

> Five engineered features were designed based on educational domain knowledge to capture multi-dimensional student characteristics that no single raw survey response can represent. **All five are constructed strictly from columns that are not part of any target definition, eliminating circularity between predictor and label.** Specifically:
>
> - `academic_index` aggregates *prior* academic record (`tenth_percentage`, `twelth_percentage`) only; `current_cgpa` is excluded because it defines both the CGPA-category and at-risk targets.
> - `engagement_score` aggregates `extracurricular_support`, `workshop_seminar_participation_yn`, and `community_service_yn`; `internship_completion` and `academic_projects_completed_yn` are excluded because they define the career-readiness target.
> - `skills_index`, `study_stress_interaction`, and `skills_study_interaction` are constructed from features that have no overlap with any target.
>
> An ablation confirming that this leakage discipline does not artificially inflate metrics is reported in §3.10 and Table 8.
