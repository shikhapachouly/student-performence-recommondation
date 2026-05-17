# Patch 14 — Corrected Table 2 (R1.6)

Replace the existing Table 2 with the corrected one whose row counts reconcile to the stated total exactly.

> The exact numbers below are populated from `revision/output/tables/feature_audit_table2.csv`, which derives them directly from the released `feature_names.json`. The placeholder values shown here track the *current* preprocessing in `src/config.py`; re-run the audit if any preprocessing config changes.

**Table 2. Feature Categories and Counts (corrected).**

| Category | Count | Example features |
| --- | --- | --- |
| Academic | « fill » | tenth_percentage, twelth_percentage, course_branch |
| Demographic | « fill » | gender, residence_type, institute_name |
| Family / Socio-economic | « fill » | mother_education_level, family_income_bracket |
| Behavioural | « fill » | daily_study_hours, workshop_seminar_participation_yn |
| Skills | « fill » | communication_skills_scale_1to5, leadership_scale_1to5 |
| Wellbeing | « fill » | health_status_scale_1to5, stress_frequency_scale_1to5 |
| Choice context | « fill » | college_choice_reason, internet_access |
| Engineered | 5 | academic_index, skills_index, engagement_score |
| **Total** | **42** | |
