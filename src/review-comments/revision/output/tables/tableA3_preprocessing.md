# Table A3. Preprocessing mappings (canonical source: `src/config.py`).

| Mapping | Definition |
| --- | --- |
| Identifiers dropped (`DROP_FEATURES`) | `Timestamp`, `Username`, `NameOfStudent`, `hobbies` |
| Target-derived features dropped (`TARGET_FEATURES`) | `current_cgpa`, `backlog_status`, `internship_completion`, `academic_projects_completed_yn`, `internship_payment_status`, `placement_status`, `higher_education_interest_yn` |
| Low-variance columns dropped (`LOW_VARIANCE_DROP`) | `age_group`, `tobacco_use`, `alcohol_use`, `other_addictions` |
| Binary Yes/No columns (`BINARY_COLUMNS`) | `backlog_status`, `nursery_education`, `internet_access`, `extracurricular_support`, `workshop_seminar_participation_yn`, `community_service_yn`, `internship_completion`, `academic_projects_completed_yn`, `stress_coping_yn`, `romantic_relationship_yn`, `paid_classes_attending_yn`, `alcohol_use`, `tobacco_use`, `other_addictions`, `higher_education_interest_yn` |
| Ordinal 1–5 scale columns (`ORDINAL_SCALE_COLUMNS`) | `family_relationship_quality_scale_1to5`, `communication_skills_scale_1to5`, `leadership_scale_1to5`, `teamwork_skills_scale_1to5`, `peer_group_quality_scale_1to5`, `time_management_scale_1to5`, `health_status_scale_1to5`, `stress_frequency_scale_1to5`, `socializing_frequency_scale_1to5` |
| Ordinal encoding for `mother_education_level` | No Education < High School < UG < PG < Any Other |
| Ordinal encoding for `father_education_level` | No Education < High School < UG < PG < Any Other |
| Ordinal encoding for `family_income_bracket` | Less than 2.5 Lakh < 2.5 Lakh to 8 Lakh < 8 Lakh to 15 Lakh < More than 15 Lakh |
| Ordinal encoding for `commute_duration` | 15 min < 15 min to 30 min < 30 min to 1 hour < 1 hour to 4 hour |
| Ordinal encoding for `daily_study_hours` | 15 to 30 min < 30 min to 1 hour < 1 hour - 4 hour < More than 4 hours |
| study_hours ordinal map | 15 to 30 min → 1, 30 min to 1 hour → 2, 1 hour - 4 hour → 3, More than 4 hours → 4 |
