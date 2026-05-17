# Patch 11 — Numerical / formatting corrections in §5.3, §5.5 (R1.8, R1.9, R1.10)

## R1.8 — `3.800d7` artefact in §5.5 / §5.5 Discussion

Replace:

> "academic_index (SHAP=0.126 for at-risk) is 3.800d7 more influential than the next raw feature (tenth_percentage, SHAP=0.033)."

With:

> "academic_index (SHAP = 0.126 for at-risk) is **approximately 3.82× more influential** than the next raw feature (tenth_percentage, SHAP = 0.033)."

## R1.9 — `6.600d7` artefact in §5.5 Discussion

Replace:

> "the highest-impact intervention (workshop participation) has a SHAP impact 6.600d7 larger than the fifth-ranked factor (teamwork skills)"

With:

> "the highest-impact intervention (workshop participation) has a SHAP impact **approximately 6.62× larger** than the fifth-ranked factor (teamwork skills)"

## R1.10 — Transformer F1 range consistency in §5.2

Replace:

> "Transformer-based models struggle: FT-Transformer and SAINT significantly underperform on all objectives (F1=0.41–0.58)"

With:

> "Transformer-based models struggle: FT-Transformer and SAINT significantly underperform on all objectives (F1 = **0.42–0.58**)"

(matches the rounded values in Table 4: 0.423–0.584)
