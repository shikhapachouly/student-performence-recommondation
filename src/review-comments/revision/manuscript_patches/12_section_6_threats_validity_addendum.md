# Patch 12 — Threats-to-Validity addendum and §6 reframing (R2.2)

## Append to §5.7 (Threats to Validity)

Add the following bullet:

> **Causal validity.** SHAP attributions are associational and not, on their own, sufficient to support intervention prioritisation. The propensity-score-matched analysis in §5.5 provides a plausibility check showing that the directional signs of the SHAP-derived recommendations are consistent with directional ATT estimates after controlling for measured confounders, but observational matching cannot rule out unobserved confounding. Recommendations should therefore be interpreted as **hypothesis-generating decision support for academic advisors**, not as established causal interventions. Validation requires longitudinal panel data or randomised cluster-level interventions, identified as principal future-work items in §6.

## Replace the future-work paragraph in §6 with

> Future work includes: (1) cluster-randomised intervention trials at the institute level for the top-ranked modifiable features (workshop participation, peer-group quality, stress coping), (2) longitudinal extension to a multi-semester panel that allows directed acyclic-graph-based causal inference, (3) external validation on multi-institutional and multi-disciplinary datasets across different educational systems, (4) integration of structural causal-effect estimators (g-formula, doubly-robust ATE estimation), (5) interactive dashboard development for real-time advisor use with embedded fairness monitors, and (6) extension to sequential prediction using temporal student data.
