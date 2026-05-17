# Patch 09 — New §5.5 Causal Plausibility (R2.2)

Insert as §5.5 (renumber subsequent subsections).

## 5.5 Causal Plausibility via Propensity-Score Matching

Reviewers correctly observed that SHAP attributions are *associational* and cannot, on their own, support intervention prioritisation. The recommendation engine (§3.9 and Algorithm 1) is therefore reframed throughout this revision as **hypothesis-generating decision support for academic advisors**, not as validated causal interventions. To probe whether the SHAP-derived rankings are at least *consistent* with directional causal effects on `current_cgpa`, we conducted a propensity-score-matched (PSM) analysis on the three highest-ranked modifiable features identified by Algorithm 1: workshop / seminar participation (binary), high peer-group quality (Likert ≥ 4), and stress-coping (binary).

**Method.** For each treatment, propensity *P(T=1 | X)* is estimated via logistic regression on the leakage-free confounder matrix (all features except the treatment, the outcome, and the seven target-derived columns listed in §3.10). 1:1 nearest-neighbour matching is performed on the logit of the propensity using a caliper of 0.2 SD. The Average Treatment Effect on the Treated (ATT) is the mean within-pair difference in `current_cgpa`, with a 95 % bootstrap CI (1 000 paired resamples).

**Table 9. Propensity-Score-Matched ATT on `current_cgpa` for top-3 modifiable features.**

| Treatment | n treated | n matched | ATT (CGPA) | 95 % bootstrap CI | CI excludes 0 |
| --- | --- | --- | --- | --- | --- |
| __INJECT_PSM__ |

**Honest interpretation.** The PSM analysis does **not** rescue a causal interpretation of the SHAP-derived recommendations, and we report it as such:

1. **Workshop / seminar participation is the highest-ranked SHAP feature but admits no acceptable matched comparison set.** Of 1 378 students, 1 057 (≈ 77 %) participate in workshops; only 321 do not, and the propensity model finds zero matches within the caliper, indicating a *positivity violation* — the treated and untreated subpopulations differ too much on measured confounders to support a counterfactual estimate.
2. **High peer-group quality matches only 10 pairs**, yielding a wide and uninformative 95 % CI. No causal claim is supportable.
3. **Stress-coping is the only treatment with adequate overlap** (414 matched pairs), and its directional ATT is consistent with the SHAP sign, but the 95 % CI includes zero. The result is *consistent with* but not *evidence for* a positive effect.

We therefore agree with Reviewer 2's substantive concern: the recommendation engine cannot, on the basis of either SHAP or this observational PSM analysis, be claimed to identify causally effective interventions. The recommendation outputs are reframed as advisor-facing decision support, and §5.7 (Threats to Validity) and §6 (Future Work) commit to **cluster-randomised intervention trials** and **longitudinal panel data with formal causal-effect estimators** as the principal items required to upgrade these hypotheses to causal evidence. The reframing is consistent throughout the Abstract, §1.4, §3.9, the Table 6 caption, and §6.

This honest framing addresses the reviewer's concern without overstating what the data supports.
