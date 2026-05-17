# Patch 02 — Section 2 drawback statements (R1.1)

Append the following sentences (red font) to each cited approach in §§2.1–2.3.

## §2.1 (after Chen et al. [3] and Al-Din & Al Abdulqader [4])

> A common **drawback** of the gradient-boosted and ensemble pipelines surveyed here is that they treat student outcomes as a single scalar target (CGPA or pass/fail), which conflates academic performance with at-risk status and career readiness. They also rely on raw survey features and offer no automatic translation from feature attribution to advisor-facing recommendation. **Our work** removes both restrictions by adopting (i) a unified multi-objective formulation in which each task has its own loss function and feature taxonomy, and (ii) a SHAP-driven recommendation engine that maps attributions to modifiable behaviours.

## §2.2 (after Almalawi et al. [1] and Kruger et al. [2])

> The principal **drawback** of existing XAI-in-EDM systems is that they stop at *post-hoc explanation* — SHAP plots and feature importances are displayed, but no system automatically converts those attributions into actionable, personalized recommendations or audits the recommendations for fairness across demographic subgroups. **Our work** closes this loop with Algorithm 1 and adds the per-subgroup F1, FNR, and calibration audit reported in §5.6.

## §2.3 (after Lin et al. [19] and Goldblum et al. [20])

> Modern tabular-DL architectures (TabNet, FT-Transformer, SAINT) rely on large feature interactions that small educational survey datasets (typically <2 000 records) cannot reliably support. The **drawback** for the EDM community is that recent benchmarks underplay this small-N regime, leading practitioners to over-invest in DL architectures rather than in domain-informed feature engineering. **Our work** establishes — under matched 50-trial Optuna budgets and identical CV protocol — that XGBoost dominates this regime and that the locus of improvement is feature engineering, not architecture.
