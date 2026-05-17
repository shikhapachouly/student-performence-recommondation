# Patch 03 — Notation Table (R1.2)

Insert this table at the end of §3, before §4. Caption: **Table 3a. Notation used throughout this paper.**

| Symbol | Meaning |
| --- | --- |
| *N* | Number of students in the dataset (= 1 378) |
| *d* | Dimensionality of the feature vector after preprocessing |
| *x_i* ∈ ℝ^d | Feature vector for student *i* |
| *g_i* | Cumulative grade-point average (CGPA) for student *i* |
| *b_i* ∈ {0, 1} | Backlog status indicator for student *i* (1 = has backlog) |
| *t_i* ∈ {0, 1} | Internship-completion indicator |
| *p_i* ∈ {0, 1} | Academic-project-completion indicator |
| *y_i^{(k)}* | Label for objective *k* ∈ {CGPA, At-Risk, Career} |
| *C_k* | Number of classes for objective *k* |
| *w_c* | Inverse-frequency class weight, used in Eq. (4) |
| φ_j | SHAP value for feature *j* (Eq. 17) |
| *F*_mod / *F*_ctx / *F*_imm | Modifiable / contextual / immutable feature tiers (§3.9) |
| λ, γ | XGBoost regularisation strengths (Eq. 11) |
| σ_i | Stress-frequency Likert score (1–5) |
| *h_i* | Daily-study-hours ordinal level (1–4) |
| *r_j* | Skill self-rating on 1–5 Likert scale |
| *T* | Number of boosting iterations |
| *K* | Top-*K* recommendations returned by Algorithm 1 |
| TPE | Tree-structured Parzen Estimator (Optuna sampler) |
| ATT | Average Treatment Effect on the Treated (PSM, §5.5) |
| FNR | False-Negative Rate, used in §5.6 fairness audit |
