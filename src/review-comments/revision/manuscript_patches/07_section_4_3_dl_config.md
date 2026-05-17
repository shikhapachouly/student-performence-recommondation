# Patch 07 — New §4.3 Deep-Learning Training Configuration (R2.6)

Insert as §4.3 (renumber subsequent subsections accordingly).

## 4.3 Deep-Learning Training Configuration

To allow direct comparison with the ML baselines, all three deep-learning architectures share the same input matrix, the same Optuna search budget (50 trials, inner 3-fold CV via the TPE sampler), and the same outer 5-fold stratified CV with SMOTE applied only on training folds. Architecture-specific settings are listed in Table 4a.

**Table 4a. Deep-learning training configuration.**

| Setting | TabNet | FT-Transformer | SAINT |
| --- | --- | --- | --- |
| Optimiser | Adam | AdamW | AdamW |
| Initial learning rate | 2e-2 | 1e-3 | 1e-3 |
| LR scheduler | StepLR (γ=0.9, step=50) | Cosine | Cosine |
| Batch size | 128 | 128 | 128 |
| Maximum epochs | 300 | 200 | 200 |
| Early-stopping patience | 20 | 20 | 20 |
| In-fold validation split | 15 % | 15 % | 15 % |
| Categorical encoding | sparsemax mask | learned token | learned token |
| Architectural hyperparameters | n_d = n_a = 8, n_steps = 3, λ_sparse = 1e-3, γ = 1.3, momentum 0.02, gradient clip 1.0 | d_token = 64, n_blocks = 2, attention dropout 0.2, FFN dropout 0.1 | dim = 32, depth = 3, heads = 4, attention dropout 0.1, FFN dropout 0.1 |
| Optuna trials (inner 3-fold) | 50 | 50 | 50 |
| Tuned ranges | per `OPTUNA_SEARCH_SPACES["TabNet"]` | per `OPTUNA_SEARCH_SPACES["FTTransformer"]` | per `OPTUNA_SEARCH_SPACES["SAINT"]` |
| Random seeds | NumPy 42, PyTorch 42 | 42, 42 | 42, 42 |

The Optuna budget for each DL architecture is identical to that for XGBoost (50 trials × 3 inner folds = 150 inner-CV evaluations), so the gap between XGBoost and the best DL model on this dataset cannot be attributed to under-tuning of the DL architectures. The persistence of that gap (Table 4) is consistent with recent results showing that gradient-boosted trees remain dominant on small tabular data when the ratio of unique feature interactions to sample count is low [20], [21]. We therefore frame the DL underperformance as **regime-specific** to small-N educational survey data, not as a categorical claim about tabular DL.
