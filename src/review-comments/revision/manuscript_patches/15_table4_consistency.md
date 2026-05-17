# Patch 15 — Table 4 / §5.2 consistency (R1.10)

The reviewer flagged that the prose claims "F1 = 0.41–0.58" while Table 4 contains FT-Transformer F1 = 0.423 (CGPA), 0.584 (Risk), 0.423 (Career) and SAINT F1 = 0.460 (CGPA), 0.565 (Risk), 0.472 (Career).

**Action.** Replace every instance of "0.41–0.58" in §5.2 with **"0.42–0.58"** (rounded to two decimals, consistent with Table 4 row precision).

**Verification snippet** for the author to grep through the manuscript before submission:

```text
0.41-0.58       → must not appear
0.41–0.58       → must not appear (en-dash variant)
0.42–0.58       → expected
```

If a future Optuna re-run shifts these values, regenerate Table 4 from `results/tables/baselines.csv` + `results/tables/dl_models.csv` and update the prose range from the actual minimum / maximum across the three transformer rows.
