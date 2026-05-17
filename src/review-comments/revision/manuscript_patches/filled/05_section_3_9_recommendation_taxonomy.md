# Patch 05 — §3.9 Recommendation taxonomy numbers (R1.7)

**Replace the sentence beginning "Features are classified into three tiers …" with the following.**

> Features are classified into three tiers (full enumeration in `src/config.py`):
> **12 modifiable** features (e.g. study hours, workshop participation, stress coping),
> **3 contextual** features (e.g. health status, internet access, stress frequency),
> **22 immutable** features (e.g. gender, parental education, family income),
> plus **5 engineered** composite indicators.
> Together with the **7 target-derived columns** that are excluded from the predictor set to prevent leakage, every column of the original 52-column survey is accounted for. The exact tier-by-tier reconciliation against the live feature matrix appears in **Appendix A, Table A5**.
