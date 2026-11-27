# Future Work — Evaluation, Tuning & Algorithm Comparison Roadmap

> **Current Status**: The modular dual-branch hierarchical 2-stage kNN classification system (`hierarchical_two_stage_knn`) has been implemented and technically validated using provisional, uncalibrated configuration parameters (`beta = 0.5`, `alpha = 0.5`, `K1 = 10`, `K2 = 10`, `provisional_not_evaluated`).
>
> **All items listed below are strictly PENDING** and were intentionally NOT executed in the current engineering phase.

---

## Pending Evaluation & Optimization Tasks

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | **Build a manually reviewed validation dataset** | `PENDING` | Curate human-verified ground-truth labels on held-out HCMUS event posts (including full title + description posts and confirmed out-of-scope announcements), ensuring zero overlap/leakage with `res/events_full.csv`. |
| 2 | **Evaluate Stage 1 Top-2 Group Recall** | `PENDING` | Measure Stage 1 Recall@1, Recall@2, and inter-stage error propagation (Stage 1 miss vs. Stage 2 ranking error) across `C1`–`C4`. |
| 3 | **Evaluate criterion-level accuracy, Macro-F1, and confusion matrix** | `PENDING` | Assess per-criterion precision, recall, Macro-F1, Micro-F1, and confusion patterns across both frequent (`1.9`, `3.5`, `1.6`) and rare/zero-shot criteria. |
| 4 | **Compare Sample and Rubric branch performance independently** | `PENDING` | Benchmark Sample-only vs. Rubric-only vs. Stage-wise Fusion across Stage 1 and Stage 2, especially on criteria with 0–3 samples in `events_full.csv`. |
| 5 | **Fine-tune Stage 1 fusion weight `beta`** | `PENDING` | Sweep `beta ∈ [0.0, 1.0]` on the validation dataset (currently provisional `beta = 0.5`). |
| 6 | **Fine-tune Stage 2 fusion weight `alpha`** | `PENDING` | Sweep `alpha ∈ [0.0, 1.0]` on the validation dataset (currently provisional `alpha = 0.5`), including criteria-coverage-aware weighting policies. |
| 7 | **Tune `K1` and `K2`** | `PENDING` | Evaluate sensitivity to Stage 1 neighbor count `K1` and Stage 2 neighbor count `K2` (currently provisional `K1 = 10`, `K2 = 10`) under class imbalance. |
| 8 | **Compare score normalization and neighbor weighting methods** | `PENDING` | Benchmark `min_max`, `sum_normalize`, `identity` normalization and `positive_clip`, `shifted`, `exp_scaled`, `power` similarity weighting plus Rubric prototype-type weights. |
| 9 | **Calibrate confidence and abstention thresholds** | `PENDING` | Fit and validate probability calibration (e.g., temperature scaling / isotonic regression), measure ECE and risk-coverage curves, and empirically lock `PREDICTED` / `NEEDS_REVIEW` / `NO_MATCH` thresholds. |
| 10 | **Compare `hierarchical_two_stage_knn` against alternative algorithms through `AlgorithmRegistry`** | `PENDING` | Implement and benchmark pluggable alternatives (flat kNN, single-stage kNN, hierarchical kNN without intermediate fusion, BM25 + semantic hybrid retrieval, alternative fusion strategies, and cross-encoder reranking) via `BaseClassificationAlgorithm` and `AlgorithmRegistry`. |
| 11 | **Review errors involving `3.1`/`3.5`, `3.3`, and `1.9`** | `PENDING` | Conduct targeted qualitative error analysis on unresolved rubric boundaries (`Q10`: `3.1` vs `3.5`, `3.3` explicit mobilization detection, `Q23`: `1.9` academic/career vs `3.5` movement/soft-skill workshops) and update rubric specifications once CTSV/human review resolves open questions. |
| 12 | **Reassess `NONE` using validated negative examples** | `PENDING` | Collect human-verified non-DRL announcements (`Q17`: commercial ads, external part-time job postings, personal parties) and design/validate an explicit negative classification mechanism before enabling automatic `NONE` predictions. |
