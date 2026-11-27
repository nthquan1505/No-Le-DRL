# Classification Run Report — HCMUS DRL Event Classification

- **Run Timestamp (UTC)**: `2026-11-25T16:12:52+00:00`
- **Selected Algorithm**: `hierarchical_two_stage_knn` (via `AlgorithmRegistry`; registered: `['hierarchical_two_stage_knn']`)
- **Parameter & Threshold Status**: `provisional_not_evaluated` (`calibrated: false`)
- **Technical Validation Status**: **ALL CHECKS PASSED**

> **Scope Notice**: This report records technical execution, artifact alignment, and provisional status counts only. Model evaluation (accuracy, precision, recall, F1) and parameter tuning are explicitly out of scope for this phase (see `FUTURE_WORK.md`).

---

## 1. Run Counts & Deliverables

| Metric | Value |
|---|---|
| Input events (`C:/Users/Dell/Desktop/HOCTAP/HCMUS/CT/VA DRL/No-Le-DRL/data/events.json`) | **108** |
| Embedded classifiable events (`events_query_embeddings.npy`) | **108** |
| Skipped non-event / empty-text records | **0** |
| Shared query embeddings reused from cache | `False` |
| Output records (`C:/Users/Dell/Desktop/HOCTAP/HCMUS/CT/VA DRL/No-Le-DRL/outputs/classification_events.json`) | **108** |
| Sample reference vectors (`sample_embeddings.npy`) | `(908, 384)` |
| Rubric reference vectors (`rubric_embeddings.npy`) | `(176, 384)` |

### Classification Status Breakdown (Provisional Settings)

| Status | Count | Share |
|---|---|---|
| `PREDICTED` | 39 | 36.1% |
| `NEEDS_REVIEW` | 69 | 63.9% |
| `NO_MATCH` | 0 | 0.0% |
| `SKIPPED_NOT_EVENT` | 0 | 0.0% |

### Predicted Criterion Distribution (`status = PREDICTED` only)

| `criterion_ref` | Count |
|---|---|
| `1.9` | 23 |
| `3.5` | 8 |
| `1.6` | 7 |
| `4.3` | 1 |

### Review / Abstention Trigger Frequency

| Trigger | Occurrences |
|---|---|
| `branch_disagreement` | 44 |
| `narrow_stage2_criterion_margin` | 19 |
| `draft_only_rubric_evidence_unverified` | 13 |
| `narrow_stage1_group_margin` | 13 |
| `unresolved_rubric_boundary_3_1_vs_3_5` | 6 |
| `criterion_3_3_missing_explicit_mobilization_cue` | 5 |
| `criterion_1_9_catch_all_guard` | 1 |

---

## 2. Shared Query Embedding Configuration

- **Model**: `intfloat/multilingual-e5-small` (revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`)
- **Dimension / Dtype / Normalization**: `384d`, `float32`, `L2` (`pooling=mean`)
- **Query Prefix**: `"query: "`
- **Query Format**: `query: [event title] [event description]` (title + description only; zero label/category/reason leakage)
- **Token Stats**: `{"count": 108, "min_tokens": 46, "median_tokens": 80.5, "max_tokens": 172, "max_seq_length": 512, "truncated_count": 0}`
- **Shared Across Branches & Stages**: `True` (single vector per eligible event passed identically to Sample Stage 1, Rubric Stage 1, Sample Stage 2, and Rubric Stage 2)

---

## 3. Selected Algorithm & Provisional Parameters

```json
{
  "algorithm": {
    "name": "hierarchical_two_stage_knn",
    "parameter_status": "provisional_not_evaluated",
    "params": {
      "k1": 10,
      "candidate_group_count": 2,
      "stage1_fusion_beta": 0.5,
      "stage1_similarity_weighting": "positive_clip",
      "stage1_weight_temperature": 0.1,
      "stage1_weight_power": 2.0,
      "stage1_score_normalization": "min_max",
      "k2": 10,
      "stage2_fusion_alpha": 0.5,
      "stage2_similarity_weighting": "positive_clip",
      "stage2_weight_temperature": 0.1,
      "stage2_weight_power": 2.0,
      "stage2_score_normalization": "min_max",
      "rubric_prototype_type_weights": {
        "criterion_definition": 1.0,
        "verified_anchor": 1.0,
        "partial_anchor": 0.85,
        "keyword_context": 0.75,
        "draft_anchor": 0.7
      },
      "missing_branch_reference_policy": "reweight_available_branch",
      "top_sample_neighbors_in_output": 5,
      "top_rubric_prototypes_in_output": 5,
      "max_suggested_candidates": 5
    }
  },
  "decision": {
    "threshold_status": "provisional_not_evaluated",
    "calibrated": false,
    "no_match_max_raw_similarity": 0.7,
    "min_top_raw_similarity_for_predict": 0.78,
    "min_group_margin": 0.05,
    "min_criterion_margin": 0.08,
    "require_branch_agreement_when_both_available": true,
    "route_draft_only_to_review": true,
    "boundary_3_1_vs_3_5_margin": 0.18,
    "guard_3_3_requires_mobilization_cue": true,
    "guard_1_9_catch_all_margin": 0.12,
    "enable_none_negative_classifier": false
  }
}
```

---

## 4. Technical Validation Results

### 4.1. Pre-flight Input & Reference Asset Checks

| Check | Status | Detail |
|---|---|---|
| `embedding_config_valid` | **PASS** | model=intfloat/multilingual-e5-small, rev=614241f622f53c4eeff9890bdc4f31cfecc418b3, dim=384, prefixes={'passage': 'passage: ', 'query': 'query: '} |
| `sample_branch_embedding_shape` | **PASS** | shape=(908, 384), expected=(N>0, 384) |
| `sample_branch_embedding_dtype` | **PASS** | dtype=float32, expected=float32 |
| `sample_branch_embedding_finite` | **PASS** | All reference vector entries finite |
| `sample_branch_l2_normalized` | **PASS** | max |norm - 1| = 1.02e-07 (tol=1.0e-04) |
| `sample_branch_metadata_schema` | **PASS** | All required columns present |
| `sample_branch_row_count_alignment` | **PASS** | metadata rows=908 vs embedding rows=908 |
| `sample_branch_embedding_row_sequential` | **PASS** | embedding_row matches exact 0..907: True |
| `rubric_branch_embedding_shape` | **PASS** | shape=(176, 384), expected=(N>0, 384) |
| `rubric_branch_embedding_dtype` | **PASS** | dtype=float32, expected=float32 |
| `rubric_branch_embedding_finite` | **PASS** | All reference vector entries finite |
| `rubric_branch_l2_normalized` | **PASS** | max |norm - 1| = 8.80e-08 (tol=1.0e-04) |
| `rubric_branch_metadata_schema` | **PASS** | All required columns present |
| `rubric_branch_row_count_alignment` | **PASS** | metadata rows=176 vs embedding rows=176 |
| `rubric_branch_embedding_row_sequential` | **PASS** | embedding_row matches exact 0..175: True |
| `sample_rubric_referential_integrity` | **PASS** | All 908 samples valid against rubric.yaml |
| `rubric_prototypes_referential_integrity` | **PASS** | Stage 1=4, Stage 2=161, reminder=5, special=6; errors=0 |
| `input_events_integrity` | **PASS** | total_events=108, non_null_post_ids=108, unique_post_ids=True |

### 4.2. Shared Query Embedding Checks

| Check | Status | Detail |
|---|---|---|
| `query_embeddings_shape` | **PASS** | shape=(108, 384), expected=(108, 384) |
| `query_embeddings_dtype` | **PASS** | dtype=float32 |
| `query_embeddings_finite` | **PASS** | All query vector values finite |
| `query_embeddings_l2_normalized` | **PASS** | max |norm - 1| = 8.91e-08 |
| `query_metadata_sequential_rows` | **PASS** | 0..107: True |
| `query_event_partition_exact` | **PASS** | embedded=108 + skipped=0 == input=108 |
| `query_post_id_and_prefix_alignment` | **PASS** | All 108 query rows match source post_id and 'query: ' prefix |

### 4.3. Output & Workflow Integrity Checks

| Check | Status | Detail |
|---|---|---|
| `output_event_count_exact` | **PASS** | input=108, output=108 |
| `output_preserves_order_and_original_fields` | **PASS** | Original order, post_ids, and all event fields preserved |
| `output_classification_schema_valid` | **PASS** | All classification objects conform to schema and status contracts |
| `output_predicted_criteria_event_matchable` | **PASS** | Every PREDICTED criterion exists in rubric.yaml with event_matchable: true |
| `output_stage2_uses_shared_stage1_groups` | **PASS** | Both branches strictly restricted Stage 2 to shared Stage 1 candidate groups |
| `output_valid_utf8_json_no_nan` | **PASS** | Valid UTF-8 JSON (3,525,487 chars), zero NaN/Infinity |
| `shared_query_reused_across_branches_and_stages` | **PASS** | embedded=108; calls: sample_s1=108, rubric_s1=108, sample_s2=108, rubric_s2=108 |

---

## 5. Processing Notes & Unresolved Issues

1. **Provisional Parameters**: `beta = 0.5`, `alpha = 0.5`, `K1 = 10`, `K2 = 10`, prototype type weights, and abstention/review thresholds are provisional (`provisional_not_evaluated`). No evaluation or hyperparameter tuning has been performed.
2. **Uncalibrated Scores**: Cosine similarities, weighted kNN votes, and Stage-wise Fusion scores are ranking scores only (`calibrated: false`), not calibrated probabilities.
3. **Sample Coverage Gap**: 10 of 21 `event_matchable: true` criteria have zero samples in `sample_embeddings.npy` (`1.5`, `3.6`, `3.9`–`3.13`, `4.2`, `4.6`, `6.4`). Stage 2 Fusion uses `reweight_available_branch` so missing Sample coverage is represented explicitly rather than treated as proof of irrelevance, while draft-only rubric matches are routed to `NEEDS_REVIEW`.
4. **Unresolved Rubric Boundaries**: Open questions from `rubric.yaml` (`Q10` for `3.1` vs `3.5`, `3.3` explicit mobilization requirement, `Q23` for `1.9` vs `3.5`, `Q17` lack of verified `NONE` samples) remain preserved as metadata and review triggers rather than hardcoded definitive rules.
5. **Classification Boundary**: Event classification classifies announcement text only; it does not verify student attendance, organizer/collaborator role eligibility, or DRL point calculation (handled downstream by Module 6 / Module 7).
