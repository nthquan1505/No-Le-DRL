# Rubric Prototype Audit Report — HCMUS DRL Classification System (Schema v2.0)

> **Scope notice:** This report audits prototype data quality, semantic structure, metadata traceability, and cross-artifact technical consistency for the Rubric Branch (`rubric_prototypes.jsonl`, `rubric_metadata.parquet`, `rubric_embeddings.npy`) derived strictly from `data/raw/rubric.yaml`. It does **not** evaluate classification accuracy, precision, recall, or F1-score, and no classification algorithm weights, kNN parameters, or fusion rules were tuned or altered.

---

## 1. Before-and-After Overview

### 1.1 High-Level Artifact Comparison

| Metric / Property | Before (Legacy JSONL) | Before (Legacy Parquet / NPY) | After (Schema v2.0 — Synced) |
|---|---:|---:|---:|
| **Total prototypes (`jsonl` / `parquet` / `npy`)** | `176` | `170` *(out of sync by 6 rows)* | **`176` / `176` / `(176, 384)`** *(1:1 synchronized)* |
| **Embedding dimension / dtype** | — | `384` / `float32` | **`384` / `float32`** (L2-normalized, max `|norm-1| = 8.80e-08`) |
| **Embedding model** | — | `intfloat/multilingual-e5-small` | **`intfloat/multilingual-e5-small`** (`614241f622f53c4eeff9890bdc4f31cfecc418b3`, unchanged) |
| **Max token length / truncated prototypes (> 512)** | Untracked (manual `[:60]` char slicing) | Untracked | **Max `484` tokens / `0` truncated** (`max_seq_length = 512`) |
| **Stage 1 (`group`) prototypes** | `4` (`C1`–`C4`, truncated `[:60]` clauses) | `4` (`C1`–`C4`) | **`4`** (`C1`–`C4`, complete untruncated structured Vietnamese) |
| **Stage 2 (`criterion`) prototypes** | `166` *(mixed `6.4` inside Stage 2)* | `166` *(mixed `6.4` inside Stage 2)* | **`161`** *(20 eligible C1–C4 criteria only)* |
| **Reminder (`reminder`) prototypes (`6.4`)** | `0` *(mixed into `stage="criterion"`)* | `0` *(mixed into `stage="criterion"`)* | **`5`** *(isolated: `stage="reminder"`, `eligibility_class="reminder_only"`)* |
| **Special (`special`) prototypes (`NONE`)** | `6` *(labeled as `criterion_definition`/`draft_anchor`/`keyword_context`)* | `0` *(dropped during embedding generation)* | **`6`** *(isolated: `stage="special"`, `prototype_type="draft_negative_reference"`, `can_decide_none=false`)* |
| **`NEEDS_REVIEW` prototypes** | `0` | `0` | **`0`** *(routing/gating decision only)* |
| **Covered positive/conditional C1–C4 criteria** | `20` | `20` | **`20`** (`15` `positive_candidate` + `5` `conditional_candidate`) |
| **Covered reminder criteria (`P`)** | `1` (`6.4`, unisolated) | `1` (`6.4`, unisolated) | **`1`** (`6.4`, isolated in `stage="reminder"`) |
| **Excluded non-event-matchable items** | `30` (`event_matchable: false`) | `30` | **`30`** (`event_matchable: false`, including all 11 `C5` items and role variants `1.7`, `3.14`, `4.4`) |

### 1.2 Counts by `stage`, `prototype_type`, `eligibility_class`, and `anchor_status`

| Dimension | Category | Before (JSONL) | Before (Parquet) | After (Schema v2.0 — JSONL & Parquet) |
|---|---|---:|---:|---:|
| **`stage`** | `group` | 4 | 4 | **4** |
| | `criterion` | 166 | 166 | **161** |
| | `reminder` | 0 | 0 | **5** |
| | `special` | 6 | 0 | **6** |
| **`prototype_type`** | `group_definition` | 4 | 4 | **4** |
| | `criterion_definition` | 22 | 21 | **20** |
| | `reminder_definition` | 0 | 0 | **1** |
| | `verified_anchor` | 91 | 89 | **91** |
| | `partial_anchor` *(legacy)* | 0 | 2 | **0** *(normalized to `verified_anchor`)* |
| | `draft_anchor` | 37 | 33 | **33** *(30 in `criterion` + 3 in `reminder`)* |
| | `keyword_context` | 22 | 21 | **21** *(20 in `criterion` + 1 in `reminder`)* |
| | `draft_negative_reference` | 0 | 0 | **6** *(all in `special` for `NONE`)* |
| **`eligibility_class`** | `positive_candidate` | *(not present)* | *(not present)* | **140** *(4 Stage 1 + 136 Stage 2)* |
| | `conditional_candidate` | *(not present)* | *(not present)* | **25** *(`1.5`, `3.6`, `3.11`, `3.12`, `3.13`)* |
| | `reminder_only` | *(not present)* | *(not present)* | **5** *(`6.4`)* |
| | `negative_reference_only` | *(not present)* | *(not present)* | **6** *(`NONE`)* |
| **`anchor_status`** | `verified_from_school_list` | 89 | 89 | **91** *(includes `3.8` verified anchors)* |
| | `verified_partial` *(prototype-level)* | 2 | 2 | **0** *(moved to `criterion_anchor_status`)* |
| | `draft` | 37 | 33 | **37** *(33 `draft_anchor` + 4 `NONE` draft examples)* |
| | `derived` | 48 | 46 | **48** *(4 `group_definition` + 20 `criterion_definition` + 1 `reminder_definition` + 21 `keyword_context` + 2 `NONE` def/kw)* |

### 1.3 Per-Criterion Prototype Breakdown (After — Schema v2.0)

| Ref | Group | Stage | `eligibility_class` | `criterion_anchor_status` | Total | Def | Verified | Draft | Kw | NegRef | `required_signals` |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `1.5` | `C1` | `criterion` | `conditional_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | `min_level:khoa`, `selection_or_completion_required` |
| `1.6` | `C1` | `criterion` | `positive_candidate` | `verified_from_school_list` | 17 | 1 | 15 | 0 | 1 | 0 | `min_level:khoa` |
| `1.9` | `C1` | `criterion` | `positive_candidate` | `verified_from_school_list` | 20 | 1 | 18 | 0 | 1 | 0 | — |
| `2.2` | `C2` | `criterion` | `positive_candidate` | `verified_from_school_list` | 5 | 1 | 3 | 0 | 1 | 0 | — |
| `2.3` | `C2` | `criterion` | `positive_candidate` | `verified_from_school_list` | 12 | 1 | 10 | 0 | 1 | 0 | — |
| `3.1` | `C3` | `criterion` | `positive_candidate` | `verified_from_school_list` | 9 | 1 | 7 | 0 | 1 | 0 | `min_level:khoa` |
| `3.2` | `C3` | `criterion` | `positive_candidate` | `verified_from_school_list` | 8 | 1 | 6 | 0 | 1 | 0 | — |
| `3.3` | `C3` | `criterion` | `positive_candidate` | `verified_from_school_list` | 5 | 1 | 3 | 0 | 1 | 0 | `explicit_student_mobilization_signal`, `min_level:khoa` |
| `3.5` | `C3` | `criterion` | `positive_candidate` | `verified_from_school_list` | 20 | 1 | 18 | 0 | 1 | 0 | — |
| `3.6` | `C3` | `criterion` | `conditional_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | `min_level:truong`, `selection_or_completion_required` |
| `3.8` | `C3` | `criterion` | `positive_candidate` | `verified_partial` | 6 | 1 | 2 | 2 | 1 | 0 | `min_level:khoa` |
| `3.9` | `C3` | `criterion` | `positive_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | — |
| `3.10` | `C3` | `criterion` | `positive_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | — |
| `3.11` | `C3` | `criterion` | `conditional_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | `selection_or_completion_required` |
| `3.12` | `C3` | `criterion` | `conditional_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | `selection_or_completion_required` |
| `3.13` | `C3` | `criterion` | `conditional_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | `selection_or_completion_required` |
| `4.2` | `C4` | `criterion` | `positive_candidate` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | — |
| `4.3` | `C4` | `criterion` | `positive_candidate` | `verified_from_school_list` | 8 | 1 | 6 | 0 | 1 | 0 | — |
| `4.5` | `C4` | `criterion` | `positive_candidate` | `verified_from_school_list` | 5 | 1 | 3 | 0 | 1 | 0 | — |
| `4.6` | `C4` | `criterion` | `positive_candidate` | `draft` | 6 | 1 | 0 | 4 | 1 | 0 | — |
| `6.4` | `P` | `reminder` | `reminder_only` | `draft` | 5 | 1 | 0 | 3 | 1 | 0 | — |
| `NONE` | `None` | `special` | `negative_reference_only` | `draft` | 6 | 0 | 0 | 0 | 0 | 6 | — |

---

## 2. Issues Identified in the Existing Prototype Dataset & Pipeline

1. **Truncated Stage 1 group prototypes (`[:60]` character slicing):**
   `scripts/generate_rubric_prototypes.py` previously sliced each criterion name to 60 characters (`[:60]`), cutting Vietnamese words and sentences mid-token (e.g., `"Là tác giả của đề tài nghiên cứu khoa học (khóa luận, tiểu l"`, `"Tham gia các hoạt động quyên góp do các tổ chức, đơn vị tron"`).
2. **Desynchronization between `rubric_prototypes.jsonl` (176 rows) and `rubric_metadata.parquet` / `rubric_embeddings.npy` (170 rows):**
   `scripts/generate_embeddings.py` previously rebuilt its own internal prototype list directly from `rubric.yaml` instead of embedding `rubric_prototypes.jsonl`, dropping all 6 `special` (`NONE`) records and renaming `3.8` verified anchors to `partial_anchor` — breaking 1:1 traceability across artifacts.
3. **Under-represented and noisy `criterion_definition` embedding texts:**
   Legacy `criterion_definition` prototypes concatenated raw `name` and raw `scope_note` verbatim, which simultaneously omitted structured `inclusion_rules`, positive `conditions`, and `min_level` cues while leaking non-semantic administrative noise (scoring points like `5đ`, CSV row indices like `dòng 145`, `QĐ 2020` notes, and cross-criterion references).
4. **Criterion `3.3` bare-title verified anchors lacking mobilization context:**
   The 3 verified anchors of `3.3` (`Let IT Shine 2025`, `Giải Bóng đá Futsal sinh viên ĐHQG - HCM năm 2025 - Trận nam ngày 08 - 11`, `Lễ Tổng kết Khóa học thử nghiệm Đợt 1 – Dự án "HCMUS Future Leaders"`) were embedded as bare event titles even though `rubric.yaml` explicitly documents that `3.3` is determined by the `"Huy động lực lượng"` mobilization signal (`roles_seen: {"Huy động lực lượng": 3}` across all 3 recorded rows), not by the event genre (arts, sports, closing ceremony).
5. **Conflation of prototype-level vs. criterion-level verification status on `3.8`:**
   Criterion `3.8` has `anchors_status: verified_partial` in `rubric.yaml` (2 verified anchors from the school list + 2 draft anchors). The legacy JSONL labeled the 2 verified anchors as `anchor_status: "verified_partial"`, while legacy Parquet relabeled their `prototype_type` as `"partial_anchor"` — conflating criterion-level partial coverage with individual prototype provenance.
6. **Mandatory reminder `6.4` (`P.sinh_hoat_cong_dan`) mixed into Stage 2 positive candidates:**
   `6.4` (`group: P`, `kind: mandatory_event`, `mvp_scope: remind`) was placed in `stage: "criterion"` alongside C1–C4 positive scoring criteria, without a dedicated `stage: "reminder"` or `eligibility_class: "reminder_only"` boundary.
7. **`NONE` prototypes using positive prototype types and missing `can_decide_none` guard:**
   In legacy JSONL, the 6 `NONE` records used `criterion_definition`, `draft_anchor`, and `keyword_context` types with no `eligibility_class: "negative_reference_only"` or `can_decide_none: false` metadata, and were completely absent from `rubric_metadata.parquet` and `rubric_embeddings.npy`.
8. **Missing structured classification boundary metadata (`boundary_metadata`) and prototype-count imbalance visibility:**
   `rubric.yaml` contains rich boundary fields (`inclusion_rules`, `exclusion_rules`, `confusable_with`, `negative_keywords`, `conditions`, `min_level`, `role_context`, `requires_role_context`, `uncertain`) that were not carried into prototype metadata. In addition, prototype counts per criterion vary heavily across criteria (from `5` on draft-only criteria like `1.5`, `3.6`, `4.2` up to `20` on `1.9` and `3.5`), making explicit `prototype_family` and `eligibility_class` tagging essential for downstream balanced/family-aware retrieval without fabricating or duplicating anchors.

---

## 3. Changes Implemented

### 3.1 Improved Prototype Generation Pipeline (`scripts/generate_rubric_prototypes.py`)

Upgraded the existing generator in place to **Schema v2.0** (also mirroring `rubric_prototypes.yaml`), with zero modification to `data/raw/rubric.yaml`:

- **Stage 1 (`group`, 4 prototypes: `C1`–`C4`):**
  - Replaced `[:60]` character slicing with structured, complete, untruncated Vietnamese prototypes using the template:
    `Nhóm <group_id> - <name>. / Phạm vi: <tieu_chi[eligible].name> / Các loại hoạt động tiêu biểu: <complete bullets per eligible criterion> / Đặc điểm nhận diện: Từ khóa tiêu biểu: <deduplicated positive keywords>.`
  - For large groups (`C3`, 11 eligible criteria), prefers concise cleaned `scope_note` phrases (omitting redundant parenthetical example lists) and 1 representative keyword per criterion so all 4 Stage 1 prototypes remain complete and strictly under the E5 `512`-token sequence limit (`stage1_C3_group_def` = `484` tokens; `C4` = `287`; `C1` = `199`; `C2` = `142`).
  - Excluded `C5` (0 `event_matchable: true` items) and `P` (penalty/reminder group) from Stage 1.
- **Stage 2 (`criterion`, 161 prototypes across 20 C1–C4 criteria):**
  - **`criterion_definition` (20 prototypes):** Built selectively from cleaned `name`, cleaned `scope_note`, cleaned `inclusion_rules`, positive `conditions`, and `min_level` (`Phạm vi / Các hoạt động thuộc tiêu chí / Điều kiện nhận diện`). Stripped scoring points (`5đ`, `10đ`), CSV row numbers (`dòng 145`), `QĐ 2020` commentary, and cross-criterion exclusion prose from positive embedding text while preserving exclusion/confusability metadata in `boundary_metadata`.
  - **`verified_anchor` (91 prototypes):** Preserved every verified school-list anchor verbatim in `anchor_text` with `source_anchor_index`, `anchor_status: "verified_from_school_list"`, and zero paraphrasing or duplication.
  - **`draft_anchor` (30 in `criterion` + 3 in `reminder`):** Strictly separated with `anchor_status: "draft"`; zero draft anchors promoted to verified.
  - **`keyword_context` (20 in `criterion` + 1 in `reminder`):** Exactly 1 per criterion using positive `keywords` only (`negative_keywords` strictly excluded from `embedding_text` and stored in `boundary_metadata` / `negative_keywords`).
- **Special Criterion Handling:**
  - **Criterion `3.3` (`C3.huy_dong_luc_luong`):** Confirmed directly from `rubric.yaml` (`school_data.rows == 3`, `school_data.roles_seen == {"Huy động lực lượng": 3}`) that all 3 verified anchors carry the mobilization role. Contextualized all 3 `verified_anchor` `embedding_text` entries with explicit Vietnamese mobilization context (`Sự kiện / Ngữ cảnh phân loại / Tín hiệu nhận diện`), retained exact original titles in `anchor_text`, and attached `required_signals: ["explicit_student_mobilization_signal", "min_level:khoa"]`.
  - **Criterion `3.8` (`C3.tu_van_tuyen_sinh_viec_lam`):** Normalized the 2 verified anchors to `prototype_type: "verified_anchor"` and `anchor_status: "verified_from_school_list"`, kept the 2 draft anchors as `prototype_type: "draft_anchor"` and `anchor_status: "draft"`, and preserved criterion-level `criterion_anchor_status: "verified_partial"` across all 6 prototypes of `3.8`.
  - **Criterion `6.4` (`P.sinh_hoat_cong_dan`):** Separated out of Stage 2 into `stage: "reminder"`, `group_id: "P"`, `eligibility_class: "reminder_only"`, with `prototype_type: "reminder_definition"` (1) + `draft_anchor` (3) + `keyword_context` (1).
  - **Meta-label `NONE`:** Isolated in `stage: "special"`, `group_id: null`, `eligibility_class: "negative_reference_only"`, `prototype_type: "draft_negative_reference"` (6 prototypes), `prototype_family: "negative_reference"`, `can_decide_none: false`, and `event_matchable: false`.
  - **Meta-label `NEEDS_REVIEW`:** Generates `0` prototypes (routing/gating only).
- **Structured Boundary & Traceability Metadata (Schema v2.0):**
  - Added `schema_version`, `prototype_family`, `criterion_anchor_status`, `anchor_text`, `source_anchor_index`, `eligibility_class`, `boundary_metadata` (`inclusion_rules`, `exclusion_rules`, `confusable_with`, `negative_keywords`, `conditions`, `min_level`, `role_context`, `requires_role_context`, `uncertain`), `required_signals`, `negative_keywords`, `source_fields`, and `can_decide_none`.
  - Across the 176 prototypes, `boundary_metadata` preserves verbatim rubric boundary fields: `inclusion_rules` (126 prototypes), `exclusion_rules` (167), `confusable_with` (141), `negative_keywords` (43), `conditions` (64), `min_level` (47), and `uncertain` (166).

### 3.2 Synchronized Rubric Embedding Generation (`scripts/generate_embeddings.py`)

- Added `load_rubric_prototypes_jsonl()`, upgraded `write_rubric_parquet()` to the full Schema v2.0 Arrow schema (retaining backward-compatible `criterion_anchors_status` alongside `criterion_anchor_status`), and added `--rubric-only` mode + `validate_rubric_cross_artifact_consistency()`.
- Regenerated **only** `data/processed/cls_index/rubric_embeddings.npy` `(176, 384)` and `data/processed/cls_index/rubric_metadata.parquet` (`176` rows) directly from `data/processed/cls_index/rubric_prototypes.jsonl` using the exact model, revision, prefix, pooling, dtype, and L2-normalization specified in `data/processed/cls_index/embedding_config.json`.
- Left `sample_embeddings.npy`, `sample_metadata.parquet`, `events_query_embeddings.npy`, `events_query_metadata.json`, `embedding_config.json`, `data/raw/rubric.yaml`, and `res/events_full.csv` completely untouched.

### 3.3 Minimal Classification Compatibility Updates

Preserved the entire Shared Query Embedding + Dual Independent Retrieval Branches + Hierarchical Two-Stage kNN algorithm (`K1`, `K2`, `alpha`, `beta`, Shared Top-2 group selection, Sample Branch, fusion rules) untouched; updated only metadata readers/validators:

- `src/classification/data/validators.py`: Accepts `stage` in `{"group", "criterion", "reminder", "special"}` and validates `eligibility_class` isolation per stage.
- `src/classification/retrieval/rubric_retriever.py`: Strictly filters Stage 1 to `stage == "group"` & `group_id in C1..C4`, Stage 2 to `stage == "criterion"` & `event_matchable == True` & `group_id in C1..C4` & `eligibility_class in {"positive_candidate", "conditional_candidate"}` (guaranteeing `6.4` reminder and `NONE` special prototypes never enter positive Stage 1 / Stage 2 scoring), and reads `criterion_anchor_status` (falling back to `criterion_anchors_status`).
- `src/classification/algorithms/hierarchical_two_stage_knn.py`: Updated `_classify_evidence_verification()` (2-line metadata read) so `criterion_anchor_status == "verified_partial"` is still reported as `"verified_partial"` now that `3.8`'s verified anchors use `prototype_type == "verified_anchor"`.

---

## 4. Unresolved Issues (Preserved from `rubric.yaml` Without Fabrication)

The following domain ambiguities and data gaps remain explicitly unresolved in `data/raw/rubric.yaml` and are preserved transparently in prototype metadata (`boundary_metadata.uncertain`, `criterion_anchor_status`, `eligibility_class`, `can_decide_none`) rather than resolved silently:

1. **Draft-only criteria with zero verified school records (10 of 21 `event_matchable: true` items):**
   `1.5`, `3.6`, `3.9`, `3.10`, `3.11`, `3.12`, `3.13`, `4.2`, `4.6`, and `6.4` (`Q14`, `Q15`). All remain strictly marked `criterion_anchor_status: "draft"`.
2. **Partially verified criterion `3.8`:**
   Only the Job Fair side (`Ngày hội việc làm khoa CNTT`, `Ngày hội việc làm My Jobs năm 2026`) is verified from school records; the admissions-advising (`tư vấn tuyển sinh`) side remains draft-only (`criterion_anchor_status: "verified_partial"`).
3. **No verified negative (`NONE` / `NO_MATCH`) examples (`Q17`):**
   `res/events_full.csv` contains only positive DRL activities. All 6 `NONE` prototypes remain `draft_negative_reference` with `can_decide_none: false`.
4. **Prototype count imbalance across Stage 2 criteria:**
   Criteria with large school activity lists (`1.9` and `3.5`: 20 prototypes each; `1.6`: 17) naturally have 3–4× more prototypes than draft-only or sparse criteria (`1.5`, `2.2`, `3.3`, `3.6`, `3.9`–`3.13`, `4.2`, `4.5`: 5 prototypes each). Because anchors must not be fabricated or duplicated, this imbalance is intentionally documented and surfaced via `prototype_family` rather than artificially equalized.
5. **Known confusable criterion boundaries & open rubric questions (`Q9`–`Q23`):**
   - `Q10`: Boundary between `3.1` (faculty-level formal contests) and `3.5` (faculty movement quizzes/activities).
   - `Q14`: Whether `4.2` (blood donation) and `4.6` (community/residence legal/health awareness) remain active as separate criteria when in-school legal/cybersecurity quizzes currently appear under `3.1` and `3.5`.
   - `Q16`: Unconfirmed organizer levels (e.g., `U-Sci Hub` appearing in `3.3` with `level: null` despite `min_level: khoa`; `KHXHNV`, `TTTT`, `CLB` abbreviations).
   - `Q18`–`Q19`: Mutual exclusivity / double-counting rules (`2.2` vs `2.3`, and `1.6` vs `3.6` / `5.1` / `5.3`).
   - `Q20`: `"Ngày Chủ nhật xanh giai đoạn 1"` recorded under `3.5` rather than `4.3`.
   - `Q23`: Boundary between academic/career workshops (`1.9`) and non-academic/emotional/craft workshops (`3.5`), plus `3.14` vs `3.2`.
   - `Q11`, `Q12`, `Q13`, `Q21`, `Q22`, `Q9`: Scoring/status/evidence-source questions outside prototype semantics, preserved unchanged in `rubric.yaml`.
6. **Role-variant criteria excluded from event-text matching (`1.7`, `3.14`, `4.4`):**
   Have `event_matchable: false` and `requires_role_context: true` in `rubric.yaml`; they remain excluded from Stage 2 prototypes until runtime role metadata is available.

---

## 5. Technical Validation Report

All checks below validate data structure, semantic completeness, rubric referential integrity, and cross-artifact alignment (no model accuracy or benchmark claims):

### 5.A Prototype Quality & Schema v2.0 Checks (`scripts/generate_rubric_prototypes.py`)

| Check | Status | Result |
|---|---|---|
| `unique_prototype_ids` | **PASS** | `176` prototypes, `176` unique IDs |
| `required_fields_present` | **PASS** | `0` missing Schema v2.0 fields across all `176` records |
| `valid_single_passage_prefix_non_empty` | **PASS** | `0` invalid or double-prefixed `embedding_text` values |
| `no_truncated_group_prototypes` | **PASS** | `0` truncated words/sentences across `C1`–`C4`; max token length `484 <= 512`, `truncated_count = 0` |
| `stage1_only_c1_to_c4` | **PASS** | Exactly `4` Stage 1 prototypes: `['C1', 'C2', 'C3', 'C4']` (`C5` and `P` excluded) |
| `stage2_only_eligible_c1_c4_candidates` | **PASS** | `161` Stage 2 prototypes covering `20/20` C1–C4 `event_matchable: true` criteria; `0` invalid groups/eligibilities |
| `criterion_6_4_separated_in_reminder_stage` | **PASS** | `5` prototypes in `stage="reminder"`, `group_id="P"`, `eligibility_class="reminder_only"` |
| `none_isolated_and_zero_needs_review_prototypes` | **PASS** | `6` `NONE` prototypes in `stage="special"` (`draft_negative_reference`, `negative_reference_only`, `can_decide_none=false`); `0` `NEEDS_REVIEW` prototypes |
| `criterion_3_3_mobilization_context_and_signals` | **PASS** | All `3` `3.3` verified anchors carry mobilization context in `embedding_text`, verbatim `anchor_text`, and `required_signals: ["explicit_student_mobilization_signal", "min_level:khoa"]` |
| `criterion_3_8_verified_partial_normalized` | **PASS** | `2` `verified_anchor` (`anchor_status="verified_from_school_list"`) + `2` `draft_anchor` (`anchor_status="draft"`), all carrying `criterion_anchor_status="verified_partial"` |
| `rubric_referential_integrity_and_no_draft_promotion` | **PASS** | `0` broken references to `rubric.yaml`; `0` draft anchors promoted to verified |

### 5.B Cross-Artifact & Embedding Consistency Checks (`scripts/generate_embeddings.py --rubric-only`)

| Check | Status | Result |
|---|---|---|
| `cross_artifact_row_counts_match` | **PASS** | `rubric_prototypes.jsonl` = `176`, `rubric_metadata.parquet` = `176`, `rubric_embeddings.npy` = `(176, 384)` |
| `npy_float32_and_finite` | **PASS** | `dtype=float32`, `0` NaN/Inf |
| `npy_l2_normalized` | **PASS** | `max |‖v‖₂ - 1| = 8.80e-08` |
| `tokenizer_zero_truncation` | **PASS** | `min=7`, `median=17.0`, `max=484`, `max_seq_length=512`, `truncated_count=0` |
| `prototype_id_and_embedding_row_exact_order` | **PASS** | Exact 1:1 row order (`embedding_row == 0..175`) across JSONL, Parquet, and NPY |
| `embedding_text_jsonl_parquet_match` | **PASS** | `176/176` exact string matches between JSONL and Parquet |
| `vector_to_prototype_alignment_spotcheck` | **PASS** | Re-encoded spot-check at indices `[0, 3, 4, 164, 165, 169, 170, 175]`: `min_cosine = 1.000000` |
| `stage_and_eligibility_isolation` | **PASS** | `group=4`, `criterion=161`, `reminder=5`, `special=6` |

### 5.C Untouched Source & Reference Artifacts Verification (SHA-256)

| Artifact | Policy | Verified SHA-256 |
|---|---|---|
| `data/raw/rubric.yaml` | Read-only (unmodified) | `d06da721c85bc79b3cd1f56c05e8a7ce3819ddff04564bab372d25e3ba1a04fd` |
| `res/events_full.csv` | Read-only (unmodified) | `23962abbb4affc970d07a5eb2d2fb4c9d314d111add89fa259ee8baee00fa279` |
| `data/processed/cls_index/sample_embeddings.npy` | Read-only (unmodified) | `98449e3d2d3b65aa84daa4d76b52abd57d8dfd150f43fb110b61cab1d4d605c5` |
| `data/processed/cls_index/sample_metadata.parquet` | Read-only (unmodified) | `79cdfd0692c2ca59d3978f0d74236ea52b28564e4b2930fd33b68fcf19c78939` |
| `data/processed/cls_index/embedding_config.json` | Read-only (unmodified) | `2c40317cda7dece055137a3c5cd561fdaab876063e3fcccfafcab0f826c924bb` |
| `data/generated/events_query_embeddings.npy` | Read-only (unmodified) | `f4dd5510d36384989b46dcc20a4a055bcb68db117688b9127dd498b96f7c7fa9` |
| `data/generated/events_query_metadata.json` | Read-only (unmodified) | `7cf2583ed9c53d4cf968e3915ea577cb81a0af9a85006cbf09a8d67310380d78` |
