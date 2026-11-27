# Embedding Generation Report — HCMUS DRL Hierarchical 2-Stage kNN

- **Generated (UTC)**: `2026-11-25T06:03:59+00:00`
- **Output Directory**: `data/processed/cls_index`
- **Model**: `intfloat/multilingual-e5-small` (revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`)
- **Embedding Spec**: `384d`, `float32`, L2-normalized (`True`), pooling=`mean`, passage prefix=`"passage: "`, query prefix=`"query: "`
- **Validation Status**: **ALL CHECKS PASSED**

---

## 1. Deliverables Summary

| Deliverable | Shape / Rows | Details |
|---|---|---|
| `rubric_embeddings.npy` | `(170, 384)` | `float32`, L2-normalized, Stage 1 + Stage 2 prototypes |
| `rubric_metadata.parquet` | `170` rows | Aligned 1:1 with `rubric_embeddings.npy` (`embedding_row` `0..169`) |
| `sample_embeddings.npy` | `(127, 384)` | `float32`, L2-normalized, eligible `events_full.csv` activities |
| `sample_metadata.parquet` | `127` rows | Aligned 1:1 with `sample_embeddings.npy` (`embedding_row` `0..126`) |
| `embedding_config.json` | 1 JSON | Reproducibility spec, model revision, tokenizer/pooling config, source SHA-256 hashes |
| `embedding_generation_report.md` | 1 Markdown | This generation, exclusion, duplicate, and validation audit report |

---

## 2. Rubric Branch Summary

- Total rubric items inspected: **51**
- Eligible (`event_matchable: true`) criteria: **21** (20 positive in C1–C4 + 1 mandatory reminder `6.4` in group P)
- Excluded (`event_matchable: false`) criteria: **30**
- Excluded non-criterion meta-labels: `NONE, NEEDS_REVIEW`
- Total rubric prototypes embedded: **170**

### Prototypes by Stage

| Stage | Count | Groups Covered |
|---|---|---|
| `group` (Stage 1) | 4 | `C1, C2, C3, C4` |
| `criterion` (Stage 2) | 166 | `C1, C2, C3, C4, P` |

### Prototypes by Type

| `prototype_type` | Count |
|---|---|
| `group_definition` | 4 |
| `criterion_definition` | 21 |
| `verified_anchor` | 89 |
| `partial_anchor` | 2 |
| `draft_anchor` | 33 |
| `keyword_context` | 21 |

### Stage 2 Prototypes by `anchor_status`

| `anchor_status` | Count |
|---|---|
| `draft` | 53 |
| `verified_from_school_list` | 109 |
| `verified_partial` | 4 |

### Major Groups Excluded from Stage 1

| Group | Name | Reason |
|---|---|---|
| `C5` | Các nội dung khuyến khích | Zero items with event_matchable: true (all items are achievements, statuses, or special circumstances). |
| `P` | Điểm trừ | Non-positive penalty/reminder group (sign=negative); not a positive Stage 1 C1-C5 event-classification target (contains only 6.4 mandatory_event / remind). |

### Token Length Audit (Rubric Branch)

- Min tokens: `7`, Median: `17.0`, Max: `467` (limit `512`)
- Truncated prototypes: `0`

---

## 3. Sample Branch Summary (`res/events_full.csv`)

- Total source rows: **153**
- Eligible positive sample embeddings: **127**
- Excluded source rows: **26**
- Invalid / missing / special (`NONE`, `NO_MATCH`, `NEEDS_REVIEW`) labels: **0**
- Detected CSV schema: `['STT', 'Ma_Muc', 'Organizer', 'Ten_Hoat_Dong', 'Vai_Tro', 'Tieu_Chi_Quy_Che', 'Diem_Toi_Da']`
- Embedded text columns: `Ten_Hoat_Dong` (title only; no description column exists in `events_full.csv`). `Organizer`, `Vai_Tro`, `Tieu_Chi_Quy_Che`, `Diem_Toi_Da`, and `STT` are preserved strictly in `source_metadata` to prevent label/role/scoring leakage and CNTT organizer skew.

### Eligible Samples by Group and Criterion

| Group | Eligible Samples | Criteria Covered |
|---|---|---|
| `C1` | 50 | `1.6` (17), `1.9` (33) |
| `C2` | 13 | `2.2` (3), `2.3` (10) |
| `C3` | 55 | `3.1` (7), `3.2` (6), `3.3` (3), `3.5` (37), `3.8` (2) |
| `C4` | 9 | `4.3` (6), `4.5` (3) |

### Token Length Audit (Sample Branch)

- Min tokens: `6`, Median: `14.0`, Max: `44` (limit `512`)
- Truncated samples: `0`

---

## 4. Excluded Source Rows (`res/events_full.csv`)

All **26** excluded rows map to valid rubric criteria whose `event_matchable` is `false` (compliance, academic record, role_variant, membership, status, post-hoc achievement, or penalty). Zero rows were dropped due to missing, unknown, or special meta-labels.

| Source Row (`STT`) | `Ma_Muc` | `criterion_key` | `group_id` | Reason | Activity Title |
|---|---|---|---|---|---|
| 1 | `1.1` | `C1.khong_vi_pham_quy_che_dao_tao` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=compliance) | Ý thức chấp hành Quy chế đào tạo, Quy định về trách nhiệm của cán bộ và người học trong các kỳ thi của Trường |
| 2 | `1.2` | `C1.diem_trung_binh_hoc_ky` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=record) | Điểm trung bình tích lũy HK1/25-26 |
| 3 | `1.2` | `C1.diem_trung_binh_hoc_ky` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=record) | Điểm trung bình tích lũy HK2/25-26 |
| 4 | `1.3` | `C1.giu_xep_loai_xuat_sac` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=record) | Giữ xếp loại xuất sắc so với học kỳ trước |
| 5 | `1.4` | `C1.tang_diem_trung_binh` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=record) | Tăng điểm trung bình tích lũy so với học kỳ trước |
| 23 | `1.7` | `C1.btc_cuoc_thi_hoc_thuat` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=role_variant) | Cuộc thi Coding Challenge |
| 24 | `1.7` | `C1.btc_cuoc_thi_hoc_thuat` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=role_variant) | Cuộc thi trực tuyến Tiếng Anh trong sinh viên Star Awards năm 2025 |
| 25 | `1.8` | `C1.cau_lac_bo_hoc_thuat` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=membership) | Thành viên CLB Design ITUS |
| 26 | `1.8` | `C1.cau_lac_bo_hoc_thuat` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=membership) | Thành viên CLB Google Developer Group on Campus - HCMUS |
| 27 | `1.8` | `C1.cau_lac_bo_hoc_thuat` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=membership) | Thành viên CLB Google Developer Group on Campus - HCMUS |
| 28 | `1.8` | `C1.cau_lac_bo_hoc_thuat` | `C1` | non_event_matchable_criterion (event_matchable=false, kind=membership) | Tham gia sinh hoạt thường xuyên tại Thư viện |
| 62 | `2.1` | `C2.khong_vi_pham_noi_quy` | `C2` | non_event_matchable_criterion (event_matchable=false, kind=compliance) | Ý thức chấp hành nội quy, quy chế |
| 83 | `3.14` | `C3.btc_ctv_hoat_dong` | `C3` | non_event_matchable_criterion (event_matchable=false, kind=role_variant) | Đại hội Đại biểu Hội Sinh viên Việt Nam Trường Đại học Khoa học tự nhiên, ĐHQG - HCM lần thứ XIII, nhiệm kỳ 2025 - 2028 |
| 84 | `3.14` | `C3.btc_ctv_hoat_dong` | `C3` | non_event_matchable_criterion (event_matchable=false, kind=role_variant) | Đại hội chi Đoàn |
| 85 | `3.14` | `C3.btc_ctv_hoat_dong` | `C3` | non_event_matchable_criterion (event_matchable=false, kind=role_variant) | Đại hội chi Đoàn nhiệm kỳ 2025 - 2026 |
| 132 | `3.7` | `C3.khong_vi_pham_te_nan` | `C3` | non_event_matchable_criterion (event_matchable=false, kind=compliance) | Tham gia phòng chống tội phạm và các tệ nạn xã hội |
| 135 | `4.1` | `C4.khong_vi_pham_phap_luat` | `C4` | non_event_matchable_criterion (event_matchable=false, kind=compliance) | Chấp hành Luật Bảo hiểm y tế |
| 136 | `4.1` | `C4.khong_vi_pham_phap_luat` | `C4` | non_event_matchable_criterion (event_matchable=false, kind=compliance) | Ý thức chấp hành pháp luật |
| 143 | `4.4` | `C4.btc_ctv_tinh_nguyen` | `C4` | non_event_matchable_criterion (event_matchable=false, kind=role_variant) | Hạ Xanh Hy Vọng |
| 147 | `4.7` | `C4.ban_can_su_lop` | `C4` | non_event_matchable_criterion (event_matchable=false, kind=status) | Ban cán sự hoàn thành tốt nhiệm vụ |
| 148 | `4.7` | `C4.ban_can_su_lop` | `C4` | non_event_matchable_criterion (event_matchable=false, kind=status) | Ban cán sự lớp hoàn thành tốt nhiệm vụ trở lên |
| 149 | `4.7` | `C4.ban_can_su_lop` | `C4` | non_event_matchable_criterion (event_matchable=false, kind=status) | Danh sách ban cán sự lớp hoàn thành tốt nhiệm vụ |
| 150 | `4.7` | `C4.ban_can_su_lop` | `C4` | non_event_matchable_criterion (event_matchable=false, kind=status) | Hoàn thành tốt nhiệm vụ Ban cán sự |
| 151 | `5.1` | `C5.khen_thuong_thanh_tich` | `C5` | non_event_matchable_criterion (event_matchable=false, kind=achievement) | Thành tích tại Kỳ thi Olympic Tin học Sinh viên HCMUS năm 2025 |
| 152 | `5.3` | `C5.doi_tuyen_giai_thuong_cap_thanh` | `C5` | non_event_matchable_criterion (event_matchable=false, kind=achievement) | Đội tuyển ICPC Bắc, Trung, Nam, Quốc gia, Châu Á |
| 153 | `6.1` | `P.khai_bao_khao_sat` | `P` | non_event_matchable_criterion (event_matchable=false, kind=penalty) | Không tham gia đánh giá môn học |

### Invalid, Special, or Unresolved Labels

- None (`0` rows had `NONE`, `NO_MATCH`, `NEEDS_REVIEW`, missing, unknown, or unmapped rubric references).

---

## 5. Missing / Uncovered Rubric Criteria in `events_full.csv`

**10** of the **21** `event_matchable: true` criteria have **0 sample rows** in `events_full.csv`. All of them are covered by Stage 2 prototypes in `rubric_embeddings.npy` (currently draft anchors only):

| `criterion_ref` | `criterion_key` | `group_id` | `anchors_status` |
|---|---|---|---|
| `1.5` | `C1.de_tai_nckh` | `C1` | `draft` |
| `3.6` | `C3.doi_tuyen_cap_truong` | `C3` | `draft` |
| `3.9` | `C3.tim_hieu_lich_su_truong` | `C3` | `draft` |
| `3.10` | `C3.gop_y_van_ban` | `C3` | `draft` |
| `3.11` | `C3.san_pham_truyen_thong` | `C3` | `draft` |
| `3.12` | `C3.dai_bieu_duoc_tuyen_chon` | `C3` | `draft` |
| `3.13` | `C3.dai_bieu_tu_ung_tuyen` | `C3` | `draft` |
| `4.2` | `C4.hien_mau` | `C4` | `draft` |
| `4.6` | `C4.tuyen_truyen_phap_luat` | `C4` | `draft` |
| `6.4` | `P.sinh_hoat_cong_dan` | `P` | `draft` |

---

## 6. Duplicate & Near-Duplicate Records Preserved

- Exact case-insensitive title duplicates among the 127 eligible samples: **0**
- Orthographic / diacritic-normalized near-duplicate groups preserved among eligible samples: **1**
  - Preserved as separate samples: `events_full_row_0086` (row 86, ref `3.2`, title="Thanh niên khoẻ") vs `events_full_row_0088` (row 88, ref `3.2`, title="Thanh niên khỏe")
- Note on role-split duplicate titles in raw CSV: 5 activities appear twice in `events_full.csv` under different roles (`1.6`/`1.7`, `3.5`/`3.14`, `4.3`/`4.4`). Because `1.7`, `3.14`, and `4.4` have `event_matchable: false` (`kind: role_variant`), only the participant event rows (`1.6`, `3.5`, `4.3`) are retained in `sample_embeddings.npy`.
- Note on `rubric.yaml` anchors: criterion `3.2` lists both `"Thanh niên khoẻ"` and `"Thanh niên khỏe"` as separate anchors; both are preserved as distinct `verified_anchor` prototypes.

---

## 7. Embedding Model & Reproducibility Configuration

```json
{
  "model_name": "intfloat/multilingual-e5-small",
  "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
  "embedding_dimension": 384,
  "dtype": "float32",
  "device": "cpu",
  "batch_size": 32,
  "pooling_method": "mean",
  "pooling_config": {
    "embedding_dimension": 384,
    "pooling_mode": "mean",
    "include_prompt": true
  },
  "l2_normalization": true,
  "prefixes": {
    "passage": "passage: ",
    "query": "query: "
  },
  "tokenizer_config": {
    "tokenizer_class": "XLMRobertaTokenizer",
    "model_max_length": 512,
    "max_seq_length": 512,
    "do_lower_case": false,
    "vocab_size": 250002,
    "pad_token": "<pad>",
    "cls_token": "<s>",
    "sep_token": "</s>"
  },
  "model_modules": [
    "Transformer",
    "Pooling",
    "Normalize"
  ],
  "shared_across_branches": true,
  "text_preprocessing_rules": {
    "unicode_normalization": "NFC",
    "whitespace_normalization": "collapse_whitespace_and_strip",
    "lowercase": false,
    "rubric_branch": {
      "stage1_fields": [
        "groups.<id>.name",
        "tieu_chi[eligible].name",
        "items[event_matchable=true].name"
      ],
      "stage2_definition_fields": [
        "name",
        "scope_note"
      ],
      "stage2_anchor_fields": [
        "anchors",
        "verified_anchors",
        "draft_anchors"
      ],
      "stage2_keyword_fields": [
        "name",
        "keywords"
      ],
      "excluded_semantics": [
        "scoring_points_and_formulas",
        "participation_verification",
        "role_eligibility",
        "attendance_and_completion_thresholds",
        "csv_row_and_administrative_metadata",
        "uncertain_review_notes_kept_in_metadata_only"
      ]
    },
    "sample_branch": {
      "source_csv_columns": [
        "STT",
        "Ma_Muc",
        "Organizer",
        "Ten_Hoat_Dong",
        "Vai_Tro",
        "Tieu_Chi_Quy_Che",
        "Diem_Toi_Da"
      ],
      "embedded_fields": [
        "Ten_Hoat_Dong"
      ],
      "format_template": "passage: {event_title}",
      "metadata_only_fields": [
        "STT",
        "Ma_Muc",
        "Organizer",
        "Vai_Tro",
        "Tieu_Chi_Quy_Che",
        "Diem_Toi_Da"
      ],
      "eligibility_filter": "rubric item exists and event_matchable == true"
    }
  },
  "token_stats": {
    "rubric": {
      "count": 170,
      "min_tokens": 7,
      "median_tokens": 17.0,
      "max_tokens": 467,
      "max_seq_length": 512,
      "truncated_count": 0
    },
    "sample": {
      "count": 127,
      "min_tokens": 6,
      "median_tokens": 14.0,
      "max_tokens": 44,
      "max_seq_length": 512,
      "truncated_count": 0
    }
  },
  "artifact_counts": {
    "rubric_embeddings": 170,
    "sample_embeddings": 127
  },
  "generation_timestamp_utc": "2026-11-25T06:03:59+00:00",
  "sources": {
    "rubric": {
      "path": "data/raw/rubric.yaml",
      "version": "khtn-2025-26-verified-v5-mvp",
      "size_bytes": 151895,
      "sha256": "d06da721c85bc79b3cd1f56c05e8a7ce3819ddff04564bab372d25e3ba1a04fd"
    },
    "samples": {
      "path": "res/events_full.csv",
      "rows": 153,
      "size_bytes": 25797,
      "sha256": "23962abbb4affc970d07a5eb2d2fb4c9d314d111add89fa259ee8baee00fa279"
    }
  },
  "runtime_versions": {
    "python": "3.13.9",
    "torch": "2.15.0+cpu",
    "transformers": "5.24.1",
    "sentence_transformers": "6.3.0",
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0"
  }
}
```

---

## 8. Validation Results

| Check | Status | Details |
|---|---|---|
| `file_loading_and_row_count_match` | **PASS** | rubric=(170, 384) vs 170 rows; sample=(127, 384) vs 127 rows |
| `embedding_dimensions_384` | **PASS** | rubric_dim=384, sample_dim=384 |
| `float32_dtype` | **PASS** | rubric_dtype=float32, sample_dtype=float32 |
| `finite_values` | **PASS** | All vector entries finite (no NaN/Inf) |
| `l2_norm_unit_length` | **PASS** | rubric max |norm-1|=8.83e-08, sample max |norm-1|=8.32e-08 |
| `sequential_unique_embedding_row` | **PASS** | rubric 0..169: True; sample 0..126: True |
| `unique_prototype_and_sample_ids` | **PASS** | prototype_id unique=True, sample_id unique=True |
| `rubric_stage_separation_and_coverage` | **PASS** | Stage 1 groups=['C1', 'C2', 'C3', 'C4']; Stage 2 covered all 21 event_matchable=true criteria=True |
| `sample_rubric_references_valid` | **PASS** | All 127 samples map to valid event_matchable=true rubric items |
| `passage_prefix_enforced` | **PASS** | rubric_prefix_ok=True, sample_prefix_ok=True |
| `zero_sample_label_leakage` | **PASS** | Zero leakage detected across all sample embedding texts |
| `rubric_semantic_exclusions_clean` | **PASS** | Zero scoring/admin/completion leakage in rubric embedding texts |
| `vector_to_metadata_alignment_spotcheck` | **PASS** | rubric spot cosines=[1.0, 1.0, 1.0]; sample spot cosines=[1.0, 1.0, 1.0] |
| `two_stage_knn_filtering_compatibility` | **PASS** | Stage1=(4, 384); Stage2(C1|C3) rubric=(120, 384), sample=(105, 384); ref 1.6 sample=(17, 384) |
| `embedding_config_completeness` | **PASS** | model=intfloat/multilingual-e5-small, rev=614241f622f53c4eeff9890bdc4f31cfecc418b3, dim=384 |

---

## 9. Unresolved Issues & Warnings

1. **Draft-only criteria (10/21)**: Criteria `1.5`, `3.6`, `3.9`, `3.10`, `3.11`, `3.12`, `3.13`, `4.2`, `4.6`, and `6.4` have zero verified samples in `events_full.csv` and rely exclusively on `draft_anchor`, `criterion_definition`, and `keyword_context` prototypes in the Rubric branch.
2. **Partially verified criterion (`3.8`)**: `C3.tu_van_tuyen_sinh_viec_lam` has `anchors_status: verified_partial` (2 `partial_anchor` from school list covering Job Fair + 2 `draft_anchor` covering admissions advising).
3. **Severe class imbalance in Sample branch**: `3.5` (37 rows) and `1.9` (33 rows) account for 70/127 (55.1%) of eligible sample embeddings, whereas `3.8` has 2 rows and 10 eligible criteria have 0 rows. Stage-wise Fusion with the Rubric branch is required to prevent high-frequency sample dominance.
4. **Mandatory event `6.4` (`P.sinh_hoat_cong_dan`)**: Included in Stage 2 (`stage='criterion'`, `group_id='P'`, `is_positive_target=False`, `mvp_scope='remind'`) because `event_matchable: true` in `rubric.yaml`, while Group `P` is excluded from Stage 1 positive C1–C5 group prototypes. Downstream pipelines should query `group_id == 'P'` separately for mandatory-event reminders rather than positive DRL scoring.
5. **Group `C5` excluded from Stage 1**: All 11 items in `C5` have `event_matchable: false` (post-hoc achievements, statuses, or sensitive circumstances), so `C5` has no Stage 1 or Stage 2 positive prototypes and 0 eligible samples.
6. **No negative (`NONE` / `NO_MATCH`) verified samples**: `events_full.csv` contains only positive DRL records (open question `Q17` in `rubric.yaml`); out-of-scope rejection thresholds must be calibrated separately once verified negative examples are collected.
7. **Result-announcement row in `4.3`**: Row 139 (`"Danh sách tình nguyện viên tiếp nhận quyên góp cho các tỉnh miền Trung"`) is labeled `4.3` in `events_full.csv` and preserved faithfully without label modification, though `rubric.yaml` flags it as a result-list post (`post_types.result_announcement`).
8. **Short title-only sample texts**: `events_full.csv` lacks an activity description column; sample embeddings encode activity titles only (`passage: <Ten_Hoat_Dong>`). Runtime queries containing long descriptions may show slight length-distribution shift relative to title-only samples.
