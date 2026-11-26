# Sample Embedding Generation Audit Report — HCMUS DRL

> **Generated (UTC):** `2026-11-25T16:02:55+00:00`  
> **Scope notice:** Technical data, embedding, metadata, and compatibility audit for the Sample Branch (`sample_embeddings.npy`, `sample_metadata.parquet`). Does **not** evaluate classification accuracy, precision, recall, or F1-score, and no classification algorithm, fusion weights, or Rubric Branch artifacts were modified.

---

## 1. Source Dataset

| Property | Value |
|---|---|
| **Authoritative input file** | `data/raw/activities_sample.csv` |
| **File format** | CSV (`utf-8`, BOM=`False`, delimiter=`,`, quotechar=`"`) |
| **File size / SHA-256** | `284771` bytes / `1c4f55bcbc70b6f1cb0002e28e2b90014fc6955ebb9e45bdf2f8d86ab7f6f04c` |
| **Total source records** | `1330` |
| **Detected columns (7)** | `STT, Ma_Muc, Organizer, Ten_Hoat_Dong, Vai_Tro, Tieu_Chi_Quy_Che, Diem_Toi_Da` |
| **Malformed rows (column-count mismatch)** | `0` |
| **Fields selected for `embedding_text`** | `Ten_Hoat_Dong` *(no separate description column present in source CSV)* |
| **Fields kept in metadata only (strictly excluded from `embedding_text`)** | `STT, Ma_Muc, Organizer, Vai_Tro, Tieu_Chi_Quy_Che, Diem_Toi_Da` |

### Detected Source Schema & Column Mapping

| Source Column | Empty Rows (of 1,330) | Mapped Role / Metadata Field | Embedded in Vector? |
|---|---:|---|---|
| `STT` | 0 | Source row identifier (`source_stt`, `source_row_index`) | No (metadata only) |
| `Ma_Muc` | 0 | Ground-truth criterion reference (`original_label`, `criterion_ref` -> rubric `criterion_key`, `group_id`) | No (metadata only) |
| `Organizer` | 120 | Organizing unit (`organizer`, nullable) | No (metadata only) |
| `Ten_Hoat_Dong` | 0 | Event title (`event_title`) | **YES** |
| `Vai_Tro` | 563 | Participation role (`role`, nullable) | No (metadata only) |
| `Tieu_Chi_Quy_Che` | 0 | Rubric criterion description (metadata only in `source_metadata`; excluded to prevent label leakage) | No (metadata only) |
| `Diem_Toi_Da` | 0 | Max DRL points (metadata only in `source_metadata`; excluded to prevent label leakage) | No (metadata only) |

---

## 2. Processing Summary

| Metric | Count |
|---|---:|
| **Total source rows (`activities_sample.csv`)** | `1330` |
| **Valid eligible samples embedded (`C1`–`C4`, `event_matchable: true`)** | **`908`** |
| **Skipped / excluded rows (non-eligible or Group `P` reminder/penalty)** | `422` |
| **Invalid / unknown / malformed label rows** | `0` |
| **Label inconsistencies (`Ma_Muc` / `group_id` / `criterion_key` / `Tieu_Chi_Quy_Che`)** | `0` |
| **Exact full-row duplicates (ignoring `STT`) — all source / eligible** | `0` / `0` |
| **Duplicate-title groups among eligible samples (preserved)** | `12` groups (`6` cross-criterion + `6` same-criterion) |
| **Orthographic / diacritic near-duplicate groups among eligible samples (preserved)** | `4` |

### Samples by Major Group (`group_id`)

| `group_id` | Eligible Samples | Share (%) |
|---|---:|---:|
| `C1` | 290 | 31.9% |
| `C2` | 8 | 0.9% |
| `C3` | 419 | 46.1% |
| `C4` | 191 | 21.0% |

### Samples by Criterion (`criterion_ref`)

| `criterion_ref` | `group_id` | `criterion_key` | Eligible Samples |
|---|---|---|---:|
| `1.5` | `C1` | `C1.de_tai_nckh` | 3 |
| `1.6` | `C1` | `C1.cuoc_thi_hoc_thuat` | 45 |
| `1.9` | `C1` | `C1.sinh_hoat_hoc_thuat` | 242 |
| `2.2` | `C2` | `C2.danh_gia_mon_hoc` | 1 |
| `2.3` | `C2` | `C2.khao_sat_dieu_tra` | 7 |
| `3.1` | `C3` | `C3.cuoc_thi_ct_vhvn_tdtt` | 30 |
| `3.2` | `C3` | `C3.chung_nhan_the_luc` | 11 |
| `3.3` | `C3` | `C3.huy_dong_luc_luong` | 47 |
| `3.5` | `C3` | `C3.hoat_dong_cap_lop` | 298 |
| `3.6` | `C3` | `C3.doi_tuyen_cap_truong` | 8 |
| `3.8` | `C3` | `C3.tu_van_tuyen_sinh_viec_lam` | 8 |
| `3.12` | `C3` | `C3.dai_bieu_duoc_tuyen_chon` | 16 |
| `3.13` | `C3` | `C3.dai_bieu_tu_ung_tuyen` | 1 |
| `4.2` | `C4` | `C4.hien_mau` | 9 |
| `4.3` | `C4` | `C4.tinh_nguyen` | 103 |
| `4.5` | `C4` | `C4.quyen_gop` | 78 |
| `4.6` | `C4` | `C4.tuyen_truyen_phap_luat` | 1 |

### Excluded / Skipped Rows Breakdown (Documented, Not Silently Dropped)

| Exclusion Reason | Row Count |
|---|---:|
| `non_event_matchable_criterion (event_matchable=false, kind=role_variant)` | 301 |
| `non_event_matchable_criterion (event_matchable=false, kind=achievement)` | 50 |
| `non_event_matchable_criterion (event_matchable=false, kind=membership)` | 48 |
| `non_event_matchable_criterion (event_matchable=false, kind=status)` | 8 |
| `non_event_matchable_criterion (event_matchable=false, kind=compliance)` | 4 |
| `non_event_matchable_criterion (event_matchable=false, kind=special_circumstance)` | 4 |
| `non_event_matchable_criterion (event_matchable=false, kind=penalty)` | 3 |
| `non_event_matchable_criterion (event_matchable=false, kind=record)` | 3 |
| `excluded_non_positive_group_or_reminder (group=P, kind=mandatory_event, mvp_scope=remind)` | 1 |

<details>
<summary>Excluded rows by criterion (`Ma_Muc`)</summary>

| `Ma_Muc` | `group_id` | `kind` | Excluded Rows |
|---|---|---|---:|
| `1.1` | `C1` | `compliance` | 1 |
| `1.2` | `C1` | `record` | 1 |
| `1.3` | `C1` | `record` | 1 |
| `1.4` | `C1` | `record` | 1 |
| `1.7` | `C1` | `role_variant` | 31 |
| `1.8` | `C1` | `membership` | 14 |
| `2.1` | `C2` | `compliance` | 1 |
| `3.4` | `C3` | `membership` | 34 |
| `3.7` | `C3` | `compliance` | 1 |
| `3.14` | `C3` | `role_variant` | 203 |
| `4.1` | `C4` | `compliance` | 1 |
| `4.4` | `C4` | `role_variant` | 67 |
| `4.7` | `C4` | `status` | 6 |
| `5.1` | `C5` | `achievement` | 24 |
| `5.2` | `C5` | `achievement` | 16 |
| `5.3` | `C5` | `achievement` | 6 |
| `5.4` | `C5` | `achievement` | 2 |
| `5.5` | `C5` | `achievement` | 2 |
| `5.7` | `C5` | `status` | 2 |
| `5.8` | `C5` | `special_circumstance` | 1 |
| `5.9` | `C5` | `special_circumstance` | 1 |
| `5.10` | `C5` | `special_circumstance` | 1 |
| `5.11` | `C5` | `special_circumstance` | 1 |
| `6.1` | `P` | `penalty` | 1 |
| `6.3` | `P` | `penalty` | 1 |
| `6.4` | `P` | `mandatory_event` | 1 |
| `6.6` | `P` | `penalty` | 1 |

</details>

### Preserved Duplicate & Multi-Label Records (Zero Silent Deletion)

- **Cross-criterion duplicate titles (`6` groups, preserved as separate labeled samples):**
  - "Talkshow Giá trị của việc đọc trên môi trường đại học": `activities_sample_row_0108` (row `108`, ref `1.9`, org=`CLBS`, role=``); `activities_sample_row_0508` (row `508`, ref `3.5`, org=`CLBS`, role=``)
  - "Giải bóng đá FIT Super League 2026": `activities_sample_row_0360` (row `360`, ref `3.1`, org=`CNTT`, role=`Tham gia`); `activities_sample_row_0393` (row `393`, ref `3.3`, org=`CNTT`, role=`Huy động lực lượng`)
  - "Story in Pixel": `activities_sample_row_0361` (row `361`, ref `3.1`, org=`CNTT`, role=`Tham gia`); `activities_sample_row_0394` (row `394`, ref `3.3`, org=`CNTT`, role=`Huy động lực lượng`)
  - "Cuộc thi hát Sing with FETEL": `activities_sample_row_0419` (row `419`, ref `3.3`, org=`LCH ĐTVT`, role=`Huy động lực lượng`); `activities_sample_row_0623` (row `623`, ref `3.5`, org=`LCH ĐTVT`, role=``)
  - "Đêm Nhạc Gây Quỹ KHẢM": `activities_sample_row_0646` (row `646`, ref `3.5`, org=`VNXK`, role=`Tham gia`); `activities_sample_row_1246` (row `1246`, ref `4.5`, org=`LCH CNTT`, role=`Tham gia`)
  - "Hành trình hạt gạo": `activities_sample_row_1115` (row `1115`, ref `4.3`, org=`ĐK ĐTVT`, role=`Tham gia`); `activities_sample_row_1262` (row `1262`, ref `4.5`, org=`ĐK ĐTVT`, role=`Tham gia`)
- **Same-criterion duplicate titles (`6` groups, differing by organizer or role — preserved):**
  - "Thi thử TOEIC 2026": `activities_sample_row_0022` (row `22`, ref `1.6`, org=`CTXH`, role=`Tham gia`); `activities_sample_row_0026` (row `26`, ref `1.6`, org=`LCH HH`, role=`Tham gia`)
  - "Thi thử Toeic": `activities_sample_row_0025` (row `25`, ref `1.6`, org=`LCH CNTT`, role=`Tham gia`); `activities_sample_row_0031` (row `31`, ref `1.6`, org=`LCH TTH`, role=`Tham gia`); `activities_sample_row_0037` (row `37`, ref `1.6`, org=`LCH ĐTVT`, role=`Tham gia`); `activities_sample_row_0044` (row `44`, ref `1.6`, org=`ĐK HH`, role=`Tham gia`)
  - "Chương trình Người trong nghề": `activities_sample_row_0110` (row `110`, ref `1.9`, org=`CNTT`, role=`Ban tổ chức`); `activities_sample_row_0111` (row `111`, ref `1.9`, org=`CNTT`, role=`Tham gia`)
  - "Hành trình đến với bảo tàng": `activities_sample_row_0491` (row `491`, ref `3.5`, org=`CDCS KHLN`, role=`Tham gia`); `activities_sample_row_0583` (row `583`, ref `3.5`, org=`LCH SH-CNSH`, role=`Tham gia`); `activities_sample_row_0662` (row `662`, ref `3.5`, org=`ĐK HH`, role=`Tham gia`); `activities_sample_row_0699` (row `699`, ref `3.5`, org=`ĐK SH-CNSH`, role=`Tham gia`)
  - "Sinh hoạt chủ điểm tháng 3": `activities_sample_row_0681` (row `681`, ref `3.5`, org=`ĐK KHCNVL`, role=`Tham gia`); `activities_sample_row_0749` (row `749`, ref `3.5`, org=`ĐK ĐTVT`, role=`Tham gia`)
  - "Chủ nhật Xanh": `activities_sample_row_1049` (row `1049`, ref `4.3`, org=`LCH MT`, role=`Tham gia`); `activities_sample_row_1094` (row `1094`, ref `4.3`, org=`ĐK SH-CNSH`, role=`Tham gia`); `activities_sample_row_1105` (row `1105`, ref `4.3`, org=`ĐK VL-VLKT`, role=`Tham gia`); `activities_sample_row_1112` (row `1112`, ref `4.3`, org=`ĐK ĐTVT`, role=`Tham gia`)
- **Orthographic / diacritic near-duplicate groups (`4` groups — preserved):**
  - `activities_sample_row_0039` (row `39`, ref `1.6`, title="Hội thi học thuật Bigbang lần thứ XIX năm 2026"); `activities_sample_row_0040` (row `40`, ref `1.6`, title="Hội thi học thuật Bigbang lần thứ XIX, năm 2026")
  - `activities_sample_row_0380` (row `380`, ref `3.2`, title="Sinh viên khỏe 2026"); `activities_sample_row_0387` (row `387`, ref `3.2`, title="Sinh viên khoẻ 2026")
  - `activities_sample_row_1011` (row `1011`, ref `4.2`, title="Hiến máu tình nguyện lần 2"); `activities_sample_row_1014` (row `1014`, ref `4.2`, title="Hiến máu tình nguyện - lần 2")
  - `activities_sample_row_1086` (row `1086`, ref `4.3`, title="HOHO DU KÝ Số 05 - Ươm mầm tri thức"); `activities_sample_row_1255` (row `1255`, ref `4.5`, title="HOHO DU KÝ Sô 05 Ươm mầm tri thức")

---

## 3. Embedding Configuration

| Parameter | Value (from `embedding_config.json`) |
|---|---|
| **Model name** | `intfloat/multilingual-e5-small` |
| **Model revision** | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| **Embedding dimension** | `384` |
| **Data type** | `float32` |
| **Pooling method** | `mean` |
| **Normalization** | L2 unit normalization (`l2_normalization: true`) |
| **Tokenizer** | `XLMRobertaTokenizer` (`max_seq_length=512`, `do_lower_case=False`) |
| **Prefix convention** | Sample passages: `passage: ...` / Queries: `query: ...` |
| **Text construction rules** | Unicode NFC + whitespace collapse (matching `src/classification/embeddings/event_text_builder.py`); embeds `Ten_Hoat_Dong` only (no description column in CSV); strictly excludes `Ma_Muc`, `group_id`, `criterion_key`, `Tieu_Chi_Quy_Che`, `Diem_Toi_Da`, `Vai_Tro`, and `Organizer` to prevent label leakage. |
| **Truncation strategy & token stats** | Model `max_seq_length = 512` tokens; observed sample lengths: `min=6`, `median=14.0`, `max=54`, **`truncated_count=0`** (no truncation needed). |

---

## 4. Generated Artifacts

| Artifact | Rows | Dimension / Columns | Alignment Status |
|---|---:|---|---|
| `data/processed/cls_index/sample_embeddings.npy` | `908` | `(908, 384)`, `float32`, L2-normalized | Exact 1:1 with `sample_metadata.parquet` (`0..907`) |
| `data/processed/cls_index/sample_metadata.parquet` | `908` | `20` columns (zero embedded vectors) | Exact 1:1 with `sample_embeddings.npy` (`embedding_row == 0..907`) |

**Metadata columns:** `embedding_row, sample_id, source_row_index, source_stt, source_dataset, event_title, event_description, embedding_text, original_label, criterion_ref, criterion_key, group_id, tieu_chi, role, organizer, label_status, embedding_model, embedding_dimension, normalization, source_metadata`

---

## 5. Technical Validation Results

| Check | Status | Details |
|---|---|---|
| `dataset_zero_malformed_rows` | **PASS** | malformed_rows=0 across 1330 source rows |
| `dataset_zero_label_inconsistencies` | **PASS** | label_inconsistencies=0, criterion_text_variants=0 |
| `dataset_zero_empty_event_texts` | **PASS** | empty_text_rows=0 |
| `embedding_count_matches_eligible_samples` | **PASS** | npy_rows=908, eligible_samples=908 |
| `embedding_dimension_384` | **PASS** | shape=(908, 384), expected_dim=384 |
| `embedding_dtype_float32` | **PASS** | dtype=float32 |
| `embedding_all_finite` | **PASS** | No NaN or Inf values |
| `embedding_l2_normalized` | **PASS** | max \|norm - 1\| = 1.02e-07 |
| `tokenizer_zero_silent_truncation` | **PASS** | min=6, median=14.0, max=54, max_seq_length=512, truncated=0 |
| `embedding_passage_prefix_and_zero_leakage` | **PASS** | prefix_errors=0, leakage_hits=0 |
| `embedding_vector_row_alignment_spotcheck` | **PASS** | spot_indices=[0, 289, 290, 297, 298, 454, 716, 717, 907], min_cosine=1.000000 |
| `metadata_schema_and_no_embedded_vectors` | **PASS** | columns=20, missing_required=[], no_vector_col=True |
| `metadata_row_count_matches_embeddings` | **PASS** | metadata_rows=908, embedding_rows=908 |
| `metadata_contiguous_embedding_row_and_source_order` | **PASS** | embedding_row_0..N-1=True, unique_sample_id=True, strictly_increasing_source_row=True |
| `metadata_rubric_referential_integrity` | **PASS** | All 908 rows match rubric.yaml C1-C4 event_matchable:true criteria |
| `metadata_embedding_provenance_recorded` | **PASS** | model=intfloat/multilingual-e5-small, dim=384, normalization=l2 |
| `sample_retriever_stage1_stage2_compatibility` | **PASS** | eligible=908, groups=['C1', 'C2', 'C3', 'C4'], stage1_hits=10 (top1_sim=1.000000), stage2_C1C3_hits=10 |

### Untouched Rubric & Read-Only Artifacts Verification (SHA-256)

| Artifact | SHA-256 |
|---|---|
| `data/raw/activities_sample.csv` | `1c4f55bcbc70b6f1cb0002e28e2b90014fc6955ebb9e45bdf2f8d86ab7f6f04c` |
| `data/raw/rubric.yaml` | `d06da721c85bc79b3cd1f56c05e8a7ce3819ddff04564bab372d25e3ba1a04fd` |
| `data/processed/cls_index/rubric_embeddings.npy` | `ad46584f7fea071d9170faf25315daefc3a5c02db634a63e5cd7f0549119e6bd` |
| `data/processed/cls_index/rubric_metadata.parquet` | `eaf1dd43e1f919f77e26db5f23c9825d737584c0b66b2c13404281b1975eb94c` |
| `data/processed/cls_index/rubric_prototypes.jsonl` | `638fa66625ce9dddee73e65a5c768c2c4eae28024efd91f0aa1a279365b76ba4` |
| `data/processed/cls_index/embedding_config.json` | `2c40317cda7dece055137a3c5cd561fdaab876063e3fcccfafcab0f826c924bb` |
| `data/generated/events_query_embeddings.npy` | `14aa5478865e571230300a63a02a9fe58ccea26cd4f68d440ece004692c54db3` |
| `data/generated/events_query_metadata.json` | `5b22fec91b2224aa91bf31c65bdcbd1e1cdbce548183c42ab3a3979336da58b8` |

---

## 6. Unresolved Issues & Data Quality Notes

1. **No separate event description column in `activities_sample.csv`:** The CSV provides `Ten_Hoat_Dong` (event title) only; `event_description` is preserved as `null` across all `908` samples without fabrication. Runtime incoming queries in `events.json` include both title and description.
2. **Missing optional metadata fields among eligible samples:** `role` (`Vai_Tro`) is empty/null on `456/908` samples, and `organizer` (`Organizer`) is empty/null on `95/908` samples (stored as `null`, not fabricated).
3. **Uncovered C1–C4 `event_matchable: true` criteria (`3/20`):** Criteria `3.9, 3.10, 3.11` have `0` rows in `activities_sample.csv` and continue to rely on Stage 2 Rubric Branch prototypes (via `missing_branch_reference_policy: reweight_available_branch`).
4. **Criterion `6.4` (`STT=1329`, `group=P`) excluded from positive Sample Branch:** Row 1329 (`"Không tham gia, không đạt Sinh hoạt công dân Cuối khóa năm học 2025-2026"`, `Diem_Toi_Da=-20`) is a mandatory-event penalty record (`group: P`, `mvp_scope: remind`) and is excluded from the positive `C1`–`C4` Sample Branch to remain aligned with `RubricRetriever` (`stage="reminder"`) and `Module3Config.excluded_parent_groups = ("P",)`.
5. **Cross-criterion identical titles (`6` groups, `12` rows) & near-duplicate cross-criterion titles (`1` group):** Identical event titles appear under different criteria (`1.9` vs `3.5`, `3.1` vs `3.3`, `3.3` vs `3.5`, `3.5` vs `4.5`, `4.3` vs `4.5`), reflecting role/mobilization/fundraising distinctions not fully expressed by title alone. All records are preserved faithfully without merging or relabeling.
6. **Criterion `3.3` role coverage:** Among the `47` eligible `3.3` samples, `21` carry explicit `Vai_Tro = "Huy động lực lượng"`, `6` carry `"Tham gia"`, and `20` have empty `Vai_Tro` — reinforcing why the existing `3.3` mobilization decision guard remains important.
7. **Severe class count imbalance:** `3.5` (`298`) and `1.9` (`242`) account for `540/908` (`59.5%`) of eligible samples, whereas `2.2`, `3.13`, and `4.6` have `1` sample each.
8. **Raw source title artifacts preserved verbatim (no rewriting):** Row `751` (`3.5`) contains a raw concatenation/truncation artifact (`"...“Theo dấu chân Bác”DTVT - Tổ chức sự kiện “Hành trình theo chân Bác"`); rows `32`, `742`, `743`, `1113` contain leading underscore artifacts (`_Chấn động Pangaea`, `_Tứ cực đại chiến`, `_25DTV CUP`, `_Thanh niên hành động xanh`); minor source typos exist on rows `402`, `416`, `549`, `1255`.
9. **No verified negative (`NONE` / `NO_MATCH`) samples:** `activities_sample.csv` contains only DRL rubric records (`0` negative/out-of-scope examples).
