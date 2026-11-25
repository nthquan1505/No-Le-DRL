# Prototype Generation Report

> **Source**: `data/raw/rubric.yaml` (v5-mvp, khtn-2025-26-verified-v5-mvp)
> **Date**: 2026-11-25
> **Script**: `scripts/generate_rubric_prototypes.py`
> **Do NOT modify**: `data/raw/rubric.yaml`, `res/events_full.csv`, or any existing code.

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Total prototypes | 176 |
| event_matchable: true items | 21 |
| event_matchable: false items (excluded from positive prototypes) | 30 |
| Stage 1 (group_definition) | 4 |
| Stage 2 (criterion-level) | 166 |
| Special (NONE meta-label) | 6 |

### Prototype types

| Type | Count |
|------|-------|
| criterion_definition | 22 |
| draft_anchor | 37 |
| group_definition | 4 |
| keyword_context | 22 |
| verified_anchor | 91 |

---

## 2. Included Items (`event_matchable: true`)

| criterion_key | ref | group | anchors_status | verified | draft |
|---|---|---|---|---|---|
| `C1.de_tai_nckh` | 1.5 | C1 | draft | 0 | 3 |
| `C1.cuoc_thi_hoc_thuat` | 1.6 | C1 | verified_from_school_list | 15 | 0 |
| `C1.sinh_hoat_hoc_thuat` | 1.9 | C1 | verified_from_school_list | 18 | 0 |
| `C2.danh_gia_mon_hoc` | 2.2 | C2 | verified_from_school_list | 3 | 0 |
| `C2.khao_sat_dieu_tra` | 2.3 | C2 | verified_from_school_list | 10 | 0 |
| `C3.cuoc_thi_ct_vhvn_tdtt` | 3.1 | C3 | verified_from_school_list | 7 | 0 |
| `C3.chung_nhan_the_luc` | 3.2 | C3 | verified_from_school_list | 6 | 0 |
| `C3.huy_dong_luc_luong` | 3.3 | C3 | verified_from_school_list | 3 | 0 |
| `C3.hoat_dong_cap_lop` | 3.5 | C3 | verified_from_school_list | 18 | 0 |
| `C3.doi_tuyen_cap_truong` | 3.6 | C3 | draft | 0 | 3 |
| `C3.tu_van_tuyen_sinh_viec_lam` | 3.8 | C3 | verified_partial | 2 | 2 |
| `C3.tim_hieu_lich_su_truong` | 3.9 | C3 | draft | 0 | 3 |
| `C3.gop_y_van_ban` | 3.10 | C3 | draft | 0 | 3 |
| `C3.san_pham_truyen_thong` | 3.11 | C3 | draft | 0 | 3 |
| `C3.dai_bieu_duoc_tuyen_chon` | 3.12 | C3 | draft | 0 | 3 |
| `C3.dai_bieu_tu_ung_tuyen` | 3.13 | C3 | draft | 0 | 3 |
| `C4.hien_mau` | 4.2 | C4 | draft | 0 | 3 |
| `C4.tinh_nguyen` | 4.3 | C4 | verified_from_school_list | 6 | 0 |
| `C4.quyen_gop` | 4.5 | C4 | verified_from_school_list | 3 | 0 |
| `C4.tuyen_truyen_phap_luat` | 4.6 | C4 | draft | 0 | 4 |
| `P.sinh_hoat_cong_dan` | 6.4 | P | draft | 0 | 3 |

---

## 3. Excluded Items (`event_matchable: false` — 30 items)

These items are **not** positive event-classification targets. No Stage 2 positive prototypes generated.

| criterion_key | ref | group | kind |
|---|---|---|---|
| `C1.khong_vi_pham_quy_che_dao_tao` | 1.1 | C1 | compliance |
| `C1.diem_trung_binh_hoc_ky` | 1.2 | C1 | record |
| `C1.giu_xep_loai_xuat_sac` | 1.3 | C1 | record |
| `C1.tang_diem_trung_binh` | 1.4 | C1 | record |
| `C1.btc_cuoc_thi_hoc_thuat` | 1.7 | C1 | role_variant |
| `C1.cau_lac_bo_hoc_thuat` | 1.8 | C1 | membership |
| `C2.khong_vi_pham_noi_quy` | 2.1 | C2 | compliance |
| `C3.clb_so_thich_doi_nhom` | 3.4 | C3 | membership |
| `C3.khong_vi_pham_te_nan` | 3.7 | C3 | compliance |
| `C3.btc_ctv_hoat_dong` | 3.14 | C3 | role_variant |
| `C4.khong_vi_pham_phap_luat` | 4.1 | C4 | compliance |
| `C4.btc_ctv_tinh_nguyen` | 4.4 | C4 | role_variant |
| `C4.ban_can_su_lop` | 4.7 | C4 | status |
| `C4.doan_vien_dang_vien_xuat_sac` | 4.8 | C4 | status |
| `C5.khen_thuong_thanh_tich` | 5.1 | C5 | achievement |
| `C5.giai_thuong_cuoc_thi` | 5.2 | C5 | achievement |
| `C5.doi_tuyen_giai_thuong_cap_thanh` | 5.3 | C5 | achievement |
| `C5.bai_bao_quoc_te` | 5.4 | C5 | achievement |
| `C5.bai_bao_quoc_gia` | 5.5 | C5 | achievement |
| `C5.giai_thuong_to_chuc` | 5.6 | C5 | achievement |
| `C5.hoc_bong_vuot_kho` | 5.7 | C5 | status |
| `C5.mo_coi` | 5.8 | C5 | special_circumstance |
| `C5.khuyet_tat` | 5.9 | C5 | special_circumstance |
| `C5.ho_ngheo` | 5.10 | C5 | special_circumstance |
| `C5.mien_giam_hoc_phi` | 5.11 | C5 | special_circumstance |
| `P.khai_bao_khao_sat` | 6.1 | P | penalty |
| `P.ngoai_tru` | 6.2 | P | penalty |
| `P.khong_hoan_thanh_nhiem_vu_can_bo` | 6.3 | P | penalty |
| `P.ky_luat_khien_trach` | 6.5 | P | penalty |
| `P.ky_luat_canh_cao` | 6.6 | P | penalty |

---

## 4. Items with Zero Verified Anchors (all draft)

> [!WARNING]
> These criteria have **only draft anchors** — not verified against `events_full.csv`.
> Classification confidence will be lower for these criteria.

| criterion_key | ref | draft_count |
|---|---|---|
| `C1.de_tai_nckh` | 1.5 | 3 |
| `C3.dai_bieu_duoc_tuyen_chon` | 3.12 | 3 |
| `C3.dai_bieu_tu_ung_tuyen` | 3.13 | 3 |
| `C3.doi_tuyen_cap_truong` | 3.6 | 3 |
| `C3.gop_y_van_ban` | 3.10 | 3 |
| `C3.san_pham_truyen_thong` | 3.11 | 3 |
| `C3.tim_hieu_lich_su_truong` | 3.9 | 3 |
| `C4.hien_mau` | 4.2 | 3 |
| `C4.tuyen_truyen_phap_luat` | 4.6 | 4 |
| `P.sinh_hoat_cong_dan` | 6.4 | 3 |

---

## 5. Special: NONE / NO_MATCH Meta-label

> [!IMPORTANT]
> `events_full.csv` contains **only positive examples** (no verified NONE examples — Q17).
> NONE prototypes use draft anchors from `meta_labels.NONE.anchors` only.
> Do NOT treat these as verified evidence.

---

## 6. Special: 6.4 Mandatory Event (remind)

> [!NOTE]
> `P.sinh_hoat_cong_dan` (ref 6.4) is `event_matchable: true` but `mvp_scope: remind`.
> It is a **mandatory obligation** — participation avoids a penalty, it does NOT earn points.
> Prototypes are generated as `draft_anchor` and clearly marked `mandatory_event / remind`.
> The classifier should route this to a reminder, not a point-earning recommendation.

---

## 7. Classification Boundary (Important)

> [!IMPORTANT]
> **Prototype generation (P2) classifies event text, NOT individual participation.**
> Matching an event to a criterion does NOT prove a student attended, holds a qualifying role,
> or meets duration/round conditions. Role verification (Module 7) and score calculation
> (Module 6) are separate concerns.

---

## 8. Unresolved Issues (Require Human Review)

The following open questions from `rubric.yaml` directly affect prototype quality:

| Q# | Topic | Impact on prototypes |
|---|---|---|
| Q10 | 3.1 vs 3.5 boundary | Ambiguous cuộc thi/quiz cấp khoa can fall in either; NEEDS_REVIEW trigger |
| Q14 | 4.2 (hiến máu) and 4.6 | Only draft anchors; no school-list verification |
| Q15 | 25 items with 0 CSV rows | 1.5, 3.6, 3.9–3.13, 4.2, 4.6, etc. have only draft anchors |
| Q16 | Organizer abbreviations | KHXHNV, U-Sci Hub, CLB BEE/LLT/SV5T level unknown |
| Q17 | No verified NONE examples | NONE prototypes are draft-only; threshold calibration unreliable |
| Q19 | 1.6 ↔ 5.1/5.3/3.6 overlap | ICPC/Olympic appear in multiple items; ambiguous multi-label |
| Q20 | 3.5 vs 4.3 (Chủ nhật xanh) | Boundary unconfirmed; NEEDS_REVIEW for similar events |
| Q23 | 1.9 vs 3.5 boundary | Workshop/talkshow academic vs phong trào still open |

---

## 9. Validation Results

> [!NOTE]
> All validation checks passed.

---

## 10. Source Traceability

Every prototype record contains:
- `criterion_key`: exact item key from `rubric.yaml` items dict
- `criterion_ref`: rubric Phụ lục ref number (e.g. '3.5')
- `source_fields`: which YAML fields the embedding_text was derived from
- `anchor_status`: `verified_from_school_list` | `verified_partial` | `draft` | `derived`

No embedding vectors are included. The `embedding_text` values are plain strings
ready for the multilingual-e5-small `passage:` format.