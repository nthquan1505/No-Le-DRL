# Module 3 Decision Log & Architecture Alignment Registry

> **Status Note**: This file records confirmed architectural decisions (`D-1`–`D-6`), resolved/deferred taxonomy decisions (`T-1`–`T-5`), open design decisions (`D-7`–`D-8`), deferred experimental parameters, and flowchart alignment checks against [`module3_engine_flowchart.drawio.svg`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/module3_engine_flowchart.drawio.svg).
>
> **Implementation Status**: None of the approved Module 3 classifier or recommendation components are implemented or experimentally validated yet ([`src/classifier.py`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/src/classifier.py) and [`src/recommender.py`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/src/recommender.py) remain `0`-byte placeholders).

---

## 1. Decision Summary Registry

| ID | Decision Topic | Status | Summary |
| :--- | :--- | :--- | :--- |
| **D-1** | Classifier Paradigm (`P2`) | **APPROVED** | Embedding-based **KNN architecture** (superseding the earlier v1 Tier-0 keyword / prototype / LLM cascade). |
| **D-2** | Hierarchical Classification Structure (`P2`) | **APPROVED** | **Two-stage hierarchical KNN classifier**: Stage 1 selects Top-k candidate parent groups (`C1`–`C5`); Stage 2 searches over the **union** of retrieved fine-grained datasets across all selected candidate groups and returns `(rubric_item_id, parent_group_id)`. Numerical hyperparameters & exact label taxonomy remain **OPEN**. |
| **D-3** | Classification vs. Credit Eligibility Boundary | **APPROVED** | **Separate classification and credit eligibility**: Module 3 outputs classification and scores only; **Module 6** remains solely responsible for credit eligibility, point calculations, caps, and penalties. |
| **D-4** | `Module 6` Delivery & Integration Format | **APPROVED** | **Shared Python Library** (deterministic rubric evaluation engine shared across Module 3 forecasts and Module 6 evaluation). |
| **D-5** | `P6` Multi-Criteria Scoring Weights | **APPROVED** | **Use `w1` through `w4`** (`Score = w1·Gain + w2·Interest + w3·Fit + w4·Urgency + Penalty`, with `4` weights and additive unweighted `Penalty`, resolving the flowchart edge typo `w₁..w₅`). |
| **D-6** | Event Ingestion & Normalization Ownership | **APPROVED** | **Upstream M1/M2 handle ingestion, normalization, and missing-field handling**; Module 3 consumes their normalized input contract. |
| **D-7** | `P8` LLM Rephrasing & Privacy Policy | **OPEN** | Choose template-only MVP vs. optional guarded LLM rephrasing (API vs. local model). |
| **D-8** | `Q9` Official Evidence Source & Git Hygiene | **OPEN** | Define `Module 7` participation verification source/format (`Q9` in [`data/raw/rubric.yaml`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/data/raw/rubric.yaml#L1324-L1326)) and confirm Git disposition of deleted `Module3_Classification_Recommendation_Design.md` and untracked [`module3_engine_flowchart.drawio.svg`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/module3_engine_flowchart.drawio.svg). |

---

## 2. Approved Architectural Decisions

### D-1: Embedding-Based KNN Architecture — **APPROVED**
- Module 3 classification (`P2`) uses an embedding-based KNN architecture over normalized embeddings (`configs/model_config.py` specifies `intfloat/multilingual-e5-small`, `384d`), rather than the standalone Tier-0 keyword engine ([`res/keyword_rules.py`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/res/keyword_rules.py)) or the v1 3-tier cascade documented in `HEAD`'s deleted `Module3_Classification_Recommendation_Design.md`.

### D-2: Two-Stage Hierarchical KNN Classifier — **APPROVED**
- **Status**: Architecture **APPROVED**; **not yet implemented or experimentally validated**.
- **Intended Architecture**: A **two-stage hierarchical KNN classifier** (distinct from a single flat KNN classifier):
  - **Stage 1 — Coarse Classification**:
    - Use KNN to classify an event into one or more candidate parent groups: `C1`, `C2`, `C3`, `C4`, `C5`.
    - Select the `Top-k` candidate groups based on Stage 1 ranking.
  - **Stage 2 — Fine-Grained Classification**:
    - For **every** selected parent group from Stage 1, retrieve its corresponding fine-grained rubric-item dataset.
    - Run the second KNN over the **union** of these retrieved candidate datasets to identify and rank specific rubric items (e.g., `1.1`, `1.2`, `1.3`, etc., within the confirmed label taxonomy).
    - Return a specific fine-grained rubric-item ID together with its parent group.
- **Mandatory Architectural Requirements**:
  1. Stage 1 selects candidate parent groups; it does **not** directly determine the final rubric item.
  2. Stage 2 must search across **all** selected candidate groups (union of retrieved per-group datasets), not only the highest-ranked group.
  3. Preserve the explicit parent-child relationship between parent group IDs (`C1`–`C5`) and their fine-grained rubric-item IDs.
  4. Keep Stage 1 scores and Stage 2 scores strictly distinguishable in intermediate structures, APIs, and logs.
  5. Do **not** treat raw KNN similarity scores or vote proportions as calibrated probabilities unless calibration has been implemented and experimentally validated.
  6. Do **not** assume Stage 1 confidence and Stage 2 confidence can simply be multiplied (`stage1 * stage2`). Any final score aggregation method must be explicitly defined and evaluated on validation data.
  7. Keep classification confidence separate from credit eligibility; **Module 6** remains solely responsible for determining credit eligibility (`D-3`).

#### Parameters Intentionally Left OPEN (Must Be Resolved via Validation Experiments)
The following parameters are **not** finalized by D-2 and must **not** be assigned hardcoded or arbitrary numerical defaults:
1. **`Top-k` value** for Stage 1 candidate parent-group selection.
2. **Minimum candidate-group score threshold** for Stage 1 inclusion/filtering.
3. **Stage 2 score aggregation and final ranking strategy** across the multi-group union of retrieved candidates.
4. **Final confidence calibration** method and calibrated acceptance/review threshold.
5. **Rejection / `UNKNOWN` / `NONE` policy** (how out-of-scope, commercial/non-ĐRL, or low-confidence events are handled).

### D-3: Separate Classification and Credit Eligibility — **APPROVED**
- Classification confidence (`P2`) and ĐRL credit eligibility (`Module 6`) are strictly decoupled.
- Module 3 classifies events and produces classification scores; **Module 6** remains solely responsible for credit eligibility, point calculation, role/level/duration rule lookups, group caps, penalties, and shortfall evaluation.

### D-4: `Module 6` Shared Python Library — **APPROVED**
- Implement deterministic rubric evaluation (`Module 6` credit calculation logic) as a **Shared Python Library** inside the codebase, usable by Module 3 (expected points / shortfall inputs) and Module 6 evaluation workflows without invoking LLMs for numeric calculation.

### D-5: `P6` Multi-Criteria Scoring Weights (`w1`–`w4`) — **APPROVED**
- Multi-criteria scoring uses **four** weights `w1` through `w4` with additive unweighted `Penalty`:
  `Score = w1·Gain + w2·Interest + w3·Fit + w4·Urgency + Penalty`.
- Resolves the internal discrepancy in [`module3_engine_flowchart.drawio.svg`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/module3_engine_flowchart.drawio.svg) where the `P4` and `P6` boxes state `w₁..w₄` while edge `P4 -> P6` states `w₁..w₅`.

### D-6: Upstream Ingestion & Normalization Boundary — **APPROVED**
- Upstream **M1 & M2** handle crawling, entity extraction, normalization, and missing-field handling.
- Module 3 consumes their normalized input contract rather than duplicating raw scraping/post-parsing logic in production. *(Note: Local normalization fixtures/adapters remain necessary strictly for offline testing against the legacy snapshot [`data/events.json`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/data/events.json).)*

---

## 3. Label Taxonomy Decisions (`T-1`–`T-5` — Resolved / Deferred)

Inspection of [`data/raw/rubric.yaml`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/data/raw/rubric.yaml), [`data/raw/events_dataset.csv`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/data/raw/events_dataset.csv), [`res/events_full.csv`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/res/events_full.csv), and [`res/drl2025.md`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/res/drl2025.md) surfaced five taxonomy questions (`T-1`–`T-5`), which have now been decided as follows:

### Current Verified Rubric Breakdown (`data/raw/rubric.yaml`, `version: khtn-qd1028-2020-v3.1-mvp`)
- **Total defined items**: `51` items (`45` items across `C1`–`C5` + `6` items in `P`), plus `1` meta-label (`meta_labels.NONE`).
- **Event-matchability breakdown in `rubric.yaml`**:

| Group | Total Items | `event_matchable: true` | `event_matchable: false` (Compliance / Record / Role Variant / Membership / Status / Achievement / Special / Penalty) |
| :--- | :---: | :---: | :--- |
| **`C1`** | `9` (`1.1`–`1.9`) | `3` (`1.5`, `1.6`, `1.9`) | `6` (`1.1` compliance, `1.2`–`1.4` GPA records, `1.7` role variant, `1.8` membership) |
| **`C2`** | `3` (`2.1`–`2.3`) | `2` (`2.2`, `2.3`) | `1` (`2.1` compliance) |
| **`C3`** | `14` (`3.1`–`3.14`) | `11` (`3.1`–`3.3`, `3.5`–`3.6`, `3.8`–`3.13`) | `3` (`3.4` membership, `3.7` compliance, `3.14` role variant) |
| **`C4`** | `8` (`4.1`–`4.8`) | `4` (`4.2`, `4.3`, `4.5`, `4.6`) | `4` (`4.1` compliance, `4.4` role variant, `4.7`–`4.8` status) |
| **`C5`** | `11` (`5.1`–`5.11`) | **`0`** | `11` (`5.1`–`5.6` achievements, `5.7` status, `5.8`–`5.11` sensitive circumstances) |
| **`P`** | `6` (`6.1`–`6.6`) | `1` (`6.4` / `P.sinh_hoat_cong_dan`, `mandatory_event`) | `5` (`6.1`–`6.3`, `6.5`–`6.6` penalties) |
| **`meta_labels`** | `1` (`NONE`) | — | Marked special classifier exclusion label (`NONE`) in `rubric.yaml` |

### Confirmed Taxonomy Decisions (`T-1`–`T-5`)

| ID | Taxonomy Topic | Status | Decision & Implementation Impact |
| :--- | :--- | :--- | :--- |
| **T-1** | **Full `51`-Item Taxonomy vs. `21` `event_matchable` Subset** | **APPROVED** | Classify across **all `51` rubric items** (full taxonomy, not restricted to the `21` `event_matchable` subset). Current CSVs (`events_dataset.csv`, `events_full.csv`) are **demo datasets only** while awaiting the full dataset; `ClsIndex` and `P2` must be data-driven so dropping in the full dataset requires no code changes. *(Note: `C1`–`C5` contain `45` items and `P` contains `6` items `6.1`–`6.6`; see `T-2` regarding Module 1 handling of `P`/`6.4`.)* |
| **T-2** | **Handling of Group `P` / Mandatory Item `6.4` (`P.sinh_hoat_cong_dan`)** | **APPROVED** | Handled upstream by **Module 1**. Module 3 Stage 1 focuses on parent groups `C1`–`C5` (or accepts whatever parent-group partitions exist in the supplied full dataset). |
| **T-3** | **Inclusion of `NONE` / `UNKNOWN` Special Labels** | **APPROVED** | Handled upstream by **Module 1** (non-event / irrelevant post filtering occurs before Module 3). Module 3 `P2` does not need a dedicated `NONE` training class in `ClsIndex`, though confidence-based low-confidence review gating remains configurable per `D-2`. |
| **T-4** | **Canonical Parent-Child ID Schema & Role-Variant Separation** | **APPROVED** | Canonical fine-grained item ID format is **`<major>.<minor>`** (matching `ref` in `rubric.yaml` and `Ma_Muc` in CSV, e.g., `"1.6"`, `"1.7"`, `"3.5"`), mapped to parent group `C1`–`C5` (e.g., `"1.6" -> "C1"`). Module 3 **only classifies rubric items** (including role-distinguished items like `1.6` vs `1.7`); all credit calculation and scoring belong strictly to `Module 6` (`D-3`). *(Note: to avoid title-only collisions between role variants like `1.6`/`1.7`, input text to `P2` should include description/role context when available.)* |
| **T-5** | **Item `3.5` Scope & Regulation Currency (`QĐ 1028` vs. Record Practice)** | **DEFERRED** | **Temporarily skipped** for the current MVP / demo phase. Keep `rubric.yaml` and dataset labels as-is for now; revisit when full dataset and active regulation alignment are needed. |

---

## 4. Read-Only Flowchart Alignment Audit (`module3_engine_flowchart.drawio.svg`)

Read-only inspection of [`module3_engine_flowchart.drawio.svg`](file:///Users/Apple/Documents/Computational%20Thinking/No-Le-DRL/module3_engine_flowchart.drawio.svg) identified **six** exact inconsistencies with the approved decisions (`D-1`–`D-6`) and unresolved parameters/taxonomy. *Per constraints, the SVG was **not** modified; the table below documents required future updates:*

| Flowchart Cell / Edge | Current Flowchart Content | Conflict / Gap vs. Approved Decisions | Proposed Flowchart Change (For Human Review) |
| :--- | :--- | :--- | :--- |
| **`P2.2`** (`aakRZ4Ocrra6V0yE2CO7-20`) & **Edge `P2.2 -> P2.3`** (`aakRZ4Ocrra6V0yE2CO7-40`) | `Lọc các nhóm ứng viên xác suất cao` and `Top nhóm (P > ~40%)` | Treats KNN group scores as probabilities (`xác suất cao`, `P`) and hardcodes `~40%`, conflicting with **D-2** (Requirement 5: raw scores != calibrated probabilities; `Top-k` & min-score threshold remain **OPEN**). | Change to: `Chọn Top-k nhóm ứng viên` (or `Score >= threshold (TBD)`), dropping unvalidated `P > ~40%` and probability phrasing. |
| **`P2.3`** (`aakRZ4Ocrra6V0yE2CO7-21`), **`ClsIndex`** (`aakRZ4Ocrra6V0yE2CO7-17`), & **Edge `P2.3 -> Pmax`** (`aakRZ4Ocrra6V0yE2CO7-41`) | `Chỉ tìm trong ứng viên Bước 1 | Phân loại mục nhỏ`, single flat `ClsIndex`, edge labeled `Xác xuất` | Does not explicitly show **per-group dataset retrieval + KNN over the union of all selected Stage 1 groups**, does not distinguish Stage 1 vs. Stage 2 scores, and mislabels raw KNN outputs as `Xác xuất` (also misspelled "Xác suất"). | Update `ClsIndex` to show coarse group index + per-group fine datasets; update `P2.3` to `Truy xuất & hợp nhất tập mục nhỏ của mọi nhóm Top-k -> KNN`; change edge label to `Điểm Stage 1 & Stage 2` (or `Ranked items + raw scores`). |
| **Decision Diamond** (`aakRZ4Ocrra6V0yE2CO7-22`) | `Pmax ≥ 0.70 (<0.70: chờ duyệt)` | Hardcodes `0.70` and assumes calibrated `Pmax`, plus omits `NONE`/`UNKNOWN` policy—all **OPEN** under **D-2**. | Replace with `Calibration & Acceptance / Rejection Gate (ngưỡng TBD)` referencing open calibration & rejection policy. |
| **Edge `P4 -> P6`** (`aakRZ4Ocrra6V0yE2CO7-62`) | `user_vec, tags, w₁..w₅` | Uses `w₁..w₅`, conflicting with **D-5 (APPROVED)** and the `P4`/`P6` boxes (`w₁..w₄` + unweighted `Penalty`). | Change edge label from `w₁..w₅` to `w₁..w₄`. |
| **`M1 & M2` / `API SYNC`** (`-12`, `-14`) & **`Edge -> P2.1`** (`-37`) | Shows `EventRaw` -> `Common DB` -> `API SYNC` -> `Event` | Does not explicitly reflect **D-6 (APPROVED)** (upstream M1/M2 own ingestion, normalization, and missing-field handling). | Annotate `M1 & M2` / `API SYNC` output explicitly as `Normalized Event Contract (D-6)`. |
| **`MODULE 6`** (`aakRZ4Ocrra6V0yE2CO7-23`) | Styled as `EXTERNAL`: `Tính toán điểm thực nhận...` | Aligns with **D-3** (separate credit eligibility), but visually marks it strictly external despite **D-4 (APPROVED: Shared Python Library)** and mentions `thực nhận` vs forecast `điểm dự kiến`. | Clarify label as `MODULE 6 (Shared Lib): Credit Eligibility & Points Calc` without weakening the D-3 separation boundary. |
