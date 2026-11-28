# Module 6 — Credit Calc & Evaluation (Replaceable Local Implementation)

> **Isolation Notice**: This directory is intentionally kept separate from `src/` (Module 3 core) so that it can be swapped out or replaced cleanly once the dedicated **Module 6** implementation is delivered.

## Contract Boundary (D-3 & D-4)
- **Strict Separation (`D-3`)**: Module 3 (`src/classifier.py`) predicts canonical `<major>.<minor>` rubric item IDs (e.g., `"1.6"`, `"3.5"`). Module 6 owns all DRL credit eligibility checks, point lookups, group caps (`30/20/25/25/30`), penalties (`P`), floor (`0`), total cap (`100`), and shortfall calculation.
- **Stable Interface**: Any replacement for this folder only needs to implement `CreditCalculatorProtocol` in [`interface.py`](./interface.py):
  - `estimate_event_credit(...) -> CreditEstimate`
  - `compute_group_shortfall(...) -> GroupShortfall`
  - `evaluate_ledger(...) -> LedgerEvaluation`
