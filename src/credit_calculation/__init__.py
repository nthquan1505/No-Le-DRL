"""Credit Calculation & Evaluation Module (`src/credit_calculation`)."""

from src.credit_calculation.attribute_extractor import AttributeExtractor
from src.credit_calculation.credit_calculator import (
    CreditCalculator,
    CreditEstimate,
    DEFAULT_MAX_CREDIT,
    SOURCE_RELIABILITY,
    TOTAL_MAX_SCORE,
    credit_calc,
    estimate_event_credit,
    finalize_completed_subscriptions,
    gain_calc,
    gain_calc_with_details,
    normalize_role,
    update_missing_min_credits,
)
from src.credit_calculation.credit_engine import RubricCreditEngine
from src.credit_calculation.interface import (
    CreditCalculatorProtocol,
    LedgerEntry,
    LedgerEvaluation,
    UserLedger,
)

__all__ = [
    "AttributeExtractor",
    "CreditCalculator",
    "CreditCalculatorProtocol",
    "CreditEstimate",
    "DEFAULT_MAX_CREDIT",
    "LedgerEntry",
    "LedgerEvaluation",
    "RubricCreditEngine",
    "SOURCE_RELIABILITY",
    "TOTAL_MAX_SCORE",
    "UserLedger",
    "credit_calc",
    "estimate_event_credit",
    "finalize_completed_subscriptions",
    "gain_calc",
    "gain_calc_with_details",
    "normalize_role",
    "update_missing_min_credits",
]
