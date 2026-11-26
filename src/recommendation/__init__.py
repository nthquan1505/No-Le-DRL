"""Modular Recommendation Engine for No-Le-DRL (Module 3 — Phases 4–5)."""

from src.recommendation.config import (
    DEFAULT_RECOMMENDATION_CONFIG_PATH,
    load_recommendation_config,
)
from src.recommendation.demo_module4 import (
    DEMO_SCHEDULE_PRESETS,
    DemoFitAssessment,
    DemoScheduleProvider,
)
from src.recommendation.demo_module6 import (
    DEMO_CREDIT_PRESETS,
    DemoCreditProvider,
    MandatoryPenaltyInfo,
)
from src.recommendation.demo_profile import (
    DEMO_PERSONAS,
    DemoUserProfile,
    available_demo_personas,
    build_demo_user_vector,
    create_mandatory_shcd_demo_event,
    resolve_demo_weights,
)
from src.recommendation.filter import (
    DroppedCandidate,
    EligibleCandidate,
    FilterReport,
    HardConstraintFilter,
    RecommendationCandidateInput,
    parse_reference_now,
    resolve_effective_deadline,
)
from src.recommendation.pipeline import (
    load_recommendation_candidate_pool,
    run_recommendation_demo,
)
from src.recommendation.schedule_adapter import (
    ConflictCheckResult,
    FitAssessment,
    LocalScheduleAdapter,
    ScheduleProvider,
    WeeklyBusySlot,
    detect_campus,
    parse_ics_schedule,
)
from src.recommendation.scoring import (
    InterestCalibrationRange,
    MultiCriteriaScorer,
    ScoredCandidate,
    compute_interest_calibration_range,
    scale_interest_clipped_minmax,
)
from src.recommendation.user_profile import (
    BehaviorSignal,
    OnboardingInput,
    UserProfileManager,
    UserProfileState,
)

__all__ = [
    "BehaviorSignal",
    "ConflictCheckResult",
    "DEFAULT_RECOMMENDATION_CONFIG_PATH",
    "DEMO_CREDIT_PRESETS",
    "DEMO_PERSONAS",
    "DEMO_SCHEDULE_PRESETS",
    "DemoCreditProvider",
    "DemoFitAssessment",
    "DemoScheduleProvider",
    "DemoUserProfile",
    "DroppedCandidate",
    "EligibleCandidate",
    "FilterReport",
    "FitAssessment",
    "HardConstraintFilter",
    "InterestCalibrationRange",
    "LocalScheduleAdapter",
    "MandatoryPenaltyInfo",
    "MultiCriteriaScorer",
    "OnboardingInput",
    "RecommendationCandidateInput",
    "ScheduleProvider",
    "ScoredCandidate",
    "UserProfileManager",
    "UserProfileState",
    "WeeklyBusySlot",
    "available_demo_personas",
    "build_demo_user_vector",
    "compute_interest_calibration_range",
    "create_mandatory_shcd_demo_event",
    "detect_campus",
    "load_recommendation_candidate_pool",
    "load_recommendation_config",
    "parse_ics_schedule",
    "parse_reference_now",
    "resolve_demo_weights",
    "resolve_effective_deadline",
    "run_recommendation_demo",
    "scale_interest_clipped_minmax",
]
