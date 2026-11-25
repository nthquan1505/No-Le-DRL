"""Modular Recommendation Engine for No-Le-DRL (Module 3 — Phases 4–7)."""

from src.recommendation.config import (
    DEFAULT_RECOMMENDATION_CONFIG_PATH,
    load_recommendation_config,
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
    "FitAssessment",
    "LocalScheduleAdapter",
    "OnboardingInput",
    "ScheduleProvider",
    "UserProfileManager",
    "UserProfileState",
    "WeeklyBusySlot",
    "detect_campus",
    "load_recommendation_config",
    "parse_ics_schedule",
]
