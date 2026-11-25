"""Public facade for the No-Le-DRL Recommendation subsystem (`src/recommendation`).

Currently exposes Step 1 (Phase 4) modules:
- Configuration: `load_recommendation_config`, `DEFAULT_RECOMMENDATION_CONFIG_PATH`
- Module 4 Schedule Adapter: `ScheduleProvider`, `LocalScheduleAdapter`,
  `WeeklyBusySlot`, `ConflictCheckResult`, `FitAssessment`, `parse_ics_schedule`, `detect_campus`
- P4 User Profile & Cold-Start: `UserProfileManager`, `UserProfileState`,
  `OnboardingInput`, `BehaviorSignal`

Phases 5–6 (`filter`, `scoring`, `selector`, `planner`, `explainer`) will be exported
here as subsequent steps are implemented.
"""

from src.recommendation import (
    DEFAULT_RECOMMENDATION_CONFIG_PATH,
    BehaviorSignal,
    ConflictCheckResult,
    FitAssessment,
    LocalScheduleAdapter,
    OnboardingInput,
    ScheduleProvider,
    UserProfileManager,
    UserProfileState,
    WeeklyBusySlot,
    detect_campus,
    load_recommendation_config,
    parse_ics_schedule,
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
