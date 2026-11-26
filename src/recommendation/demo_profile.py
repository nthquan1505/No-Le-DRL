"""Lightweight Demo User Profile (P4) for testing P5 Hard Filter and P6 Scoring.

Intentionally omits online Rocchio adaptation and dynamic weight learning so teams
can deterministically inspect how simple interests, hard constraints, Demo M4/M6
settings, and static w1..w4 weights affect filtering and ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.embedder import BaseEmbedder
from src.recommendation.config import load_recommendation_config
from src.recommendation.demo_module4 import DEMO_SCHEDULE_PRESETS
from src.recommendation.demo_module6 import DEMO_CREDIT_PRESETS
from src.schemas import EventNormalized, ScoringWeights, TargetAudienceStructured, UserProfile
from src.vector_ops import l2_normalize

DEFAULT_TZ = timezone(timedelta(hours=7))


@dataclass
class DemoUserProfile:
    """Minimal demo student profile exercising every P5 hard filter & P6 scoring axis."""

    user_id: str = "demo_sv_01"
    cohort: Optional[str] = "K24"
    faculty: Optional[str] = "CNTT"
    class_id: Optional[str] = None

    # Simple interest declaration (static e5-small vector, no online Rocchio)
    interest_chips: List[str] = field(default_factory=lambda: ["ai_tech", "academic_nckh"])
    free_text: Optional[str] = None
    user_vec: Optional[List[float]] = None

    # Hard & soft personal constraints
    home_campus: Optional[str] = "cs1_nvc"  # "cs1_nvc" | "cs2_lt" | None
    max_travel_min: Optional[int] = 45
    strict_max_travel: bool = True
    max_budget_vnd: Optional[int] = 30000
    pref_hours: Tuple[float, float] = (8.0, 21.0)
    participated_event_ids: List[str] = field(default_factory=list)
    hidden_event_ids: List[str] = field(default_factory=list)

    # Demo Module 6 state
    credit_preset: str = "mid_semester"
    current_by_group: Dict[str, float] = field(
        default_factory=lambda: {"C1": 22.0, "C2": 15.0, "C3": 10.0, "C4": 10.0, "C5": 0.0}
    )
    target_total: float = 85.0
    target_by_group: Dict[str, float] = field(
        default_factory=lambda: {"C1": 25.0, "C2": 18.0, "C3": 22.0, "C4": 20.0, "C5": 0.0}
    )

    # Demo Module 4 state
    schedule_preset: str = "light_schedule"

    # Scoring weights (static preset or custom w1..w4)
    weight_preset: str = "balanced"
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Evaluation clock for historical fixtures
    reference_now: str = "2026-10-01T08:00:00+07:00"

    def to_user_profile(self) -> UserProfile:
        return UserProfile(
            user_id=self.user_id,
            cohort=self.cohort,
            faculty=self.faculty,
            class_id=self.class_id,
            interest_tags=list(self.interest_chips),
            user_vec=list(self.user_vec) if self.user_vec is not None else None,
            target_total=float(self.target_total),
            target_by_group=dict(self.target_by_group),
            max_budget_vnd=self.max_budget_vnd,
            max_travel_min=self.max_travel_min,
            hidden_event_ids=list(self.hidden_event_ids),
            participated_event_ids=list(self.participated_event_ids),
            saved_event_ids=[],
            weights=self.weights,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "cohort": self.cohort,
            "faculty": self.faculty,
            "class_id": self.class_id,
            "interest_chips": list(self.interest_chips),
            "free_text": self.free_text,
            "has_user_vec": self.user_vec is not None,
            "home_campus": self.home_campus,
            "max_travel_min": self.max_travel_min,
            "strict_max_travel": self.strict_max_travel,
            "max_budget_vnd": self.max_budget_vnd,
            "pref_hours": [float(self.pref_hours[0]), float(self.pref_hours[1])],
            "participated_event_ids": list(self.participated_event_ids),
            "hidden_event_ids": list(self.hidden_event_ids),
            "credit_preset": self.credit_preset,
            "current_by_group": dict(self.current_by_group),
            "target_total": float(self.target_total),
            "target_by_group": dict(self.target_by_group),
            "schedule_preset": self.schedule_preset,
            "weight_preset": self.weight_preset,
            "weights": self.weights.to_dict(),
            "reference_now": self.reference_now,
        }


def resolve_demo_weights(
    weight_preset: str = "balanced",
    custom_weights: Optional[Dict[str, float]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, ScoringWeights]:
    """Resolve and strictly normalize w1..w4 (sum == 1.0, finite >= 0)."""
    cfg = config if config is not None else load_recommendation_config()
    presets = cfg.get("weight_presets") or {}

    if weight_preset == "custom" and custom_weights:
        raw = custom_weights
        name = "custom"
    else:
        if weight_preset not in presets:
            raise ValueError(
                f"Unknown weight_preset '{weight_preset}'. Available: {sorted(presets.keys())}"
            )
        raw = presets[weight_preset]["weights"]
        name = weight_preset

    vals = [
        max(0.0, float(raw.get("w1_gain", 0.40))),
        max(0.0, float(raw.get("w2_interest", 0.25))),
        max(0.0, float(raw.get("w3_fit", 0.20))),
        max(0.0, float(raw.get("w4_urgency", 0.15))),
    ]
    total = sum(vals)
    if total <= 0.0:
        vals = [0.40, 0.25, 0.20, 0.15]
        total = 1.0
    norm = [round(v / total, 6) for v in vals]
    # Exact sum adjustment on largest coordinate
    norm[0] = round(norm[0] + (1.0 - sum(norm)), 6)
    return name, ScoringWeights(
        w1_gain=norm[0],
        w2_interest=norm[1],
        w3_fit=norm[2],
        w4_urgency=norm[3],
    )


def build_demo_user_vector(
    interest_chips: Sequence[str],
    free_text: Optional[str],
    embedder: BaseEmbedder,
    *,
    config: Optional[Dict[str, Any]] = None,
    chip_vec_cache: Optional[Dict[str, List[float]]] = None,
) -> Optional[List[float]]:
    """Build static L2-normalized 384d E5 interest vector from selected chips + optional text."""
    cfg = config if config is not None else load_recommendation_config()
    chips_cfg = cfg.get("interest_chips") or {}

    unique_chips = [
        c for c in dict.fromkeys(str(x).strip() for x in interest_chips if str(x).strip())
        if c in chips_cfg
    ]
    clean_text = (free_text or "").strip()
    if not unique_chips and not clean_text:
        return None

    cache = chip_vec_cache if chip_vec_cache is not None else {}
    missing = [cid for cid in unique_chips if cid not in cache]
    if missing:
        texts = [str(chips_cfg[cid]["text"]) for cid in missing]
        encoded = embedder.encode_queries(texts)
        for cid, vec in zip(missing, encoded):
            cache[cid] = l2_normalize(vec)

    vecs: List[List[float]] = [cache[cid] for cid in unique_chips]
    if clean_text:
        vecs.append(l2_normalize(embedder.encode_query(clean_text)))

    dim = len(vecs[0])
    summed = [0.0] * dim
    for v in vecs:
        for i in range(dim):
            summed[i] += v[i]
    return l2_normalize(summed)


DEMO_PERSONAS: Dict[str, Dict[str, Any]] = {
    "balanced_cs1": {
        "id": "balanced_cs1",
        "label": "SV CNTT CS1 · Cân bằng",
        "description": "Học tại CS1, thích AI & Học thuật, ngân sách ≤30k, thiếu điểm C3/C4.",
        "cohort": "K24",
        "faculty": "CNTT",
        "interest_chips": ["ai_tech", "academic_nckh"],
        "free_text": "thích hội thảo AI, lập trình, seminar chuyên môn",
        "home_campus": "cs1_nvc",
        "max_travel_min": 45,
        "strict_max_travel": True,
        "max_budget_vnd": 30000,
        "pref_hours": [8.0, 21.0],
        "credit_preset": "mid_semester",
        "current_by_group": dict(DEMO_CREDIT_PRESETS["mid_semester"]["current_by_group"]),
        "target_total": 85.0,
        "target_by_group": dict(DEMO_CREDIT_PRESETS["mid_semester"]["target_by_group"]),
        "schedule_preset": "light_schedule",
        "weight_preset": "balanced",
        "reference_now": "2026-10-01T08:00:00+07:00",
        "include_mandatory_demo": False,
    },
    "c1_capped_need_points": {
        "id": "c1_capped_need_points",
        "label": "Chạm trần C1 (30/30) · Cần điểm",
        "description": "C1 đã chạm trần 30/30đ -> P5 loại toàn bộ nhóm C1, đẩy C3/C4 lên đầu.",
        "cohort": "K24",
        "faculty": "CNTT",
        "interest_chips": ["ai_tech", "volunteering"],
        "free_text": "",
        "home_campus": "cs1_nvc",
        "max_travel_min": 45,
        "strict_max_travel": False,
        "max_budget_vnd": 50000,
        "pref_hours": [8.0, 21.0],
        "credit_preset": "c1_capped",
        "current_by_group": dict(DEMO_CREDIT_PRESETS["c1_capped"]["current_by_group"]),
        "target_total": 85.0,
        "target_by_group": dict(DEMO_CREDIT_PRESETS["c1_capped"]["target_by_group"]),
        "schedule_preset": "light_schedule",
        "weight_preset": "need_points",
        "reference_now": "2026-04-01T08:00:00+07:00",
        "include_mandatory_demo": False,
    },
    "exam_week_busy": {
        "id": "exam_week_busy",
        "label": "Tuần thi · Bận cuối tuần · 0đ",
        "description": "Đang tuần thi 09/11–22/11, trùng lịch thi 16/11 & chiều T7, chỉ đi sự kiện miễn phí.",
        "cohort": "K24",
        "faculty": "CNTT",
        "interest_chips": ["soft_skills", "english_global"],
        "free_text": "",
        "home_campus": "cs1_nvc",
        "max_travel_min": 30,
        "strict_max_travel": True,
        "max_budget_vnd": 0,
        "pref_hours": [8.0, 21.0],
        "credit_preset": "mid_semester",
        "current_by_group": dict(DEMO_CREDIT_PRESETS["mid_semester"]["current_by_group"]),
        "target_total": 85.0,
        "target_by_group": dict(DEMO_CREDIT_PRESETS["mid_semester"]["target_by_group"]),
        "schedule_preset": "exam_week_heavy",
        "weight_preset": "exam_week",
        "reference_now": "2026-11-01T08:00:00+07:00",
        "include_mandatory_demo": False,
    },
    "mandatory_shcd_test": {
        "id": "mandatory_shcd_test",
        "label": "Test Bắt buộc (6.4 Tuần SHCD)",
        "description": "Chèn sự kiện bắt buộc 6.4 (0đ cộng, bỏ lỡ trừ 20đ) -> kiểm chứng P5 giữ lại & P6 Penalty đẩy lên #1.",
        "cohort": "K24",
        "faculty": "CNTT",
        "interest_chips": ["ai_tech"],
        "free_text": "chỉ thích lập trình và AI",
        "home_campus": "cs1_nvc",
        "max_travel_min": 45,
        "strict_max_travel": True,
        "max_budget_vnd": 30000,
        "pref_hours": [8.0, 21.0],
        "credit_preset": "mid_semester",
        "current_by_group": dict(DEMO_CREDIT_PRESETS["mid_semester"]["current_by_group"]),
        "target_total": 85.0,
        "target_by_group": dict(DEMO_CREDIT_PRESETS["mid_semester"]["target_by_group"]),
        "schedule_preset": "light_schedule",
        "weight_preset": "balanced",
        "reference_now": "2026-11-01T08:00:00+07:00",
        "include_mandatory_demo": True,
    },
}


def available_demo_personas() -> List[Dict[str, Any]]:
    return [dict(v) for v in DEMO_PERSONAS.values()]


def create_mandatory_shcd_demo_event(reference_now: datetime) -> Tuple[EventNormalized, Dict[str, Any]]:
    """Create a realistic synthetic 6.4 mandatory SHCD event anchored near `reference_now`."""
    ref = reference_now if reference_now.tzinfo is not None else reference_now.replace(tzinfo=DEFAULT_TZ)
    ref_local = ref.astimezone(DEFAULT_TZ)
    # Pick the next Thu (3) or Fri (4) at least 2 days after reference_now (14:00-16:30)
    # so it does not collide with standard demo presets (Mon/Wed morning or Sat/Sun).
    start_day = (ref_local + timedelta(days=2)).date()
    while start_day.weekday() not in (3, 4):
        start_day += timedelta(days=1)
    start_dt = datetime(start_day.year, start_day.month, start_day.day, 14, 0, tzinfo=DEFAULT_TZ)
    end_dt = datetime(start_day.year, start_day.month, start_day.day, 16, 30, tzinfo=DEFAULT_TZ)
    reg_day = max(ref_local.date(), start_day - timedelta(days=1))
    reg_end_dt = datetime(reg_day.year, reg_day.month, reg_day.day, 23, 59, tzinfo=DEFAULT_TZ)

    event = EventNormalized(
        event_id="demo_mandatory_6_4",
        title="[BẮT BUỘC] Tuần Sinh hoạt Công dân Sinh viên đầu năm học 2026–2027",
        description=(
            "Chương trình sinh hoạt công dân bắt buộc do Phòng Công tác Sinh viên tổ chức. "
            "Sinh viên không tham gia đầy đủ hoặc không hoàn thành bài thu hoạch sẽ bị trừ 10–20 điểm rèn luyện theo quy chế."
        ),
        organizer="Phòng Công tác Sinh viên HCMUS",
        start_datetime=start_dt.isoformat(),
        end_datetime=end_dt.isoformat(),
        reg_start_datetime=ref_local.isoformat(),
        reg_end_datetime=reg_end_dt.isoformat(),
        online_mode="offline",
        is_online=False,
        location="Giảng đường 1 - Trường ĐH Khoa học tự nhiên, ĐHQG-HCM (227 Nguyễn Văn Cừ, Cơ sở 1)",
        fee_vnd=0,
        fee_raw="Miễn phí",
        target_audience=TargetAudienceStructured(
            raw_texts=["Toàn thể sinh viên chính quy HCMUS"],
            all_students=True,
        ),
        source_type="demo_fixture",
        categories=["Mandatory", "Orientation"],
        metadata={
            "is_mandatory": True,
            "is_demo_synthetic": True,
            "rubric_item_id": "6.4",
            "penalty_if_missed_points": -20.0,
        },
    )

    classification_stub = {
        "status": "PREDICTED",
        "predicted_group_id": "P",
        "predicted_criterion_ref": "6.4",
        "predicted_criterion_key": "P.sinh_hoat_cong_dan",
        "predicted_criterion_name": "Không tham gia đầy đủ tuần sinh hoạt công dân đầu khóa, đầu năm học và cuối khóa",
    }
    return event, classification_stub
