"""Unit and SQLite integration tests for `[CRD]` (`src/utility/credit_calculator.py`)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from database import DRLDatabaseManager
from src.utility.credit_calculator import (
    CreditCalculator,
    CreditEstimate,
    credit_calc,
    estimate_event_credit,
    finalize_completed_subscriptions,
    gain_calc,
    gain_calc_with_details,
    normalize_role,
    update_missing_min_credits,
)


class TestCreditCalcAndRoles(unittest.TestCase):
    """Verify deterministic `credit_calc` across event types, attributes, and user roles."""

    def test_role_normalization(self) -> None:
        self.assertEqual(normalize_role(None), "participant")
        self.assertEqual(normalize_role("participant"), "participant")
        self.assertEqual(normalize_role("Tình nguyện viên"), "participant")
        self.assertEqual(normalize_role("host"), "host")
        self.assertEqual(normalize_role("organizer"), "host")
        self.assertEqual(normalize_role("Ban tổ chức"), "host")
        self.assertEqual(normalize_role("BTC"), "host")
        self.assertEqual(normalize_role("collaborator"), "collaborator")
        self.assertEqual(normalize_role("Cộng tác viên"), "collaborator")
        self.assertEqual(normalize_role("CTV"), "collaborator")

    def test_academic_contest_roles_1_6_and_1_7(self) -> None:
        ev = {
            "event_id": 101,
            "title": "Cuộc thi Học thuật Lập trình ICPC / AI Challenge 2026",
            "description": "Cuộc thi học thuật dành cho sinh viên toàn trường.",
            "topic": ["học thuật", "cuộc thi"],
            "format": ["thi", "đồng đội"],
            "category": 1,
        }
        self.assertEqual(credit_calc(ev, role="participant"), 5)
        self.assertEqual(credit_calc(ev, role="host"), 10)
        self.assertEqual(credit_calc(ev, role="collaborator"), 5)

        est_host = estimate_event_credit(ev, role="host")
        self.assertEqual(est_host.rubric_item_id, "1.7")
        self.assertEqual(est_host.source, "rule")

    def test_academic_workshop_1_9_roles(self) -> None:
        ev = {
            "event_id": 102,
            "title": "Workshop Viết và Chấm điểm CV - Chinh phục nhà tuyển dụng",
            "description": "Buổi hội thảo kỹ năng nghề nghiệp dành cho sinh viên.",
            "topic": ["workshop", "hướng nghiệp"],
            "category": 1,
        }
        self.assertEqual(credit_calc(ev, role="participant"), 5)
        self.assertEqual(credit_calc(ev, role="host"), 5)
        self.assertEqual(credit_calc(ev, role="collaborator"), 5)

    def test_category_3_level_and_role_redirection_3_1_and_3_14(self) -> None:
        ev_school = {
            "event_id": 103,
            "title": "Hội thi Olympic các môn Khoa học Mác - Lênin cấp Trường",
            "description": "Cuộc thi chính trị do Đoàn Trường tổ chức.",
            "sources": "organizer: Đoàn Trường ĐH Khoa học Tự nhiên",
            "category": 3,
        }
        self.assertEqual(credit_calc(ev_school, role="participant"), 10)
        self.assertEqual(credit_calc(ev_school, role="host"), 10)
        self.assertEqual(credit_calc(ev_school, role="collaborator"), 5)

        ev_faculty = {
            "event_id": 104,
            "title": "Giải Thể thao Điện tử Khoa Công nghệ Thông tin",
            "description": "Giải đấu cấp Khoa do Đoàn Khoa tổ chức.",
            "sources": "organizer: Đoàn Khoa Công nghệ thông tin",
            "category": 3,
        }
        self.assertEqual(credit_calc(ev_faculty, role="participant"), 5)
        self.assertEqual(credit_calc(ev_faculty, role="host"), 5)
        self.assertEqual(credit_calc(ev_faculty, role="collaborator"), 3)

        ev_class = {
            "event_id": 105,
            "title": "Sinh hoạt văn nghệ Chi đoàn 25CTT1",
            "description": "Hoạt động giao lưu cấp lớp do Chi đoàn tổ chức.",
            "sources": "organizer: Chi đoàn 25CTT1",
            "category": 3,
        }
        self.assertEqual(credit_calc(ev_class, role="participant"), 5)
        self.assertEqual(credit_calc(ev_class, role="host"), 5)
        # Quy che 1028 does not grant CTV points at class ('lop') level -> 0
        self.assertEqual(credit_calc(ev_class, role="collaborator"), 0)

        # Faculty-level "Liên chi Hội" must resolve to 'khoa' (CTV = 3), not 'lop'
        ev_lch = {
            "event_id": 1051,
            "title": "Lễ kỷ niệm Ngày Nhà giáo Việt Nam - Tri ân Thầy Cô tại ĐH Khoa học Tự nhiên",
            "description": "Chương trình do Liên chi Hội Khoa Vật lý và Đoàn Khoa Sinh học tổ chức.",
        }
        est_lch_c = estimate_event_credit(ev_lch, role="collaborator")
        self.assertEqual(est_lch_c.category, 3)
        self.assertEqual(est_lch_c.inferred_level, "khoa")
        self.assertEqual(est_lch_c.credit, 3)

    def test_volunteering_duration_and_roles_4_3_and_4_4(self) -> None:
        ev_short = {
            "event_id": 106,
            "title": "Ngày Chủ nhật xanh",
            "description": "Hoạt động tình nguyện dọn vệ sinh khuôn viên trong 4 giờ.",
            "event_start_datetime": "2026-11-22T07:30:00+07:00",
            "event_end_datetime": "2026-11-22T11:30:00+07:00",
            "sources": "organizer: Liên chi Hội Khoa Hóa học",
            "category": 4,
        }
        self.assertEqual(credit_calc(ev_short, role="participant"), 5)
        self.assertEqual(credit_calc(ev_short, role="host"), 5)
        self.assertEqual(credit_calc(ev_short, role="collaborator"), 3)

        ev_mid = {
            "event_id": 107,
            "title": "BƯỚC CHÂN YÊU THƯƠNG",
            "description": "Chương trình tình nguyện hè kéo dài 2 ngày 1 đêm tại Vĩnh Long.",
            "event_start_datetime": "2026-05-23T04:00:00+07:00",
            "event_end_datetime": "2026-05-24T20:30:00+07:00",
            "sources": "organizer: Hội Sinh viên Trường",
            "category": 4,
        }
        self.assertEqual(credit_calc(ev_mid, role="participant"), 10)
        self.assertEqual(credit_calc(ev_mid, role="host"), 10)
        self.assertEqual(credit_calc(ev_mid, role="collaborator"), 5)

        ev_long = {
            "event_id": 108,
            "title": "Chiến dịch Tình nguyện Mùa Hè Xanh 2026",
            "description": "Chiến dịch tình nguyện dài hạn 3 tuần.",
            "category": 4,
        }
        self.assertEqual(credit_calc(ev_long, role="participant"), 15)

    def test_explicit_points_and_negation(self) -> None:
        ev_explicit = {
            "event_id": 109,
            "title": "Tọa đàm Kỹ năng mềm đặc biệt",
            "description": "Sinh viên tham dự đầy đủ được cộng 8 điểm rèn luyện. BTC: +12 ĐRL, CTV: +6 ĐRL.",
            "award": "+8 điểm rèn luyện",
            "category": 1,
        }
        est_p = estimate_event_credit(ev_explicit, role="participant")
        self.assertEqual(est_p.source, "explicit")
        self.assertEqual(est_p.credit, 8)
        self.assertEqual(credit_calc(ev_explicit, role="host"), 12)
        self.assertEqual(credit_calc(ev_explicit, role="collaborator"), 6)

        ev_negated = {
            "event_id": 110,
            "title": "Đêm nhạc giao lưu tự do",
            "description": "Lưu ý: chương trình này không cộng điểm rèn luyện.",
            "category": 3,
        }
        est_neg = estimate_event_credit(ev_negated, role="participant")
        self.assertFalse(est_neg.is_credit_eligible)
        self.assertEqual(est_neg.source, "explicit")
        self.assertEqual(credit_calc(ev_negated, role="participant"), 0)

    def test_mandatory_shcd_6_4_zero_bonus_credit(self) -> None:
        ev_shcd = {
            "event_id": 111,
            "title": "Tuần Sinh hoạt Công dân - Sinh viên đầu năm học 2026-2027",
            "description": "Bắt buộc tham gia đầy đủ để không bị trừ điểm.",
        }
        est = estimate_event_credit(ev_shcd, role="participant")
        self.assertEqual(est.rubric_item_id, "6.4")
        self.assertFalse(est.is_credit_eligible)
        self.assertEqual(est.credit, 0)
        self.assertEqual(gain_calc(ev_shcd, user_credits=[0, 0, 0, 0, 0]), 0.0)

    def test_classification_sync_contracts(self) -> None:
        # 1. Module-3 nested `classification` payload (`predicted_criterion_ref`)
        ev_cls_pred = {
            "event_id": 112,
            "title": "Sự kiện bất kỳ",
            "sources": "organizer: Đoàn Trường",
            "classification": {"status": "PREDICTED", "predicted_criterion_ref": "3.1"},
        }
        self.assertEqual(credit_calc(ev_cls_pred, role="participant"), 10)
        self.assertEqual(credit_calc(ev_cls_pred, role="host"), 10)
        self.assertEqual(credit_calc(ev_cls_pred, role="collaborator"), 5)

        # 2. Module-3 `NO_MATCH` abstention -> 0 points & Gain = 0.0
        ev_cls_nomatch = {
            "event_id": 113,
            "title": "Thông báo lịch thi cuối kỳ",
            "classification": {"status": "NO_MATCH", "predicted_criterion_ref": None},
        }
        self.assertEqual(credit_calc(ev_cls_nomatch, role="participant"), 0)
        self.assertEqual(gain_calc(ev_cls_nomatch, user_credits=[0, 0, 0, 0, 0]), 0.0)


class TestGainCalcFormula(unittest.TestCase):
    """Verify exact Gain formula in `[0, 1]` from Module 6 specification."""

    def test_ineligible_or_zero_effective_points_returns_zero(self) -> None:
        est_inelig = CreditEstimate(
            event_id=1,
            category=1,
            rubric_item_id="1.9",
            role="participant",
            is_credit_eligible=False,
            expected_points=5.0,
            min_points=5.0,
            max_points=5.0,
            effective_points=0.0,
            source="rule",
            capped_by="category",
        )
        self.assertEqual(gain_calc(est_inelig), 0.0)

        # Category 1 already at max cap (30/30) -> P_eff = 0 -> Gain = 0.0
        ev = {"event_id": 2, "title": "Workshop AI", "category": 1}
        self.assertEqual(
            gain_calc(ev, user_credits=[30, 0, 0, 0, 0], max_credit=[30, 20, 25, 25, 30]),
            0.0,
        )

    def test_exact_formula_for_explicit_rule_and_rule_fallback(self) -> None:
        # 1) source = 'explicit' (r_src = 1.00)
        # P_eff = 8.0 -> min(1, 8/10) = 0.8
        # Cat 1: user=15, M_1=30 -> target_gap_1 = 15 -> m_gap = 0.5 + 0.5*(15/30) = 0.75
        # Gain = 0.8 * 0.75 * 1.00 = 0.6
        est_exp = CreditEstimate(
            event_id=10,
            category=1,
            rubric_item_id="1.9",
            role="participant",
            is_credit_eligible=True,
            expected_points=8.0,
            min_points=8.0,
            max_points=8.0,
            effective_points=8.0,
            source="explicit",
        )
        g_exp, details_exp = gain_calc_with_details(
            est_exp,
            user_credits=[15, 15, 10, 10, 0],
            max_credit=[30, 20, 25, 25, 30],
        )
        self.assertAlmostEqual(g_exp, 0.6, places=6)
        self.assertAlmostEqual(details_exp["r_src"], 1.00, places=6)

        # 2) source = 'rule' (r_src = 0.90)
        # P_eff = 10.0 -> min(1, 10/10) = 1.0
        # Cat 4: user=0, M_4=25 -> target_gap_4 = 25 -> m_gap = 0.5 + 0.5*(25/25) = 1.0
        # Gain = 1.0 * 1.0 * 0.90 = 0.9
        est_rule = CreditEstimate(
            event_id=11,
            category=4,
            rubric_item_id="4.2",
            role="participant",
            is_credit_eligible=True,
            expected_points=10.0,
            min_points=10.0,
            max_points=10.0,
            effective_points=10.0,
            source="rule",
        )
        g_rule = gain_calc(
            est_rule,
            user_credits=[0, 0, 0, 0, 0],
            max_credit=[30, 20, 25, 25, 30],
        )
        self.assertAlmostEqual(g_rule, 0.9, places=6)

        # 3) source = 'rule_fallback' (r_src = 0.80)
        # P_min = 5.0, P_exp = 10.0, P_eff = 10.0
        # P_use = min(10.0, 0.7 * 5.0 + 0.3 * 10.0) = min(10.0, 6.5) = 6.5
        # min(1, 6.5 / 10.0) = 0.65
        # Cat 3: user=12.5, M_3=25 -> target_gap_3 = 12.5 -> m_gap = 0.5 + 0.5 * 0.5 = 0.75
        # Gain = 0.65 * 0.75 * 0.80 = 0.39
        est_fb = CreditEstimate(
            event_id=12,
            category=3,
            rubric_item_id="3.1",
            role="participant",
            is_credit_eligible=True,
            expected_points=10.0,
            min_points=5.0,
            max_points=10.0,
            effective_points=10.0,
            source="rule_fallback",
        )
        g_fb, details_fb = gain_calc_with_details(
            est_fb,
            user_credits=[0, 0, 12.5, 0, 0],
            max_credit=[30, 20, 25, 25, 30],
        )
        self.assertAlmostEqual(details_fb["p_use"], 6.5, places=6)
        self.assertAlmostEqual(g_fb, 0.39, places=6)

    def test_partial_remaining_cap_clamps_p_eff(self) -> None:
        # Blood donation gives 10 pts in Cat 4 (max 25), user already has 22 -> P_eff = 3
        # target_gap_4 = 3 / 25 = 0.12 -> m_gap = 0.5 + 0.06 = 0.56
        # Gain = (3 / 10) * 0.56 * 0.90 = 0.1512
        ev = {"event_id": 20, "title": "Ngày hội Hiến máu tình nguyện", "category": 4}
        g, details = gain_calc_with_details(
            ev,
            user_credits=[15, 15, 10, 22, 0],
            max_credit=[30, 20, 25, 25, 30],
        )
        self.assertAlmostEqual(details["p_eff"], 3.0, places=4)
        self.assertAlmostEqual(g, 0.1512, places=6)


class TestDatabaseWorkflows(unittest.TestCase):
    """Verify CRD(S) and CRD(U) workflows against an isolated SQLite `DRLDatabaseManager`."""

    def setUp(self) -> None:
        DRLDatabaseManager.reset_instance()
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="crd_test_"))
        self.db_path = self.tmp_dir / "test_drl.db"
        self.db = DRLDatabaseManager(db_path=self.db_path)
        self.calc = CreditCalculator(db=self.db)

        self.db.insert_record(
            "rubric",
            {"school_id": 1, "school_name": "HCMUS", "max_credit": [30, 20, 25, 25, 30]},
        )
        self.db.insert_record(
            "users",
            {
                "user_id": "u_01",
                "username": "student1",
                "password": "pw",
                "email": "s1@hcmus.edu.vn",
                "school_id": 1,
                "credits": [15, 15, 10, 20, 0],
            },
        )
        self.db.insert_record(
            "events",
            {
                "event_id": 1,
                "title": "Cuộc thi Học thuật AI Hackathon 2026",
                "description": "Cuộc thi lập trình trí tuệ nhân tạo.",
                "topic": ["học thuật", "cuộc thi"],
                "format": ["thi"],
                "category": 1,
                "min_credit": None,
                "created_at": "2026-11-28T00:00:00+07:00",
                "updated_at": "2026-11-28T00:00:00+07:00",
            },
        )
        self.db.insert_record(
            "events",
            {
                "event_id": 2,
                "title": "Ngày hội Hiến máu Nhân đạo Giọt Hồng",
                "description": "Tham gia hiến máu tình nguyện.",
                "topic": ["thiện nguyện"],
                "format": ["offline"],
                "category": 4,
                "min_credit": None,
                "created_at": "2026-11-28T00:00:00+07:00",
                "updated_at": "2026-11-28T00:00:00+07:00",
            },
        )

    def tearDown(self) -> None:
        DRLDatabaseManager.reset_instance()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_crd_server_and_client_lifecycle(self) -> None:
        # 1. CRD(S): populate missing min_credit for participant
        updated_cnt = update_missing_min_credits(db=self.db)
        self.assertEqual(updated_cnt, 2)

        ev1 = self.db.fetch_one("SELECT * FROM events WHERE event_id = 1;", table="events")
        ev2 = self.db.fetch_one("SELECT * FROM events WHERE event_id = 2;", table="events")
        self.assertEqual(ev1["min_credit"], 5)
        self.assertEqual(ev2["min_credit"], 10)

        # 2. CRD(U) feed gain_calc using event_id & user_id from DB
        # For ev2 (Blood donation, +10 raw), user has 20/25 in Cat 4 -> P_eff = 5
        g2 = gain_calc(2, "u_01", db=self.db)
        # g_pts = 5/10 = 0.5; target_gap = 5/25 = 0.2 -> m_gap = 0.5 + 0.1 = 0.6; r_src = 0.9 -> 0.27
        self.assertAlmostEqual(g2, 0.27, places=6)

        # 3. CRD(U) completion finalization:
        #    - ev1 completed as 'host' (1.6 -> 1.7 BTC = +10 pts in Cat 1: 15 + 10 = 25 <= 30)
        #    - ev2 completed as 'participant' (+10 pts in Cat 4: 20 + 10 -> capped at 25!)
        self.db.insert_record(
            "event_subscription",
            {"user_id": "u_01", "event_id": 1, "gain": 0.3375, "role": "host", "status": "done", "credit": None},
        )
        self.db.insert_record(
            "event_subscription",
            {"user_id": "u_01", "event_id": 2, "gain": g2, "role": "participant", "status": "done", "credit": None},
        )

        finalized = finalize_completed_subscriptions("u_01", db=self.db)
        self.assertEqual(len(finalized), 2)

        sub1 = self.db.fetch_one(
            "SELECT * FROM event_subscription WHERE user_id = 'u_01' AND event_id = 1;",
            table="event_subscription",
        )
        sub2 = self.db.fetch_one(
            "SELECT * FROM event_subscription WHERE user_id = 'u_01' AND event_id = 2;",
            table="event_subscription",
        )
        self.assertEqual(sub1["credit"], 10)
        self.assertEqual(sub2["credit"], 10)

        user_after = self.db.fetch_one("SELECT * FROM users WHERE user_id = 'u_01';", table="users")
        self.assertEqual(user_after["credits"], [25, 15, 10, 25, 0])

        # 4. Calling finalize_completed_subscriptions again does NOT double-count
        finalized_again = finalize_completed_subscriptions("u_01", db=self.db)
        self.assertEqual(finalized_again, [])
        user_after_2 = self.db.fetch_one("SELECT * FROM users WHERE user_id = 'u_01';", table="users")
        self.assertEqual(user_after_2["credits"], [25, 15, 10, 25, 0])


if __name__ == "__main__":
    unittest.main()
