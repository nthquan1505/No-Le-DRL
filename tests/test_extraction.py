"""Unit and SQLite integration tests for `[EXT]` (`src/extraction`)."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Tuple
import unittest

from database import DRLDatabaseManager
from src.credit_calculation.credit_calculator import update_missing_min_credits
from src.extraction import (
    DESCRIPTION_SUMMARIZATION_GUIDELINES,
    SYSTEM_INSTRUCTION,
    DirectLLMExtractor,
    EventGeocoder,
    ExtractionPipeline,
    GeminiKeyPool,
    RawIngestionItem,
    build_fallback_description,
    clean_registration_url,
    clean_summary_description,
    reason_about_post,
    resolve_registration_url_from_posters,
)
from src.extraction.contracts import UTC7


class TestPreFilter(unittest.TestCase):
    """Verify Stage-1 deterministic noise rejection and lifecycle pass-through."""

    def test_rejects_recaps_admin_scholarships_and_jobs(self) -> None:
        recap = "[RECAP] HÀNH TRÌNH KHÉP LẠI – Cảm ơn sự góp mặt của hơn 300 bạn sinh viên!"
        self.assertFalse(reason_about_post(recap).is_candidate)

        admin = "Thông báo lịch thi chi tiết cuối kỳ HK1 năm học 2026-2027 dành cho sinh viên đại học."
        self.assertFalse(reason_about_post(admin).is_candidate)

        scholarship = "THÔNG BÁO HỌC BỔNG VALLET 2026: Sinh viên nộp hồ sơ học bổng trước ngày 20/10."
        self.assertFalse(reason_about_post(scholarship).is_candidate)

        job = "BẠN ĐÃ SẴN SÀNG CHO ENGINEERING DEVELOPMENT PROGRAM 2026? Làm việc toàn thời gian cho sinh viên mới tốt nghiệp."
        self.assertFalse(reason_about_post(job).is_candidate)

    def test_rejects_behind_the_scenes_summaries_and_copy_pasted_footers(self) -> None:
        bts_with_footer = (
            "❄️ LET IT SHINE 2026 ❄️ - BEHIND THE FINAL STAGE\n"
            "Phía sau những màn trình diễn được mong chờ là những giờ luyện tập miệt mài trong phòng tập.\n"
            "_____________________________\n"
            "⭐️ THÔNG TIN CUỘC THI LET IT SHINE 2026 ⭐️\n"
            "Thời gian: 16h00 - 20h00 22/11/2026\n"
            "Địa điểm: Giảng đường 1, cơ sở phường Chợ Quán."
        )
        self.assertFalse(reason_about_post(bts_with_footer).is_candidate)

        judge_intro_footer = (
            "❄️ LET IT SHINE 2026 ❄️ - GIỚI THIỆU BAN GIÁM KHẢO\n"
            "Let IT Shine 2026 vinh dự chào đón sự đồng hành của Hội đồng Ban Giám khảo.\n"
            "_____________________________\n"
            "⭐️ THÔNG TIN VÒNG CHUNG KẾT LET IT SHINE 2026 ⭐️\n"
            "👥 Mở đơn đăng ký cho tất cả sinh viên đang theo học tại Trường Đại học Khoa học tự nhiên.\n"
            "Thời gian: 16h00 - 20h00 22/11/2026"
        )
        self.assertFalse(reason_about_post(judge_intro_footer).is_candidate)

        judge_thanks = "❄️ LET IT SHINE 2026 ❄️ - CẢM ƠN BAN GIÁM KHẢO\nVậy là sân khấu Winter Aria đã chứng kiến những bản hòa ca cuối cùng!"
        self.assertFalse(reason_about_post(judge_thanks).is_candidate)

        round_summary = (
            "MYOR 2026 | SÔI NỔI VÒNG BÁN KẾT 2 TẠI TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN\n"
            "Ngày 21/11/2026, Vòng Bán kết 2 Cuộc thi Lắp ráp và Lập trình Robot – MYOR 2026 đã diễn ra tại Trường Đại học Khoa học tự nhiên."
        )
        self.assertFalse(reason_about_post(round_summary).is_candidate)

        event_summary = (
            "TUỔI TRẺ VẬT LÝ GÓP SỨC TẠI NGÀY HỘI TRI ÂN & KẾT NỐI 2026\n"
            "Ngày 14/11/2026 vừa qua, đông đảo đoàn viên đã tích cực tham gia hỗ trợ. Ngày hội khép lại nhưng nụ cười sẽ còn đọng mãi."
        )
        self.assertFalse(reason_about_post(event_summary).is_candidate)

        attendance_check = (
            "[THÔNG BÁO] KIỂM TRA THÔNG TIN THAM GIA CHUỖI HOẠT ĐỘNG TRỰC TUYẾN 2026\n"
            "Link đăng ký xác minh bổ sung: https://forms.gle/bgnzjYK4hwNKNVLWA"
        )
        self.assertFalse(reason_about_post(attendance_check).is_candidate)

    def test_qualifies_events_scholarship_seminars_and_lifecycle_notices(self) -> None:
        workshop = "🔥 [WORKSHOP: VIẾT VÀ CHẤM ĐIỂM CV] Thời gian: 14h00 ngày 22/11/2026. Link đăng ký: https://forms.gle/abc"
        self.assertTrue(reason_about_post(workshop).is_candidate)

        sch_seminar = "THÔNG BÁO HỌC BỔNG CHÍNH PHỦ NHẬT BẢN & BUỔI HỘI THẢO GIỚI THIỆU trực tiếp tại Hội trường I."
        self.assertTrue(reason_about_post(sch_seminar).is_candidate)

        cancel_notice = "[THÔNG BÁO TẠM HOÃN] BTC xin thông báo tạm hoãn chương trình Ngày hội AI vì lý do thời tiết."
        decision = reason_about_post(cancel_notice)
        self.assertTrue(decision.is_candidate)
        self.assertTrue(decision.is_lifecycle_signal)


class TestDirectLLMAndHeuristics(unittest.TestCase):
    """Verify Option B Direct REST client, key rotation, Groq failover, and URL/date sanitizers."""

    def test_clean_registration_url_unwraps_fb_redirect_and_rejects_placeholders(self) -> None:
        self.assertIsNone(clean_registration_url("free"))
        self.assertIsNone(clean_registration_url("null"))
        self.assertIsNone(clean_registration_url("Quét mã QR"))

        fb_wrapped = (
            "https://l.facebook.com/l.php?u=https%3A%2F%2Fforms.gle%2FtestForm99%3Fusp%3Dsend_form"
            "&h=AT123&fbclid=IwAR999"
        )
        self.assertEqual(
            clean_registration_url(fb_wrapped),
            "https://forms.gle/testForm99?usp=send_form",
        )

    def test_summarization_guidelines_and_description_sanitizers(self) -> None:
        # Ensure CLF-aligned summarization rules are embedded in SYSTEM_INSTRUCTION
        self.assertIn(DESCRIPTION_SUMMARIZATION_GUIDELINES, SYSTEM_INSTRUCTION)
        self.assertIn("fundraising workshop is not necessarily the same as directly performing charitable work", SYSTEM_INSTRUCTION)
        self.assertIn("guaranteed Điểm rèn luyện points unless the post explicitly states that", SYSTEM_INSTRUCTION)
        self.assertIn("2–3 concise sentences", SYSTEM_INSTRUCTION)

        # Strip accidental bullets/headings from LLM output into plain prose
        messy_llm_desc = "Summary: - Workshop hướng dẫn làm gốm gây quỹ.\n- Người tham gia được nhận sản phẩm mang về."
        self.assertEqual(
            clean_summary_description(messy_llm_desc),
            "Workshop hướng dẫn làm gốm gây quỹ. Người tham gia được nhận sản phẩm mang về.",
        )

        # Offline fallback skips greetings, duplicate title, URLs, and raw logistics lines
        raw_post = (
            "Thân chào các bạn sinh viên,\n"
            "🔥 WORKSHOP LÀM NẾN THƠM GÂY QUỸ XUÂN TÌNH NGUYỆN\n"
            "Chương trình hướng dẫn sinh viên tự tay pha chế mùi hương và đổ khuôn nến thơm thủ công.\n"
            "Toàn bộ lợi nhuận bán vé được đóng góp vào quỹ quà Tết cho trẻ em vùng cao.\n"
            "Thời gian: 14h00 ngày 10/12/2026\n"
            "Địa điểm: Sảnh I, Cơ sở 1\n"
            "Link đăng ký: https://forms.gle/candle2026"
        )
        fb_desc = build_fallback_description(raw_post, title="WORKSHOP LÀM NẾN THƠM GÂY QUỸ XUÂN TÌNH NGUYỆN")
        self.assertNotIn("Thân chào", fb_desc)
        self.assertNotIn("forms.gle", fb_desc)
        self.assertNotIn("Thời gian:", fb_desc)
        self.assertIn("nến thơm thủ công", fb_desc)

    def test_gemini_key_rotation_429_failover_and_schema_payload(self) -> None:
        calls: List[Tuple[str, str, Dict[str, Any]]] = []

        def fake_http_post(
            url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float
        ) -> Tuple[int, Dict[str, str], str]:
            key = headers.get("x-goog-api-key", "")
            calls.append((url, key, payload))
            if key == "key_1":
                return 429, {"retry-after": "120"}, '{"error": "quota exceeded"}'
            gemini_items = [
                {
                    "source_id": "post_01",
                    "classification": "new_or_active_event",
                    "reason": "Workshop học thuật cho sinh viên",
                    "title": "Workshop Thực Chiến AI Agent 2026",
                    "description": "Buổi workshop hướng dẫn xây dựng AI Agent và cấp giấy chứng nhận tham dự.",
                    "keywords": ["workshop", "AI Agent", "sinh viên"],
                    "is_online": False,
                    "reg_start_datetime": None,
                    "reg_end_datetime": "2026-11-29T23:59:00+07:00",
                    "event_start_datetime": "2026-11-30T14:00:00+07:00",
                    # Intentionally inverted end time to verify chronological sanitizer clears it
                    "event_end_datetime": "2026-11-30T09:00:00+07:00",
                    "location": "Hội trường I, Cơ sở 1 Trường ĐH Khoa học Tự nhiên",
                    "fee": "Miễn phí",
                    "target_audience": ["sinh viên KHTN"],
                    "topic": ["AI", "học thuật"],
                    "format": ["workshop", "offline"],
                    "award": "Điểm rèn luyện, Giấy chứng nhận",
                    "organizer": "Đoàn Khoa CNTT",
                    "registration_url": "free",
                }
            ]
            envelope = {
                "candidates": [{"content": {"parts": [{"text": json.dumps(gemini_items, ensure_ascii=False)}]}}]
            }
            return 200, {}, json.dumps(envelope, ensure_ascii=False)

        pool = GeminiKeyPool(keys=["key_1", "key_2"])
        extractor = DirectLLMExtractor(key_pool=pool, gemini_model="gemini-2.5-flash-lite", http_post=fake_http_post)
        item = RawIngestionItem(
            source_type="facebook",
            source_id="post_01",
            source_name="Đoàn Khoa CNTT - HCMUS",
            content=(
                "🔥 WORKSHOP THỰC CHIẾN AI AGENT 2026 dành cho sinh viên! "
                "Đăng ký ngay tại https://l.facebook.com/l.php?u=https%3A%2F%2Fforms.gle%2FaiAgent2026&fbclid=abc"
            ),
            published_at=datetime(2026, 11, 25, 10, 0, tzinfo=UTC7),
        )

        drafts = extractor.extract_batch([item])
        self.assertEqual(len(drafts), 1)
        draft = drafts[0]

        # Key 1 hit 429 -> rotated automatically to Key 2
        self.assertEqual([c[1] for c in calls], ["key_1", "key_2"])
        # Verified server-side structured output config was sent
        gen_cfg = calls[0][2]["generationConfig"]
        self.assertEqual(gen_cfg["responseMimeType"], "application/json")
        self.assertIn("responseSchema", gen_cfg)

        self.assertEqual(draft.extraction_method, "gemini")
        self.assertEqual(draft.title, "Workshop Thực Chiến AI Agent 2026")
        # "free" placeholder cleared and replaced by unwrapped URL from raw post text!
        self.assertEqual(draft.registration_url, "https://forms.gle/aiAgent2026")
        # Inverted end time cleared safely
        self.assertIsNone(draft.event_end_datetime)

    def test_groq_failover_and_offline_heuristic_fallback(self) -> None:
        def fake_groq_only_http(
            url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float
        ) -> Tuple[int, Dict[str, str], str]:
            if "generativelanguage.googleapis.com" in url:
                return 503, {}, "unavailable"
            groq_body = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "source_id": "post_groq",
                                            "classification": "new_or_active_event",
                                            "reason": "Cuộc thi học thuật",
                                            "title": "Cuộc thi Lập trình ICPC HCMUS",
                                            "description": "Sân chơi thuật toán cho sinh viên.",
                                            "keywords": ["cuộc thi", "lập trình"],
                                            "is_online": False,
                                            "location": "Cơ sở 2 ĐH Khoa học Tự nhiên",
                                            "fee": "",
                                            "target_audience": ["sinh viên"],
                                            "topic": ["lập trình"],
                                            "format": ["cuộc thi"],
                                            "award": "Giải thưởng",
                                            "organizer": "Khoa CNTT",
                                            "registration_url": "https://forms.gle/icpc2026",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
            return 200, {}, json.dumps(groq_body)

        item = RawIngestionItem(
            source_type="facebook",
            source_id="post_groq",
            source_name="Khoa CNTT",
            content="Cuộc thi Lập trình ICPC HCMUS ngày 12/12/2026 tại Cơ sở 2. Đăng ký: https://forms.gle/icpc2026",
            published_at=datetime(2026, 11, 28, 8, 0, tzinfo=UTC7),
        )

        # 1. Gemini 503 -> fails over to Groq
        groq_extractor = DirectLLMExtractor(
            key_pool=GeminiKeyPool(keys=["bad_key"]),
            gemini_model="gemini-2.5-flash-lite",
            groq_api_key="gsk_test",
            http_post=fake_groq_only_http,
        )
        with self.assertLogs("src.extraction.llm_client", level="WARNING"):
            d_groq = groq_extractor.extract_batch([item])[0]
        self.assertEqual(d_groq.extraction_method, "groq")
        self.assertEqual(d_groq.event_start_datetime, "2026-12-12T08:00:00+07:00")

        # 2. Zero cloud keys -> falls back to deterministic offline heuristic
        offline_extractor = DirectLLMExtractor(key_pool=GeminiKeyPool(keys=[]), groq_api_key="")
        d_heur = offline_extractor.extract_batch([item])[0]
        self.assertEqual(d_heur.extraction_method, "heuristic")
        self.assertEqual(d_heur.classification, "new_or_active_event")
        self.assertEqual(d_heur.registration_url, "https://forms.gle/icpc2026")


class TestQrAndGeocoder(unittest.TestCase):
    """Verify local-only QR decoding and campus/venue coordinate resolution."""

    def test_qr_decoder_filters_self_fb_and_returns_form_url(self) -> None:
        url = resolve_registration_url_from_posters(
            images_url=["https://cdn.example.com/poster1.jpg"],
            image_fetcher=lambda u, t: b"fake_image_bytes",
            qr_decoder=lambda b: [
                "https://www.facebook.com/doankhoacntt",
                "https://l.facebook.com/l.php?u=https%3A%2F%2Fforms.gle%2FqrDecoded77",
            ],
        )
        self.assertEqual(url, "https://forms.gle/qrDecoded77")

    def test_geocoder_campus_rules_shorthand_and_online_skip(self) -> None:
        geo = EventGeocoder(cache_path=None, enable_network_geocoding=False)

        c1, _ = geo.resolve("Hội trường I", sources="facebook (Đoàn Khoa CNTT - HCMUS)", is_online=False)
        self.assertEqual(c1, "10.762622,106.682172")

        c2, _ = geo.resolve("Nhà điều hành Cơ sở Linh Trung, ĐH Khoa học Tự nhiên", is_online=False)
        self.assertEqual(c2, "10.875809,106.799178")

        c_nv, _ = geo.resolve("Nhà Văn hóa Sinh viên TP.HCM tại Khu đô thị ĐHQG-HCM", is_online=False)
        self.assertEqual(c_nv, "10.875120,106.800720")

        c_online, _ = geo.resolve("Google Meet", is_online=True)
        self.assertIsNone(c_online)


class TestEndToEndDatabaseLifecycle(unittest.TestCase):
    """Verify full `EXT -> drl_database.db` flow: create, QR backfill, dedupe, update, soft-cancel, and CRD handoff."""

    def setUp(self) -> None:
        DRLDatabaseManager.reset_instance()
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="drl_ext_test_"))
        self.db_path = self.tmp_dir / "test_drl.db"
        self.db = DRLDatabaseManager(db_path=self.db_path)

        # Seed a tracked fanpage and a student user
        self.fanpage_id = int(
            self.db.insert_record(
                "fanpages",
                {
                    "fanpage_name": "Đoàn Khoa CNTT - HCMUS",
                    "facebook_url": "https://www.facebook.com/doankhoacntt.hcmus",
                    "number_of_events": 0,
                },
            )
        )
        self.user_id = "stu_001"
        self.db.insert_record(
            "users",
            {
                "user_id": self.user_id,
                "username": "student1",
                "password": "hashed_password",
                "email": "student1@student.hcmus.edu.vn",
                "start_year": 2025,
            },
        )

    def tearDown(self) -> None:
        if hasattr(self.db, "close"):
            self.db.close()
        DRLDatabaseManager.reset_instance()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_full_lifecycle_create_duplicate_update_cancel_and_crd_integration(self) -> None:
        pipeline = ExtractionPipeline(
            db=self.db,
            llm_extractor=DirectLLMExtractor(key_pool=GeminiKeyPool(keys=[]), groq_api_key=""),
            geocoder=EventGeocoder(cache_path=None, enable_network_geocoding=False),
            qr_resolver=lambda urls: "https://forms.gle/fromPosterQR" if urls else None,
            seen_hashes_path=self.tmp_dir / "seen_hashes.json",
        )

        # 1. Ingest a scraped fanpage batch (1 recap noise + 1 valid event with QR-only poster)
        scraped_posts = [
            {
                "post_id": "p_recap",
                "post_url": "https://www.facebook.com/doankhoacntt.hcmus/posts/100",
                "content": "[RECAP] Nhìn lại những khoảnh khắc tuyệt vời của đêm nhạc vừa qua!",
                "media_urls": [],
                "published_at": "2026-11-20T18:00:00+07:00",
            },
            {
                "post_id": "p_event_1",
                "post_url": "https://www.facebook.com/doankhoacntt.hcmus/posts/101",
                "content": (
                    "📢 WORKSHOP KỸ NĂNG NGHIÊN CỨU KHOA HỌC TRONG KỶ NGUYÊN AI\n"
                    "Thời gian tổ chức: 08h30 - 11h30 ngày 05/12/2026\n"
                    "Hạn đăng ký: 23h59 ngày 03/12/2026\n"
                    "Địa điểm: Hội trường I, 227 Nguyễn Văn Cừ\n"
                    "Đối tượng tham gia: Sinh viên toàn trường\n"
                    "Quyền lợi: Cộng điểm rèn luyện và Giấy chứng nhận.\n"
                    "Quét mã QR trên poster để đăng ký tham gia!"
                ),
                "media_urls": ["https://cdn.example.com/poster_nckh.png"],
                "published_at": "2026-11-25T09:00:00+07:00",
            },
        ]

        summary1 = pipeline.extract_scraped_posts(
            scraped_posts,
            fanpage_name="Đoàn Khoa CNTT - HCMUS",
            fanpage_id=self.fanpage_id,
        )
        self.assertEqual(summary1.prefilter_rejected, 1)
        self.assertEqual(summary1.created, 1)

        ev_id = summary1.results[1].event_id
        self.assertIsNotNone(ev_id)
        assert ev_id is not None

        ev_row = self.db.fetch_one("SELECT * FROM events WHERE event_id = ?;", (ev_id,), table="events")
        assert ev_row is not None
        self.assertIsNone(ev_row["category"])
        self.assertIsNone(ev_row["min_credit"])
        self.assertFalse(ev_row["is_cancelled"])
        self.assertEqual(ev_row["registration_url"], "https://forms.gle/fromPosterQR")
        self.assertEqual(ev_row["coordinate"], "10.762622,106.682172")
        self.assertEqual(ev_row["event_start_datetime"], "2026-12-05T08:30:00+07:00")
        self.assertEqual(ev_row["event_end_datetime"], "2026-12-05T11:30:00+07:00")

        fp_row = self.db.fetch_one("SELECT * FROM fanpages WHERE fanpage_id = ?;", (self.fanpage_id,), table="fanpages")
        assert fp_row is not None
        self.assertEqual(fp_row["number_of_events"], 1)

        # 2. Re-running exact same batch skips via content hash
        summary_repeat = pipeline.extract_scraped_posts(
            scraped_posts,
            fanpage_name="Đoàn Khoa CNTT - HCMUS",
            fanpage_id=self.fanpage_id,
        )
        self.assertEqual(summary_repeat.skipped_seen, 2)
        self.assertEqual(summary_repeat.created, 0)

        # 3. Cross-channel email announcing the exact same event -> deduplicated + sources merged
        dup_email_res = pipeline.extract_email(
            subject="Workshop Kỹ năng Nghiên cứu Khoa học trong Kỷ nguyên AI",
            sender="ctsv@hcmus.edu.vn",
            content=(
                "Thân mời sinh viên tham gia Workshop Kỹ năng Nghiên cứu Khoa học trong Kỷ nguyên AI.\n"
                "Thời gian: 08h30 - 11h30 ngày 05/12/2026 tại Hội trường I, 227 Nguyễn Văn Cừ.\n"
                "Link đăng ký: https://forms.gle/fromPosterQR"
            ),
            received_at="2026-11-26T10:00:00+07:00",
        )
        self.assertEqual(dup_email_res.action, "duplicate")
        self.assertEqual(dup_email_res.event_id, ev_id)
        ev_after_dup = self.db.fetch_one("SELECT * FROM events WHERE event_id = ?;", (ev_id,), table="events")
        assert ev_after_dup is not None
        self.assertIn("email (ctsv@hcmus.edu.vn)", ev_after_dup["sources"])

        # 4. Simulate downstream CLF (`category = 1`) + CRD (`update_missing_min_credits`) + Student Subscription
        self.db.update_records("events", {"category": 1}, {"event_id": ev_id})
        crd_updated = update_missing_min_credits(db=self.db)
        self.assertEqual(crd_updated, 1)
        ev_credited = self.db.fetch_one("SELECT * FROM events WHERE event_id = ?;", (ev_id,), table="events")
        assert ev_credited is not None
        self.assertGreater(ev_credited["min_credit"], 0)

        self.db.insert_record(
            "event_subscription",
            {
                "user_id": self.user_id,
                "event_id": ev_id,
                "role": "participant",
                "status": "registered",
            },
        )

        # 5. Ingest an explicit schedule/venue UPDATE -> updates event without clobbering registration_url & notifies subscriber
        upd_res = pipeline.extract_email(
            subject="[THÔNG BÁO GIA HẠN ĐĂNG KÝ & THAY ĐỔI ĐỊA ĐIỂM] Workshop Kỹ năng Nghiên cứu Khoa học trong Kỷ nguyên AI",
            sender="doankhoacntt@hcmus.edu.vn",
            content=(
                "BTC thông báo gia hạn đăng ký và thay đổi địa điểm tổ chức Workshop Kỹ năng Nghiên cứu Khoa học trong Kỷ nguyên AI:\n"
                "Hạn đăng ký: 23h59 ngày 04/12/2026\n"
                "Địa điểm: Nhà điều hành Cơ sở 2 ĐH Khoa học Tự nhiên (Linh Trung)"
            ),
            received_at="2026-12-01T12:00:00+07:00",
        )
        self.assertEqual(upd_res.action, "updated")
        self.assertEqual(upd_res.event_id, ev_id)
        self.assertEqual(upd_res.notifications_created, 1)

        ev_updated = self.db.fetch_one("SELECT * FROM events WHERE event_id = ?;", (ev_id,), table="events")
        assert ev_updated is not None
        self.assertEqual(ev_updated["coordinate"], "10.875809,106.799178")
        # Registration URL and category preserved!
        self.assertEqual(ev_updated["registration_url"], "https://forms.gle/fromPosterQR")
        self.assertEqual(ev_updated["category"], 1)

        # 6. Ingest a CANCELLATION post -> soft-cancels (`is_cancelled = 1`), keeps FK valid, notifies subscriber
        cancel_summary = pipeline.extract_scraped_posts(
            [
                {
                    "post_id": "p_cancel_1",
                    "post_url": "https://www.facebook.com/doankhoacntt.hcmus/posts/105",
                    "content": (
                        "[THÔNG BÁO HỦY TỔ CHỨC] BTC xin cáo lỗi và thông báo hủy sự kiện "
                        "Workshop Kỹ năng Nghiên cứu Khoa học trong Kỷ nguyên AI vì lý do khách quan."
                    ),
                    "media_urls": [],
                    "published_at": "2026-12-03T08:00:00+07:00",
                }
            ],
            fanpage_name="Đoàn Khoa CNTT - HCMUS",
            fanpage_id=self.fanpage_id,
        )
        self.assertEqual(cancel_summary.cancelled, 1)
        ev_cancelled = self.db.fetch_one("SELECT * FROM events WHERE event_id = ?;", (ev_id,), table="events")
        assert ev_cancelled is not None
        self.assertTrue(ev_cancelled["is_cancelled"])

        notifs = self.db.fetch_all(
            "SELECT * FROM notifications WHERE user_id = ? ORDER BY notification_id ASC;",
            (self.user_id,),
            table="notifications",
        )
        self.assertEqual([n["type"] for n in notifs], ["update", "cancellation"])

    def test_extractor_signals_classifier_to_query_db_on_completion(self) -> None:
        from src.classification.pipeline import connect_extractor_to_classifier
        from src.embedder import BaseEmbedder
        from src.signals import SIGNAL_CLASSIFICATION_COMPLETED, SIGNAL_EXTRACTION_COMPLETED, SignalBus
        from src.vector_ops import l2_normalize
        import numpy as np

        # Use a known PREDICTED query vector from precomputed multivector embeddings
        npz_path = Path(__file__).resolve().parent.parent / "data" / "generated" / "events_cls_multivector_embeddings.npz"
        with np.load(npz_path) as npz:
            anchor_vec = l2_normalize(npz["title_description"][0].tolist())

        class _SignalTestEmbedder(BaseEmbedder):
            def encode_queries(self, texts):
                return [list(anchor_vec) for _ in texts]

            def encode_passages(self, texts):
                return [list(anchor_vec) for _ in texts]

        bus = SignalBus()
        ext_signals = []
        cls_signals = []
        bus.subscribe(SIGNAL_EXTRACTION_COMPLETED, lambda s: ext_signals.append(s))

        # Subscribe Classifier to the signal bus (it will query self.db by itself when signaled)
        unsub_cls = connect_extractor_to_classifier(
            db=self.db,
            only_unclassified=True,
            embedder=_SignalTestEmbedder(),
            bus=bus,
        )
        from src.signals import signal_bus as global_bus

        unsub_cls_done = global_bus.subscribe(SIGNAL_CLASSIFICATION_COMPLETED, lambda s: cls_signals.append(s))
        try:
            llm = DirectLLMExtractor(key_pool=GeminiKeyPool(keys=[]), groq_api_key="")
            geocoder = EventGeocoder(cache_path=None, enable_network_geocoding=False)
            pipeline = ExtractionPipeline(
                db=self.db,
                llm_extractor=llm,
                geocoder=geocoder,
                enable_qr_scan=False,
                signal_bus=bus,
            )

            res = pipeline.extract_email(
                subject="Cuộc thi Học thuật An toàn Thông tin HCMUS CTF 2026",
                sender="cntt@hcmus.edu.vn",
                content=(
                    "Khoa Công nghệ Thông tin tổ chức Cuộc thi Học thuật An toàn Thông tin HCMUS CTF 2026.\n"
                    "Thời gian: 08h00 - 16h00 ngày 12/12/2026 tại Hội trường I, 227 Nguyễn Văn Cừ.\n"
                    "Link đăng ký: https://forms.gle/ctf2026"
                ),
                received_at="2026-11-28T10:00:00+07:00",
            )
            self.assertEqual(res.action, "created")
            self.assertIsNotNone(res.event_id)

            # 1. Extractor emitted exactly 1 signal with the created event_id (and 0 JSON event bodies)
            self.assertEqual(len(ext_signals), 1)
            self.assertEqual(ext_signals[0].created_event_ids, (res.event_id,))

            # 2. Classifier received the signal, queried `self.db` itself, and populated `events.category`
            self.assertEqual(len(cls_signals), 1)
            self.assertEqual(cls_signals[0].total_queried_events, 1)
            row = self.db.fetch_one("SELECT * FROM events WHERE event_id = ?;", (res.event_id,), table="events")
            assert row is not None
            self.assertIn(row["category"], (1, 2, 3, 4, 5))

            # 3. Duplicate email causes 0 DB mutations -> no redundant signal emitted
            dup = pipeline.extract_email(
                subject="Cuộc thi Học thuật An toàn Thông tin HCMUS CTF 2026",
                sender="cntt@hcmus.edu.vn",
                content=(
                    "Khoa Công nghệ Thông tin tổ chức Cuộc thi Học thuật An toàn Thông tin HCMUS CTF 2026.\n"
                    "Thời gian: 08h00 - 16h00 ngày 12/12/2026 tại Hội trường I, 227 Nguyễn Văn Cừ.\n"
                    "Link đăng ký: https://forms.gle/ctf2026"
                ),
                received_at="2026-11-28T10:00:00+07:00",
            )
            self.assertEqual(dup.action, "skipped_seen")
            self.assertEqual(len(ext_signals), 1)
        finally:
            unsub_cls()
            unsub_cls_done()


if __name__ == "__main__":
    unittest.main()
