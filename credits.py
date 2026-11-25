from datetime import datetime
import json
from pathlib import Path


class CreditCalculationModule:
    def __init__(self, drl_rules):
        self.rules = drl_rules

    @staticmethod
    def _parse_duration_hours(event_dict):
        """Tính số giờ diễn ra hoạt động từ start_datetime và end_datetime (nếu có)."""
        start_str = event_dict.get("start_datetime")
        end_str = event_dict.get("end_datetime")
        if not start_str or not end_str:
            return None
        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)
            hours = (end_dt - start_dt).total_seconds() / 3600.0
            return max(0.0, hours)
        except Exception:
            return None

    def _infer_canonical_key_from_dict(self, event_dict):
        """
        Phân tích ngữ cảnh (category, format, title, description, thời lượng)
        cho dữ liệu thô từ scrape_1311.json theo Quy chế 1028/QĐ-KHTN-CTSV.
        """
        title = (event_dict.get("title") or "").lower()
        desc = (event_dict.get("description") or "").lower()
        text = f"{title} {desc}"

        raw_cats = event_dict.get("category") or []
        if isinstance(raw_cats, str):
            raw_cats = [raw_cats]
        raw_fmts = event_dict.get("format") or []
        if isinstance(raw_fmts, str):
            raw_fmts = [raw_fmts]
        cat_tokens = [str(c).strip().lower() for c in raw_cats]
        fmt_tokens = [str(f).strip().lower() for f in raw_fmts]

        # 1. Kiểm tra từ khóa chuyên biệt theo Phụ lục Quy chế 1028 (ưu tiên cao nhất)
        if any(k in text for k in ["hiến máu", "giọt hồng", "giọt máu"]):
            return "Blood_Donation"

        if any(k in text for k in ["mùa hè xanh", "xuân tình nguyện"]):
            return "Volunteer_Long"

        if any(k in text for k in ["olympic các môn khoa học mác", "mác - lênin", "mác-lênin", "tư tưởng hồ chí minh", "tầm nhìn xuyên thế kỷ", "lý luận chính trị", "theo dấu chân người"]):
            return "Political_Contest"

        if any(k in text for k in ["hiểu luật", "bắt cóc", "buôn bán người", "lừa đảo mạng", "an toàn", "chủ quyền", "biển đông", "hiến pháp"]):
            return "Legal_Propaganda"

        if any(k in text for k in ["việt dã", "chạy bộ", "uprace", "marathon", "thanh niên khỏe", "kéo co", "cờ vua", "thể thao điện tử", "tournament", "sport"]):
            if any(c in cat_tokens for c in ["competition", "competitions", "sporting events", "sports"]) or "tournament" in title or "giải" in title:
                return "Sports_Culture_Contest"
            return "Sport_Running"

        if any(k in text for k in ["việc làm", "career", "tuyển dụng", "cv", "phỏng vấn", "tham quan công ty", "vng", "netcompany", "saritasa"]):
            if any(k in text for k in ["hội thảo", "workshop", "cv", "phỏng vấn"]):
                return "Academic_Workshop"
            return "Career_Fair"

        if any(k in text for k in ["nghiên cứu khoa học", "hội nghị khoa học", "mini-course", "summer school", "phd", "nghiên cứu"]):
            if "hội nghị" in title or "nckh" in title:
                return "Academic_Research"
            return "Academic_Workshop"

        # 2. Hoạt động Tình nguyện -> xét thời lượng theo Mục 4.3
        if any(c in cat_tokens for c in ["volunteering", "volunteer", "volunteer / community"]) or "tình nguyện" in text:
            hours = self._parse_duration_hours(event_dict)
            if hours is not None:
                if hours >= 24 * 7:
                    return "Volunteer_Long"
                if hours >= 24:
                    return "Volunteer_Medium"
                return "Volunteer_Short"
            return "Volunteer_Short"

        # 3. Cuộc thi / Sân chơi
        if any(c in cat_tokens for c in ["competition", "competitions", "competitions & tournaments"]) or any(f in fmt_tokens for f in ["competition", "contest"]):
            if any(k in text for k in ["văn hóa", "nghệ thuật", "ảnh", "âm nhạc", "out r space", "the stage", "nét chữ", "lịch sử"]):
                return "Sports_Culture_Contest"
            if any(k in text for k in ["khởi nghiệp", "sáng tạo trẻ quốc gia"]):
                return "Academic_Competition"
            return "Academic_Competition"

        # 4. Thi thử / Khảo thí / Hội thảo / Seminar / Workshop
        if "exam" in cat_tokens or "thi thử toeic" in text:
            return "Academic_Workshop"

        if any(c in cat_tokens for c in ["academic", "workshop", "seminar", "academic & skill sessions", "academic / skills", "skill-building", "technology", "education", "conference", "scholarship", "tọa đàm"]):
            return "Academic_Workshop"

        # 5. Khám sức khỏe, Chào tân sinh viên, Tuyển thành viên CLB, Văn nghệ, Ngày hội
        if "health" in cat_tokens or "khám sức khỏe" in text:
            return "School_Mobilization"
        if "orientation" in cat_tokens or "hành trang tân sinh viên" in text:
            return "Class_Faculty_Activity"
        if "recruitment" in cat_tokens or "câu lạc bộ" in text:
            return "Academic_Club"
        if any(c in cat_tokens for c in ["festival", "ngày hội", "ceremony", "exhibition", "cultural", "arts", "music", "entertainment", "arts & culture", "arts & culture/music events", "cultural / community", "culture", "community", "environment", "activity"]):
            if "truyền thống" in title or "nhà giáo" in title or "văn hóa" in title:
                return "School_History_Event"
            return "Class_Faculty_Activity"

        # 6. Fallback qua bảng alias của category
        aliases = self.rules.get("raw_category_aliases", {})
        for c in cat_tokens:
            if c in aliases:
                return aliases[c]

        return None

    def map_event_to_drl_criteria(self, event_category):
        """
        1. Đối chiếu loại hoạt động vào đúng Điều/Khoản (Điều 4, Điều 5) và Tiểu mục (Phụ lục).
        Hỗ trợ:
          - Chuỗi mã chuẩn (VD: "Academic_Workshop")
          - Chuỗi hoặc list category thô từ scrape_1311.json (VD: ["Workshop", "Academic"])
          - Dict sự kiện đầy đủ từ Vault hoặc scrape_1311.json
        """
        mapping = self.rules.get("criteria_mapping", {})
        aliases = self.rules.get("raw_category_aliases", {})

        # Trường hợp truyền vào nguyên object sự kiện (dict)
        if isinstance(event_category, dict):
            explicit_key = event_category.get("event_category")
            if isinstance(explicit_key, str) and explicit_key in mapping:
                matched = dict(mapping[explicit_key])
                matched["matched_key"] = explicit_key
                return matched

            inferred_key = self._infer_canonical_key_from_dict(event_category)
            if inferred_key and inferred_key in mapping:
                matched = dict(mapping[inferred_key])
                matched["matched_key"] = inferred_key
                return matched

            if isinstance(explicit_key, str) and explicit_key.strip().lower() in aliases:
                alias_key = aliases[explicit_key.strip().lower()]
                if alias_key in mapping:
                    matched = dict(mapping[alias_key])
                    matched["matched_key"] = alias_key
                    return matched

        # Trường hợp truyền vào chuỗi chuẩn hoặc chuỗi category thô
        elif isinstance(event_category, str):
            if event_category in mapping:
                matched = dict(mapping[event_category])
                matched["matched_key"] = event_category
                return matched
            lowered = event_category.strip().lower()
            if lowered in aliases and aliases[lowered] in mapping:
                alias_key = aliases[lowered]
                matched = dict(mapping[alias_key])
                matched["matched_key"] = alias_key
                return matched

        # Trường hợp truyền vào list category (VD: ["Seminar", "Academic"])
        elif isinstance(event_category, list):
            for cat in event_category:
                if isinstance(cat, str):
                    if cat in mapping:
                        matched = dict(mapping[cat])
                        matched["matched_key"] = cat
                        return matched
                    lowered = cat.strip().lower()
                    if lowered in aliases and aliases[lowered] in mapping:
                        alias_key = aliases[lowered]
                        matched = dict(mapping[alias_key])
                        matched["matched_key"] = alias_key
                        return matched

        return {
            "category_id": "Unmapped",
            "sub_code": "N/A",
            "article_ref": "Chưa đối chiếu được Điều 4/Điều 5",
            "description": "Chưa phân loại trong Phụ lục",
            "default_points": 0,
            "matched_key": "Unmapped",
        }

    def validate_max_cap_limit(self, drl_category, current_category_points):
        """
        2. Kiểm tra giới hạn điểm tối đa (max cap) cho từng mục theo Khoản 2 Điều 4
        """
        cat_info = self.rules["categories"].get(drl_category, {})
        max_cap = cat_info.get("max_cap", None)

        # Mục 6 (Điểm trừ) không giới hạn mức tối đa (max_cap = None)
        if max_cap is None:
            return current_category_points, None, False

        is_capped = current_category_points > max_cap
        # Giới hạn mức tối đa theo Khoản 2 Điều 4 và không để âm cho các mục điểm cộng (Mục 1-5)
        valid_points = max(0, min(current_category_points, max_cap))
        return valid_points, max_cap, is_capped

    def _classify_rank(self, final_score):
        """Phân loại kết quả rèn luyện theo Điều 7"""
        if final_score >= 90: return "Xuất sắc"
        if final_score >= 80: return "Tốt"
        if final_score >= 65: return "Khá"
        if final_score >= 50: return "Trung bình"
        if final_score >= 35: return "Yếu"
        return "Kém"

    def calculate_earned_points(self, completed_events_list):
        """
        3. Cộng dồn điểm thực tế, áp dụng giới hạn 2 lớp (Mục & Tổng 100đ) và xuất báo cáo.
        Hỗ trợ cả event có field 'event_category' lẫn event thô từ scrape_1311.json.
        """
        print("=" * 72)
        print("🚀 WORKFLOW: CREDIT CALCULATION MODULE (QUY CHẾ KHTN 1028/QĐ-KHTN-CTSV)")
        print("=" * 72)

        # Khởi tạo điểm cơ bản (mặc định không vi phạm quy chế đào tạo, nội quy, pháp luật...)
        raw_points = {
            cat_id: info.get("base_points", 0)
            for cat_id, info in self.rules["categories"].items()
        }
        unmapped_events = []

        print("\n[BƯỚC 1] Ánh xạ sự kiện vào Điều 4, Điều 5 & Phụ lục ĐRL (map_event_to_drl_criteria):")
        for ev in completed_events_list:
            # Nếu ev là dict đầy đủ (có thể từ scrape_1311.json hoặc Vault), truyền nguyên dict để đối chiếu ngữ cảnh
            mapped = self.map_event_to_drl_criteria(ev if isinstance(ev, dict) else {"event_category": ev})
            cat_id = mapped["category_id"]
            pts = ev.get("points", mapped["default_points"]) if isinstance(ev, dict) else mapped["default_points"]
            title = ev.get("title", "Không tên") if isinstance(ev, dict) else str(ev)

            if cat_id in raw_points:
                raw_points[cat_id] += pts
                sign = "+" if pts >= 0 else ""
                print(
                    f"  ✔ [{mapped['sub_code']} | {mapped.get('article_ref', '')}] {title} "
                    f"-> {cat_id} ({sign}{pts}đ) | Tạm tính: {raw_points[cat_id]}đ"
                )
            else:
                unmapped_events.append(title)
                print(f"  ⚠️ [Unmapped] {title} -> Không tìm thấy trong Phụ lục (0đ)")

        print("\n[BƯỚC 2] Kiểm tra giới hạn điểm tối đa từng Mục - Lớp 1 (validate_max_cap_limit):")
        capped_points = {}
        alerts = []

        for cat_id, raw_val in raw_points.items():
            valid_val, max_cap, is_capped = self.validate_max_cap_limit(cat_id, raw_val)
            capped_points[cat_id] = valid_val
            cat_name = self.rules["categories"][cat_id]["name"]

            if max_cap is None:
                print(f"  ➖ {cat_id} ({cat_name}): {valid_val}đ (Không giới hạn)")
            elif is_capped:
                msg = f"{cat_id} đạt mức tối đa: Điểm thô {raw_val}đ -> Quy về {valid_val}/{max_cap}đ"
                alerts.append(msg)
                print(f"  ⚠️ {msg}")
            else:
                print(f"  ✅ {cat_id} ({cat_name}): {valid_val}/{max_cap}đ")

        # Giới hạn Lớp 2: Tổng điểm tối đa 100 điểm theo Khoản 3, 4 Điều 4
        subtotal = sum(capped_points.values())
        total_max_cap = self.rules.get("total_max_cap", 100)
        final_score = max(0, min(subtotal, total_max_cap))

        if subtotal > total_max_cap:
            alerts.append(f"Tổng điểm {subtotal}đ vượt 100 -> Quy về {total_max_cap}đ (Khoản 4 Điều 4)")

        payload = {
            "raw_by_category": raw_points,
            "capped_by_category": capped_points,
            "subtotal_points": subtotal,
            "total_earned_points": final_score,
            "classification": self._classify_rank(final_score),
            "max_cap_alerts": alerts,
            "unmapped_events": unmapped_events,
        }

        print("\n[BƯỚC 3] Xuất Standardized Payload lên Web Dashboard:")
        print(json.dumps(payload, indent=4, ensure_ascii=False))
        print("=" * 72)
        return payload

    def load_and_map_scraped_events(self, json_path):
        """Đọc danh sách sự kiện từ file JSON scrape và ánh xạ sẵn vào tiêu chí ĐRL."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw_events = json.load(f)

        mapped_list = []
        for ev in raw_events:
            mapped = self.map_event_to_drl_criteria(ev)
            mapped_list.append({
                "post_id": ev.get("post_id"),
                "title": ev.get("title", ""),
                "date": ev.get("date"),
                "organizer": ev.get("organizer"),
                "raw_categories": ev.get("category") or [],
                "raw_format": ev.get("format") or [],
                "post_url": ev.get("post_url"),
                "event_category": mapped.get("matched_key"),
                "category_id": mapped.get("category_id"),
                "sub_code": mapped.get("sub_code"),
                "article_ref": mapped.get("article_ref"),
                "criteria_description": mapped.get("description"),
                "points": mapped.get("default_points", 0),
                "source_event": ev,
            })
        return mapped_list


# =====================================================================
# DỮ LIỆU CẤU HÌNH TRÍCH XUẤT TRỰC TIẾP TỪ FILE DRL2025.PDF (ĐH KHTN)
# =====================================================================
DRL_KHTN_RULES = {
    "rule_version": "1028/QĐ-KHTN-CTSV",
    "total_max_cap": 100,
    "categories": {
        "Muc_1": {"name": "Học tập & NCKH", "max_cap": 30, "base_points": 15, "article_ref": "Khoản 2a Điều 4; Khoản 1 Điều 5 (Tiêu chí 1–3)"},
        "Muc_2": {"name": "Chấp hành nội quy", "max_cap": 20, "base_points": 15, "article_ref": "Khoản 2b Điều 4; Khoản 2 Điều 5 (Tiêu chí 4)"},
        "Muc_3": {"name": "Chính trị, VH-VN, TDTT", "max_cap": 25, "base_points": 10, "article_ref": "Khoản 2c Điều 4; Khoản 3 Điều 5 (Tiêu chí 5–7)"},
        "Muc_4": {"name": "Xây dựng cộng đồng", "max_cap": 25, "base_points": 10, "article_ref": "Khoản 2d Điều 4; Khoản 4 Điều 5 (Tiêu chí 8–10)"},
        "Muc_5": {"name": "Nội dung khuyến khích", "max_cap": 30, "base_points": 0, "article_ref": "Khoản 2e Điều 4; Khoản 5 Điều 5 (Tiêu chí 11–13)"},
        "Muc_6": {"name": "Điểm trừ", "max_cap": None, "base_points": 0, "article_ref": "Khoản 2f Điều 4; Mục 6 Phụ lục"},
    },
    "criteria_mapping": {
        # Mục 1: Học tập & NCKH
        "Academic_Research":      {"category_id": "Muc_1", "sub_code": "1.5", "article_ref": "Khoản 1c Điều 5 (Tiêu chí 3)", "description": "Tác giả đề tài NCKH từ cấp Khoa trở lên", "default_points": 15},
        "Academic_Competition":   {"category_id": "Muc_1", "sub_code": "1.6", "article_ref": "Khoản 1c Điều 5 (Tiêu chí 3)", "description": "Thi học thuật cấp Khoa trở lên", "default_points": 5},
        "Academic_Club":          {"category_id": "Muc_1", "sub_code": "1.8", "article_ref": "Khoản 1c Điều 5 (Tiêu chí 3)", "description": "Sinh hoạt CLB học thuật / PTN / Thư viện", "default_points": 5},
        "Academic_Workshop":      {"category_id": "Muc_1", "sub_code": "1.9", "article_ref": "Khoản 1c Điều 5 (Tiêu chí 3)", "description": "Hội thảo học thuật, kỹ năng, hướng nghiệp", "default_points": 5},

        # Mục 2: Chấp hành nội quy
        "Course_Evaluation":      {"category_id": "Muc_2", "sub_code": "2.2", "article_ref": "Khoản 2 Điều 5 (Tiêu chí 4)", "description": "Đánh giá trên 50% số môn học đã đăng ký", "default_points": 5},
        "Course_Survey":          {"category_id": "Muc_2", "sub_code": "2.3", "article_ref": "Khoản 2 Điều 5 (Tiêu chí 4)", "description": "Khảo sát, điều tra xã hội", "default_points": 5},

        # Mục 3: Chính trị - XH, VH-VN, TDTT
        "Political_Contest":      {"category_id": "Muc_3", "sub_code": "3.1", "article_ref": "Khoản 3a Điều 5 (Tiêu chí 5)", "description": "Cuộc thi chính trị, Mác-Lênin cấp Trường trở lên", "default_points": 10},
        "Sports_Culture_Contest": {"category_id": "Muc_3", "sub_code": "3.1", "article_ref": "Khoản 3a Điều 5 (Tiêu chí 5)", "description": "Cuộc thi văn hóa - văn nghệ, thể dục - thể thao cấp Khoa trở lên", "default_points": 5},
        "Sport_Running":          {"category_id": "Muc_3", "sub_code": "3.2", "article_ref": "Khoản 3a Điều 5 (Tiêu chí 5)", "description": "Giải chạy bộ, Thanh niên khỏe", "default_points": 5},
        "School_Mobilization":    {"category_id": "Muc_3", "sub_code": "3.3", "article_ref": "Khoản 3a Điều 5 (Tiêu chí 5)", "description": "Đợt huy động lực lượng cấp Trường, cấp Khoa", "default_points": 5},
        "Hobby_Club":             {"category_id": "Muc_3", "sub_code": "3.4", "article_ref": "Khoản 3b Điều 5 (Tiêu chí 6)", "description": "Thành viên tích cực CLB sở thích / đội an ninh", "default_points": 5},
        "Class_Faculty_Activity": {"category_id": "Muc_3", "sub_code": "3.5", "article_ref": "Khoản 3a Điều 5 (Tiêu chí 5)", "description": "Hoạt động chung cấp lớp, Khoa", "default_points": 5},
        "Career_Fair":            {"category_id": "Muc_3", "sub_code": "3.8", "article_ref": "Khoản 3c Điều 5 (Tiêu chí 7)", "description": "Tư vấn tuyển sinh, Ngày hội việc làm", "default_points": 5},
        "School_History_Event":   {"category_id": "Muc_3", "sub_code": "3.9", "article_ref": "Khoản 3c Điều 5 (Tiêu chí 7)", "description": "Chương trình tìm hiểu lịch sử, truyền thống Nhà trường", "default_points": 5},

        # Mục 4: Xây dựng cộng đồng
        "Blood_Donation":         {"category_id": "Muc_4", "sub_code": "4.2", "article_ref": "Khoản 4b Điều 5 (Tiêu chí 9)", "description": "Hiến máu tình nguyện", "default_points": 10},
        "Volunteer_Short":        {"category_id": "Muc_4", "sub_code": "4.3", "article_ref": "Khoản 4b Điều 5 (Tiêu chí 9)", "description": "Tình nguyện ngắn hạn (<24 giờ)", "default_points": 5},
        "Volunteer_Medium":       {"category_id": "Muc_4", "sub_code": "4.3", "article_ref": "Khoản 4b Điều 5 (Tiêu chí 9)", "description": "Tình nguyện trung hạn (24 giờ đến <1 tuần)", "default_points": 10},
        "Volunteer_Long":         {"category_id": "Muc_4", "sub_code": "4.3", "article_ref": "Khoản 4b Điều 5 (Tiêu chí 9)", "description": "Tình nguyện dài hạn (>=1 tuần)", "default_points": 15},
        "Legal_Propaganda":       {"category_id": "Muc_4", "sub_code": "4.6", "article_ref": "Khoản 4c Điều 5 (Tiêu chí 10)", "description": "Tuyên truyền Hiến pháp, pháp luật, phòng chống tệ nạn", "default_points": 10},

        # Mục 5: Nội dung khuyến khích
        "Competition_FirstPrize": {"category_id": "Muc_5", "sub_code": "5.2", "article_ref": "Khoản 5a Điều 5 (Tiêu chí 11)", "description": "Giải Nhất cuộc thi cấp Khoa trở lên", "default_points": 5},
        "Scholarship_Hardship":   {"category_id": "Muc_5", "sub_code": "5.7", "article_ref": "Khoản 5b Điều 5 (Tiêu chí 12)", "description": "Học bổng vượt khó vươn lên trong học tập", "default_points": 10},

        # Mục 6: Điểm trừ
        "Penalty_Survey":         {"category_id": "Muc_6", "sub_code": "6.1", "article_ref": "Khoản 2f Điều 4 (Mục 6 Phụ lục)", "description": "Không thực hiện khai báo, điều tra xã hội", "default_points": -5},
        "Penalty_CivicWeek":      {"category_id": "Muc_6", "sub_code": "6.4", "article_ref": "Khoản 2f Điều 4 (Mục 6 Phụ lục)", "description": "Không tham gia đầy đủ tuần sinh hoạt công dân", "default_points": -20},
        "Penalty_Reprimand":      {"category_id": "Muc_6", "sub_code": "6.5", "article_ref": "Khoản 2f Điều 4; Khoản 1 Điều 8", "description": "Bị kỷ luật mức Khiển trách", "default_points": -10},
    },
    # Ánh xạ 43 nhãn category thô xuất hiện trong scrape_1311.json về key chuẩn
    "raw_category_aliases": {
        "academic": "Academic_Workshop",
        "academic & skill sessions": "Academic_Workshop",
        "academic / skills": "Academic_Workshop",
        "workshop": "Academic_Workshop",
        "seminar": "Academic_Workshop",
        "conference": "Academic_Workshop",
        "skill-building": "Academic_Workshop",
        "education": "Academic_Workshop",
        "technology": "Academic_Workshop",
        "exam": "Academic_Workshop",
        "talkshow": "Academic_Workshop",
        "tọa đàm": "Academic_Workshop",
        "scholarship": "Academic_Workshop",
        "career": "Career_Fair",
        "career/job fair": "Career_Fair",
        "competition": "Academic_Competition",
        "competitions": "Academic_Competition",
        "competitions & tournaments": "Academic_Competition",
        "sporting events": "Sport_Running",
        "sports": "Sport_Running",
        "volunteering": "Volunteer_Short",
        "volunteer": "Volunteer_Short",
        "volunteer / community": "Volunteer_Short",
        "festival": "Class_Faculty_Activity",
        "ngày hội": "Class_Faculty_Activity",
        "activity": "Class_Faculty_Activity",
        "entertainment": "Class_Faculty_Activity",
        "cultural": "Class_Faculty_Activity",
        "cultural / community": "Class_Faculty_Activity",
        "culture": "Class_Faculty_Activity",
        "community": "Class_Faculty_Activity",
        "environment": "Class_Faculty_Activity",
        "arts": "Class_Faculty_Activity",
        "arts & culture": "Class_Faculty_Activity",
        "arts & culture/music events": "Class_Faculty_Activity",
        "music": "Class_Faculty_Activity",
        "exhibition": "Class_Faculty_Activity",
        "ceremony": "School_History_Event",
        "orientation": "Class_Faculty_Activity",
        "health": "School_Mobilization",
        "recruitment": "Academic_Club",
        "khác": "Class_Faculty_Activity",
    },
}

# Danh sách hoạt động sinh viên đã hoàn thành (Input mẫu từ Vault)
DEFAULT_COMPLETED_EVENTS = [
    {"title": "Hội thảo AI & Hướng nghiệp", "event_category": "Academic_Workshop", "points": 5},
    {"title": "Cuộc thi Học thuật Khoa CNTT", "event_category": "Academic_Competition", "points": 5},
    {"title": "Khảo sát lấy ý kiến sinh viên", "event_category": "Course_Survey", "points": 5},
    {"title": "Hội thi Olympic Mác-Lênin", "event_category": "Political_Contest", "points": 10},
    {"title": "Giải chạy Việt dã KHTN", "event_category": "Sport_Running", "points": 5},
    {"title": "Ngày hội Việc làm IT", "event_category": "Career_Fair", "points": 5},
    {"title": "Hiến máu nhân đạo đợt 1", "event_category": "Blood_Donation", "points": 10},
    {"title": "Chiến dịch Mùa Hè Xanh", "event_category": "Volunteer_Long", "points": 15},
]

if __name__ == "__main__":
    engine = CreditCalculationModule(DRL_KHTN_RULES)
    engine.calculate_earned_points(DEFAULT_COMPLETED_EVENTS)

    scrape_file = Path(__file__).resolve().parent / "scrape_1311.json"
    if scrape_file.exists():
        mapped_scraped = engine.load_and_map_scraped_events(scrape_file)
        unmapped = [x["title"] for x in mapped_scraped if x["category_id"] == "Unmapped"]
        print(f"\n[SCRAPE CHECK] Đã đọc {len(mapped_scraped)} sự kiện từ scrape_1311.json | Unmapped: {len(unmapped)}")
