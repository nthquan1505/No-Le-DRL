# Báo cáo Triển khai & Future Work — Hệ thống Đề xuất Hoạt động ĐRL (P4 Demo · Demo M4/M6 · P5 · P6 · UI)

> **Ngày lập báo cáo:** 27/11/2026
> **Phạm vi:** Triển khai bản Demo đầy đủ luồng đề xuất hoạt động (P4 Demo User Profile, Demo Module 4, Demo Module 6, P5 Hard Filter, P6 Multi-Criteria Scoring và Giao diện Web UI Tab 3) dựa theo phương án thiết kế đã duyệt tại `docs/module3_p4_p5_p6_design.md` (v2.0) và đặc tả bàn giao Module 4 tại `docs/module4_fit_handoff_spec.md`.
> **Trạng thái kiểm thử:** **40/40 unit & integration tests PASS** + **Live HTTP API & `e5-small` Smoke Test PASS**.

---

## PHẦN A — NHỮNG GÌ ĐÃ THỰC HIỆN ĐƯỢC

### 1. Kiến trúc & Danh sách mã nguồn đã triển khai

Toàn bộ các thành phần được xây dựng tách biệt theo đúng ranh giới kiến trúc đã phê duyệt (`D-3`, `D-4`, `D-5`, `D-6`), sử dụng các giao diện chuẩn (`ScheduleProvider`, `CreditCalculatorProtocol`) để có thể cắm thay thế bằng Module 4 và Module 6 bản chính thức mà không cần sửa logic P5/P6.

| Thành phần | File triển khai | Vai trò & Điểm nổi bật |
|---|---|---|
| **Cấu hình P5/P6 & Demo M4/M6** | `configs/recommendation_config.yaml` | Tham số hóa đầy đủ `demo_module4`, `filter` (P5), `scoring` (`gain`, `interest`, `urgency`, `penalty_mandatory`), giữ tương thích ngược hoàn toàn với cấu hình Phase 4. |
| **Demo Module 4 (Schedule & Fit)** | `src/recommendation/demo_module4.py` | Lớp `DemoScheduleProvider` kế thừa `LocalScheduleAdapter`, triển khai đầy đủ `check_conflict`, `estimate_travel_minutes`, `nearest_slack_minutes`, 3 preset lịch (`light_schedule`, `busy_weekend`, `exam_week_heavy`) và hàm `compute_fit` liên tục 5 bước (đạt **8/8** test case nghiệm thu `TC1–TC8` của `docs/module4_fit_handoff_spec.md`, không dính lệ phí). |
| **Demo Module 6 (Credit & Mandatory)** | `src/recommendation/demo_module6.py` | Lớp `DemoCreditProvider` bọc `RubricCreditEngine` (`data/raw/rubric.yaml`, trần thật HCMUS `30/20/25/25/30`), cho phép truyền trực tiếp 5 con số điểm hiện có `C1..C5` (3 preset: `mid_semester`, `c1_capped`, `near_target`) và trích xuất `MandatoryPenaltyInfo` cho hoạt động bắt buộc cao (`6.4` Tuần SHCD / cờ `is_mandatory`). |
| **P4 · Demo User Profile** | `src/recommendation/demo_profile.py` | Lớp `DemoUserProfile` gọn nhẹ (chọn chip sở thích + mô tả tự do $\to$ vector tĩnh `384d` từ `e5-small`, các ràng buộc lọc cứng, mốc `reference_now`, preset/custom `w1..w4`), tích hợp sẵn 4 Persona Demo và hàm sinh sự kiện mẫu bắt buộc `6.4`. |
| **P5 · Hard Constraint Filter** | `src/recommendation/filter.py` | Cổng lọc đầu vào **chỉ nhận `classification.status == "PREDICTED"`** (39/108 sự kiện, tách riêng 69 `NEEDS_REVIEW`), kiểm tra **không dừng sớm (non-short-circuit)** cả 8 luật (`already_participated`, `hidden_by_user`, `expired`, `audience_mismatch`, `schedule_conflict`, `over_budget`, `too_far`, `no_credit_gain`), xử lý an toàn `reg_end_after_start` và danh sách đối tượng rỗng, đồng thời **giữ ngoại lệ cho hoạt động bắt buộc cao** đi qua lọc `effective_points == 0`. |
| **P6 · Multi-Criteria Scoring** | `src/recommendation/scoring.py` | Công thức chuẩn `D-5`: `Score = w1·Gain + w2·Interest + w3·Fit + w4·Urgency + Penalty`.<br>• `Gain`: từ Demo M6 (`effective_points`, `target_gap_by_group`, `source`).<br>• `Interest`: **Clipped Min-Max** trên cosine E5, tự động duyệt qua toàn bộ **908 vector** trong `sample_embeddings.npy` để tính dải thực nghiệm `[c_min, c_max]` theo từng hồ sơ (hỗ trợ chuyển đổi `[min, max]`, `[p1, p99]`, hoặc ngưỡng cố định).<br>• `Fit`: từ Demo M4 `compute_fit`.<br>• `Urgency`: thuần thời gian `exp(-d / 5.0)` (đã bỏ `seats_left`).<br>• `Penalty`: **chỉ can thiệp đẩy ưu tiên cho hoạt động bắt buộc cao** (`+1.00` cho `6.4` trừ `-20`đ; `+0.50` cho `is_mandatory`; `0.00` cho sự kiện thường). |
| **Pipeline & Facade** | `src/recommendation/pipeline.py`, `src/recommendation/__init__.py`, `src/recommender.py` | Hàm `load_recommendation_candidate_pool()` hợp nhất `test_fixtures_events.json` + `classification_events.json` + `events_query_embeddings.npy`; hàm `run_recommendation_demo()` điều phối trọn gói `P4 -> M4/M6 -> P5 -> P6`. |
| **Backend API** | `app.py` | Bổ sung 2 route tách biệt `GET /api/recommendation/bootstrap` và `POST /api/recommendation/run` (giữ nguyên 100% `/api/bootstrap` để không ảnh hưởng kiểm thử hiện có). |
| **Web UI Demo (Tab 3)** | `prototype/index.html`, `prototype/app.js`, `prototype/styles.css` | Bổ sung **Tab 3: "Đề xuất hoạt động (Demo)"** với 2 cột trực quan: bảng điều khiển Persona / Sở thích / Ràng buộc P5 / Demo M6 (`C1..C5`) / Demo M4 (thêm-xóa slot bận & tuần thi) / Trọng số `w1..w4` và bảng kết quả gồm KPI pipeline, Sub-tab **Xếp hạng P6** (thanh phân rã 5 thành phần + nút test nhanh *Đã tham gia / Ẩn*) và Sub-tab **Bị lọc cứng P5 ("Vì sao bị loại?")**. |
| **Bộ kiểm thử tự động** | `tests/test_recommendation_p5_p6.py` | 16 test case bao phủ 8 TC chuẩn của `compute_fit`, Demo M6, cổng `PREDICTED`-only, đủ 8 luật P5, ngoại lệ bắt buộc `6.4`, Clipped Min-Max `Interest`, `Urgency` thuần thời gian, và tích hợp trên toàn bộ 4 persona thực tế. |

---

### 2. Kết quả Kiểm thử Tự động & Kiểm chứng Thực tế

#### 2.1. Kết quả `pytest` & HTTP API Smoke Test

| Bộ kiểm thử | Kết quả | Chi tiết xác minh |
|---|---|---|
| `tests/test_recommendation_p5_p6.py` | **16 / 16 PASSED** | Đạt 100% cả 8 Acceptance Tests (`TC1–TC8`) của Demo Module 4, toàn bộ 8 luật P5, tính toán P6 và 4 Persona trên tập 108 sự kiện thật. |
| `tests/test_schedule_adapter.py` | **4 / 4 PASSED** | Không làm ảnh hưởng `LocalScheduleAdapter` gốc. |
| `tests/test_module6_credit_engine.py` | **11 / 11 PASSED** | Không làm ảnh hưởng `RubricCreditEngine` gốc. |
| `tests/test_schemas.py` | **9 / 9 PASSED** | Mọi contract schema (`D-1`–`D-6`) giữ nguyên tính toàn vẹn. |
| **Tổng bộ test xác minh** | **40 / 40 PASSED** *(7.53s)* | Không phát sinh regression ở bất kỳ module liên quan nào. |
| **Live HTTP Smoke Test** (`app.py`) | **PASSED (HTTP 200)** | Xác minh `GET /api/recommendation/bootstrap` (trả `predicted_count = 39`, `sample_embeddings_count = 908`) và `POST /api/recommendation/run` trên `ThreadingHTTPServer` thực với `intfloat/multilingual-e5-small`. |

*(Lưu ý minh bạch về test có sẵn ngoài phạm vi: `tests/test_user_profile.py::test_behavior_vector_adaptation_and_reset` thất bại sẵn từ trước trong `user_profile.py` chưa chỉnh sửa do phép cộng/trừ cùng 1 vector đơn vị sau chuẩn hóa L2 cho cùng hướng vector; lỗi này được ghi nhận vào mục Future Work.)*

#### 2.2. Kết quả chạy thực tế với `intfloat/multilingual-e5-small` trên 4 Persona Demo

Dữ liệu đầu vào cố định: **108 sự kiện** $\to$ **39 sự kiện `PREDICTED`** (bỏ qua **69 `NEEDS_REVIEW`**), hiệu chuẩn `Interest` trên **908 mẫu** `sample_embeddings.npy`:

| Persona Demo | Mốc ngày (`reference_now`) | Qua P5 / Bị loại P5 | Lý do loại P5 chính | Dải `Interest` `[c_min, c_max]` | Hạng #1 (Score & phân rã) |
|---|---|---|---|---|---|
| **1. `balanced_cs1`** *(SV CNTT CS1, thích AI & NCKH, ngân sách 30k)* | `2026-10-01` | **17** đạt / **22** bị loại | `expired` (21), `schedule_conflict` (6), `over_budget` (1) | `[0.785, 0.921]` | **#1** *Tham quan Công ty Netcompany VN* (`1.9/C1`, +5đ) — **Score = `0.494`** *(G=0.247, I=0.685, F=0.760, U=0.480, P=0.000)*; **#2** *Lớp PP NCKH 2026* (`0.480`, F=0.920); **#3** *Kỹ năng thời đại số* (`0.473`, I=0.868) |
| **2. `c1_capped_need_points`** *(Chạm trần `C1 = 30/30`, ưu tiên điểm)* | `2026-04-01` | **9** đạt / **30** bị loại | `no_credit_gain` (**30** — loại sạch 100% nhóm C1), `schedule_conflict` (6), `expired` (3), `over_budget` (1) | `[0.788, 0.916]` | **#1** *BƯỚC CHÂN YÊU THƯƠNG* (`4.3/C4`, +10đ) — **Score = `0.541`** *(G=0.630, I=0.743, F=0.550, U=0.000, P=0.000)*; các hạng #2–#5 đều thuộc `3.5/C3` |
| **3. `exam_week_busy`** *(Tuần thi `09/11–22/11`, bận T7 & thi `16/11`, ngân sách `0đ`, `max_travel=30p`)* | `2026-11-01` | **4** đạt / **35** bị loại | `expired` (28), `schedule_conflict` (16), `too_far` (11), `over_budget` (5) | `[0.764, 0.898]` | **#1** *Series "Bản đồ Văn hóa" — Số 4 (Online)* (`3.5/C3`, +5đ) — **Score = `0.523`** *(F=0.711 dù trong tuần thi)*; sự kiện offline trong tuần thi (`18/11`) bị hạ mạnh `Fit` xuống `0.396` (hạng #4) |
| **4. `mandatory_shcd_test`** *(Chèn sự kiện bắt buộc mẫu `6.4` Tuần SHCD)* | `2026-11-01` | **12** đạt / **28** bị loại *(trên 40 PREDICTED)* | `expired` (28), `schedule_conflict` (6), `over_budget` (1) | `[0.785, 0.907]` | **#1** *[BẮT BUỘC] Tuần Sinh hoạt Công dân* (`6.4/P`, **+0đ**) — **Score = `1.393`** *(G=0.000, I=0.548, F=0.920, U=0.480, **P=+1.000**)* — vượt lên Top 1 đúng thiết kế |

---

### 3. Hướng dẫn chạy Demo UI & Kiểm thử nhanh

1. Khởi động web server bằng môi trường Python đã cài đầy đủ thư viện (`PyYAML`, `numpy`, `pandas`, `pyarrow`, `sentence-transformers`, `torch`):
   ```bash
   python app.py
   ```
2. Mở trình duyệt tại `http://127.0.0.1:8765` và chọn **Tab 3: "Đề xuất hoạt động (Demo)"**.
3. Thao tác kiểm chứng nhanh:
   - Bấm lần lượt 4 nút **Kịch bản Persona nhanh** ở góc trên bên trái để đối chiếu P5 và P6.
   - Chuyển qua lại giữa **Xếp hạng Đề xuất P6** và **Bị lọc cứng P5 · Vì sao bị loại?** để kiểm tra từng lý do loại.
   - Kéo điểm `C1` lên `30` (Demo Module 6) hoặc thêm một khung giờ bận mới (Demo Module 4) rồi bấm **Chạy Đề Xuất (P5 + P6)**.
   - Bấm **Đã tham gia** hoặc **Ẩn** ngay trên thẻ sự kiện để thấy sự kiện di chuyển tức thì sang tab **Bị lọc cứng P5** (riêng sự kiện bắt buộc `6.4` sẽ không bị ẩn bởi nút **Ẩn**).

---

## PHẦN B — HẠN CHẾ HIỆN TẠI & FUTURE WORK CẦN LÀM ĐỂ HOÀN THIỆN / CẢI TIẾN

### 1. Ưu tiên Cao (`P0` — Cần hoàn thiện cho bản tích hợp hệ thống)

1. **Thay `DemoScheduleProvider` bằng Module 4 bản chính thức:**
   - Hiện Demo M4 dùng lịch mẫu/nhập tay và bảng ước lượng thô thời gian di chuyển theo 2 cơ sở (`10p` cùng cơ sở, `45p` khác cơ sở, `30p` chưa rõ).
   - **Cần làm:** kết nối Module 4 thật đọc TKB trực tiếp từ Portal / file `.ics`, sử dụng vị trí tiết học kề trước để tính `travel_min` thực tế (qua OSRM / bản đồ / ma trận khoảng cách khu vực TP.HCM) và tuân thủ đúng 8 test case hợp đồng trong `docs/module4_fit_handoff_spec.md`.
2. **Thay `DemoCreditProvider` bằng Module 6 bản chính thức & đồng bộ Module 7:**
   - Hiện Demo M6 nhận 5 con số tổng điểm `C1..C5` nên chưa kiểm tra được luật `exclusive_with` theo từng mã mục đã tham gia thực tế hoặc xác nhận trạng thái hoàn thành minh chứng.
   - **Cần làm:** đọc `UserLedger` thật từ Module 6 kết hợp xác nhận minh chứng hoàn thành từ Module 7 (`Evidence Vault`), đồng thời tự động cập nhật `participated_event_ids`.
3. **Giảm tỷ lệ `NEEDS_REVIEW` ở P2 (Classification) để mở rộng nguồn sự kiện cho Recommendation:**
   - Khi chỉ sử dụng kết quả `PREDICTED`, nguồn đầu vào Recommendation hiện chỉ còn **39/108 sự kiện (36.1%)** và bị lệch phân bố mạnh về nhóm `C1` (**30/39** sự kiện là `1.9`/`1.6`, chỉ có `8` sự kiện `C3`, `1` sự kiện `C4`, `0` sự kiện `C2/C5` và chưa có bài đăng `6.4` thật).
   - **Cần làm:** triển khai các khuyến nghị từ báo cáo Decision Policy Audit (cải thiện phân biệt `3.1` vs `3.5`, `1.9` vs `3.5`, bổ sung mẫu cho nhóm `C2`/`C4`, xử lý các câu hỏi mở `Q9–Q23` trong `rubric.yaml`) và bổ sung cơ chế duyệt tay (human-in-the-loop) cho các ca `NEEDS_REVIEW` trên UI.
4. **Chuẩn hóa trích xuất thời gian & đối tượng từ upstream M1/M2:**
   - Trong `test_fixtures_events.json` vẫn còn `47/108` sự kiện thiếu `reg_end_datetime`, `16/108` sự kiện có `reg_end_datetime > start_datetime`, và `79/108` sự kiện có `target_audience` chưa tách được `cohorts/faculties/classes` có cấu trúc. Cần cải thiện prompt/parser của M2 để P5 lọc đối tượng và P6 tính `Urgency` chính xác hơn.

### 2. Ưu tiên Trung bình (`P1` — Hoàn thiện đầy đủ kiến trúc Module 3 theo Flowchart)

5. **Triển khai P7 · Selection (Lựa chọn danh sách đa dạng & Lập kế hoạch lịch trình):**
   - P6 hiện chấm điểm từng sự kiện **độc lập**, chưa ngăn việc Top-5 chứa 2 sự kiện trùng giờ với nhau hoặc tổng điểm cộng dồn của Top-5 vượt trần nhóm `remaining_cap_by_group`.
   - **Cần làm:**
     - **Chế độ danh sách "Dành cho bạn":** cài đặt **MMR (Maximal Marginal Relevance)** kèm **1 slot khám phá** để đa dạng hóa chủ đề và nhóm tiêu chí.
     - **Chế độ "Lập kế hoạch học kỳ/tuần":** mô hình hóa bài toán tối ưu tập chọn bằng **ILP / OR-Tools CP-SAT** (có greedy fallback) đảm bảo không trùng lịch giữa các sự kiện được chọn và tôn trọng trần điểm từng nhóm.
6. **Triển khai P8 · Explanation Engine & kết nối Module 5 (Notification):**
   - Sinh tự động tối đa 3 lý do tự nhiên có kiểm chứng số liệu (numeric grounding) từ breakdown điểm của P6 và bảng `DropReason` của P5.
   - Đẩy các sự kiện có `Urgency` cao, sự kiện đã lưu và đặc biệt là sự kiện bắt buộc (`Penalty > 0`) sang **Module 5** để gửi nhắc hạn kịp thời.
7. **Mở rộng quy tắc nhận diện hoạt động bắt buộc (`Penalty`):**
   - Xây dựng danh mục chuẩn hóa các hoạt động bắt buộc theo quy chế hiện hành của Trường (không chỉ mục `6.4` Tuần SHCD mà cả `2.2` Đánh giá môn học liên kết `6.1` trừ `-5`đ, các đợt huy động lực lượng bắt buộc của Khoa/Trường, v.v.) và có tín hiệu nhận diện ổn định từ M1/M2 thay vì chỉ phụ thuộc fixture demo.

### 3. Ưu tiên Nâng cao (`P2` — Cá nhân hóa, Hiệu chỉnh Định lượng & Đánh giá)

8. **Khôi phục & sửa lỗi vòng học thích nghi P4 (`UserProfileManager` + `Feedback Store`):**
   - Lưu bền vững hành vi (`click`, `save`, `joined`, `dismiss`, `label_correction`) vào `Feedback Store` (SQLite/JSONL).
   - Bổ sung cập nhật trọng số động theo ngữ cảnh (`Preset + Shortfall M6 + Lịch bận M4` như thiết kế hộp P4 trên flowchart).
   - Khắc phục lỗi toán học đã ghi nhận ở `tests/test_user_profile.py::test_behavior_vector_adaptation_and_reset`: khi `u_learned_accum` chỉ chứa 1 hướng vector đơn vị duy nhất (`1.5 * v - 0.4 * v = 1.1 * v`), phép chuẩn hóa L2 giữ nguyên vector `u_learned = v` khiến tín hiệu `dismiss` không giảm độ tương đồng; cần chuyển sang công thức online Rocchio trực tiếp `u <- normalize(u - eta_neg * e)` hoặc điều chỉnh trọng số trộn `beta` theo độ lớn tích lũy.
9. **Hiệu chỉnh bộ trọng số `(w1, w2, w3, w4)` và đánh giá `Interest` bằng dữ liệu người dùng thực:**
   - So sánh thực nghiệm giữa 3 phương án scale `Interest` (`[min, max]` động trên 908 sample vs `[p1, p99]` vs ngưỡng cố định) để chọn phương án ít nhạy với nhiễu nhất khi tập sample mở rộng.
   - Tổ chức khảo sát so sánh cặp trên 15–20 sinh viên, khớp mô hình **Bradley–Terry (Logistic Regression 4 tham số)** để tìm bộ trọng số thực nghiệm tối ưu, kiểm chứng độ bền vững bằng **phân tích độ nhạy ($\pm 20\%$, Kendall's $\tau$)** và đánh giá offline bằng **nDCG@5 / Overlap@5** so với baseline sắp xếp theo thời gian hoặc chỉ theo điểm.
10. **Cập nhật sơ đồ `module3_engine_flowchart.drawio.svg` & đồng bộ quy chế mới nhất:**
    - Sửa 6 điểm lệch đã ghi trong `docs/module3_decision_log.md` (bao gồm lỗi typo `w₁..w₅` trên cạnh `P4 → P6`) và cập nhật `data/raw/rubric.yaml` ngay khi có khung điểm ĐRL hiện hành thay thế QĐ 1028/2020.
