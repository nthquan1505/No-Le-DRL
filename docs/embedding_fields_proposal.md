# Đề xuất Lựa chọn Trường Thông tin cho Embedding: Phạm vi Classification (P2) & Recommendation (P6)

> **Ngày lập:** 27/11/2026
> **Trạng thái:** `PROPOSAL — ĐỀ XUẤT THIẾT KẾ CHO GIAI ĐOẠN CẢI TIẾN TIẾP THEO (Chưa thay đổi pipeline embedding hiện tại)`
> **Cơ sở phân tích thực tế:**
> - Toàn bộ **32 trường** trên **108 sự kiện** trong `data/events.json`.
> - Cấu trúc **908 mẫu** tham chiếu trong `data/raw/activities_sample.csv` (`STT, Ma_Muc, Organizer, Ten_Hoat_Dong, Vai_Tro, Tieu_Chi_Quy_Che, Diem_Toi_Da`).
> - Cấu trúc **176 prototype** trong `data/raw/rubric.yaml` / `rubric_prototypes.jsonl`.
> - Model hiện hành: `intfloat/multilingual-e5-small` (`384d`, mean pooling, chuẩn hóa L2, prefix `query: ` / `passage: `).

---

## 1. Hiện trạng Embedding trong Hệ thống

Hiện tại cả hai chức năng **Classification (P2)** và **Recommendation (`Interest` — P6)** đang dùng chung một file vector `data/generated/events_query_embeddings.npy`, được tạo bởi `src/classification/embeddings/event_text_builder.py`:

| Vector | Trường đưa vào embedding | Prefix E5 | Độ dài tiêu biểu |
|---|---|---|---|
| **Event Query** (`108` sự kiện) | `title` + `description` | `query: ` | 46–172 tokens (trung vị 80.5) |
| **Sample Branch** (`908` mẫu `activities_sample.csv`) | Chỉ `Ten_Hoat_Dong` (tên hoạt động) | `passage: ` | 6–44 tokens (trung vị ~14) |
| **Rubric Branch** (`176` prototype `rubric.yaml`) | `name`, `scope_note`, `anchors`, `keywords` | `passage: ` | 7–467 tokens |
| **User Profile** (`user_vec` P4) | Text của các chip sở thích + `free_text` | `query: ` | ~15–60 tokens |

**Thống kê độ phủ các trường chính trong `data/events.json` (108 sự kiện):**
- `title`: `108/108` | `description`: `108/108` (20–98 từ, trung vị 47 từ)
- `category`: `108/108` (43 chuỗi tiếng Anh/Việt chưa chuẩn hóa: `Competition`, `Academic`, `Workshop`, `Volunteer`, `Festival`, `Sports`, `Technology`, `khác`…)
- `format`: `107/108` (trộn lẫn hình thức tổ chức `Offline/In-person/Online` với loại hình `Seminar/Workshop/Talkshow/Contest/Competition/Festival/Tour`)
- `organizer`: `101/108` (**88/101** không xuất hiện nguyên văn trong `title + description`)
- `target_audience`: `107/108` (phần lớn lặp `"Sinh viên"`, `"Sinh viên HCMUS"`)
- `location`: `98/108` | `online_mode`: `106/108` | `fee`: `24/108` | `fanpage_name`: `26/108` | `reason`: `108/108` (câu tiếng Anh do LLM M2 giải thích `is_event=true`)

---

## 2. Đề xuất cho Phạm vi Classification (P2 — Phân loại Mục ĐRL 2 nhánh)

### 2.1. Đặc thù của bài toán Classification cần tôn trọng
1. **Chống rò rỉ nhãn (Zero Data Leakage):** Tuyệt đối không đưa các trường chứa nhãn quy chế (`Ma_Muc`, `Tieu_Chi_Quy_Che`, `Diem_Toi_Da`) hay câu giải thích suy đoán của LLM (`reason`).
2. **Tính đối xứng miền giữa Query và Reference (`Domain Alignment`):** Vector query sự kiện được so trực tiếp với **Sample Branch** (hiện chỉ embed `Ten_Hoat_Dong` tiếng Việt ngắn) và **Rubric Branch** (văn bản quy chế tiếng Việt). Nếu nhồi thêm nhiều trường dài hoặc tiếng Anh chưa chuẩn hóa vào Query mà không đồng bộ với Sample/Rubric sẽ làm **giảm cosine** và tăng tỷ lệ bất đồng giữa hai nhánh (`branch_disagreement`).
3. **Trọng tâm phân biệt các cặp mục hay nhầm trong báo cáo Audit:** `1.6` (cuộc thi học thuật) vs `1.9` (hội thảo/workshop chuyên môn) vs `3.1` (cuộc thi chính trị-văn hóa-thể thao) vs `3.5` (hoạt động phong trào/ngày hội) vs `3.8` (ngày hội việc làm/tư vấn tuyển sinh) vs `4.3`/`4.5` (tình nguyện/quyên góp).

### 2.2. Đánh giá từng trường trong `data/events.json` cho Classification

| Trường | Khuyến nghị | Lý do & Điều kiện áp dụng |
|---|---|---|
| **`title` + `description`** | **GIỮ (Bắt buộc)** | Nguồn ngữ nghĩa trung thực nhất, không gây rò rỉ nhãn. |
| **`format` *(chỉ phần loại hình hoạt động, dịch sang tiếng Việt)*** | **NÊN THÊM (Ưu tiên #1)** | Phần loại hình như `Competition/Contest → Cuộc thi`, `Seminar/Workshop/Talkshow → Hội thảo chuyên đề`, `Festival → Ngày hội`, `Volunteer → Tình nguyện`, `Exam → Kỳ thi` giúp phân tách trực diện các biên dễ nhầm (`1.6` vs `1.9`, `3.1` vs `3.5`, `4.3`). **Phải loại bỏ `Offline`/`In-person`/`Online`** vì hình thức tổ chức không quyết định mục ĐRL. |
| **`organizer` *(đã chuẩn hóa & đồng bộ với Sample Branch)*** | **CÂN NHẮC CAO (Ưu tiên #2 — cần đồng bộ 2 phía)** | Mang tín hiệu rất mạnh về nhóm ĐRL (`Khoa CNTT / Vật lý` $\to$ `C1`; `CTSV` $\to$ `C2/3.3`; `ĐTN / HSV / CTXH` $\to$ `C3/C4`).<br>**Điều kiện bắt buộc:**<br>1. Bỏ cụm lặp `"Trường Đại học Khoa học tự nhiên, ĐHQG-HCM"` để tránh kéo mọi vector sát nhau.<br>2. Ánh xạ chuẩn hóa viết tắt (`CNTT ↔ Khoa Công nghệ Thông tin`, `ĐTN ↔ Đoàn Thanh niên`, `HSV ↔ Hội Sinh viên`, `CTSV ↔ Phòng Công tác Sinh viên`, `CTXH ↔ Đội Công tác Xã hội`).<br>3. **Đồng bộ thêm `Organizer` (đã chuẩn hóa) vào `sample_embeddings.npy`** (vì `1,210/1,330` dòng trong `activities_sample.csv` có cột `Organizer`!) — hoặc giữ `organizer` làm **prior/feature ngoài embedding** ở bước fusion để tránh lỗi đồng tổ chức đa CLB (như `sample_005`). |
| **`category` *(ánh xạ sang tập từ vựng tiếng Việt cố định)*** | **CHỈ THỬ NGHIỆM CÓ KIỂM SOÁT (Ưu tiên #3)** | `category` giúp tách nhanh nhóm lớn Stage 1 (`Academic`, `Volunteering`, `Sports`, `Career/Job Fair`), **nhưng tiềm ẩn 2 rủi ro lớn cho P2:** (a) `activities_sample.csv` không có cột `category`; (b) `category` là nhãn tự do do LLM ở M2 sinh ra (43 chuỗi lộn xộn) nên nếu M2 gắn sai sẽ truyền nhiễu thẳng vào P2.<br>$\Rightarrow$ Nếu dùng, phải **ánh xạ về 6–8 nhãn tiếng Việt cố định**, loại bỏ `"khác"`/`"Activity"`, và **đo ablation trên tập validation có nhãn người thật** trước khi bật chính thức. |
| **`role` *(tương lai — khi bật phân loại biến thể vai trò)*** | **ĐỂ SAU (theo `T-4`)** | Hiện các mục vai trò BTC/CTV (`1.7`, `3.14`, `4.4`) do Module 6 chuyển hướng. Chỉ thêm `Vai trò: {role}` vào embedding khi có dữ liệu bài đăng tuyển BTC/CTV thực tế và quyết định đưa `role_variant` trở lại P2. |
| **`location`, `online_mode`, `is_online`, `fee`, `start/end/reg_datetime`, `date`, `time`** | **KHÔNG THÊM** | Không quyết định tiêu chí ĐRL; đưa vào chỉ làm loãng trọng tâm ngữ nghĩa văn bản. |
| **`target_audience`** | **KHÔNG THÊM** | `79+` dòng lặp chung chung `"Sinh viên"`, `"Sinh viên HCMUS"`; không giúp phân biệt mục ĐRL (trừ trường hợp đặc thù tuyển đại biểu `3.12`/`3.13` vốn đã thể hiện rõ trong `title` + `description`). |
| **`reason`, URLs, `post_id`, `fanpage_name`, `scraped_at`, `extraction_method`** | **TUYỆT ĐỐI KHÔNG THÊM** | `reason` mang văn phong tiếng Anh lặp của M2 và dễ rò rỉ suy đoán ngoài nội dung bài đăng; các trường còn lại là metadata kỹ thuật thuần túy. |

### 2.3. Mẫu chuỗi Embedding đề xuất để thử nghiệm cho Classification (P2)

- **Baseline hiện tại (`v1`):**
  ```text
  query: {title} {description}
  ```
- **Ứng viên cải tiến `C-v2` (nhẹ, không cần sinh lại Sample Branch):**
  ```text
  query: {title}. Loại hình: {format_activity_vi}. {description}
  ```
- **Ứng viên cải tiến `C-v3` (đồng bộ cả Query và Sample Branch — cần tái sinh `sample_embeddings.npy`):**
  - Event Query: `query: {title}. Loại hình: {format_activity_vi}. Đơn vị: {organizer_normalized}. {description}`
  - Sample Passage: `passage: {Ten_Hoat_Dong}. Đơn vị: {organizer_normalized}` *(khi `Organizer` khác rỗng)*

> **Bổ sung kỹ thuật đáng chú ý cho P2:** Vì `Ten_Hoat_Dong` trong Sample Branch rất ngắn (~14 tokens) còn `title + description` của Event Query dài (~80 tokens), ngoài việc chọn trường, nhóm có thể thử nghiệm **tách trọng số `title` cao hơn** (hoặc multi-vector: `0.6 * emb(title) + 0.4 * emb(title + description)`) để giảm chênh lệch độ dài giữa Event Query và Sample Branch!

---

## 3. Đề xuất cho Phạm vi Recommendation (P6 — Khớp Sở thích `Interest`)

### 3.1. Đặc thù của bài toán Recommendation khác với Classification
1. **Mục tiêu là khớp chủ đề/trải nghiệm sinh viên yêu thích**, không phải khớp điều khoản quy chế — nên **không lo rò rỉ nhãn ĐRL**.
2. **Độ đồng nhất prefix E5 (`query ↔ passage`):** Hiện `user_vec` dùng prefix `query: ` và 908 `sample_embeddings` (dùng đo dải `[c_min, c_max]`) dùng prefix `passage: `, nhưng `event_vec` lại đang mang prefix `query: ` (do mượn từ P2). Tách riêng embedding Recommendation với prefix **`passage: `** giúp phép so sánh `user_vec (query) ↔ event_vec (passage)` cùng đúng không gian với `user_vec (query) ↔ sample_embeddings (passage)`.
3. **Các yếu tố thời gian, địa điểm, hình thức online/offline, học phí, đối tượng đã có tiêu chí chuyên trách riêng** (`P5 Hard Filter`, `M4 Fit`, `P6 Urgency`), nên embedding `Interest` chỉ nên tập trung 100% vào **chủ đề & trải nghiệm hoạt động**.

### 3.2. Đánh giá từng trường trong `data/events.json` cho Recommendation

| Trường | Khuyến nghị | Lý do & Cách xử lý |
|---|---|---|
| **`title` + `description`** | **GIỮ (Bắt buộc)** | Mô tả trực tiếp nội dung và trải nghiệm sinh viên nhận được. |
| **`category` *(đã chuẩn hóa & dịch sang tiếng Việt)*** | **NÊN THÊM (Ưu tiên #1)** | Độ phủ `108/108`. Giúp các bài đăng có tiêu đề ẩn dụ (vd *"BƯỚC CHÂN YÊU THƯƠNG"*, *"NÉT SON"*, *"SẮC MÀU QUÂN NGŨ"*) khớp chuẩn với 8 chip sở thích tiếng Việt. Cần loại bỏ `"khác"`, `"Activity"` và chuẩn hóa 43 chuỗi về bảng ánh xạ gọn (vd `Volunteering/Volunteer → Tình nguyện, cộng đồng`; `Technology → Công nghệ, lập trình`; `Career/Job Fair → Hướng nghiệp, việc làm`; `Sports/Sporting Events → Thể thao`; `Arts/Cultural/Music → Văn hóa, nghệ thuật, âm nhạc`). |
| **`format` *(chỉ phần loại hình, bỏ `Offline/Online`)*** | **NÊN THÊM (gộp cùng `category`)** | Bổ sung phong cách tham gia (`Workshop`, `Seminar`, `Talkshow`, `Cuộc thi`, `Ngày hội`, `Tham quan doanh nghiệp`) nếu chưa có trong `category`. **Tuyệt đối lọc bỏ `Offline`/`In-person`/`Online`** để không trùng với `Fit`. |
| **`organizer` *(rút gọn tên trường)*** | **CÂN NHẮC THẬN TRỌNG (Ưu tiên #2 — cần test A/B)** | Hữu ích khi sinh viên thích theo dõi sự kiện từ Khoa/CLB cụ thể (vd `Khoa CNTT`, `CLB Kỹ Năng`), nhưng dễ gây **khớp giả khi nhiều CLB đồng tổ chức** (vd `sample_005` workshop vẽ quạt/nặn tò he có `CLB Anh văn BEE` đồng tổ chức). Chỉ nên bật sau khi cắt bỏ tên trường lặp lại và kiểm chứng không làm giảm chất lượng Top-5. |
| **`start/end/reg_datetime`, `location`, `online_mode`, `fee`, `target_audience`, `reason`, URL/metadata crawl** | **KHÔNG THÊM** | Đã xử lý chính xác ở P5/M4/Urgency hoặc là nhiễu kỹ thuật; đưa vào sẽ gây tính trùng tiêu chí hoặc làm co cụm cosine. |

### 3.3. Mẫu chuỗi Embedding đề xuất cho Recommendation (P6)

- **Hiện tại (`R-v1` — dùng chung vector P2):**
  ```text
  query: {title} {description}
  ```
- **Đề xuất chuẩn (`R-v2` — tách riêng file `data/generated/events_rec_embeddings.npy`):**
  ```text
  passage: {title}. Chủ đề: {category_vi_normalized}. {description}
  ```
  *(Trong đó `category_vi_normalized` gộp `category` đã chuẩn hóa + phần loại hình hoạt động từ `format`, đã loại bỏ `Offline/In-person/Online/khác/Activity`.)*

---

## 4. So sánh Kiến trúc: 1 Vector Dùng Chung vs 2 Vector Chuyên Biệt

| Tiêu chí so sánh | Phương án 1: 1 Shared Vector *(Hiện tại)* | Phương án 2: Tách 2 Vector (`P2` & `P6` riêng) *(Khuyến nghị tương lai)* |
|---|---|---|
| **Số file embedding / chi phí tính** | 1 file (`events_query_embeddings.npy`), nhẹ nhất (~165 KB cho 108 sự kiện) | 2 file (`events_query_embeddings.npy` + `events_rec_embeddings.npy`), tăng thêm ~165 KB và < 1 giây CPU |
| **Prefix E5** | Dùng chung `query: ` $\to$ `P6` bị so `query ↔ query` lệch nhẹ với `[c_min, c_max]` (`query ↔ passage`) | `P2` giữ `query: `; `P6` dùng `passage: ` $\to$ **đúng thiết kế gốc của E5** |
| **Thêm `category` / `format`** | Rủi ro làm lệch kết quả P2 hoặc truyền nhiễu M2 vào phân loại | `P6` tự do bổ sung `category` + loại hình để tăng độ khớp sở thích mà **không đụng chạm đến P2** |
| **Độ phù hợp theo giai đoạn** | Phù hợp MVP / Demo hiện tại | Phù hợp giai đoạn tối ưu độ chính xác (Phase cải tiến sau đánh giá) |

---

## 5. Bảng Ánh xạ Chuẩn hóa Gợi ý cho `category` & `format` (dùng khi triển khai)

| Giá trị thô trong `events.json` | Nhãn tiếng Việt chuẩn hóa đề xuất |
|---|---|
| `Academic`, `Education`, `Conference`, `Seminar`, `Academic & Skill Sessions`, `Academic / Skills` | `Học thuật, hội thảo chuyên đề` |
| `Technology` | `Công nghệ, kỹ thuật` |
| `Competition`, `Competitions`, `Contest`, `Competitions & Tournaments`, `Exam` | `Cuộc thi, thi đấu` |
| `Workshop`, `Talkshow`, `tọa đàm`, `Skill-building`, `Course` | `Workshop, tọa đàm, kỹ năng` |
| `Career`, `Career/Job Fair`, `Recruitment`, `Tour` | `Hướng nghiệp, kết nối doanh nghiệp` |
| `Volunteer`, `Volunteering`, `Community`, `Volunteer / Community`, `Environment`, `Health` | `Tình nguyện, hoạt động cộng đồng` |
| `Sports`, `Sporting Events` | `Thể thao, rèn luyện thể lực` |
| `Arts`, `Music`, `Cultural`, `Culture`, `Arts & Culture`, `Arts & Culture/Music Events`, `Festival`, `Exhibition`, `Entertainment`, `ngày hội` | `Văn hóa, văn nghệ, lễ hội` |
| `Orientation`, `Ceremony`, `Scholarship` | `Sinh hoạt định hướng, học bổng` |
| `Offline`, `In-person`, `Online`, `offline`, `Activity`, `khác` | *(Loại bỏ — không đưa vào chuỗi embedding)* |

---

## 6. Kế hoạch Triển khai & Kiểm chứng Thực nghiệm (A/B)

Trước khi thay đổi embedding sản xuất, cần chạy đánh giá định lượng theo 3 bước:

1. **Bước 1 — Chốt bảng chuẩn hóa chung (`src/embedding_text_utils.py`):** cài đặt hàm chuẩn hóa `category`/`format` (bảng Mục 5) và hàm chuẩn hóa `organizer` (xóa tên trường lặp, đồng bộ viết tắt).
2. **Bước 2 — Đánh giá riêng cho Classification (`C-v1` vs `C-v2` vs `C-v3`):**
   - Đo trên tập validation có nhãn xác thực (như kế hoạch `FUTURE_WORK.md`): so sánh Stage 1 Recall@2, Stage 2 Macro-F1, tỷ lệ `branch_agreement`, và tỷ lệ `NEEDS_REVIEW`.
   - Chỉ giữ `organizer` trong embedding nếu `C-v3` (đồng bộ cả Sample Branch) cải thiện rõ Macro-F1 mà không gây tụt trên các ca đồng tổ chức đa CLB.
3. **Bước 3 — Đánh giá riêng cho Recommendation (`R-v1` vs `R-v2`):**
   - Sinh thử `events_rec_embeddings.npy` với prefix `passage: ` + `category_vi_normalized`, đo lại phân bố cosine `user_vec ↔ events_rec` so với `user_vec ↔ sample_embeddings` (kiểm tra hết lệch `query ↔ query`) và đánh giá chất lượng Top-5 theo 4 Persona Demo (`Overlap@5`, `nDCG@5`).
