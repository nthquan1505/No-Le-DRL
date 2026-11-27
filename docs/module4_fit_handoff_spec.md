# Tài liệu Bàn giao cho Module 4 — Đặc tả Giá trị `Fit` & Hàm `compute_fit`

> **Từ:** Nhóm Module 3 (Classification & Recommendation Engine)
> **Gửi tới:** Nhóm Module 4 (Schedule Conflict Detection)
> **Mục đích:** Chốt hợp đồng giao tiếp (I/O Contract), yêu cầu nghiệp vụ và công thức chuẩn hóa cho giá trị `fit_score` cùng hàm `compute_fit(...)` để Module 4 triển khai và tích hợp trực tiếp vào P5 (Lọc cứng) và P6 (Chấm điểm đề xuất) của Module 3.
> **Tham chiếu mã nguồn hiện có:** `src/recommendation/schedule_adapter.py` (`ScheduleProvider`, `ConflictCheckResult`, `FitAssessment`), `src/schemas.py` (`EventNormalized`, `BusyInterval`, `ScheduleContext`).

---

## 1. Ranh giới trách nhiệm giữa Module 4 và Module 3

Trên sơ đồ hệ thống (`module3_engine_flowchart.drawio.svg`), Module 4 đảm nhận toàn bộ phần lịch trình cá nhân và khoảng cách di chuyển:

- **Module 4 cung cấp cho Module 3:**
  1. `check_conflict(event)` $\to$ dùng ở **P5 (Hard Filter)** để loại sự kiện trùng lịch học/lịch thi (có tính đệm di chuyển).
  2. `estimate_travel_minutes(event, home_campus)` $\to$ dùng ở **P5** (nếu bật lọc quá xa `travel_min > max_travel_min`) và **P6** (`f_travel`).
  3. `nearest_slack_minutes(event)` $\to$ khoảng trống nhỏ nhất giữa sự kiện với tiết học kề trước/sau trong ngày (dùng cho `f_slack`).
  4. `is_in_exam_week(dt)` $\to$ xác định sự kiện có rơi vào tuần thi hay không (dùng cho `f_exam`).
  5. **`compute_fit(event, ...)`** $\to$ tổng hợp các tín hiệu trên thành `FitAssessment` chứa `fit_score ∈ [0.0, 1.0]` truyền trực tiếp vào công thức chấm điểm **P6**:
     $$\text{Score}(u, e) = w_1 \cdot \text{Gain} + w_2 \cdot \text{Interest} + w_3 \cdot \mathbf{Fit} + w_4 \cdot \text{Urgency} + \text{Penalty}$$
- **Module 4 KHÔNG xử lý các yếu tố sau** *(để tránh tính trùng với Module 3 / Module 6)*:
  - Không xét lệ phí sự kiện (`fee_vnd`) — đã do P5 lọc cứng theo ngân sách.
  - Không xét điểm ĐRL (`Gain`), sở thích (`Interest`), hạn đăng ký (`Urgency`), hay cờ bắt buộc (`Penalty`).

---

## 2. Đặc tả Hợp đồng Dữ liệu (Input / Output Contract)

### 2.1. Đầu vào của `compute_fit`

```python
def compute_fit(
    self,
    event: EventNormalized,
    *,
    home_campus: Optional[str] = None,
    max_travel_min: Optional[int] = None,
    pref_hours: Tuple[float, float] = (8.0, 21.0),
) -> FitAssessment:
    ...
```

| Tham số | Kiểu dữ liệu | Ý nghĩa & Giá trị mặc định |
|---|---|---|
| `event.event_id` | `str` | Mã định danh sự kiện |
| `event.start_datetime` | `Optional[str]` | Thời điểm bắt đầu dạng ISO-8601 (vd `"2026-11-14T13:30:00+07:00"`). Có thể `None` |
| `event.end_datetime` | `Optional[str]` | Thời điểm kết thúc dạng ISO-8601. Nếu `None` hoặc `<= start_datetime`, tự suy ra `start_datetime + 120 phút` |
| `event.online_mode` | `Optional[str]` | `"online"` \| `"offline"` \| `"hybrid"` \| `None` (nếu `None`, fallback sang `event.is_online`; nếu vẫn `None`, coi là `"offline"`) |
| `event.location` | `Optional[str]` | Chuỗi địa điểm diễn ra sự kiện (dùng ước lượng thời gian di chuyển) |
| `home_campus` | `Optional[str]` | Cơ sở/vị trí xuất phát chính của sinh viên (vd `"cs1_nvc"` = Cơ sở 1 Nguyễn Văn Cừ, `"cs2_lt"` = Cơ sở 2 Linh Trung, hoặc tọa độ/địa chỉ nếu M4 dùng bản đồ) |
| `max_travel_min` | `Optional[int]` | Thời gian di chuyển tối đa sinh viên chấp nhận (phút). Nếu `None` hoặc `<= 0` $\to$ dùng mặc định **`45` phút** |
| `pref_hours` | `Tuple[float, float]` | Khung giờ ưu tiên trong ngày `(start_hour, end_hour)` theo giờ thập phân. Mặc định **`(8.0, 21.0)`** (tức 08:00–21:00) |

### 2.2. Đầu ra `FitAssessment` (Python Dataclass / JSON tương đương)

```python
@dataclass(frozen=True)
class FitAssessment:
    fit_score: float                       # Bắt buộc trong [0.0, 1.0], làm tròn 6 chữ số
    has_conflict: bool                     # True nếu trùng lịch học / lịch thi (có đệm)
    in_exam_week: bool                     # True nếu ngày diễn ra thuộc exam_weeks
    estimated_travel_min: int              # Thời gian di chuyển ước tính (phút, >= 0)
    nearest_slack_min: Optional[int] = None  # Khoảng trống (phút) tới tiết kề sát nhất; None nếu không có tiết cùng ngày / online
    conflict_detail: Optional[str] = None  # Lý do trùng lịch chi tiết (nếu has_conflict=True)
    sub_scores: Dict[str, float] = field(default_factory=dict)  # {"f_travel", "f_slack", "f_pref", "f_exam"}
    reasons: List[str] = field(default_factory=list)            # 1-3 câu giải thích ngắn cho UI/P8
```

Ví dụ payload JSON nếu giao tiếp qua REST API:

```json
{
  "fit_score": 0.92,
  "has_conflict": false,
  "in_exam_week": false,
  "estimated_travel_min": 10,
  "nearest_slack_min": 60,
  "conflict_detail": null,
  "sub_scores": {
    "f_travel": 0.822222,
    "f_slack": 1.0,
    "f_pref": 1.0,
    "f_exam": 1.0
  },
  "reasons": [
    "Không trùng lịch học TKB",
    "Di chuyển ngắn (~10 phút)",
    "Nằm trong khung giờ rảnh ưa thích"
  ]
}
```

---

## 3. Yêu cầu nghiệp vụ cho các hàm phụ trợ trong Module 4

### 3.1. `check_conflict(event) -> ConflictCheckResult`
- Mọi timestamp `naive` (không có timezone) phải được gán mặc định múi giờ **`UTC+7` (`Asia/Ho_Chi_Minh`)** trước khi so sánh.
- **Đệm di chuyển (`travel_buffer_min`):**
  - Sự kiện `online`: `buffer = 0` phút. Mặc định **không xung đột** với tiết học thường, **chỉ xung đột** với slot lịch thi (`is_exam == True`), trừ khi cấu hình `online_conflicts_with_regular_classes = True`.
  - Sự kiện `offline` / `hybrid`: `buffer = default_travel_buffer_min` (mặc định **`30` phút**; nếu M4 tính được thời gian di chuyển thực tế từ địa điểm tiết học kề trước sang sự kiện thì có thể dùng chính thời gian đó làm buffer động).
- **Điều kiện trùng khoảng thời gian:**
  $$\text{event\_start} < (\text{busy\_end} + \text{buffer}) \quad \text{VÀ} \quad \text{event\_end} > (\text{busy\_start} - \text{buffer})$$
- Phải quét cả lịch cụ thể (`busy_intervals`) lẫn lịch lặp hằng tuần (`weekly_slots`, có tôn trọng `valid_from` / `valid_until`) trên **tất cả các ngày** mà khoảng `[event_start, event_end]` đi qua.

### 3.2. `estimate_travel_minutes(event, home_campus) -> int`
- Nếu sự kiện `online` $\Rightarrow$ luôn trả về **`0`**.
- Nếu Module 4 tích hợp bản đồ / ma trận khoảng cách thật (từ vị trí tiết học kề trước hoặc `home_campus` tới `event.location`): trả về số phút nguyên không âm ($\ge 0$).
- Nếu không định vị được địa chỉ hoặc chưa tích hợp bản đồ, dùng bảng fallback chuẩn HCMUS:
  - Cùng cơ sở (`cs1_nvc == cs1_nvc` hoặc `cs2_lt == cs2_lt`): **`10` phút**
  - Khác cơ sở (CS1 Nguyễn Văn Cừ $\leftrightarrow$ CS2 Linh Trung): **`45` phút**
  - Không xác định được cơ sở/địa điểm: **`30` phút**

### 3.3. `nearest_slack_minutes(event) -> Optional[int]`
- Chỉ áp dụng khi sự kiện **không phải `online`**, có `start_datetime`, và **không trùng lịch** (`has_conflict == False`).
- Xét tất cả các khoảng bận (cả `busy_intervals` lẫn `weekly_slots` có hiệu lực) trong **cùng ngày** diễn ra sự kiện (theo `UTC+7`):
  - Với mọi slot kết thúc trước sự kiện (`busy_end <= event_start`): `gap_before = (event_start - busy_end)` tính bằng phút.
  - Với mọi slot bắt đầu sau sự kiện (`busy_start >= event_end`): `gap_after = (busy_start - event_end)` tính bằng phút.
- Trả về `min` của tất cả các `gap_before` và `gap_after` tìm được. Nếu ngày đó sinh viên không có tiết học/lịch bận nào khác $\Rightarrow$ trả về **`None`**.

---

## 4. Công thức tính `compute_fit` (Chi tiết từng bước)

### Bước 0 — Kiểm tra xung đột cứng
Gọi `conflict = check_conflict(event)`.
- Nếu `conflict.has_conflict == True` $\Rightarrow$ **trả về ngay `fit_score = 0.0`**, `has_conflict = True`, kèm `conflict_detail`.

### Bước 1 — Thành phần di chuyển: $f_{\text{travel}} \in [0.0, 1.0]$
Đặt $T_{\max} = \text{max\_travel\_min}$ nếu hợp lệ ($> 0$), ngược lại $T_{\max} = 45$; đặt $t = \text{estimated\_travel\_min}$.

- Nếu `online`:
  $$f_{\text{travel}} = 1.0$$
- Nếu `offline`:
  $$f_{\text{travel}}^{\text{offline}} = \begin{cases} 1.0 - 0.8 \cdot \dfrac{t}{T_{\max}} & \text{nếu } t \le T_{\max} \\[6pt] \max\!\left(0.0,\; 0.2 - 0.2 \cdot \dfrac{t - T_{\max}}{T_{\max}}\right) & \text{nếu } t > T_{\max} \end{cases}$$
- Nếu `hybrid`:
  $$f_{\text{travel}} = 0.5 \cdot \big(1.0 + f_{\text{travel}}^{\text{offline}}\big)$$

*(Ý nghĩa với $T_{\max} = 45$p: Online `0p` $\to 1.00$; cùng cơ sở `10p` $\to 0.822$; chưa rõ `30p` $\to 0.467$; khác cơ sở `45p` $\to 0.200$; quá xa `90p` $\to 0.000$.)*

### Bước 2 — Thành phần độ thoáng lịch học kế cận: $f_{\text{slack}} \in [0.4, 1.0]$
- Nếu `online`, hoặc `start_datetime is None`, hoặc `nearest_slack_min is None` (ngày đó không có tiết kề cận):
  $$f_{\text{slack}} = 1.0$$
- Ngược lại, gọi $t_{\text{buf}} = \text{conflict.travel\_buffer\_min}$ (mặc định `30` phút) và $B_{\text{comfort}} = 30$ phút:
  $$\Delta t_{\text{dư}} = \max\!\big(0,\; \text{nearest\_slack\_min} - t_{\text{buf}}\big)$$
  $$f_{\text{slack}} = \min\!\left(1.0,\; 0.4 + 0.6 \cdot \frac{\Delta t_{\text{dư}}}{30.0}\right)$$

*(Ý nghĩa: vừa sát ngưỡng đệm $\Delta t_{\text{dư}} = 0$p $\to 0.40$; dư `15p` $\to 0.70$; dư $\ge 30$p $\to 1.00$.)*

### Bước 3 — Thành phần khung giờ ưa thích: $f_{\text{pref}} \in \{0.3, 0.5, 0.7, 1.0\}$
Gọi `pref_hours = (p_start, p_end)` (mặc định `(8.0, 21.0)`).
- Nếu thiếu `start_datetime`:
  $$f_{\text{pref}} = 0.5 \quad (\text{trung tính})$$
- Nếu có `start_datetime` (đổi sang giờ thập phân `h_s`, `h_e` theo `UTC+7`):
  - Nếu sự kiện diễn ra **trong cùng 1 ngày** (`start.date() == end.date()`):
    $$\text{overflow} = \max(0.0,\; p_{\text{start}} - h_s) + \max(0.0,\; h_e - p_{\text{end}})$$
  - Nếu sự kiện **kéo dài nhiều ngày** (`end.date() > start.date()`, vd đi tình nguyện 2N1Đ): chỉ tính độ lệch giờ xuất phát $\text{overflow} = \max(0.0,\; p_{\text{start}} - h_s)$ (tránh phạt nhầm vì thời lượng vượt 24h).
  - Ánh xạ:
    $$f_{\text{pref}} = \begin{cases} 1.0 & \text{nếu } \text{overflow} = 0 \\ 0.7 & \text{nếu } 0 < \text{overflow} \le 1.0 \text{ (lệch tối đa 1 giờ)} \\ 0.3 & \text{nếu } \text{overflow} > 1.0 \text{ (quá sớm hoặc quá khuya)} \end{cases}$$

### Bước 4 — Hệ số áp lực tuần thi: $f_{\text{exam}} \in \{0.45, 0.70, 0.90, 1.00\}$
- Nếu `in_exam_week == False`: $f_{\text{exam}} = 1.00$.
- Nếu `in_exam_week == True`:
  - `online` $\to 0.90$
  - `hybrid` $\to 0.70$
  - `offline` $\to 0.45$

### Bước 5 — Công thức tổng hợp `fit_score`
$$\text{fit\_score} = \text{round}\!\Big(\text{clip}_{[0.0,\; 1.0]}\!\Big(\big(0.45 \cdot f_{\text{travel}} + 0.25 \cdot f_{\text{slack}} + 0.30 \cdot f_{\text{pref}}\big) \cdot f_{\text{exam}}\Big),\; 6\Big)$$

---

## 5. Tham số cấu hình mặc định (để đồng bộ YAML)

```yaml
schedule:
  default_timezone_offset_hours: 7
  default_event_duration_min: 120
  default_travel_buffer_min: 30
  default_max_travel_min: 45
  comfort_slack_min: 30
  default_pref_hours: [8.0, 21.0]
  online_conflicts_with_regular_classes: false
  campus_travel_minutes:
    online: 0
    same_campus: 10
    cross_campus: 45
    unknown: 30
  fit_scoring:
    conflict_score: 0.0
    weights:
      travel: 0.45
      slack: 0.25
      pref_hours: 0.30
    pref_hours_scores:
      inside: 1.0
      near_1h: 0.7
      outside: 0.3
      missing_time: 0.5
    exam_week_multipliers:
      online: 0.90
      hybrid: 0.70
      offline: 0.45
```

---

## 6. Mã giả Python tham chiếu cho hàm `compute_fit`

```python
def compute_fit(
    self,
    event: EventNormalized,
    *,
    home_campus: Optional[str] = None,
    max_travel_min: Optional[int] = None,
    pref_hours: Tuple[float, float] = (8.0, 21.0),
) -> FitAssessment:
    conflict = self.check_conflict(event)
    travel_min = max(0, int(self.estimate_travel_minutes(event, home_campus=home_campus)))
    window = self._parse_event_window(event)
    in_exam = self.is_in_exam_week(window[0]) if window is not None else False

    # Bước 0: Xung đột lịch cứng
    if conflict.has_conflict:
        return FitAssessment(
            fit_score=0.0,
            has_conflict=True,
            in_exam_week=in_exam,
            estimated_travel_min=travel_min,
            nearest_slack_min=0,
            conflict_detail=conflict.overlap_detail,
            sub_scores={"f_travel": 0.0, "f_slack": 0.0, "f_pref": 0.0, "f_exam": 0.0},
            reasons=[conflict.overlap_detail or "Trùng thời khóa biểu"],
        )

    is_online = self._is_online_event(event)
    is_hybrid = event.online_mode == "hybrid"
    t_max = float(max_travel_min) if (max_travel_min is not None and max_travel_min > 0) else 45.0

    # Bước 1: f_travel
    if is_online:
        f_travel = 1.0
    else:
        if travel_min <= t_max:
            f_travel_offline = 1.0 - 0.8 * (travel_min / t_max)
        else:
            f_travel_offline = max(0.0, 0.2 - 0.2 * ((travel_min - t_max) / t_max))
        f_travel = 0.5 * (1.0 + f_travel_offline) if is_hybrid else f_travel_offline

    # Bước 2: f_slack
    slack_min = None if (is_online or window is None) else self.nearest_slack_minutes(event)
    if is_online or window is None or slack_min is None:
        f_slack = 1.0
    else:
        excess = max(0.0, float(slack_min - conflict.travel_buffer_min))
        f_slack = min(1.0, 0.4 + 0.6 * (excess / 30.0))

    # Bước 3: f_pref
    if window is None:
        f_pref = 0.5
    else:
        start_local = window[0].astimezone(self._default_tz)
        end_local = window[1].astimezone(self._default_tz)
        h_s = start_local.hour + start_local.minute / 60.0
        h_e = end_local.hour + end_local.minute / 60.0
        p_start, p_end = pref_hours
        if end_local.date() > start_local.date():
            overflow = max(0.0, p_start - h_s)
        else:
            overflow = max(0.0, p_start - h_s) + max(0.0, h_e - p_end)
        if overflow <= 0.0:
            f_pref = 1.0
        elif overflow <= 1.0:
            f_pref = 0.7
        else:
            f_pref = 0.3

    # Bước 4: f_exam
    if not in_exam:
        f_exam = 1.0
    elif is_online:
        f_exam = 0.90
    elif is_hybrid:
        f_exam = 0.70
    else:
        f_exam = 0.45

    # Bước 5: Tổng hợp
    raw_score = (0.45 * f_travel + 0.25 * f_slack + 0.30 * f_pref) * f_exam
    fit_score = round(max(0.0, min(1.0, raw_score)), 6)

    return FitAssessment(
        fit_score=fit_score,
        has_conflict=False,
        in_exam_week=in_exam,
        estimated_travel_min=travel_min,
        nearest_slack_min=slack_min,
        conflict_detail=None,
        sub_scores={
            "f_travel": round(f_travel, 6),
            "f_slack": round(f_slack, 6),
            "f_pref": round(f_pref, 6),
            "f_exam": round(f_exam, 6),
        },
        reasons=self._build_fit_reasons(is_online, is_hybrid, travel_min, slack_min, in_exam, t_max),
    )
```

---

## 7. Bộ Test Cases nghiệm thu bàn giao (Acceptance Tests)

Module 4 cần vượt qua tối thiểu 8 test case chuẩn sau (với `max_travel_min = 45`, `pref_hours = (8.0, 21.0)`, `travel_buffer_min = 30`):

| Case | Kịch bản đầu vào | `f_travel` | `f_slack` | `f_pref` | `f_exam` | `fit_score` kỳ vọng |
|---|---|---:|---:|---:|---:|---:|
| **TC1** | Trùng lịch học / lịch thi (`has_conflict = True`) | 0.000 | 0.000 | 0.000 | 0.000 | **`0.000000`** |
| **TC2** | Online (`0p`), `19:00–20:30`, ngoài tuần thi | 1.000 | 1.000 | 1.000 | 1.000 | **`1.000000`** |
| **TC3** | Offline cùng cơ sở (`10p`), cách tiết gần nhất `60p` ($\Delta t_{\text{dư}} = 30$p), `14:00–16:00`, ngoài tuần thi | 0.822222 | 1.000 | 1.000 | 1.000 | **`0.920000`** |
| **TC4** | Offline khác cơ sở (`45p`), sát đệm lịch học (`slack = 30p`, $\Delta t_{\text{dư}} = 0$p), `14:00–16:00`, ngoài tuần thi | 0.200 | 0.400 | 1.000 | 1.000 | **`0.490000`** |
| **TC5** | Offline khác cơ sở (`45p`), không có tiết cùng ngày (`slack = None`), `14:00–16:00`, **trong tuần thi** | 0.200 | 1.000 | 1.000 | 0.450 | **`0.288000`** |
| **TC6** | Online (`0p`), `19:00–20:30`, **trong tuần thi** | 1.000 | 1.000 | 1.000 | 0.900 | **`0.900000`** |
| **TC7** | Offline quá xa (`60p > 45p`), không có tiết cùng ngày, `07:30–10:00` (lệch 0.5h), ngoài tuần thi | 0.133333 | 1.000 | 0.700 | 1.000 | **`0.520000`** |
| **TC8** | Offline thiếu `start_datetime`, không rõ cơ sở (`30p`), ngoài tuần thi | 0.466667 | 1.000 | 0.500 | 1.000 | **`0.610000`** |

---

## 8. Edge Cases bắt buộc xử lý an toàn
1. **`start_datetime is None`:** không báo trùng lịch (`has_conflict = False`), `in_exam_week = False`, `f_slack = 1.0`, `f_pref = 0.5`.
2. **`end_datetime is None` hoặc `end_datetime <= start_datetime`:** mặc định `end_datetime = start_datetime + 120 phút`.
3. **`online_mode is None`:** fallback sang `is_online`; nếu vẫn `None` coi là `offline` với `travel_min = 30`.
4. **Sự kiện nhiều ngày (vd đi tình nguyện 2 ngày 1 đêm):** `check_conflict` quét tất cả các ngày liên quan; `f_pref` chỉ kiểm tra giờ khởi hành (`start_datetime`) để không phạt oan.
5. **Chia cho 0 / dữ liệu âm:** `max_travel_min <= 0` fallback về `45`; `travel_min < 0` kẹp về `0`; `fit_score` luôn kẹp trong `[0.0, 1.0]`.
6. **Không tính lệ phí trong `Fit`:** bỏ hoàn toàn `free_event_bonus` (khác với bản nháp cũ trong `schedule_adapter.py`) để không trùng với lọc ngân sách của P5.
