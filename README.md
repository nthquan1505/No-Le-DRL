# No-Le-DRL — Module 4

Module 4 đánh giá mức độ phù hợp về lịch của một sự kiện với một sinh viên. Nó kiểm tra xung đột lịch học và lịch thi, ước lượng thời gian đi ô tô, rồi trả `FitAssessment` cho Module 3. Lõi chạy với Python 3.10+ và có thể tính Fit offline bằng bảng di chuyển dự phòng.

## Luồng vận hành

1. **Chuẩn hóa giờ.** Sự kiện và lịch bận được đưa về cùng múi giờ, mặc định UTC+7. Nếu giờ kết thúc bị thiếu, không hợp lệ hoặc không sau giờ bắt đầu, lấy giờ bắt đầu cộng 120 phút. Nếu thiếu giờ bắt đầu, kết quả ghi rõ chưa thể kiểm tra lịch.
2. **Kiểm tra xung đột.** Module 4 xét các khoảng Busy và lịch tuần còn hiệu lực trên mọi ngày sự kiện đi qua. Offline và hybrid dùng đệm 30 phút ở hai đầu khoảng bận. Online dùng đệm 0 và mặc định chỉ kiểm tra lịch thi. Nếu khoảng trống đúng bằng đệm, sự kiện vừa đủ qua kiểm tra. Mục Free trùng giờ xuất hiện trong `flexible_overlaps` và không giảm Fit.
3. **Ước lượng di chuyển hai phía.** Đầu ngày lấy nhà làm điểm xuất phát. Mục Busy ghi “Ở nhà” dùng địa chỉ nhà trong hồ sơ. Mục Busy thiếu địa điểm có thể kế thừa điểm đã biết trong cùng ngày. Chặng đến dùng để tính `f_travel`, chặng rời dùng để cảnh báo thiếu giờ, và `f_slack` dùng khoảng trống ngắn hơn ở hai phía. Thứ tự tra tuyến là bảng nội bộ → tọa độ có sẵn hoặc Nominatim riêng → OSRM → Google Routes → 30 phút dự phòng. Online cần 0 phút.
4. **Trả kết quả cho Module 3.** Trùng lịch cho `Fit = 0`. Nếu qua kiểm tra lịch, điểm Fit là `(0.45 × f_travel + 0.25 × f_slack + 0.30 × f_pref) × f_exam`, kẹp trong `[0, 1]` và làm tròn 6 chữ số. P5 dùng cờ xung đột và phút di chuyển để lọc theo cấu hình. P6 dùng điểm Fit để xếp hạng. Payload đề xuất giữ điểm thành phần và lý do để giao diện giải thích.

Ví dụ, nếu lịch trống 40 phút và tuyến ô tô cần 55 phút, sự kiện qua quy tắc đệm 30 phút. Fit vẫn dùng 55 phút để chấm điểm di chuyển, đồng thời kết quả báo thiếu 15 phút để đến sự kiện.

## Đầu vào

Điểm vào của lõi là `ScheduleEngine.compute_fit(event, *, home_campus=None, max_travel_min=None, pref_hours=None)`.

| Dữ liệu | Trường sử dụng |
|---|---|
| Sự kiện `event` | Tương thích với `EventNormalized` của Module 3. Gồm `event_id`, `start_datetime`, `end_datetime`, `online_mode`, `is_online` và `location`. Ngày giờ dùng ISO-8601. Khi thiếu `online_mode`, lõi dùng `is_online`. Nếu cả hai đều thiếu, dùng chế độ offline với 30 phút di chuyển dự phòng. |
| Lịch sinh viên `ScheduleContext` | `busy_intervals` chứa các khoảng bận cụ thể, kể cả việc cá nhân. `weekly_slots` chứa lịch lặp với thứ, giờ và ngày hiệu lực tùy chọn. `exam_weeks` chứa các cặp ngày bắt đầu và kết thúc tuần thi. `default_travel_buffer_min` mặc định là 30. |
| Tùy chọn sinh viên | `home_address` là địa chỉ nhà, `home_campus` là điểm xuất phát dự phòng, `max_travel_min` mặc định 45 phút, `pref_hours` mặc định `(8.0, 21.0)`. Ví dụ, đặt `(8.0, 20.0)` để ưu tiên hoạt động kết thúc sớm. |
| Cấu hình hệ thống | Lõi nhận dict có khóa `schedule`. [Cấu hình tham chiếu](configs/schedule.yaml) mô tả các giá trị mặc định. Khi tích hợp, pipeline truyền cấu hình YAML đã nạp của Module 3. |

### Nguồn Google Calendar

Pipeline Module 3 nhận các mục lịch đã được mở rộng tại `schedule.calendar_items`. Dùng `calendar_exam_event_ids` để đánh dấu ca thi và `calendar_excluded_event_ids` để bỏ lịch đã đồng bộ mà sự kiện đang đánh giá cần tránh tự trùng.

Module 4 xin quyền đọc và ghi sự kiện. Tích hợp hiện tại đọc lịch bận. Để kết nối lần đầu, bật Google Calendar API trong Google Cloud, tạo OAuth Client kiểu Desktop, lưu file OAuth ngoài repo rồi chạy:

```powershell
python -m pip install -r requirements-calendar.txt
python -m module4.oauth --client-secrets "C:/path/to/client_secret.json"
```

Token mặc định nằm ở `~/.no-le-drl/calendar-token.json`. Dùng `--token` hoặc biến môi trường `MODULE4_CALENDAR_TOKEN` để chọn đường dẫn khác. Khi vận hành nhiều tài khoản, mỗi người dùng cần token riêng và backend phải chọn token theo danh tính đã xác thực. `FitEvaluator.evaluate()` nhận đường dẫn tại `calendar_token` hoặc client đã cấp quyền tại `calendar_service`, đọc một khoảng ngày cho cả lượt và xử lý phân trang. Khi mất kết nối, đầu ra trả `calendar_status="unavailable"` cùng kết quả tạm từ lịch cục bộ.

## Đầu ra

| Trường trong `FitAssessment` | Ý nghĩa |
|---|---|
| `fit_score` | Điểm phù hợp từ 0 đến 1 mà P6 dùng để xếp hạng. |
| `has_conflict`, `conflict_detail` | Cờ xung đột và chi tiết khoảng bận hoặc phần đệm bị vi phạm. |
| `in_exam_week` | Ngày bắt đầu sự kiện có nằm trong tuần thi hay không. |
| `estimated_travel_min` | Số phút đi ô tô ước tính, hoặc giá trị từ bảng dự phòng. |
| `nearest_slack_min` | Khoảng trống ngắn nhất đến lịch kề trước hoặc kề sau. Có thể là `null` khi không áp dụng. |
| `sub_scores` | `f_travel` cho di chuyển, `f_slack` cho độ thoáng lịch, `f_pref` cho khung giờ ưu tiên và `f_exam` cho hệ số tuần thi. |
| `reasons` | Tối đa 3 lý do tiếng Việt, gồm cảnh báo thiếu thời gian di chuyển hoặc ghi chú dùng số phút dự phòng nếu có. |
| `inbound`, `outbound` | Chi tiết từng chặng gồm số phút đi, khoảng trống, số phút thiếu, nguồn ước lượng và cờ địa điểm kế thừa. Chặng rời có thể là `null` khi không có lịch Busy tiếp theo trong ngày. |
| `flexible_overlaps` | Các mục lịch Free trùng giờ, lấy từ lịch trong bộ nhớ. |
| `calendar_status`, `provisional` | Trạng thái lịch cục bộ, đã đồng bộ hoặc chưa đồng bộ, cùng cờ cho kết quả tạm thời. |

### Ví dụ chạy offline

```python
from types import SimpleNamespace

from module4 import ScheduleEngine

event = SimpleNamespace(
    event_id="demo",
    start_datetime="2026-11-14T14:00:00+07:00",
    end_datetime="2026-11-14T16:00:00+07:00",
    online_mode="offline",
    is_online=False,
    location="cs2_lt",
)

assessment = ScheduleEngine().compute_fit(event, home_campus="cs2_lt")
print(assessment.fit_score, assessment.estimated_travel_min)  # 0.92 10
```

## Cấu hình dịch vụ định tuyến

Để bật Google Routes, nhập khóa trong PowerShell rồi chạy backend Module 3 từ cùng cửa sổ. Khóa được truyền qua biến môi trường `MODULE4_GOOGLE_ROUTES_KEY`. File `.env` được Git bỏ qua và mã hiện tại chỉ đọc biến môi trường của tiến trình.

```powershell
$key = Read-Host "Google Routes API key" -AsSecureString
$env:MODULE4_GOOGLE_ROUTES_KEY = [System.Net.NetworkCredential]::new('', $key).Password
Remove-Variable key
```

| Biến môi trường | Tác dụng |
|---|---|
| `MODULE4_GOOGLE_ROUTES_KEY` | Bật Google Compute Routes với `DRIVE` và `TRAFFIC_UNAWARE`. |
| `MODULE4_QUOTA_DB` | Đường dẫn tệp SQLite đếm lượt gọi Google, mặc định `.module4-route-quota.db`. Module 4 giới hạn 250 lượt mỗi ngày UTC trên mỗi tệp quota. Các tiến trình dùng chung khóa nên dùng chung tệp. Bộ đếm chỉ theo dõi yêu cầu do Module 4 gửi. |
| `MODULE4_OSRM_URL` | URL gốc của OSRM chạy profile ô tô. Module 4 thử OSRM trước Google khi có tọa độ cho cả hai đầu chặng. |
| `MODULE4_NOMINATIM_URL` | URL gốc của Nominatim riêng để tra tọa độ từ địa chỉ ngoài cơ sở khi cần gọi OSRM. |
| `MODULE4_LOCATION_CATALOG` | Đường dẫn JSON tùy chọn để bổ sung địa điểm đã xác minh. Mục `places` nhận `id`, `address`, `latitude`, `longitude` và `aliases`. Mục `travel_minutes` nhận `origin`, `destination` và `minutes` cho từng chiều. |

## Tính theo lô và quản lý cache

Điểm vào mới của Module 4 là `FitEvaluator.evaluate(user_id, events, context, ...)`. Bên gọi đưa vào các ứng viên trước khi lọc trùng lịch để hoạt động từng bị ẩn có thể được đánh giá lại sau khi xóa lịch. Bộ định tuyến `CarRouting.from_environment()` và `FitCache()` được truyền vào khi tạo evaluator. Điểm vào `ScheduleEngine.compute_fit()` vẫn dùng được để đánh giá một sự kiện riêng.

```python
from module4 import FitCache, FitEvaluator, ScheduleContext
from module4.oauth import token_path
from module4.routing import CarRouting

evaluator = FitEvaluator(cache=FitCache(), routing=CarRouting.from_environment())
results = evaluator.evaluate(
    "student-demo",
    [event],  # Dùng event ở ví dụ trên hoặc ứng viên do Module 3 cung cấp.
    ScheduleContext(),
    calendar_token=token_path(),
    home_campus="cs2_lt",
    pref_hours=(8.0, 20.0),
)
```

SQLite mặc định nằm tại `runtime/module4/fit_cache.sqlite3`. Mỗi dòng giữ ID sinh viên, ID hoạt động, dấu kiểm tra đầu vào và kết quả Fit đã bỏ chi tiết lịch cá nhân. Khi lịch đổi, dấu kiểm tra thay đổi cho các ngày liên quan, kể cả lịch sát nửa đêm và hoạt động nhiều ngày. Ứng viên có đầu vào giữ nguyên dùng lại kết quả. Kết quả tạm thời và kết quả dùng tuyến Google chỉ dùng trong lượt hiện tại. Trong một lượt, cùng chặng ô tô chỉ tra một lần. Nếu bên gọi thay đổi cách định tuyến tùy chỉnh, đặt giá trị `routing_version` mới để làm mới Fit đã lưu.

Cache được dọn trong lượt đánh giá đầu tiên sau 24 giờ. Các dòng cho hoạt động đã kết thúc hoặc không được dùng trong 30 ngày sẽ bị xóa. Khi nguồn dữ liệu xác nhận xóa một hoạt động, bên gọi có thể dùng `FitCache.delete_event(event_id)`. Dùng `FitCache.delete_user(user_id)` để xóa toàn bộ cache của một sinh viên.

## Tích hợp và kiểm thử

Bản vá [Module 3](integration/module3.patch) có từ đợt tích hợp trước và dành cho pipeline đề xuất tại commit `b520038`. Bản vá dùng `ScheduleEngine` và chưa tự kích hoạt OAuth hay cache theo lô. Mã Module 3 và bản vá này được giữ nguyên trong đợt triển khai hiện tại. Để kiểm tra lại hợp đồng cũ trong một checkout tích hợp riêng, chép thư mục `module4/` vào thư mục gốc của checkout đó rồi chạy:

```powershell
git apply --check D:/No-Le-DRL/integration/module3.patch
git apply D:/No-Le-DRL/integration/module3.patch
python -m unittest tests.test_recommendation_p5_p6.TestDemoModule6AndP5P6 tests.test_schedule_adapter -v
python -c "import runpy; runpy.run_path('D:/No-Le-DRL/integration/smoke_module3.py', run_name='__main__')"
```

Để chạy bộ kiểm thử độc lập của Module 4, gồm TC1–TC8 và các ca chuyển nguồn định tuyến giả lập, chạy lệnh sau từ thư mục gốc nhánh `module-4`.

```powershell
python -m unittest discover -s tests -v
```
