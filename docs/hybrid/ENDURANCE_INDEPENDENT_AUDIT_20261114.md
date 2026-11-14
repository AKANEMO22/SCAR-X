# Kiểm toán endurance: chưa hoàn tất, chưa là video liên tục

**Trạng thái tại snapshot a2ccf7: PRELIMINARY, chưa PASS8giờ.** Tiến trình41904
và nativeVLM39372 còn đúng identity; source đang khóa vẫn khớp. Auditor mới
không chạy inference/học, không restart, không sửa source active hoặc nới gate.

[Protocol audit](../../experiments/hybrid-endurance-independent-audit-v1.md),
[snapshot a2ccf7](../../runs/20261114T072942Z-endurance-data-audit-v1-a2ccf7/result.json),
[raw observer events](../../runs/20261114T072942Z-endurance-data-audit-v1-a2ccf7/observer-events.json).
Các số dưới là ảnh chụp thời điểm, không tự đổi thành kết quả cuối khi job đủ giờ.

## Đã đo được gì?

739heartbeat records và75video/QAcycles trong observer snapshot. File heartbeat
được monitor đọc ngay trước đó còn74cycles; hai database không được đọc nguyên
tử chung nên auditor ghi rõ boundary này, không sửa log cho khớp giả tạo.
Checkpoint riêng hiện80updates/80hiddenupdates,16mẫu replay; có13pinnedchecks,
mọi snapshot còn lưu khớp SHA/bounds. Không có cognitive_quantum_failed hoặc
ASRerror trong dữ liệu đã đọc. Một family số học không được gọi là20kỹ năng.

| Đo tại snapshot | Giá trị | Giới hạn diễn giải |
|---|---:|---|
| Tổng thời lượng75clip đã xử lý |1.575s =26phút15giây|Cùng một video21s lặp lại |
| Nguồn video gốc duy nhất |1|Không phải1–2giờ video đa dạng |
| Trung vị bắt đầu quan sát→có đáp án |26,46s|Bao gồm phát/capture21s và xử lý; không là tốc độ decode thuần |
| Trung vị5chu kỳ đầu/cuối |25,75 /26,77s|Mô tả, chưa kết luận có/không memoryleak hoặc drift |
| Skip/ASRerror trong completedcycle |0 /0|Không chấm khả năng hiểu video hoặc transcript |
| VRAM vật lý trống thấp nhất trong heartbeat |1.888MiB|Lấy mẫu, có thể bỏ sót đỉnh ngắn |

Thời gian chương trình còn sống, thời lượng video và chất lượng hiểu là ba
thước đo khác nhau. Khoảng rảnh giữa các clip giúp phục vụ chat/review/replay;
lượt này chưa đo sustainedcapture/phân tích mọi giây của một video dài.
Các answer vẫn giữ answer_quality_scoredfalse, không sinh điểm QA sau khi xem.

## Khoảng trống đo RAM đã xác minh

Monitor của lượt dài giữ các clientroots ban đầu: SCAR-X35036, Desktop35648 và
Hermesgateway43608. Gateway mới47768 sau các lần sửa/nạp lại không nằm trong
sample cuối. Do đó peak đã ghi là của nhóm tiến trình được theo dõi, **không
phải peak đầy đủ của Hermes hiện tại**. Không cộng một mức RAM hiện tại vào
peak lịch sử để giả khôi phục phần thiếu.

Peak được báo trong nhóm gốc đến lúc snapshot:

- RSS5.924.352.000byte; privatecommit8.681.582.592byte, hai số không cộng nhau.
- ConservativeRAM6.215.282.688byte có cộng sharedGPU theo monitor gốc, có thể
  tính bảo thủ phần chia sẻ. DedicatedGPU báo riêng~3,043GB.
- Đây là GPUmode đã được người dùng bỏ caphostRAM. **Không dùng các số này để
  chứng nhận CPUmode4,5GB**, cũng không coi vượt4,5GB là vi phạm GPUmode.

[Snapshot bổ sung chỉ một thời điểm](../../runs/20261114T072942Z-endurance-data-audit-v1-a2ccf7/gateway-coverage-supplement-v1.json)
xác minh PID/create-time của bốn root hiện tại, gộp descendant không trùng:
RSS~4,202GB, privatecommit~6,943GB. Riêng gateway47768 có RSS~322MB và
privatecommit~301MB. Đây là điểm hiện tại khác thời điểm/đỉnh gốc; không suy ra
peak toàn hệ hoặc khẳng định phần thiếu trong quá khứ luôn bằng mức này.

Source monitor active được giữ nguyên để bảo toàn thí nghiệm. Sau khi kết thúc,
cần bật cập nhật clientroots trong profile đo tiếp theo và giữ explicitroots.
[P0 sau đó](RESOURCE_ROOTS_AND_ASR_SOAK_20261114.md) cho thấy discovery gốc vẫn
tìm đúng gateway khi bật; patch đã được thu hẹp, không thay discovery đang đúng.
Đây là công việc engineering được phép, độc lập với rulehistory đang chờ duyệt.

## Điều kiện ra kết luận cuối

Auditor tính lại ngưỡng gốc bằng đường code riêng: elapsed≥28.800s,≥80video/QA,
≥8checkpointchecks, localupdates+replay trênGPU, không criticalfailure/unfinished
capture; kiểm source/preflight và sampled512MiBreserve. Không chỉ import hàm
qualifies gốc rồi lặp lại verdict của runner.

Chỉ có thể cho finalengineering auditPASS khi finalreceipt tồn tại, worker thật
đã terminal và receipt/observer/private records khớp. Nếu chưa có receipt mà
PID còn sống, luôn PRELIMINARY; nếu PID mất và thiếu receipt, không tự restart.
Checkpointauditor kiểm bytehash/metadata/bounds, không chạy lại neuralforward
hoặc giả làm đã tự kiểm toàn bộ coldGPUloader. Exactclone/rollback vẫn là các
actualcalls/assertions của primaryrunner và các test persistence trước đó.

Đã kiểm auditor trên [lượt0acab3 bị ngắt trước đây](../../runs/20261114T073327Z-endurance-data-audit-v1-16e3bd/result.json):
verdict INTERRUPTED_OR_MISSING_RECEIPT_NOT_PASS, không biến heartbeat cuối thành
một lượt8h. Negativecase này không tạo thêm video, học hoặc retryjob cũ.

Ngay cả finalengineeringPASS cũng không đổi PIVOT về hiddenadvantage/retention,
[QA sai qua restart](VIDEO_COLD_MEMORY_20261114.md) hoặc các gateP1/P2/P3.
Thêm VRAM chỉ có ích khi cải thiện thành phần đã đo; hiện [WhispersmallFP32](ASR_CANDIDATE_AND_TOOL_DATA_20261114.md)
có gain phiên âm DEV nhưng chưa được đưa vào profile live/longrun. Không tăng
context/model, hạ precision hoặc nhồi bộ nhớ để làm GPU trông đầy hơn.

### Tra trạng thái hiện tại

~~~powershell
.\local\hybrid-v1\runtime-env\Scripts\python.exe scripts/monitor_hybrid_endurance.py runs/20261114T011841Z-shared-endurance-v1-554c19
~~~

Auditor chỉ nên chạy lại ở mốc kết thúc/biến cố có ý nghĩa; mỗi snapshot tạo run
riêng. [Trace](http://127.0.0.1:8775/?after=7357&tab=trace) tách walltime, video đã
xử lý, source count và cảnh báo processcoverage khỏi số neuralupdates live.
