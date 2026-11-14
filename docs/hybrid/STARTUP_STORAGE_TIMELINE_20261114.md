# Khởi động mất thời gian ở đâu? — mốc thực và hash chỉ đọc

Mốc khởi động đã ghi ở57a681 cho thấy **15,086s** từ lúc tiến trình gateway
bắt đầu đến nativegate ngay trước ready. Con số21,25s của giao dịch activation
còn gồm stop/backup/source/restore/probe, không phải thời gian nạp model riêng.

[add9f5 actualtimestamps/hashstages](../../runs/20261114T110937Z-startup-storage-audit-v1-add9f5/result.json),
[dataaudit](../../runs/20261114T110937Z-startup-storage-audit-v1-add9f5/independent-data-audit-v1.json),
[phạm vi đã đăng ký](../../experiments/hybrid-startup-storage-audit-v1.md).

| Khoảng giữa mốc đã lưu | Thời gian | Giới hạn quy nguyên nhân |
|---|---:|---|
|Processstart→storageverified|5,682s|Gồm import/khôi phục/init/kiểmfiles; không phải hash thuần|
|Storageverified→ALIFdeviceevent|0,501s|Đưa ALIF lênCUDA; không phải toàn coldstart hoặc học|
|ALIFdeviceevent→spawnnativeVLM|3,704s|Có selectedASRcheck/prelaunch; chưa tách được riêng từng việc|
|SpawnVLM→SearXNGready|4,887s|Gồm chờVLMhealth và startupsearch; không quy hết cho một thành phần|
|Searchready→3paritycases→nativegate|0,311s|Kiểm numericalparity và chốtgate|

Đây là một lần khởi động có log, hiệu epochwalltime, không có coldcachecontrol
hoặc nhiều lượt để suy p95/CI. `gateway_native_gate` là boundaryproxy trước
ready, không phải thời điểm khung chat Hermes hoàn toàn sẵn sàng. MCP/login/
Hermesbootstrap có đường khởi động riêng, chưa được đo ở bảng này.

## Đọc file lặp đã đo trực tiếp

Không restart/khởi tạo neuralmodel hoặc chạy ASR/VLM trong audit. Các function
verify_storage thật chỉ đọc file; wrapper SHA256 trong tiến trình riêng đếm
bytes rồi chuyển cho implementation thật, giữ nguyên digest. Backend riêng
không gọi start/generate/transcribe. OS/cache đang có không bị flush hoặc ép
warm; không gọi số này là coldI/O hay chứng minh đã tăng tốc.

| Stage thực | Bytes đưa vàoSHA256 | SHAobjects | Walltime |
|---|---:|---:|---:|
|Qualificationbaseline3exports+selectedsmall|3.023.502.362|4|2,359s|
|Freshbackendselectedsmall|967.535.936|1|0,860s|
|Cùngbackend,ensurelần2|0|0|0,000s theo độ phân giải timer|

Tổng3.991.038.298byte được hash, trong đó967.535.936byte của ASR bị đọc lặp
giữa hai component. Đây là byte Python đưa vào hash, **không phải lượng dữ
liệu bắt buộc đọc từ SSD vật lý hoặc tăng RAM/VRAM tương ứng**. Chỉ buffer
hash được giữ; RSS tiến trình audit riêng~41,0MB, không phải footprint AI.

Cache theo statsignature trong cùngbackend đã hoạt động. Tối ưu handoff giữa
qualification và backend có thể tránh một lượt hash, nhưng cần proof nội bộ
gắn fileidentity và invalidation, không được chỉ tin `verified=true`.0,86s vừa
đo không được trừ thẳng khỏi mốc3,704s của lần boot khác để hứa tốc độ.

**Chưa áp tối ưu mới.** Ưu tiên giữ kiểm tra toàn vẹn, keepwarm hiện có và
hoàn tất mixed8h. Không bỏ parity, giảm precision hoặc giả sourcefile để boot
nhanh. Mọi sourcePY của32cd13 và nativeworker4728 được giữ nguyên.

Localhost đổi nhãn `VLM đang nạp` thành `VLM đang giữ tài nguyên` khi trạng thái
resident=true, tránh hiểu đó là vòng loading chưa dứt.20tracechecks hiện có
qua; đây là sửa diễn đạt, không phải số đo hiệu năng. Trace tự ghi audit thật,
không thêm một tiến trình/gateway hoặc cập nhật synapse nào.
