# Kiểm tra cơ chế giữ kiến thức và lượt chạy dài — 14/11/2026

## Phát hiện về cơ chế học hiện tại

Trong **LocalALIF của nhánh Hybrid**, không được coi mảng có tên importance hoặc
stable là một cơ chế chống quên đang hoạt động. Source learn chỉ dùng eligibility,
syn_trace/syn_energy để tính delta; importance và stable_* không tham gia công
thức đó. consolidate gọi lại learn trên mẫu replay, không gọi EMA stable của
SpikingCore. importance không tự tăng theo trải nghiệm trong đường chạy này.

Điều này khác với PlasticCore của Phase2.1 lịch sử: _update ở đó thực sự dùng
importance để điều tiết update và kéo về anchor. Không suy phát hiện Hybrid
thành kết luận rằng mọi metaplasticity trong SCAR-X đều không hoạt động.

[Protocol P0](../../experiments/hybrid-retention-mechanism-audit-p0-v1.md) đã khóa
trước phép can thiệp. Ba seed260113/260137/260171, mỗi seed32 local updates trên
hai clone CUDA cùng input/nhãn/RNG. Clone đối chứng có importance=1000 và toàn
stable_* dịch+0,25; các trọng số đang học lúc khởi đầu giống nhau.

| Seed | Sai khác lớn nhất của trọng số học ở mọi bước | Importance mặc định sau32 bước | Stable mặc định đổi? |
|---|---:|---:|---|
| 260113 | 0, chính xác | 0 | Không |
| 260137 | 0, chính xác | 0 | Không |
| 260171 | 0, chính xác | 0 | Không |

Các weight đang học có thay đổi thật; không phải hai clone đều khóa học.
Đổi những buffer trên không ảnh hưởng loss/delta ở phép thử. Checkpoint đầy đủ,
next continuation và provenance được giữ; không lấy weights live vào lab.

Can thiệp thứ hai giữ cùng synapse sau học nhưng thay f_mean/f_var bằng giá trị
ban đầu. Xác suất đầu ra đổi tối đa lần lượt0,02430/0,02118/0,01819; argmax không
đổi trên bốn input. Đây là bằng chứng normalization state có ảnh hưởng tới
forward, **chưa phải bằng chứng nó là nguyên nhân của toàn bộ forgetting**,
và không phải đo accuracy hoặc useful hidden representation.

- [Raw run9d1545](../../runs/20261114T012120Z-retention-mechanism-p0-9d1545/result.json)
- [Auditor data-only và kết quả độc lập](../../runs/20261114T012120Z-retention-mechanism-p0-9d1545/independent-audit-v1.json)
- [Script tái kiểm raw/checkpoint](../../scripts/verify_hybrid_retention_mechanisms.py)

Tín hiệu target0..3 trong P0 là nhãn kỹ thuật thuộc input fixture sở hữu riêng,
không là bộ kiểm chứng sự thật ngữ nghĩa. Không gọi192 lượt cập nhật riêng này
là đã học sửa lỗi, tổng quát hóa hoặc giữ kỹ năng dài hạn.

## Đã tách version cho shadow recovery policy

[WrapperP0ac7bb8](../../runs/20261114T013315Z-recovery-policy-p0-ac7bb8/result.json)
chạy CUDA, ba seed và observation từ filetool thật; không train hay thực thi
action. [Research wrapper](../../src/scarx/evaluation/recovery_policy.py) giữ
nguyên64input/H128/sáu slot ALIF, nhưng action order và causalencoder được khóa
trong envelope checkpoint riêng. Loader live không nhận envelope này; kiểm
checksum/encoder/actionorder/innerseed/mode/duplicate/size trước khi nạpCUDA.
Cold proposal và full state khớp; không mang weights lab vào Hermes.

Policy chưa học có thể đề xuất finish khi file còn thiếu hoặc cứ retry. Đó là
đầu ra chưa qualified, không phải hành động đã chạy hay giải thích nguyên nhân.
Không dùng P0 để bật policy;8regressionchecks và9proposals chỉ kiểm đường dữ
liệu/compatibility. [Đề xuất DEV tiếp theo](../../experiments/hybrid-recovery-policy-dev-proposal-v1.md)
tách RAG/rule/frozen/readout/hidden/shuffle/reset, verifiedeffect/support,
unseen templates, budget và retention; chưa bắt đầu training từ proposal.

Kiểm tra tổng hợp hiện tại213Python và9Node qua. Một lượt tổng hợp trước có
WinError10053 ở positive HTTP nonce fixture; lần chạy riêng qua. Fixture nay
đọc body request được chấp nhận và đóng HTTPError response bị từ chối đúng cách;
không đổi nonce gate hoặc retry model để ép điểm. Giữ lịch sử failed run trong
validation receipt; chưa xác định nguyên nhân hệ điều hành của lỗi10053.
Warnings còn lại: CuPy không có fulltoolkitpath dù kernel/parity qua, và archive
trùng member được test cố ý tạo để loader bác bỏ.

Đã quan sát trang localhost thật qua browser: panel ghi đúng run554c19, thời
gian/QA/checkpoint và câu trả lời hình tròn màu vàng, tách bộ đếm live5updates.
Đây là kiểm UI/trace, không phải điểm chất lượng model hoặc hình ảnh mock.

## Hệ quả đối với kế hoạch tiếp theo

Ba vấn đề riêng cần kiểm chứng:

1. Neural input đang thiếu error/probe; cần version encoder/checkpoint rõ ràng
   trước khi nối policy mới. [P0 đường thông tin](INCIDENT_INFORMATION_AND_BENCHMARK_AUDIT_20261114.md).
2. Cơ chế bảo vệ synapse phải thực sự tham gia update và chứng minh giữ được kỹ
   năng đồng thời còn học mới. Không quảng bá buffer được lưu là metaplasticity
   hiệu quả; cũng không mặc định thêm anchor sẽ tốt hơn, vì có thể mất plasticity.
3. Reward phải dựa vào đúng effect và câu trả lời có hỗ trợ, không chỉ API thành
   công. [Executor và lỗi Qwen thực](RECOVERY_EXECUTOR_AND_NATIVE_FAILURES_20261114.md).

Đây là các giả thuyết để đăng ký cơ chế/đối chứng mới, không phải thay công thức
học trong lượt audit. Không sửa ALIF, không backprop/BPTT, không đổi model live
hoặc bật LoRA để che kết quả PIVOT.

## Trạng thái phép đo 8 giờ

Preflight240s7b589e hoàn tất: hai chu kỳ video/QA, bốn checkpoint/rollback,
sáu local/hidden updates, có replay; CPU/native/CUDA checks trước đó203 qua.
9 Node checks kiểm trang trace và composer: lab không ghi vào livecounter,
heartbeat cũ chỉ cảnh báo chậm chứ không tự kết luận process đã dừng.

Lượt [0acab3](../../runs/20261114T005920Z-shared-endurance-v1-0acab3/interruption-audit-v1.json)
dừng sau heartbeat1024,594s,4video cycles,1checkpoint. Không còn PID/exec handle
và không có final receipt. Giữ original heartbeat, ghi interruption audit riêng.
**Không đạt8h; nguyên nhân dừng chưa xác định**, không suy thành lỗi model hoặc
OOM nếu thiếu bằng chứng.

Lượt mới554c19 dùng cùng nguồn/hash và preflight đã qua, chạy bằng Start-Process
ẩn với log local riêng. [Biên bản kiểm PID](../../runs/20261114T011841Z-shared-endurance-v1-554c19/process-identity-v1.json)
xác minh tiến trình đang tồn tại và cmd đúng. Không khởi chạy lại chỉ vì đọc
heartbeat timeout. [Monitor chỉ đọc](../../scripts/monitor_hybrid_endurance.py)
kiểm process/source/final receipt, không stop/start bất kỳ service nào.

~~~powershell
.\local\hybrid-v1\runtime-env\Scripts\python.exe scripts/monitor_hybrid_endurance.py runs/20261114T011841Z-shared-endurance-v1-554c19
~~~

[Mở đúng trace của lượt554c19](http://127.0.0.1:8775/?after=7357&tab=trace).
Trang tự hiện thời gian, video/QA, checkpoint, neural lab và đáp án thật.
Số cập nhật trên cùng trang của model live vẫn tách riêng. Nếu heartbeat quá90s
thì báo chưa nhận nhịp mới; final verdict chỉ đến từ receipt của orchestrator.

Lượt đang chạy chưa được tính là hoàn tất. Không cam kết trạng thái “đang chạy”
của tài liệu luôn là trạng thái hiện tại; dùng monitor/receipt để đối chiếu.
Source của lượt dài được khóa; chỉ thêm auditor/docs/UI không thuộc manifest,
không sửa module học/cognition/media/shared worker.

Ngay cả khi đủ8giờ, đây vẫn là engineering endurance trên clip21s lặp mỗi5phút,
không phải8giờ video liên tục, bộP1 video đa dạng, hoặc lifelong learning PASS.
Main gốc và R13 giữ nguyên; live vẫn Qwen0.8B/ASR/ALIF trên GPU.
