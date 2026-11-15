# P1A — data binding có bằng chứng, toàn lượt vẫn STOP/FAIL vì harness guard

**Kết quả chính: `P1A_STOPPED_NOT_PASS`. Không chạy lại.** User duyệt riêng P1A
ngày15/11/2026. Code/source trước execution tại commit863513e5; source copy và
raw result bất biến trong [runf39bc6](../../runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6/result.json).
Không chấm P1A PASS từ các kiểm tra thành phần hoặc tự sửa tiêu chí sau kết quả.

## Đã thực thi thật

-18 CUDA0 forwards,2 seed kỹ thuật260113/260137, ALIF H128/FP32 **frozen**.
-10 dispatch attempts:8 actual local macro executions và2 scope-lease denials.
  Ordinary masks đều có2 choices; action do neural argmax chọn, không fixed rule.
-0 learn calls,0 native VLM/ASR calls, không backward/BPTT.13 pure bridge tests
  +12 checker tests qua trước execution; không có forward trong preflight.
-11,192s elapsed; process peak RSS725.966.848 byte, private commit751.398.912
  byte. Phạm vi là worker P1A, không peak RAM toàn hệ AI/Hermes.
-Raw artifacts khoảng11,3MB ở lúc audit, dưới20MB. Model snapshots ZIP_STORED,
  giữ FP32; không quantization/precision change. CuPy build cache giữ local,
  không phải model weights hoặc dữ liệu để học.

| Case ordinary | Seed260113 | Seed260137 | Outcome đo được |
|---|---|---|---|
|1: source có thật|inspect_status|inspect_status|Cả hai đọc đúng bound file và có task proof|
|2: source thiếu|inspect_status|retry|Cả hai nhận FileNotFoundError thật; không lấy file khác làm proof|
|3: job có progress|inspect_status|inspect_status|Đọc trạng thái thực; chưa hoàn thành job, không tự báo resolved|
|4: field thiếu|show_schema|clarify|Một lần đọc schema; một lần queue đúng alarm_time. Không thành task completion|

Case5 kiểm4 biến thể foreign/swapped receipt mỗi seed, gồm receipt hợp lệ của
decision khác. Case6 future/private input reject trước forward; ledger không
tăng. Case7 revoke lease sau proposal:2 denial có origin runtime_scope_guard,
không quy công cho SNN. Case8 model/pending checkpoint, cold load và rollback
cho continuation probabilities/inner-model bytes exact; journal/lease không rewind.

Đây là model chưa học; khác nhau giữa seed không chứng minh intelligence. Hai
action file ở case1 đều là capability an toàn có thể đọc đúng nguồn, nên task
proof không là thắng rule/RAG hoặc học representation. Không dùng các kỹ thuật
fixture này làm finalTEST hoặc cập nhật synapse.

## Vì sao toàn run không PASS?

Guard v1 của Codex dùng `event.startswith('socket.')`, nên chặn cả local
`socket.gethostname`. Nó cũng chặn3 subprocess attempts. Callback chỉ raise
PermissionError và append tên event; SDK/stdlib bắt lỗi rồi tiếp tục fallback.
Runner chỉ xem danh sáchblocked ở cuối, nên **không dừng ngay như protocol**.
Đây là lỗi của harness, không bằng chứng kiến trúc ALIF hoặc rule học bị lỗi.

Truy vấn hostname cục bộ khác network request theo [Python3.11 docs](https://docs.python.org/3.11/library/socket.html#socket.gethostname).
Local platform.py có `_node` bắt OSError và `_syscmd_ver` thử các VER commands;
CuPy/cuda-pathfinder cũng có compiler/canary subprocess paths. **Không có raw
argv/stack ở3 event cũ**, nên không khẳng định tất cả đến từ một nguồn cụ thể.
Các attempts đã bị chặn; không suy rằng chúng đã spawn thành công hoặc là API
model. Source snippets/hash và guard diagnosis được giữ riêng, không giả peer
approval hoặc nới gate để gắn PASS cho v1.

Resource log v1 dùng CUDA-runtime reported availability và đặt tên
`physical_cuda_free_bytes`. Chưa có simultaneous dedicated-GPU telemetry đủ
để chấm global physical headroom trong lượt này. Post-run nvidia-smi là phép
đo thời điểm khác, không backfill peak/guard v1. V2 đề xuất pin dedicated query
và giữ hai nguồn riêng; không đổi raw resource records cũ.

## Data-only diagnostic sau khi dừng

[Diagnostic riêng](../../runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6/poststop-data-diagnostic.json)
re-encode input, tính readout từ saved features/weights, so neural argmax và
journal/source/effect với8 macro thực thi. Max probability error4,4512×10⁻⁸,
thấp hơn tolerance3×10⁻⁶ đã khóa.18 forwards/10 dispatches được đối chiếu bằng
attempt ID/hash-chain, không bằng timestamp gần nhau. All saved frozen parameters
và learned metadata giữ nguyên; same-device continuation exact có dữ liệu hỗ trợ.

Auditor có flag `--diagnostic-stopped` bổ sung sau run; default vẫn yêu cầu PASS.
Diagnostic lưu cả SHA auditor đã đăng ký và SHA code đọc sau run, ghi rõ
`original_result_gate_pass=false`/`original_guard_gate_satisfied=false`.
Không gọi model/tool lại, không thay result/ledger/threshold; đây không là một
confirmation mới hoặc external peer approval.

[Live-state audit](../../runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6/live-state-audit.json)
giữ exact27 saved arrays/learned metadata/RNG/replay, updates live5.
[Post-runtime audit](../../runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6/post-runtime-audit.json)
xác nhận RC14 vẫn58 payload khớp, GPU/ready,0 failure; main/R13/serving source
không thay. User có thể xem actual proposals/probabilities, raw tool results,
outcome và STOP event tại `http://127.0.0.1:8775/?tab=trace`.

## Candidate sửa đã chuẩn bị, chưa confirmation

`bridge_effect_guard.py` phân biệt local identity query; giữ sticky stop dù
library nuốt exception; ghi callsite và argv hash thay vì môi trường/keys.
Windows VER probe đã pin chỉ được decline như unavailable, không spawn shell;
chỉ dedicated GPU-memory query đã pin được phép spawn. Unknown process/network
vẫn bị chặn. `bridge_environment_metrics.py` kiểm exact executable
và strict telemetry parse.7 pure tests qua,0 real network/process/GPU/model calls.
Candidate chưa import vào runner v1 hoặc Hermes; không restart serving.

[Đề xuất confirmation v2](../../experiments/hybrid-neural-bridge-guard-confirmation-v2.md)
chờ duyệt riêng theo stop rule. Old18 forwards và10 dispatches vẫn charged,
marker/ledger không reset. Chưa P1B/train, chưa hidden-over-RAG/retention PASS.

Antigravity3.8Flash high/effort high nhận một inline binding review; CLI hết
print_timeout90s và0verdict. Không resubmit/khẳng định remote turn đã hoàn tất,
không tính là approval. Source/prompt/result hashes được giữ trong receipt.

Tái lập diagnostic không thêm forward/tool:

```powershell
& 'local/hybrid-v1/runtime-env/Scripts/python.exe' -I -B `
  scripts/audit_hybrid_neural_outcome_bridge_p1a.py `
  runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6 `
  --diagnostic-stopped
```
