# P1A-v2 — đề xuất sửa guard và một lượt confirmation kỹ thuật

**CHỜ DUYỆT; chưa chạy SDK preflight hoặc forward mới.** V1f39bc6 giữ nguyên
`P1A_STOPPED_NOT_PASS`; không dùng diagnostic để đổi FAIL thành PASS. Protocol
v1 cấm tự mở confirmation khi fail, nên cần phê duyệt riêng cho lượt này.

## Bằng chứng và thay đổi cần thiết

V1 đã ghi18 CUDA forwards/10 dispatches,8 macro executions/2 scope denials.
Data-only diagnostic khớp input/readout/action/effect và cold/rollback exact,
nhưng guard chặn `socket.gethostname` cùng3 `subprocess.Popen`. PermissionError
bị thư viện bắt, còn cờ lỗi chỉ được kiểm cuối run: đã không dừng đúng boundary.
Raw args/callsite của3 Popen không được log, nên chưa thể kết luận chúng chạy
lệnh gì. Không tự whitelist tất cả probe hoặc gọi v1 an toàn/PASS.

Đã đọc source Python3.11 `platform._node`/`_syscmd_ver` và CuPy/compiler/
cuda-pathfinder đã cài. `_node` dùng local gethostname; `_syscmd_ver` có các
Windows VER fallbacks và bắt OSError. Đây là candidate explanation, không là
chứng minh callsite của3 event v1. [Python docs](https://docs.python.org/3.11/library/socket.html#socket.gethostname)
phân biệt local hostname với DNS/network lookup. Không cần đổi ALIF hoặc học
thêm để sửa lỗi harness này; không rerun benchmark SNN/RAG đã biết.

Candidate code đã có, chưa tích hợp vào runner đang khóa v1:

-`src/scarx/evaluation/bridge_effect_guard.py`: cho phép local hostname query;
  unknown network/process attempts bị chặn và đặt cờ sticky. `check()` phải là
  bước đầu của mọi resource check và trước mọi precharge; SDK nuốt exception
  vẫn không cho forward tiếp theo.
-Windows VER probes từ `_syscmd_ver` của platform.py đã pin chỉ được nhận dạng
  để **decline như unavailable**, không spawn shell. Chỉ exact nvidia-smi
  dedicated-memory query từ `sample_physical_gpu` đã pin được phép spawn. Pin
  absolute executable/source path +SHA256;4 declined VER probes/128 GPU queries
  tối đa. Chỉ3 literal VER forms trong stdlib; không cho compiler, Python canary,
  model worker, shell command khác, URL/DNS hoặc chuỗi thêm metacharacter.
-`src/scarx/evaluation/bridge_environment_metrics.py`: query GPU qua executable
  nvidia-smi tuyệt đối/pin, kiểm1 dòng/free≤total, timeout3s. Dùng làm guard cho
  dedicated-memory point samples, ghi CUDA runtime availability riêng. Không
  gọi CUDA allocatable bytes là phép đo toàn hệ hoặc backfill memory peak v1.
-7 pure regression checks qua: local query, network/process denial, sticky stop
  sau swallowed exception, pinned source/executable/exact command và parser
  telemetry. Không chạy subprocess, network, GPU hoặc model trong các tests này.

V2 integration chỉ thay namespace/manifest, hook installation, resource checks
và log chi tiết guard. `FrozenBridgePolicy`, adapter, macro dispatch, action
semantics, public tasks, two seeds, ALIF/FP32/readout/learning equations giữ
nguyên. Per-seed loop phải AST-identical với v1 trước thực thi. Không refactor
Hermes/gateway hoặc sửa numerical gates để khiến kết quả qua.

## Lượt confirmation được đề nghị

1. Khóa source và pin metadata/executables trước chạy. Namespace/marker/ledger
   v2 riêng, ghi SHA của result/ledger v1 đã tiêu18 forwards10 dispatches.
   Không reset/refund/ghi đè ledger hay raw artifacts v1.
2. E0: cài sticky guard, kiểm SDK CUDA0 và dedicated-GPU query **trước khi tạo
   ALIF/forward**. Chỉ metadata handling đã mô tả; declined VER probes được log
   riêng, không được ghi là subprocess đã chạy. Unknown call
   ghi stack/executable basename/argv hash và dừng. Không suy whitelist mới từ
   một event vừa gặp. Không log env/keys hoặc tên máy trả về.
3. Chỉ khi E0 qua mới chạy cùng8 technical cases/two seeds260113/260137 của v1.
   Dự kiến18 forward, cap **24 forward MỚI**,10 dispatch MỚI gồm8 macro/2 denials.
   Đây là quota confirmation bổ sung, không che18/10 đã chạy của v1.
4. Full frozen arrays/learned metadata/RNG/replay giữ nguyên trong mỗi model;
   journal không rewind; malformed/foreign receipt reject trước forward;
   same-device checkpoint/cold/rollback exact. Independent data audit không thêm
   model calls, giữ readout math tolerance3e-6 và không nới history CPU/CUDA FAIL.
5. Max10phút gồmE0/queries/matrix, process RSS/private commit≤1GB, toàn artifacts
   gồm build cache≤20MB, dedicated GPU free≥512MiB ở điểm đo. Bất kỳ vi phạm
   sticky/budget/source/device/persistence nào đều STOP, không tự retry/seed mới.

V1→v2 thay SDK/control guard và nguồn telemetry nên một lượt correctness
confirmation là cần thiết; không phải học lại để tìm điểm chất lượng tốt hơn.
Không mở language/video benchmark, không research learning rule mới hoặc grid.

## Phạm vi kết luận

V2 PASS chỉ GO cho frozen action/outcome bridge tronglab. Không skill/hidden
representation/retention hoặc SNN-over-RAG PASS; không train/LoRA/replay/weight
updates, native VLM/ASR calls hoặc live promotion. Giữ main/R13/Hermes RC14
nguyên vẹn. P1B/data acquisition/train phải được duyệt riêng sau kết quả.

Đề nghị duyệt **tích hợp candidate guard + E0 + một confirmation v2** trong
budget trên; không cần hỏi lại từng fixture. Chưa thực hiện từ phê duyệt v1.
