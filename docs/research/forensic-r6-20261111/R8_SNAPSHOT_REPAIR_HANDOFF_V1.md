# R8 snapshot repair P0 — bàn giao kỹ thuật

**Kết quả: `P0_STATE_REPAIR_PASS_RESEARCH_ONLY`.** Sửa bằng wrapper data-only
ở `src/scarx/evaluation/forensic/r8_snapshot.py`; không sửa candidate equation,
forward, update rule, checkpoint v1 của Gemini, main hoặc phiên live. Đây là
engineering trong yêu cầu tiếp tục xử lý vấn đề, chưa là quyền mở quality study.

## Phần đã sửa

- Format riêng, explicit model template và sourceSHA. Không import class từ
  checkpoint, không pickle; chỉ hỗ trợ MTSG/BF public research H48/D8/O2/K8/R12.
- MTSG giữ đủ threshold, gamma vector, PCG64 RNG, weights/stable weights, neuron
  state, traces, normalizer, counters và initial hidden weights.
- BF giữ đủ fast/slow input/recurrent/readout, bias, context weights kể cả khi
  gating bị tắt, neuron state, traces/normalizer và mọi scalar đã biết.
- Caller JSON controller/replay/cursor/RNG lưu nguyên, reject tuple/non-string
  keys/numpy scalars thay vì tự ép kiểu và làm đổi trạng thái. Không tự tạo một
  controller học mới hoặc nhập state app ngoài schema.
- Metadata và từng array có checksum. Mọi NPY header/shape/dtype/body length được
  kiểm trước materialization; reject float64/NaN/Inf/object/duplicate/huge header,
  field lạ, index/decay/counter/RNG sai. Template không đổi khi payload lỗi.
- Atomic save và rollback bằng replacement model; injected os.replace failure
  giữ checkpoint đã commit và dọn temporary file.

## Kiểm chứng đã chạy

Run cuối: `runs/20261112T021800Z-r8-snapshot-repair-p0-v1-5da0b6`;
receipt `artifacts/r8-snapshot-repair-p0-v1.json`.

- **76/76test qua, không skip**, gồm15test adapter và gate/forensic/core/G1/native
  regression. Bốn chế độ native parity frozen/readout/STDP/full đều được chạy.
- Với cả MTSG và BF, **fresh process** nạp snapshot rồi chạy next-forward và một
  tiny local-update fixture: loss, controller RNG draw và hash toàn state đều
  exact so với process không bị ngắt. Saved snapshot không đổi.
- Bốn actual fixture update calls cho parent/cold verification, ngoài các tiny
  update của pytest; không train dataset hoặc chấm acquisition/retention.
- Peak sampled process-tree RSS **231,08MB**, commit **187,03MB** trong nhóm
  regression; Windows per-process2GB cap, không iGPU. Sampling không phải hard
  cap tổng process tree và không chứng minh stress8giờ.
- Peer tracked source giữ nguyên; main giữSHA
  `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`. Không restart Hermes, không mở/ghi
  SQLite live hoặc dùng checkpoint private.

Lượt sơ bộ `20261112T021445Z-r8-snapshot-repair-p0-v1-86efaa` với72test được giữ;
lượt cuối bổ sung kiểm JSON type coercion và atomic replace failure. Không sửa
run lịch sử để làm đẹp số test. Protocol trước lượt xác nhận:
`experiments/r8-snapshot-repair-p0-v1.md`.

## Dùng/tái kiểm

```powershell
python -X utf8 scripts/verify_r8_snapshot_repair.py
```

API explicit cho một model nghiên cứu đã được caller tạo từ source đã pin:

```python
from scarx.evaluation.forensic import r8_snapshot
r8_snapshot.save(model, path, controller=caller_json_state)
replacement_model, caller_state = r8_snapshot.load(fresh_matching_template, path)
```

Phải dùng explicit API này; serializer gốc R8 vẫn chưa sửa. Adapter không tự nối
vào learner/app. Chưa chứng nhận migration của PlasticCore/importance/optimizer,
capacity48–128, SQLite hoặc Hermes; không khôi phục state mà caller không cung
cấp. Cold fixture kiểm next-forward và next-episode update, **không** tuyên bố
đã có resume controller của một phiên training đang giữa episode.

Các nút thắt representation, credit attribution, distractor và forgetting vẫn
còn. Bước tiếp theo là sửa matched controls và protocol P1 riêng, không dùng
P0PASS này để promote MTSG hoặc BF vào production.
