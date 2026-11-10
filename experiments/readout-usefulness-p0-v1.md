# P0 kỹ thuật — usefulness tại readout, không pilot P2

2026-11-10. Người dùng trả lời: **“Chỉ duyệt kiểm tra P0”**. Phạm vi này ghi
đè phần chờ duyệt P0 của v2; pilot168job, trainingchat/vision và deploy vẫn chưa
được duyệt. Master plan và các gate chất lượng không thay đổi.

## Phạm vi triển khai

Prototype riêng trong `evaluation/readout_usefulness.py`, không import từ
serving/neural/learning. Giữ `UnifiedCore` G1 forward48ALIF và hidden scalar
STDP nguyên bản. Chỉ thay delta Wout bằng g từ **omega trước update**; bias,
reward advantage, hidden traces/updates và clipping giữ rule G1. Tính benefit
trên weights/features trước update bằng float64; sau update mới cập nhật omega.
Beta4, decay0.995; beta0 chỉ dùng kiểm tương đương, không tuning.

Không mượn rule có anchor của `SpikingCore.learn` làm baseline G1, vì
`UnifiedCore.learn` là implementation khác. Bảng so Phase2.1/SI chỉ là đối chiếu
phương trình, không gán chúng thành baseline numeric tương đương.

Nhãn P0 chỉ từ registry fixture công khai: p0-alpha→alpha., p0-beta→beta.,
p0-gamma→gamma. với input `P0 fixture alpha/beta/gamma`. Đây là test oracle,
không tra đáp án khi serving và không xác minh kiến thức người dùng. Không
chấp nhận cờ `verified=True` hoặc cặp ngoài registry như bằng chứng đúng.

## Kiểm tra khóa trước đo

1. Deletion benefit: ma trận đầu ra3×4,9×48,266×48; seed42001; mask tùy chọn.
   So công thức log-space với forward xóa từng synapse trên bản sao float64.
   Maxabs error<=1e-9; finite-difference output-loss epsilon1e-6, atol1e-7.
   Stress logits tới±1000, zero features, một class active, masked target invalid.
   Không gradient hidden, autograd, BPTT hoặc quality score.
2. Beta0:3seed42201,42203,42207; mỗi seed3fixtures×4cycles (12segments,
   36endpoint). So đầy đủ base neural payload với `UnifiedCore.train_pair`
   sau từng segment, bao gồm điện thế, eligibility, RNG, clocks, stable weights.
3. Beta4: cùng3seed/12segments chỉ technical trajectory/timing, không chấm
   accuracy/generalization/retention hoặc chọn model. So hidden/bias/reward
   một endpoint với reference cùng state; omega đúng/sai label gating được
   kiểm riêng bằng feature/logit fixtures có chủ đích, không thành quality pass.
4. Codec/checkpoint: snapshot ở token boundary giữa segment; restore/tiếp tục
   khớp weights/state/omega/RNG/controller/replay; serialization có format riêng
   và checksums. Rejected transaction/fault save phải rollback nguyên trạng,
   payload hỏng không được mutate. Không hứa snapshot giữa Python instruction.
5. Norm matching: hướng delta đã clip, radius0.5×min norm, relative applied
   norm error<=1e-5. Seed42003,32test nhóm3proposal interior. Zero radius phải
   báo rõ; float32 half-ULP case bắt buộc từ chối nếu không biểu diễn được bán
   kính. Không nới tolerance, không xóa invalid case khỏi báo cáo; guard đúng
   không có nghĩa matched bank luôn có hiệu lực cho mọi state tương lai.
6. Budget: một thread, cap2.000.000.000byte, <=180giây P0 runner. Đo RSS/commit
   cả dump/load/rollback và utility với maxoutput266×48 (100calls, không học).
   Tính importance/scratch riêng; latency/throughput là microbenchmark, chưa
   dự toán168job hoặc bằng chứng học liên tục. Không iGPU/điện năng ước lượng.

## Quyết định

Technical pass chỉ cho công thức/parity/codec/guard trong test đã đăng ký. Mọi
quality gate vẫn **chưa đo**. Nếu operator/parity fail, dừng trước pilot, báo
invalid. Nếu matched radius không biểu diễn được, helper phải fail closed và
contrast đó invalid; không đổi thành “matched” nhờ raw norm. Không thay g/công
thức từ kết quả chất lượng vì lượt này không được mở dữ liệu chất lượng.

Pilot P2 cần người dùng duyệt riêng dù P0 pass. Không tải Qwen/Antigravity mới,
không dùng127wordingdevelopment, không đọc privateweights/DB hoặc gọi teach/
restore main. Chỉ quan sát status/export nếu cần chứng minh phiên thật không đổi.
