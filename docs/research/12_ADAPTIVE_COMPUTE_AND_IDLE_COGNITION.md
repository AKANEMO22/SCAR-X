# Ba chế độ hoạt động trong cùng neural core

2026-11-09. Tham chiếu cho master, không là kế hoạch chính thứ hai. Người dùng
đã làm rõ “ba tầng” là ba chế độ, chưa phải ba tầng neuron vật lý hoặc ba model.

| Chế độ | Hiện có | Còn thiếu |
|---|---|---|
| Fast Neural Processing | Foreground ưu tiên, recurrent core, local teaching thử nghiệm | Skill coverage/quality |
| Deep Thinking | Dynamics hồi quy, fixed3ticks/word step | Adaptive Computation Time, thử giả thuyết/kiểm tra và calibrated halting |
| Autonomous Learning / Idle Thinking | Idle dynamics, review/ôn accepted labels, hỏi người dùng | Tự chọn mục tiêu, tạo thí nghiệm và học quy luật đã kiểm chứng |

Fast/stable synapses là thang thời gian học, không đồng nghĩa với ba chế độ
hoặc ba lớp vật lý. Đề xuất topology nhiều lớp cần signal audit và protocol
riêng theo P1; không tự suy ra đã có từ cách gọi “suy nghĩ sâu”.

## Hai trục thích ứng độc lập

Adaptive Neural Capacity chọn neuron/synapse active: prototype48–128 hiện dùng
input volume/modality, chưa đo độ khó ngữ nghĩa. Adaptive Computation Time chọn
số tick/vòng trước khi dừng: chưa có policy thích ứng trong serving. Output
length không là số bước nghĩ. Ví dụ20/128–3bước,55/128–12bước,105/128–40bước
chỉ minh họa, không là thông số hoặc số đo đã triển khai.

[ACT](https://arxiv.org/abs/1603.08983) và
[PonderNet](https://arxiv.org/abs/2107.05407) nghiên cứu học số bước xử lý/dừng.
Chúng dùng tối ưu khả vi/end-to-end; không tự đáp ứng local-only của SCAR-X.
Không đưa backward toàn mạng của phương pháp đối chiếu vào nhánh chính.
[DECOLLE](https://pmc.ncbi.nlm.nih.gov/articles/PMC7235446/) có ví dụ SNN3lớp,
local error/readout phụ theo lớp. Đây là nguồn đối chiếu cho P1, không phải
kiến trúc SCAR-X đã triển khai. Local surrogate derivative cần audit đường tín
hiệu; “không BPTT” chưa tự chứng minh mọi điều kiện local learning.

## Autonomous Cognitive Loop — Proposed

Chọn vấn đề chưa giải từ trải nghiệm → lưu giả thuyết/nguồn → tạo biến thể hoặc
counterexample → verifier độc lập → chọn mẫu học → local update → đo acquisition,
retention và transfer → commit/rollback → hỏi người dùng khi thiếu bằng chứng.

Vòng đầu chỉ dùng tác vụ đóng có generator/verifier như số học parser hiểu.
Draft tự sinh/độ tự tin/đồng thuận không là nhãn đúng. Antigravity là teacher
tạo curriculum có kiểm tra, không thay core và không tự gọi cloud liên tục khi
idle. Memory lưu ví dụ/bằng chứng; controller chọn mục tiêu/budget; neural core
phải học representation và giải tình huống mới sau restart với retrieval tắt.

## Gate theo roadmap hiện có

- P0: snapshot goal/scheduler/RNG/counters đầy đủ, budget và nhường foreground.
- P1/P2: sửa representation/credit assignment và readout stability; không tăng
  data/model để che failed acquisition hoặc nới các ngưỡng đã đăng ký.
- ACT: so fixed-small/fixed-large/adaptive cùng RAM và matched compute/time;
  đo chất lượng theo độ khó giữ kín, sai khi dừng, calibration và no-progress.
  Ablate số ticks và số neuron riêng; hết budget không được báo đã giải xong.
- P3–P5: no-idle/replay-only/exploration với cùng update/time/RAM; chỉ kết luận
  có ích khi transfer và retention đạt. Số hypotheses/ticks/spikes không là IQ.

Budget gồm neuron active, ticks, elapsed time, queued updates, RAM và giới hạn
không tiến bộ. Nhường người dùng ở quantum tiếp theo; không sửa weights giữa
lượt trả lời. Tab Hỏi để học nhận câu hỏi chủ động, không tự đổi chat/chiếm focus.
Trace ghi mode, input/source, active/allocated, ticks, stop reason, verifier và
accepted/rejected updates; không giả độc thoại nội tâm.
