# Codex review Gemini R6C — 8f24a5a

Nhánh: `research/gemini/r6-local-associative-learning`.
Commit đã push: **`8f24a5abde0598201e558eee559564ccde889145`**.
Quyết định: **REQUEST_CHANGES; giữ CKCER làm ứng viên nghiên cứu readout; chưa đưa vào main/serving.**

## Có gì thực sự đáng giữ?

1. Chuẩn hóa **trước** gating giữ nhánh không được chọn bằng đúng0, thay vì z-score biến nhánh tắt thành `−mean/std`. Codex kiểm trực tiếp class thật: cập nhật trọng số nhánh không chọn =0.
2. Đọc spike count + adaptation **trước query** tránh việc xung query ghi đè trạng thái đang đọc. Đây là lựa chọn biểu diễn thời gian đáng kiểm chứng thêm, không phải neuron tự khám phá kiến thức.
3. Gate Oja ở Mode B học phân nhóm tín hiệu query từ transient neural. Có trọng số gate thích nghi, nhưng chúng thuộc `CKCERHead.w_q`; `w_in/w_rec` của recurrent ALIF vẫn đóng băng.
4. Kết quả đọc liên kết hai bit rất mạnh trên các condition đã chạy. Codex đã replay đúng8seed lịch sử và chấm lại **344 nhóm prediction** bằng evaluator độc lập. Các accuracy/both_A của CKCER A/B khớp bản công bố. Đây là tín hiệu đủ đáng nghiên cứu, không nên bỏ vì báo cáo có lỗi.

| Kết quả CKCER-A | Giá trị xấp xỉ | Phạm vi đúng |
|---|---:|---|
| Clean both_A | 99,60% | H48 frozen + readout/gate học trên bài hai key/bit |
| Paired same-sample distractors | 96,58% | Đã học trên train có distractor |
| Noise both_A | 93,68% | Đã học lại ở noise cao; chưa là clean→noise zero-shot |

## Các finding phải sửa

### G-C01 [P1] Confirmation bị dùng lại sau khi đã FAIL

Raw run `20261111T095646Z-r6c-confirmation-v1-d215bc` ghi **STOP/G8=False**.
Sau đó source hash `r6_associative.py` đổi từ
`2e1e8a4ccff709a7bce5ba3c28db5d5215efec586623eac1caa9fa79ae9fd5ac`
sang `4f23d511e511accc26312627130c9ad73b0cbee7553319857bd47d5ad19f37bf`,
và cùng8seed chạy lại ở `20261111T100240Z-r6c-confirmation-v1-f8d321` thành GO.
Đối chiếu source.zip cho thấy không chỉ đổi log: generator protected XOR/order
được thay bằng `tasks21.representation`, baseline sim_xor/XOR đổi từ `sum_s` sang
`m_pre`. Đây có thể là sửa thiết kế/harness có ích, nhưng là **corrected replay
sau khi thấy confirmation**, không phải untouched pre-registered10/10 PASS.
`argument-audit.md` nói không có post-hoc selection cần đính chính và công bố
chronology/amendment. Giữ nguyên cả run STOP và GO.

### G-C02 [P1] Protocol khác thực thi ở nhiều condition

Protocol §4.2: noise clean.01→test.03, distractor chỉ lúc test, shifted write
windows `{1,2,3}`/`{5,6,7}`, query ticks10/11/12 vớiT13. Code thực thi:
clean.08, noise.16 **ở cả train và test**; distractor **cũng vào train**;
shift là+1tick toàn chuỗi; query variation chỉ chuyển10→11 trênT12.
Xem `r6_associative.py:1097–1161` và protocol:107–114.
Không so96,58% này như phép thử tương đương R6D55,02% clean-trained→distractor.
Cần bảng registered/executed và gắn nhãn exploratory/corrected cho đúng.

### G-C03 [P1] “Single-factor ablation” đổi cả learning rule

`R5AblationHead.train_online:846–868` và `MatchedCapacityReadout:444–459`
dùng lỗi hard-argmax perceptron. `CKCERHead.step_update:315–326` dùng softmax
probability error. Không thể quy toàn bộ gain Step2→Step3 cho bỏ shared trunk/P.
Counterexample nhỏ trên class thật: cùng ba mẫu/nhãn0, ablation có max|W|=0,
CKCER max|W|=.075. Bằng chứng ở `GEMINI_MATH_COUNTEREXAMPLES.json`.
Cần matched learner trong protocol sau; không sửa score cũ thành causal PASS.

### G-C04 [P1] Chưa chứng minh hidden representation learning

Source `capture_r6` gọi `P.capture`, chạy frozen ALIF. Chỉ readout và gate Oja học.
Mode B không đọc trực tiếp cue values ở head, nhưng vẫn cần biết query onset để
latch transient; nó chưa tự tìm ranh giới task trong hội thoại liên tục.
Gates ở `evaluate_protocol_gates:1347–1398` thực tế lấy **Mode A**, nên câu “cả
Mode A và Mode B pass10/10” thiếu bảng gate thực thi riêng cho Mode B.
Không gọi Oja query head là hidden recurrent synapses đã học kỹ năng mới.

### G-C05 [P2] “Zero cross-talk” chưa đúng cho toàn bộ đầu ra

Trọng số nhánh tắt không cập nhật, nhưng bias `b` dùng chung vẫn cập nhật.
Codex đo sau một update queryA: trọng số nhánhB đổi0, còn scores queryB đổi
**.02500000037** do shared bias. Đây chưa phải phép đo forgetting, nhưng bác
bỏ suy luận branch isolation tự bảo đảm hoàn toàn không có interference.

### G-C06 [P1] Theorem về shared trunk dùng sai điều kiện cho R5 gốc

R5 gốc đọc **post-query** state. State của qA/qB không nhất thiết giống nhau,
nên shared trunk không nhất thiết đóng góp0 vào query-switching margin.
Counterexample trên frozen seed121011,32cặp: chênh post-state lớn nhất
**1,80248046**; một readout chỉ đọc coordinate đó đã có chênh margin khác0.
Định lý giả sử cùng `s` có thể áp dụng cho pre-query ablation, không tự động
áp dụng cho R5 post-state. Sửa điều kiện định lý và phần quy nguyên nhân.

### G-C07 [P2] Seed và protected-task claims cần thu hẹp

Bank121011…121043 đã xuất hiện trong DeepSeek R6A; không fresh trên toàn dự án.
Protocol còn ghi confirmation “run once”, nhưng đã có hai lần với code khác.
Protected tasks dùng các head mới được fit riêng, có mode `ungated`, không là
bằng chứng một learner giữ kỹ năng cũ sau học mới. R7/P2 vẫn chưa được chứng minh.

### G-C08 [P2] Ridge/numerical provenance

`argument-audit.md` nói ridge bilinear chỉ both_A.6744 nhưng pipeline đã commit
đo `OfflineRidgeDiagnostic` trên **CKCER features**, both_A≈.9960. Không có cùng
feature-tensor/source digest để dùng lời giải thích đó bác bỏ ridge của R6B
(.9858056 trên features R5 gốc). Ghi rõ đó là hai probe khác nhau; ridge cũng
không phải upper bound toán học.

Replay nghiêm ngặt **không bit-exact toàn bộ seed_results**:220 margin entries
khác làm tròn, và baseline R5 reversed-write seed121037 lệch một prediction
(both_A .761905 vs.769841). CKCER A/B accuracy/both_A không đổi. Không che khác
biệt này bằng nhãn “exact reproduction”. Việc set env thread bên trong main sau
khi NumPy đã import chưa chứng minh thread pool thực tế đã là1.

## Đề xuất tận dụng

Giữ ba thành phần làm ứng viên cho protocol sau: pre-gating normalization,
pre-query temporal summary và neural query gate. Chưa copy vào learner G1.
Trước lượt mới: sửa báo cáo/chronology, khóa đúng baseline/condition, cùng
learning rule cho ablation, kiểm shared bias, seed mới được inventory và đo
hidden-learning bằng frozen/zero/shuffled + common refit probe. Không tự mở P2.

## Chứng cứ của Codex

-5/5test class thật qua trong detached checkout riêng.
-Historical replay: `runs/20261111T102246Z-forensic-gemini-replay-v1-8ad15f`.
-344nhóm prediction, source/log/command/resource giữ trong run trên.
-Peak sampled tree commit145.879MB; child hard per-process2GB bật; không iGPU.
-`GEMINI_MATH_COUNTEREXAMPLES.json` và script tái lập đi kèm.
-Không sửa main, branch Gemini, live weights/SQLite, Hermes hoặc kiến trúc core.
