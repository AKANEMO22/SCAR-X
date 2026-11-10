# P0 — Kiểm tra kỹ thuật usefulness ở readout

2026-11-10. **Đã kiểm P0; chưa chạy pilot P2.** Người dùng phê duyệt chính xác:
“Chỉ duyệt kiểm tra P0”. [Master1.32](../RESEARCH_PLAN.md) và
[protocol P0 đã khóa](../../experiments/readout-usefulness-p0-v1.md).

**Kết luận:** công thức, beta0 parity, checkpoint/rollback và restart riêng
process qua kiểm tra kỹ thuật đã đăng ký. Matched applied-norm có giới hạn
float32 phải fail closed. Chưa có bằng chứng usefulness cải thiện retention,
representation, acquisition, chat hoặc vision. Không suy ra GO cho pilot.

## Implementation thực tế và ranh giới

[Prototype](../../src/scarx/evaluation/readout_usefulness.py) nằm ở evaluation,
không import từ app/neural/learning. Khởi tạo mới48ALIF, dùng G1 forward và
`UnifiedCore.learn` cho bias/reward/hidden scalar STDP nguyên bản. Chỉ thay
candidate Wout bằng delta nhân g của omega trước update. Benefit dùng pre-update
weights/features; chỉ prediction đúng với nhãn fixture có nguồn mới tạo
importance, rồi mới cập nhật omega cho bước sau. Không có Wout.T×error tới hidden
trong prototype này, không backward toàn mạng/autograd/Transformer/LLM proxy.

Đây không phải model thứ hai được khai báo trong Hermes. Không import code hay
weights P0 vào serving, không lấy snapshot64neuron riêng làm điểm xuất phát.
Fixtures alpha/beta/gamma là test oracle công khai, không là nhãn sự thật cho
trò chuyện. API không cho cờ `verified=True` biến cặp tùy ý thành nhãn đã kiểm.

| Cơ chế | Update/importance trong code | Có tương đương numeric với G1 không? |
|---|---|---|
| G1 baseline | Output-local error×feature, bias delta, hidden scalar advantage×STDP; clipping | Reference chính |
| Phase2.1 meta | Importance cộng magnitude chuẩn hóa, chặn20; chia delta theo1+5omega và kéo stable anchor | Không; còn khác stochastic forward và anchor |
| SI-readout adaptation | Tích signal×applied delta, omega theo work/(movement²+0.1) mỗi64update; output/bias anchor | Không; là control riêng, không whole-network SI |
| Draft A | EMA magnitude rho0.95 và g=1/(1+4omega) | Chưa triển khai/chạy ở P0 này |
| U P0 | Deletion loss-benefit dương, omega=max(0.995omega,benefit), g=1/(1+4omega) | Beta0 đã so toàn base payload chính xác; beta4 là giả thuyết mới |

Nguồn đối chiếu: [PlasticCore](../../src/scarx/neural/plasticity.py),
[SI-readout](../../src/scarx/learning/lifelong.py),
[G1](../../src/scarx/neural/unified.py). Không đổi tên magnitude thành usefulness
rồi coi đó là cơ chế mới đã thắng các phương pháp cũ.

## Bằng chứng P0

[Receipt mới nhất](../../artifacts/readout-usefulness-p0.json),
[run/source snapshot](../../runs/20261110T135250Z-readout-usefulness-p0-v1-ca2357/report.md).
Runner: [validate_readout_usefulness_p0](../../scripts/validate_readout_usefulness_p0.py).

| Kiểm tra | Kết quả thực đo | Phạm vi |
|---|---|---|
| Forward xóa từng synapse | 13.092 trường hợp, maxabs error2,44×10⁻¹⁵ | Arithmetic tại readout, không gradient hidden |
| Finite difference output loss | 56 phần tử, maxabs error5,70×10⁻¹⁰ | Kiểm delta output-local, không dùng để train hidden |
| Beta0 parity | 3seed, mỗi seed12segment/36endpoint; base payload exact sau từng segment | Gồm neural state/traces/stable arrays/topology/clocks/RNG G1 |
| Beta4 technical trajectory | 3seed×12segment,36endpoint/seed | Chỉ kiểm vận hành/timing; không chấm chất lượng |
| Save/load giữa segment | Pending token boundary, omega/RNG/replay/controller khớp | Snapshot không ở giữa Python instruction |
| Fresh process continuation | Worker riêng load, dùng cả2RNG, học tiếp pending+2fixtures; toàn payload exact | Restart OS process thật, không chỉ instance mới |
| Fault/reject rollback | State/weights/omega/RNG/controller/replay khớp; giữ core object | Fault atomic replace giữ nguyên checkpoint cũ |
| Source không hợp lệ | Reject arbitrary verified flag, unknown fixture, image teaching trước mutation | Chỉ oracle P0, chưa verifier dữ kiện người dùng |

Stress có masked class, zero features, logits±1000 và trường hợp p32[target]=1
nhưng xóa contribution20 làm loss tăng≈log3. Không tính log(1-p32) trực tiếp vì
làm tròn có thể xóa mất tail. Code dùng logits/float64 cho utility, vẫn dùng
probability/error float32 G1 cho delta gốc. Không đổi neural forward để qua test.

### Giới hạn matched update được phát hiện

32nhóm interior×3proposal khớp appliedΔWnorm, relative error lớn nhất
2,15×10⁻⁸, dưới ngưỡng1e-5. Hàm không mutate bất kỳ input nào trước validation.

Negative control: weight float32 bằng1, candidate tăng một ULP. Radius0.5×min
norm là5,9604644775×10⁻⁸; áp vào float32 cho actualnorm0. Hàm phát hiện và
**từ chối**, không báo matched từ norm ở float64/raw gradient. Radius0 được ghi
riêng, không là một update học mới hoặc một contrast thành công.

Do đó matched bank **không được bảo đảm có hiệu lực ở mọi state**. Nếu pilot
tương lai gặp trường hợp này, contrast invalid theo protocol; không nới
tolerance, đổi dtype, loại sample khỏi metric hoặc dùng điều đó làm retention
win. P0 kiểm guard hoạt động đúng, không chứng minh matched controls tương lai
sẽ luôn chạy hợp lệ. Bias-matching/group scheduling trong full7arm pilot chưa
được triển khai ở P0; bias tự nhiên đã giữ rule cũ và có parity ở endpoint.

## RAM, timing và kiểm thử

Runner P0 hết1,66giây, một numerical thread, Windows cap2.000.000.000byte.
Sampled process-tree peakRSS95.494.144byte và commit92.352.512byte, gồm worker
restart. Đây là measurement nhỏ, chưa chứng minh RAM của fullpilot hoặc model
học liên tục lâu dài. Monitor có thể double-count sharedpages; cap per-process
và process-tree observation được ghi riêng, không dùng swap để hạ số đo.

Importance trong fixture16×48 là3.072byte. Với maxoutput266×48, riêng importance
float32 là51.072byte; 100utilitycalls hết0,298giây (~2,98ms/call). Ba beta4
trajectory12segment mất khoảng0,073–0,076giây/seed. Đây không là tốc độ chat,
không áp vào mục tiêu5token/s và chưa đủ dự toán168job với probe/guard đầy đủ.

**276/276tests qua**, gồm15test P0 mới và regression/native CPU parity cũ.
[Receipt phần mềm](../../artifacts/readout-usefulness-p0-software-validation.json).
Kiểm tra độc lập phiên thật xác nhận weights/topology/vocabulary/encoder,
updatecounter và64neuron đã cấp giữ nguyên; không llama-server còn chạy.
[Receipt bảo toàn main](../../artifacts/readout-usefulness-p0-live-preservation.json).

## Lịch sử kiểm tra và vấn đề còn mở

Run đầu `c9a25c` giữ nguyên rawresult/source. Field cold của nó chỉ là instance
mới cùng process; amendment P0-A thêm worker riêng và saturated-transition case.
[Coverage note](../../runs/20261110T134710Z-readout-usefulness-p0-v1-c9a25c/coverage-note.json)
chỉ rõ dùng run `ca2357` cho claim restart. Trong unit-test development, một
fixture sai chưa ép được prediction sai do feature contribution lớn; sửa bias
test trong bound trước formalrun, không sửa learning rule/gate để qua test.

Chưa được chứng minh: usefulness có phân biệt đúng synapse cần giữ, giảm quên
mà vẫn học mới, hidden học representation, khái quát ngôn ngữ/ảnh và chuyển
trải nghiệm thành kỹ năng. P2 và representation ngôn ngữ/ảnh vẫn chưa đạt; P1 order/XOR hẹp ở báo cáo20
đã có bằng chứng riêng, không bị P0 này thay đổi. Fullpilot còn cần khóa
inputmapping/readoutmask của stream20context, conditionbuilders, equal-time
budget và controls theo protocol; P0 G1 fixture không thay các bước đó.

**Dừng ở phạm vi P0 đã duyệt. Pilot P2 chưa chạy và chưa được duyệt.** Không dùng
P0 pass làm lý do mở trainingchat/vision, scale hoặc tích hợp U vào Hermes.

[Receipt bàn giao P0](../../artifacts/readout-usefulness-p0-delivery.json).
