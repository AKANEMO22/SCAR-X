# P2 — Readout usefulness / bản thiết kế v2

2026-11-10. **CHỜ DUYỆT CODE/THÍ NGHIỆM HỌC.** Chỉ cập nhật thiết kế và chuẩn
bị dữ liệu được phép ở lượt này. Không thay draft v1, kết quả lịch sử hoặc gate.
Master: `docs/RESEARCH_PLAN.md` §3.3, §7.2/P1/P2 và §11. Review căn cứ: báo cáo40.

## Câu hỏi và giới hạn

Readout có thể ghi đè một cặp đã biết. Chưa biết bảo vệ theo usefulness có giữ
được kiến thức mà vẫn học mới hay không. Đây là P2 trong master, không là giải
pháp thay representation/credit assignment. Không mở lại family35–37 bằng
tuning, không tăng neuron/model, không Transformer/BPTT/autograd ở candidate.

Phase2.1 `PlasticCore._update` đã có magnitude importance cộng dồn, chia LR và
stable anchor. Draft A mới dùng EMA magnitude trên G1, bỏ anchor: khác lịch
importance, không phải một nguyên lý chưa từng thử. Giữ cả hai làm đối chứng.

## Đề xuất U — usefulness ở chính readout

Chỉ khi nhãn train có nguồn rõ ràng và readout dự đoán đúng target y, đánh giá
mức hỗ trợ của từng synapse bằng loss tại đầu ra khi xóa tạm đóng góp synapse
đó. Không xóa weight thật, không truy cập test, không backward hidden.

Với logit z, softmax p, feature f, synapse W[k,j] và a=W[k,j]f[j]:

    loss_without(k,j) - loss_with
      = log((1 - p[k]) + p[k] exp(-a)) + 1[k=y] a

Lấy phần dương, chặn về [0,1], rồi cập nhật:

    omega <- max(0.995 * omega, positive_benefit)
    g = 1 / (1 + 4 * omega)

Không có nhãn được xác minh thì không học và không tạo importance mới. Khi
prediction sai, chỉ decay omega; không bảo vệ nhãn sai bằng độ lớn error cao.
Đây là lợi ích counterfactual **tại output train hiện tại**, chưa là importance
nhân quả dài hạn hoặc chứng minh một kỹ năng. Softmax/error output vốn là tín
hiệu cho phép; đường hidden không đổi. Bias giữ rule cũ để tách giả thuyết Wout.

P0 phải đối chiếu công thức với forward xóa từng synapse trên bản sao, dùng
log-space float64 để tránh overflow/cancellation, gồm p gần0/1 và weightclip.
Beta4/decay0.995 khóa trước pilot; không sweep/tune sau kết quả. Với omega trong
[0,1], g trong[0.2,1]; bỏ STOP g<0.1 bất khả thi của draft A, thay bằng đo
acquisition/plasticity thực. Đây là thay định nghĩa cho run tương lai, không
sửa stop rule hoặc số đo run lịch sử.

## Baseline và control

| ID | Điều kiện | Câu hỏi |
|---|---|---|
| B | Full local G1 nguyên bản | Reference đúng scope sản phẩm |
| M | EMA magnitude draft A: rho0.95, beta4, không anchor | Usefulness có hơn magnitude? |
| U | Counterfactual usefulness ở trên | Giả thuyết chính |
| R | Shuffle omega của U, cùng multiset g, RNG riêng | Chọn đúng synapse có ích? |
| G | Một scalar thu nhỏ delta readout | Chỉ giảm tốc học có đủ? |
| H | Readout-only, hidden frozen | Hidden có đóng góp? |
| F | Tất cả weights frozen | Learning có tạo ích lợi? |

SI-readout/Phase2.1 meta không được gọi là G1 baseline tương đương khi topology,
forward hoặc update khác. P0 phải dựng bảng phương trình và parity; nếu không
chuyển tương đương được, báo control không tương đương, không trộn vào contrast
chính. Chưa cho phép tự thêm một signal/rule hidden để làm control thắng.

## Hai phép đo khác nhau, không lẫn causal với hiệu quả tự nhiên

1. **Chẩn đoán cùng applied-delta norm:** tại từng example, mỗi learner lập
   delta đã clip trong state của nó. Lấy radius chung bằng 0.5 lần norm nhỏ
   nhất của các proposal không-frozen; chuẩn hóa các hướng U/R/G/B/M/H về cùng
   radius. Scalar G đi theo hướng raw update. Scale chỉ giảm nên vẫn trong
   weight/update box. Bias dùng cùng ngân sách riêng; không bảo vệ bias.
   Mọi arm được tính cùng các proposal phụ để cân chi phí diagnostic. Log norm
   sau áp dụng, sai khác tương đối>1e-5 thì contrast matched không hợp lệ. Radius0
   được công bố, không tự bỏ sample. Đây là controlled comparison, chưa là
   autonomous learner có thể deploy. Mỗi arm vẫn có hidden state/weights riêng.
2. **Rule tự nhiên:** bỏ norm matching, U/M/R dùng g tự tính; B/H/G dùng baseline
   hoặc scalar0.25 đã khóa. Đo equal-update và equal-time riêng. Khác biệt ở lượt
   này có thể do effective LR; chỉ lượt1 mới hỗ trợ claim chọn lọc theo hướng.

Không chọn arm thắng ở final test. Nếu matched control không hợp lệ hoặc U chỉ
thắng do lượng update thấp, kết luận chưa chứng minh selective usefulness.

## Stages, dữ liệu, seed và job table dự kiến

**P0 trước hết, chưa code ở lượt chuẩn bị này:** local-signal audit; math finite
difference output-only; beta0 parity; full save/load/rollback gồm omega, RNG,
controller và replay; actual delta budget/clip; đo throughput để chốt số bước
equal-time trước formal run. Chỉ train fixtures của P0 được đọc. P0 fail thì
dừng, không tự sửa công thức từ kết quả chất lượng.

**Pilot P2:** tái dùng generator context/rule quan sát được của master, 20 giai
đoạn; task boundary/ID evaluator không được vào learner. Fixed topology48ALIF,
fan-in8; không live checkpoint chọn vì điểm cao. Seed mới:
41201,41203,41207,41209,41213,41217,41219,41221. Mỗi seed là mạng khởi tạo mới,
cùng initialization/data-order giữa các arm. Test tổ hợp/noise độc lập với train.
Dataset/schema, số train/test, checkpointtimes và equal-time seconds phải được
đăng ký bằng amendment sau P0 timing và **trước bất kỳ quality score**.

| Bank | Arm | Seed | Job |
|---|---:|---:|---:|
| matched-direction / equal-update | 7 | 8 | 56 |
| natural / equal-update | 7 | 8 | 56 |
| natural / equal-time | 7 | 8 | 56 |
| Tổng tối đa sau P0 và amendment | | | 168 |

Không mở toàn bộ168job khi P0/resource prerequisite chưa qua. Một thread,
2.000.000.000 byte cho worker/tree và cả checkpoint/rollback. Pilot tối đa3giờ
wall; dự toán vượt thì trình giảm ngân sách **trước** khi mở chất lượng, không
chạy thêm để ép gate. Không tính 3checkpoint tốt là 3seed.

Nhánh chat/vision: bộ wording AGY chuẩn bị riêng chỉ là **development data**, không
final unseen test và không tự thành training labels. Chưa chạy learner U trên
word/visual family đã STOP. Muốn bridge mới phải có protocol/duyệt riêng, sau
khi kết quả P2 cho phép. Câu trả lời Qwen chỉ đối chiếu, không là truth oracle.

## Gate giữ nguyên và phân tích

- Mean forgetting <=3đ, worst<=5đ; acquisition/final>=65% theo master cho từng
  seed. Không thắng nhờ freezing hoặc chỉ giữ đúng một cặp greeting.
- Cải thiện tương đối: cận dưới CI95% giảm forgetting>=5đ so B, acquisition
  không kém B quá3đ; cần cả equal-update/equal-time.
- Primary selective contrast U-R ở matched bank: paired seed-level CI95% lower
  >0 cho gain retention, acquisition noninferiority3đ. U-M/G phụ, Holm điều
  chỉnh các contrast đã khóa. Report mỗi seed, paired exact sign-permutation và
  bootstrap seed-cluster; 8seed vẫn là nghiên cứu nhỏ, không general AI claim.
- Hidden: freeze trained hidden/refit cùng budget so initial và readout-only,
  state lesion, hidden/head transplants đúng array. Tăng weight không là pass.
- Hai vòng hợp lệ tối đa; pilotfail thì STOP v2, không tự beta/LR/data tuning.
  P0 invalid không tính là bằng chứng U vô ích; phải ghi rõ invalid.

**GO** chỉ khi joint gates/controls đều qua, mở nghiên cứu kế tiếp chứ không
deploy. **PIVOT** nếu giữ cũ nhưng acquisition thấp hoặc không có hidden gain.
**NO-GO cho U** sau hai vòng hợp lệ không lợi thế. Không NO-GO toàn bộ SNN từ
một phép đo output; không GO normalchat/vision từ bộ synthetic.

## Phạm vi xin duyệt

Chỉ xin duyệt code P0 trên prototype riêng và pilot P2 đúng stages/gates này;
amendment tài nguyên bắt buộc trước formal score. Không sửa main, không dạy hoặc
restore phiên Hermes, không scale hoặc huấn luyện ngôn ngữ lớn. Mọi khác biệt
signal/architecture mới ngoài các phương trình trên phải trình lại.
