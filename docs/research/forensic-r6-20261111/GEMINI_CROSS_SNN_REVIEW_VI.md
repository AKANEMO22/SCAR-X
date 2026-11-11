# Review độc lập khảo sát Cross-SNN của Gemini

Ngày người dùng: **12/11/2026**. Source được chốt ở Gemini
`631c89041ec7ed3ce5d6b08373b68e9dc2e4a122`; so với `bad0582`, commit mới chỉ thêm
`cross-snn-mechanism-survey.md`, không thêm code hoặc raw experiment. Đã đọc toàn
bộ mission người dùng gửi (SHA-256
`42d34d9c8dd2e6b58275503a1b247f312186920ca7f56e514b6703f36cc4d490`).

**Kết luận: có hướng hữu ích; `REQUEST_CHANGES` cho khảo sát hiện tại.** Giữ ba
ý tưởng để nghiên cứu, chưa chấp nhận các lời khẳng định “fixes”, điểm efficacy
5/5, mức rủi ro LOW đã được chứng minh hoặc đưa learner vào main. Không mở train
DEV, confirmation, không sửa core, Hermes hoặc phiên live trong review này.

## 1. Những phần có thể tận dụng

| Cơ chế | Giá trị cụ thể với SCAR-X | Quyết định |
|---|---|---|
| Eligibility nhiều hằng số thời gian | Có thể kiểm tra tradeoff giữa credit ngắn của binding và credit xa của XOR; không cần tăng neuron | Ưu tiên P1: thử yếu tố multiscale riêng sau khi có code/raw đúng; chưa coi đã thắng |
| Chọn lọc thời điểm/tín hiệu cập nhật | Có thể giảm update do distractor và phân biệt học có ích với nhiễu | Giữ giả thuyết; cần định nghĩa gate không dùng lịch sự kiện từ generator và tách khỏi multiscale |
| Synapse fast/slow Benna–Fusi | Phù hợp mục tiêu nhiều thang thời gian, không cần task ID về mặt phương trình | Ưu tiên P2 sau bằng chứng P1; phải vượt baseline EMA+anchor, replay và bảo vệ readout |
| Homeostasis chống neuron chết/bão hòa | Là một chẩn đoán đúng hướng trước khi tăng model | Trước hết đo firing, surrogate slope, rank và clipping; gain mới thay forward nên cần duyệt riêng |

Giá trị tốt nhất là **chọn câu hỏi nhân quả nhỏ và ablation**, không phải nhập
SpikingLM/SpikeGPT hay foundation model khác vào SCAR-X. Các cơ chế trên chưa
chứng minh chuyển được trải nghiệm thành kỹ năng ngôn ngữ hoặc chống quên lâu dài.

## 2. Finding P1 — +22,33 điểm distractor dùng baseline khác lượt

Survey dòng90 ghép `52.34% → 74.67%`. `52.34%` là probe EW-HE ở campaign DEV cũ;
`74.67%` là trung bình arm giảm eligibility trong `investigation-suite.json`.
Runner chẩn đoán có chính baseline không suppression, cùng seed/task/budget.
Tính lại đúng các cặp từ raw đó:

| DEV seed | Standard probe | Suppressed probe | Chênh lệch |
|---|---:|---:|---:|
| 140011 | 68,16% | 73,63% | +5,47 điểm |
| 140017 | 63,09% | 71,68% | +8,59 điểm |
| 140023 | 83,40% | 78,71% | −4,69 điểm |
| **Trung bình** | **71,55%** | **74,67%** | **+3,125 điểm** |

Bootstrap đúng ba seed cluster cho CI95% thăm dò **[−4,69; +8,59] điểm**. Direct
accuracy cùng lượt tăng 66,93% → 71,48%, tức +4,56 điểm; không đồng nhất với
representation gain. Không coi đây là tái lập training: Codex chỉ tính lại JSON
lịch sử, không chạy learner. Chưa có frozen/zero/shuffled matched controls cho
biến thể suppression mới này. Vì vậy chưa chứng minh nó đạt gate hidden gain.

**Yêu cầu sửa:** giữ cả số lịch sử, bỏ phép trừ xuyên campaign để kết luận nhân
quả; báo paired per-seed, uncertainty và khác biệt config/source của hai lượt.

## 3. Finding P1 — ba gate khác nhau đang được gọi cùng một cơ chế

`scripts/r7_investigation_suite.py:427–430` thực thi:

```python
if suppress_delay_elig and (3 <= t <= 8):
    z_e = z_e * 0.10
e_f = 0.25 * e_f + 0.75 * z_e
```

Đó là **damping theo tick biết trước**, không phải cổng sensory không task ID.
Generator `r7_hidden_plasticity.py:432–451` đặt lần ghi key/value thứ hai tại
tick4/5, nằm ngay trong đoạn3–8. Gate do đó giảm cả eligibility của sự kiện hữu
ích, không chỉ distractor. OOD đổi sự kiện thứ hai sang tick7 cũng chồng gate.
Không thể dùng việc điểm tăng để kết luận riêng “đã loại nhiễu”.

Survey lại đề xuất `G=1[||x||_1 > epsilon_event]`. Epsilon chưa được khóa; background
Gaussian và distractor có biên độ khác zero. Gate theo độ lớn không tự nhận ra
sự kiện nào liên quan đến câu hỏi. Một fixture với noise L1=0,16 hoặc distractor
L1=1,2 đều mở gate nếu epsilon=1e-6. Đây là phản ví dụ cho diễn giải tự loại nhiễu,
không phải chứng minh mọi threshold đều thất bại.

Ngoài ra, công thức `e_new=gamma*e_old+(1-gamma)*G*z_e` vẫn làm trace suy giảm khi
G=0: với e_old=1 và gamma=0,25 thì e_new=0,25, **không freeze thành1**. Cần chốt
đang gate đóng góp mới, gate decay hay gate cả hai. Damping10%, freeze hoàn toàn
và amplitude gate chưa là cùng thuật toán.

Cuối cùng, gate eligibility không ngăn distractor làm thay đổi voltage/spikes
trong forward. Nếu vấn đề chính là working state đã mất thông tin, sửa trace
có thể không đủ. Phải đo state retention và learned representation riêng.

## 4. Finding P1 — multiscale chưa có provenance cho các điểm thắng

Claim binding85,29% / XOR96,61% tiếp tục xuất hiện nhưng Git tree đã khóa không
có `test_multiscale.py` hoặc raw arm `[.25,.92]`. Candidate hiện có CG-MHE dùng
`gamma_e_vec=linspace(.15,.55,48)` kèm coincidence gate; nó khác MTSG được đề xuất.
`investigation-suite.json` cũng không có arm multiscale tương ứng.

**Trạng thái: `REPORTED_NOT_TRACEABLE`.** Chưa kết luận thí nghiệm không tồn tại;
cần cung cấp code/config/seed/update counters/checkpoint hoặc predictions/raw
metrics và hash trong commit. Một file survey mới không khép khoảng trống này.
Các nhóm “neuron nhanh/chậm đã chuyên môn hóa” cũng cần phép đo; đặt decay theo
ID neuron không tự chứng minh chuyên môn hóa.

## 5. Finding P1 — BF-Cascade là giả thuyết rút gọn, chưa sửa được forgetting

Benna–Fusi gốc dùng tương tác hai chiều giữa nhiều quá trình nhanh/chậm để cải
thiện dung lượng ký ức. Card rút xuống hai state/weight: fast nhận update, slow
đuổi theo fast và kéo fast trở lại. Đây là lựa chọn nghiên cứu hợp lý nhưng
**không kế thừa tự động bảo đảm capacity/retention của mô hình gốc**.
[Paper gốc](https://www.nature.com/articles/nn.4401).

SCAR-X đã có stable-weight EMA và anchoring trong các nhánh lịch sử. Vì vậy phải
so với **EMA+anchor cùng hai hệ số/thời điểm cập nhật** để biết cascade có khác
gì thực chất, thay vì đổi tên cơ chế đã thử. Slow state cũng sẽ bị kéo theo nếu
nhiệm vụ mới đẩy lâu; hai state hữu hạn không cho phép nhớ vô hạn. Card chỉ bảo
vệ `w_in/w_rec`, trong khi readout overwrite là lỗi đã quan sát; chưa có cơ sở
để hứa giải quyết toàn bộ P2 mà không đụng readout.

Hai phép gán ở dòng128–129 còn mơ hồ: update slow dùng gap cũ hay fast vừa đổi?
Với u1=1,u2=0,k1=.005,k2=.001, update đồng thời cho (0,995;0,001), update tuần tự
cho (0,995;0,000995). Không phải một lỗi lớn được chứng minh, nhưng cần thống
nhất phương trình, source và kiểm tra ổn định trước bất kỳ train nào.

## 6. Finding P1 — DAS-Homeo không phải DAS nguyên bản và làm đổi forward

SpikingLM chính thức là **Yu Liang và cộng sự, ICML2026**, mô hình BERT-style;
DAS là các hệ số Q/K/V trainable, có thể gộp vào projection lúc inference. Quy tắc
gain dựa trên firing-rate mục tiêu0,05 trong card là **bản chuyển ý tưởng mới**,
không phải thuật toán local-online mà source SpikingLM cung cấp. Cần đặt tên và
giả định đúng. [Nguồn tác giả](https://github.com/hamings1/SpikingLM/blob/6d323f2a0ed2758850d2b27c3f8393660b0b4d81/README.md),
[proceedings](https://proceedings.mlr.press/v306/liang26ae.html).

Nhân input/recurrent current với g_j thay đổi dòng vào và spike trajectory. Giữ
ALIF và topology chưa đồng nghĩa **giữ cùng forward baseline**. Hướng này cần
được duyệt là thay đổi forward, có frozen-gain/matched-rate controls, giới hạn
gain và state persistence. Chỉ tăng participation ratio không đủ chứng minh
accuracy hoặc năng lực học representation tốt hơn.

## 7. Finding P2 — bibliographic/license và RAM chưa đủ để tái sử dụng

- e-prop revision `efd02e6` có ba điều khoản phân phối kiểu BSD, không phải
  Apache2 như survey. [LICENSE đã pin](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/LICENSE).
- DECOLLE README/LICENSE từng có conflict đã lưu trong six-source audit; chưa
  có bằng chứng cho phép gọi nó “GPL3/Apache2” rồi sao chép tùy ý.
- SpikingLM tên tác giả/năm bị ghi sai, license Apache2 chưa được xác minh trong
  source đã kiểm. Không suy ra license code từ giấy phép paper.
- SpikingBrain2.0 README trỏ **arXiv2604.22575, 2026** và repository ghi MIT;
  survey ghi Apache2 và arXiv2502.14447. ID sau là một paper vật lý lượng tử.
  [Repository gốc](https://github.com/BICLab/SpikingBrain2.0),
  [ID bị dẫn nhầm](https://arxiv.org/abs/2502.14447).
- Survey không chốt exact revisions/source anchors cho chín nguồn. Review này
  chỉ xác minh các nguồn chọn lọc; không giả vờ chứng nhận cả chín implementation.
- `gamma_vec`48 float32 là192byte, không phải “0 additional array bytes”. Có thể
  bằng0 **gia tăng** nếu thật sự tái sử dụng buffer cũ, nhưng cần nói rõ baseline.
- Một slow state cho H48, D8, fan8 là3.072byte; public G1 D222 là44.160byte, bản
  codec V55 D253 là50.112byte. Đây chỉ là một array, chưa tính trace/readout/RNG/
  replay/snapshot/runtime; các con số vài KB không chứng minh peak process2GB.

## 8. Protocol đề xuất: sửa bằng chứng trước, rồi một thay đổi mỗi lượt

1. **Khép source/raw:** sửa các finding trên; giữ lịch sử, commit đủ multiscale
   evidence nếu có. Không tự chạy lại một campaign để lấp bảng số còn thiếu.
2. **P1 ưu tiên multiscale riêng:** cùng forward H48/fan8, không sensory gate,
   không đổi normalization/optimizer; homogeneous EW-HE, multiscale, frozen,
   zero/shuffled signal và R35 cùng source/budget. Strict binding, XOR và
   distractor có common-refit/direct/OOD; ghi actual applied update/norm/compute.
3. **Gate riêng sau đó:** gate chỉ dựa vào tín hiệu runtime thực sự có, threshold
   khóa từ train/dev trước đánh giá. OOD phải dịch timing và distractor độc lập;
   phase/tick oracle chỉ là diagnostic control có nhãn rõ. Không ghép multiscale
   và gate rồi quy hết hiệu quả cho một thành phần.
4. **P2 sau P1 và duyệt riêng:** cascade vs EMA+anchor, replay-only và combination
   với RAM/counters/compute tương đương; không task ID/future calibration, có
   stream không ranh giới công khai. Acquisition/final≥65%, meanF≤3đ/worstF≤5đ
   trên mọi seed, không chỉ trung bình. Theo dõi hidden và readout riêng.
5. **DAS-Homeo để sau chẩn đoán:** đo dead/saturated neurons trước; không âm thầm
   thay ALIF current để làm đẹp điểm.

Ba seed140011/17/23 đã được xem nhiều lần: chỉ là DEV thăm dò. Không chọn thêm
hyperparameters trên các kết quả đó rồi gọi là independent confirmation. Giữ
mọi ngưỡng cũ; không mở fresh confirmation nếu chưa được duyệt.

## 9. Bàn giao và trạng thái production

Runner `scripts/review_cross_snn_transfer.py`; helpers
`src/scarx/evaluation/cross_snn_review.py`; protocol
`experiments/cross-snn-transfer-review-v1.md`; receipt
`artifacts/cross-snn-transfer-review-v1.json`. Run chứa input Git blobs, SHA256,
paired metrics, counterexamples số học và JUnit. Các test chỉ bảo vệ phép tính
audit, không được dùng làm bằng chứng model thông minh hơn. Raw/source assertions
mới của Gemini chưa được tái lập training trong review này.

**Approve vào luồng research:** cơ chế nhiều thang thời gian, chẩn đoán firing,
ablation một yếu tố và audit provenance. **Chưa approve:** claim MTSG đã giải
XOR/distractor, cascade đã giải continual learning, nhập source chưa rõ license,
thay forward/learner hoặc merge main. G1 và phạm vi codec P0 giữ nguyên;
production vẫn `NOT_PRODUCTION_READY`.
