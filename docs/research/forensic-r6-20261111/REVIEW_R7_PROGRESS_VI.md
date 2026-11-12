# Review cập nhật DeepSeek / Gemini R7 — nên giữ gì để đi nhanh hơn?

## Kết luận

**Có tiến bộ đáng giữ, nhưng chưa nhánh nào đủ điều kiện thay learner main.**
Ưu tiên hiện tại: xác minh **EW-HE của Gemini trên strict_binding** bằng cùng
refit probe và ablation công bằng; dùng các công cụ replay/R-matrix của DeepSeek
làm đối chứng. Không gom hai benchmark khác nhau thành một bảng winner.

Các commit được chốt:

- DeepSeek R7: `aacde3fa0983b9cebc585e19980022a52a45ccad`, đến round37.
- Gemini R7: `64441fb9be162864cf2e1252638d101a2031f2e4`.
- Gemini R6 erratum: `df683923722409ab00ee40213e7cda29b2003900`.

Source/main/live của các agent không bị sửa. Scope review được ghi trước tại
R7_PROGRESS_REVIEW_SCOPE.md (commit4312042). Không mở confirmation/P2/thí nghiệm
mới; chỉ source/raw arithmetic, P0 và một cell lịch sử được tái lập.

## 1. Gemini: đã có learner cập nhật hidden thực sự

`EventWindowedHomeostaticEligibility` cập nhật `w_in/w_rec` bằng own-neuron
forward eligibility và one-step readout error, kèm moving feature statistics và
synaptic trace/energy normalization. Không còn chỉ là frozen ALIF + query head
như R6C. Source không có backward thời gian/whole-network trong candidate.
Tín hiệu `W_out.T @ err` vẫn là ngoại lệ local một bước, không thuần biological-local.

**Đã kiểm:**5/5test P0 của class thật qua. Tái lập đúng một cell lịch sử
strict_binding/seed140011 với toàn bộ conditions và budget gốc: mọi primary
direct/probe ID/OOD score khớp artifact. Probe của hidden đã học82,42%, frozen
64,06% ở cell đó. Đây là kiểm một kết quả positive có chủ đích, không phải fresh
overall confirmation hoặc tái lập cả15trial.

Tính lại15raw P1rows (3seed mỗi task):

| Task | Frozen probe ID | EW-HE probe ID | Gain | Bootstrap95% gain, seed-level |
|---|---:|---:|---:|---:|
| strict_binding |62,30%|83,14%|+20,83đ|[+16,60;+27,54]đ|
| context_selective_rule |58,01%|68,68%|+10,68đ|[+4,30;+16,99]đ|
| distractor_binding |53,84%|52,34%|−1,50đ|[−5,08;+2,54]đ|
| xor |83,27%|86,46%|+3,19đ|[−2,73;+13,67]đ|
| order |93,68%|97,85%|+4,17đ|[−0,78;+10,35]đ|

**Điểm tốt nhất:** hidden representation trên strict_binding có tín hiệu thật ở
DEV, gồm common-refit gain. Đây là bài hai giá trị complementary/hẹp; chưa được
đánh đồng với hai independent facts, ngôn ngữ hay kỹ năng đa năng.

**Phải sửa trước confirmation:**

- `scripts/r7_dev_screen.py:333` đặt cứng câu “passes ... across all5tasks”. Raw
  distractor/XOR/order không hỗ trợ câu đó. Phải tính gate từ dữ liệu, giữ thất
  bại lịch sử và đăng ký scope tiếp theo; không chọn một task thắng rồi gọi cả
  vòng đãPASS.
- `r35_baseline` là wrapper control, không đúng toàn pipeline35 cũ. Source cho
  EW-HE hiddenLR.008/clip.01, baseline.002/clip.002; fast/slow trace, RMS và
  statistics còn khác nhau. Chưa thể quy gain riêng cho event-window decay.
- Geometry rank/Fisher và trained-head causal revert không thay common refit.
  Giữ cả hai loại bằng chứng và đo transfer trên heldout mới trước GO rộng.

## 2. Gemini retention: có ích nhưng chưa đạt, có trả giá acquisition

Đã tính lại18raw cells,3seed×6orders:

| Arm | Acquisition | Final | Mean forgetting | Cell qua joint gate |
|---|---:|---:|---:|---:|
| Không consolidate/replay |90,05%|68,35%|21,70đ|0/18|
| EW-HE+CA-MSC+replay |84,98%|84,49%|2,80đ|**8/18**|

Mean giảm quên18,90đ, final tăng16,14đ, **acquisition giảm5,08đ**. Đây là trade-off,
không phải giải xong continual learning. Có cell meanF9,11đ/worst18,75đ, và cell
acquisition61,20% thấp hơn65%. Không cho mean2,80đ che các cell không đạt.

Những khác biệt source phải đối chiếu:

- `wrap_skill_pulse` thêm3kênh one-hot skillID, pulse.8 ở tick11; có task identity.
- Calibration đọc trước48sample của cả3skill. Các mốc consolidate biết ranh giới.
- Arm replay gọi thêm learn_episode sau mỗi experience; không khớp update budget
  với no-replay. Không tách được contribution của CA-MSC và thêm compute/replay.
- CI cũ bootstrap18cells như độc lập; chỉ có3initialization seeds. Codex tính lại
  primary CI trên seed means qua6orders. Hiệu ứng vẫn có, nhưng bằng chứng DEV
  ít seed và chưa đáp ứng setting không task marker/biên task rõ.

**Nên tận dụng:** activity-gated momentum/importance và code resume/P0 làm
prototype nghiên cứu, sau khi có replay-only, consolidation-only và matched-update
controls. Chưa đưa vào idle learner sản phẩm.

## 3. DeepSeek: nhiều chẩn đoán hữu ích, tiến bộ mới chủ yếu ở readout

Round31 dùng frozen ALIF spike-sum + head32tanh units + replay/tail. `MLP.step`
trong scriptR32:192–200 tính gradient qua hai tầng readout bằng NumPy. Không có
BPTT qua recurrent SNN, nhưng kết quả đó **không phải hidden ALIF synapses học**
và không tự chứng minh rule chính local-SNN. Giữ như control/readout diagnostic.

Tính lại raw:

- T=3:87/90cells qua acquisition/final/meanF/worstF;3orders đạt15/15seeds.
  “90/90mọi cell qua mọi gate” trong report không đúng raw. Mức meanF.42đ là
  một order tốt, không phải worst order của mọi stream.
- T=6corrected:0/36cells qua joint gate. Order tốt nhất meanF3,84đ, mean worstF
  9,12đ; chỉ là gần hơn, chưa qua3đ/5đ.
- Round32–35 đã bị rút sau phát hiện ba task cùng input/khác nhãn. Codex tái
  hiện generator cũ trên128row lịch sử: cả tensor trùng,50/93/43row có nhãn
  đối nghịch giữa các cặp task. Specimen corrected round37 có0exact sample-label
  conflict; xnor còn dùng marker kênh2 riêng. Đây là sửa testbed đáng giữ, chưa
  là bằng chứng mọi distribution hoặc capacity đều đã kiểm xong.
- Không dùng signed forgetting để gains của task này che loss task khác; không
  gọi15seeds được dùng lặp từr24tớir31 là fresh lần nữa.

**Nên tận dụng:** full R-matrix, acquisition/forgetting đồng thời, kiểm mọi thứ
tự task, replay-tail như control, và row-level conflict validator. Chưa copy
MLP-trained solution thành “hidden local synapses đã học”.

## 4. Gemini R6 đã đính chính gì?

Commitdf68392 chỉ sửa `argument-audit.md`: rút các con số/arm không có trong
artifact và nhận CKCER là handcrafted query×state readout trên hidden frozen.
Đây là tiến bộ về trung thực/provenance; không phải accuracy mới. Các finding
ở report/math/protocol khác cần đồng bộ, không chỉ sửa một bản argument.

## 5. Cách rút ngắn đường đi — chia việc từ bằng chứng hiện có

1. **Gemini:** giữ candidate64441fb ở research; sửa gate tự tính, tập trung giả
   thuyết mới về strict-binding credit có positive signal. Thêm matched LR/norm
   và replay/computation controls trước khi xin confirmation. Không mở rộng model.
2. **DeepSeek:** dừng suy GO từ best order/MLP head; chuẩn hóa R-matrix, retention
   denominator và budget. Giao phần công cụ benchmark/replay-control, tránh thêm
   sweep chưa gắn với lỗi nhân quả mới. Giữ các retraction rõ ràng.
3. **Luồng chung:** cùng generator, input cues, sample/label oracle, forgetting
   definition, seed clusters và budget. Sau đó mới xét hidden gain/retention.
   Hai nhánh hiện khác tasks, ALIF config, input dimensions và head, nên không
   dùng0,42đvs2,80đ để tuyên bố bên nào thắng.

**Quyết định:** ưu tiên Gemini EW-HE strict_binding cho nghiên cứu xác minh;
ưu tiên công cụ kiểm định/replay của DeepSeek; **chưa approve learner nào vào
main**, chưa tự chạy confirmation/P2. Main/live/architecture giữ nguyên.

## Chứng cứ

- `R7_PROGRESS_RECALCULATION.json`: per-task gains, seed-cluster intervals,
  joint-gate counts và command/resource của kiểm tra thực.
- `R7_BENCHMARK_VALIDITY.json`: counterexample generator cũ và specimen sửa.
- Run `runs/20261111T155855Z-r7-progress-review-v1-a52c77`:
  P0log/JUnit, một historical positive-cell replay và raw/source snapshot.
- Không khẳng định đã chạy lại37round DeepSeek hoặc toàn bộ Gemini R7.
