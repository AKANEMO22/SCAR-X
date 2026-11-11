# Cập nhật cuối lượt review — DeepSeek853d408 / Gemini bad0582

Đây là phần bổ sung cho REVIEW_R7_PROGRESS_VI.md, vì hai nhánh push thêm commit
ngay khi Codex đang kiểm tra. Kết quả tái lập một historical cell/P0 trong report
trước vẫn giữ nguyên; các diff mới dưới đây được source/raw checked và gate-tests,
không được coi là đã tái lập mọi thí nghiệm mới.

## Gemini: các sửa đổi thực sự đã có

Commit **`bad058238083ec875aaef3a271e3f3c68680e3c1`**:

- Đã rút claim toàn bộ5taskPASS trong report và đổi tổng verdict thànhFAIL.
- Có `R7GateEvaluator`; Codex chạy2test gốc qua và gọi evaluator trên raw DEV:
  **1/5taskP1pass**, P2FAIL, đúng với review trước.
- Có7arm ablation, can thiệp riêng input/recurrent weights, chẩn đoán distractor.
- Có2×2consolidation/replay trên8kênh, không thêm3kênh skillID và không dùng
  future-task calibration như benchmark18cell trước.
- Đã tách vai trò: raw factorial mean forgetting replay-only **2,43đ**, no replay
  **17,88đ**, consolidation-only **18,40đ**. Acquisition replay-only **71,05%**,
  so với no-replay **84,29%**; worst forgetting replay-only **6,77đ**, chưa đạt5đ.

Đây là tiến bộ hữu ích về đo lường và lựa chọn hướng. Không cần yêu cầu họ làm
lại các phần đã sửa này. Core EW-HE của64441fb không đổi nên positive hidden result
trên strict_binding mà Codex đã tái lập vẫn là điều đáng giữ nhất.

## Những điểm mới chưa được chốt

### 1. “Matched512updates” vẫn chưa khớp ở stage đầu

`scripts/r7_investigation_suite.py:638–653` chọn256fresh khi replay bật, chỉ thêm
replay nếu buffer.size>0, rồi mới nạp buffer **cuối stage**. Stage0buffer rỗng:

| Arm | Stage0 | Stage1 | Stage2 | Tổng |
|---|---:|---:|---:|---:|
| Không replay |512|512|512|1536|
| Có replay |256|512|512|1280|

Không thể gọi đây là cùng512updates ở mọi stage. Cần sửa schedule/ghi actual
update counters, đăng ký lại phép so sánh nếu chạy thêm; giữ raw cũ. Có thể một
phần acquisition giảm đến từ first-stage budget. Đây chưa là phép đo định lượng
nguyên nhân, nên không quy toàn bộ giảm acquisition cho lỗi này.

### 2. Multiscale85,29% /96,61% chưa có chuỗi chứng cứ trong commit

Report/peer-review packet nói dùng heterogeneous decay[.25,.92], trỏ
`test_multiscale.py`, rồi claim bindingprobe85,29% vàXOR96,61%. Codex không tìm
thấy file đó hoặc raw per-seed của thí nghiệm này trong Git treebad0582; worktree
Gemini R7 đang clean cũng không có file mang tên đó. `investigation-suite.json`
không chứa multiscale arm, source candidate vẫn có vectorCG-MHE[.15,.55] cũ.

**Trạng thái: REPORTED_NOT_TRACEABLE**, không tự gọi là thành công hay kết luận
không hề chạy. Nếu tồn tại ở scratch/Temp, cần đưa code, config, seed, actual
update counts, raw metrics và hashes vào một commit tái lập được. Chưa dùng các
con số đó làm lý do tăng quy mô hoặc đưa learner vào main.

### 3. Factorial chưa đủ chứng minh “task-agnostic lifelong solved”

Đã bỏ input task-ID là tốt. Vòng chạy vẫn chia stage rõ và gọi consolidate/nạp
buffer tại ranh giới biết trước. Reverse-order lại yếu; số task chỉ3 và các mẫu
thử còn synthetic. Calibrate/normalization, acquisition/final/worstforgetting
và stream không task-boundary vẫn cần kiểm theo protocol mới được duyệt.

### 4. “Synergy mandatory” còn là giả thuyết

7arm hiện gồm baseline, từng thành phần riêng và full combination. Điều đó cho
thấy các singleton đã thử không đủ, nhưng chưa là full-factorial/leave-one-out
để chứng minh từng thành phần đều cần hoặc bắt buộc có interaction. Giữ kết luận
phù hợp với số control đã đo, tránh mở sweep lớn chỉ vì câu diễn giải quá mạnh.

## DeepSeek: round38 có cải thiện nhỏ, vẫn chưa qua gate

Commit **`853d40832b41286046772cfc006ecc0fa548863c`** thêm replay weighting:

| Arm,12seed cùng phép thử | Mean forgetting | Mean worst forgetting | Acquisition |
|---|---:|---:|---:|
| Uniform replay |4,03đ|9,92đ|92,16%|
| Aged replay |3,39đ|8,54đ|92,56%|

Tính lại từ12raw cells: aged giảm meanF.63đ và worstF1,375đ, acquisition tăng.40đ.
**0/12aged seeds qua joint gate**, đồng thời không đạt prediction giảm worstF≥2đ.
Report đã nhận predictionFAIL; không hồi tố đổi nó thànhGO. Đây là phần có thể
giữ làm replay scheduling control; hidden ALIF vẫn đóng băng, MLP head vẫn học
bằng gradient nội bộ. Nó chưa giải quyết câu hỏi neural hidden representation.

## Chốt phần nên tận dụng để đi nhanh

1. **Ưu tiên tuyến hidden của Gemini:** EW-HE strict_binding đã có common-refit
   gain và Codex tái lập một historical cell. Sửa các control/gates rồi mới đề
   xuất confirmation; không thử thêm hàng loạt cơ chế/neurons/corpus.
2. **Ưu tiên hạ tầng chung của cả hai:** gate evaluator, full R-matrix, row-level
   label-conflict checks, actual update/episode counters và cùng chuẩn metrics.
3. **Replay làm baseline bắt buộc; chưa ưu tiên CA-MSC đơn lẻ.** Artifact hiện
   chưa cho lợi ích retention của consolidation-only. Chốt budget/first-stage
   behavior trước khi đổ thêm công vào metaplasticity.
4. **Chưa promote multiscale khi thiếu raw/code.** Có thể là hướng tốt, nhưng
   cần provenance trước, rồi protocol-v2/fresh approved evaluation; không lấy
   con số chỉ có trong prose làm acceleration shortcut.

**Approve:** giữ các phần trên trong luồng research review. **Chưa approve
learner vào main/deploy/confirmation mới.** Source production và weights/live
không thay đổi trong lượt kiểm tra.

Chứng cứ bổ sung: R7_INCREMENT_CHECK.json, R7_LATEST_GATE_OUTPUT.json,
`artifacts/forensic-r6/r7-latest-gates.{log,xml}`; so sánh trước đó ở
R7_PROGRESS_RECALCULATION.json và R7_BENCHMARK_VALIDITY.json.
