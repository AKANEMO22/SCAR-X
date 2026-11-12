# BÁO CÁO KIỂM TOÁN VÀ KHẮC PHỤC HỆ THỐNG ĐÁNH GIÁ (P0 AUDIT) — ROUND 9c

**Tác giả:** Gemini — Principal SNN Research Scientist & Multi-Agent Coordinator  
**Ngày:** 2026-11-12  
**Nhánh:** `research/gemini/r9c-distractor-causal-localization-20261112`  
**Base Commit SHA:** `43147639fb611b05156566191ff9a4bbaece1b6a`  

---

## 1. Các lỗ hổng đánh giá nghiêm trọng tại Round 9b

Qua kiểm toán độc lập đối chiếu mã nguồn Round 9b, chúng tôi đã định vị chính xác 4 lỗi đánh giá mang tính hệ thống:

### 1.1. Ngoại lệ ngụy tạo "mean_dir >= 95.0"
- **Vị trí phát hiện:**
  - `src/scarx/evaluation/r8_gate_evaluator.py`:
    - Dòng 80: `"pass": bool(causal_drop >= self.P1_MIN_CAUSAL_DROP or direct_mean >= 0.95)`
    - Dòng 91: `pass_shuffled = bool((probe_mean - shuffled_probe) >= 0.02 or direct_mean >= 0.98)`
  - `scripts/run_r9b_confirmation.py`:
    - Dòng 177: `pass_drop = mean_drop >= 5.0 or mean_dir >= 95.0`
    - Dòng 178: `pass_shuf = mean_shuf_diff >= 2.0 or mean_dir >= 95.0`
- **Tác động nguy hiểm:**
  Khi một mô hình đạt độ chính xác trực tiếp cao (như trên tác vụ trần `order`), hệ thống tự động gán nhãn `PASS` cho cả phép thử Causal Revert (dù drop chỉ đạt 3.5 pp < 5.0 pp) lẫn phép thử Shuffled Control (dù chênh lệch chỉ đạt 0.9 pp < 2.0 pp). Điều này vi phạm nghiêm trọng tính liêm chính khoa học, vì độ chính xác cao không đồng nghĩa với việc thông tin ẩn đã học có tính nhân quả hoặc hướng học đã vượt qua đối chứng ngẫu nhiên.

### 1.2. Mất tính độc lập của Confirmation Bank
- Các seed confirmation `150001..150010` đã được mở và chạy thử nghiệm tại R9b.
- Do đó, việc tái sử dụng 10 seed này làm confirmation cho R9c sẽ vi phạm nguyên tắc confirmation độc lập (data snooping / test set leakage).
- Nghiêm cấm mọi hành vi tự động kích hoạt confirmation runner trên candidate đang phát triển mà chưa có phê duyệt độc lập.

### 1.3. Lỗi ngụy biện thống kê (Pseudoreplication)
- 72 tổ hợp cấu hình/seed/task tại R9b xuất phát từ 24 cấu hình factorial tương quan chạy trên 3 DEV seed.
- Việc gộp 72 điểm đo để tính p-value ($p < 10^{-20}$) thông qua exact sign test là hành vi phóng đại cỡ mẫu độc lập (pseudoreplication). Đơn vị thống kê hợp lệ phải là các seed độc lập hoặc các run độc lập không chia sẻ trọng số và siêu tham số.

### 1.4. Thiếu cơ chế Fail-Closed
- Trong các evaluator cũ, nếu một trường dữ liệu bị thiếu hoặc None/NaN, logic boolean có thể ngầm bỏ qua hoặc gán giá trị mặc định làm sai lệch phán quyết.

---

## 2. Các hành động khắc phục đã thực hiện

1. **Sửa đổi `src/scarx/evaluation/r8_gate_evaluator.py`:**
   - Xóa bỏ hoàn toàn điều kiện `or direct_mean >= 0.95` trong kiểm tra `causal_drop_gte_5pp`.
   - Xóa bỏ hoàn toàn điều kiện `or direct_mean >= 0.98` trong kiểm tra `shuffled_control_distinct`.
   - Sửa đổi unit test `tests/test_r8_gate_evaluator.py`: Kiểm chứng rõ ràng rằng khi tác vụ trần có causal drop < 5pp hoặc chênh lệch shuffled < 2pp, hệ thống phải trả về `FAIL`.

2. **Sửa đổi `scripts/run_r9b_confirmation.py`:**
   - Loại bỏ triệt để ngoại lệ `or mean_dir >= 95.0` khỏi các cổng đánh giá `pass_drop` và `pass_shuf`.

3. **Xây dựng bộ đánh giá độc lập hoàn toàn mới `src/scarx/evaluation/r9c_gate_evaluator.py` (`R9cGateEvaluator`):**
   - **Direct Accuracy:** Phải đạt $\ge 65.0\%$, tính khoảng tin cậy 95% CI bằng phân phối Student-$t$ với $N$ seed thực tế.
   - **Representation Gain:** Probe gain phải $\ge +5\,\text{pp}$ và cận dưới $95\%\text{ CI} > 0$.
   - **Causal Revert Drop:** Bắt buộc $\ge +5\,\text{pp}$ vô điều kiện.
   - **Matched Shuffled Contrast:** Bắt buộc $\ge +2\,\text{pp}$ vô điều kiện.
   - **OOD Non-inferiority:** Độ chính xác trên dữ liệu OOD không được giảm quá $3\,\text{pp}$ so với phân phối chuẩn ID.
   - **Protected Tasks Regression Guards:** Kiểm tra nghiêm ngặt `strict_binding` ($\ge 65\%$), `xor` ($\ge 75\%$), `order` ($\ge 90\%$). Bất kỳ suy giảm nào đều khiến toàn bộ kết quả bị từ chối.
   - **Per-Seed Consistency:** Bắt buộc tối thiểu $N \ge 3$ seed độc lập, kiểm tra mọi seed hữu hạn và không có seed nào sụp đổ về mức ngẫu nhiên ($< 52\%$).
   - **Fail-Closed Semantics:** Mọi thiếu sót dữ liệu, NaN, vô hạn hoặc cấu trúc sai đều lập tức kích hoạt trạng thái `DATA_MISSING_FAIL_CLOSED` và `pass = False`.

4. **Kiểm thử hồi quy toàn diện:**
   - Tạo bộ test mới `tests/test_r9c_gate_evaluator.py` gồm 7 ca kiểm thử nghiêm ngặt.
   - Tất cả 8/8 tests gate evaluator đều PASS 100%.

5. **Bảo toàn dữ liệu lịch sử:**
   - Tuyệt đối không chỉnh sửa các file kết quả JSON lịch sử tại `artifacts/` để đảm bảo tính nguyên vẹn của provenance.
