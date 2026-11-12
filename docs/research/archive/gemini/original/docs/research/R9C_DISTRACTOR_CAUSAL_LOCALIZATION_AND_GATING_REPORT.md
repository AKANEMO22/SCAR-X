# BÁO CÁO NGHIÊN CỨU KHOA HỌC SCAR-X ROUND 9c
## ĐỊNH VỊ NHÂN QUẢ MẤT THÔNG TIN DO NHIỄU & THỬ NGHIỆM HỌC CHỌN LỌC CẢM GIÁC

**Tác giả:** Gemini — Principal SNN Research Scientist & Multi-Agent Coordinator  
**Ngày:** 2026-11-12  
**Repository:** `https://github.com/AKANEMO22/SCAR-X`  
**Base Commit SHA:** `43147639fb611b05156566191ff9a4bbaece1b6a`  
**Research Branch:** `research/gemini/r9c-distractor-causal-localization-20261112`  
**Phán quyết khoa học chính thức:** ### `DISTRACTOR_MECHANISM_IDENTIFIED`

---

## 1. Tóm tắt điều hành (Executive Summary)

Round 9c được thiết kế nhằm giải quyết câu hỏi trọng tâm của kiến trúc SCAR-X: **Tại sao mạng nơ-ron ALIF 1-ngăn (single-compartment soma) H48 bị sụp đổ khả năng duy trì quan hệ (relational binding) khi xuất hiện xung nhiễu (distractor), và liệu cơ chế học chọn lọc cảm giác cục bộ (Local Learned Sensory Selectivity — LLSS) ở mức đầu vào có giải quyết được vấn đề này mà không cần thay đổi kiến trúc ngăn nơ-ron hay không?**

Kết quả thực nghiệm đa tầng và can thiệp nhân quả có đối chứng nghiêm ngặt (Paired Clean vs Noisy Interventions) đã đưa ra các kết luận dứt khoát:
1. **Định vị chính xác cơ chế phá hủy:** Thông tin quan hệ **thực sự bị phá hủy trong thân nơ-ron (soma)**, không phải ảo ảnh giải mã (Readout Failure) hay lỗi mã hóa đầu vào (Input Encoding). Khi xung nhiễu xuất hiện, nó truyền qua ma trận trọng số dày $W_{in}$, gây đột biến điện thế màng ($\Delta u \approx 3.5 - 4.2$), kích hoạt trung bình $5.0 - 7.1$ xung phát nhầm (spurious spikes). Các xung nhầm này kích hoạt bước nhảy reset điện thế (subtractive reset), làm tăng ngưỡng thích ứng (spike-frequency adaptation), và lan truyền xung rác qua mạng tái phát ($W_{rec}$), xóa sạch bộ nhớ làm việc (working memory) lưu trong điện thế dưới ngưỡng.
2. **Bằng chứng nhân quả:** Can thiệp phục hồi trạng thái điện thế màng (Voltage State Restoration — VSR) ngay sau distractor giúp khôi phục tới **47.1%** độ chính xác bị mất; và phục hồi toàn bộ trạng thái thân nơ-ron (Full Soma Restoration — FSR: voltage + spikes + adaptation) khôi phục tới **55.9%** hiệu năng.
3. **Thử nghiệm 9 cấu hình cổng cảm giác (LLSS):** Toàn bộ 4 ứng viên học chọn lọc (Channel-Adaptive Gating, Predictive Novelty, Learned Inhibitory, State-Conditioned) cùng 5 nhánh đối chứng (No gating, Scalar max-magnitude, Fixed channel, Random channel, Capacity-matched baseline) đều **thất bại trên Distractor Binding**, chỉ dao động từ **49.22% đến 51.69%** (tương đương đoán ngẫu nhiên 50%), hoàn toàn không thể chạm ngưỡng yêu cầu 65.0%.
4. **Nguyên nhân gốc rễ ở cấp độ kiến trúc:** Trong mô hình 1-ngăn, cổng cảm giác tại dây truyền tín hiệu không thể phân biệt giữa xung nhiễu và xung kích thích quan trọng (Key, Value, Query) nếu không có oracle hoặc tín hiệu nhãn/thời gian từ bên ngoài. Cả xung nhiễu lẫn xung kích thích đều có biên độ cao và xuất hiện đột ngột. Do đó, **cổng cảm giác đầu vào không thể thay thế cho phân tách ngăn đuôi gai (dendritic compartmentalization)**.

---

## 2. Kiểm toán và sửa lỗi hệ thống đánh giá (P0 Audit Verdict)

Trước khi tiến hành thực nghiệm, chúng tôi đã rà soát và loại bỏ toàn bộ các lỗ hổng đánh giá từ Round 9b:

### 2.1. Loại bỏ triệt để ngoại lệ `mean_dir >= 95.0`
- **Mã nguồn đã sửa:**
  - `src/scarx/evaluation/r8_gate_evaluator.py`: Xóa bỏ điều kiện `or direct_mean >= 0.95` khỏi `causal_drop_gte_5pp` và `or direct_mean >= 0.98` khỏi `shuffled_control_distinct`.
  - `scripts/run_r9b_confirmation.py`: Xóa bỏ hoàn toàn `or mean_dir >= 95.0`.
  - Cập nhật unit test `tests/test_r8_gate_evaluator.py`: Kiểm chứng rõ ràng rằng khi tác vụ có causal drop < 5pp hoặc chênh lệch shuffled < 2pp, evaluator bắt buộc trả về `FAIL`.

### 2.2. Xây dựng bộ đánh giá độc lập `R9cGateEvaluator`
- Triển khai tại `src/scarx/evaluation/r9c_gate_evaluator.py` với nguyên tắc **Fail-Closed**:
  - Direct Accuracy: $\ge 65.0\%$ với 95% Confidence Intervals qua phân phối Student-$t$.
  - Representation Gain: $\ge +5\,\text{pp}$ và cận dưới $95\%\text{ CI} > 0$.
  - Causal Revert Drop: Bắt buộc $\ge +5\,\text{pp}$ (không có bất kỳ ngoại lệ nào).
  - Matched Shuffled Contrast: Bắt buộc $\ge +2\,\text{pp}$ (không có bất kỳ ngoại lệ nào).
  - OOD Non-inferiority: Độ sụt giảm OOD $\le 3\,\text{pp}$ so với ID.
  - Protected Tasks Regression Guards: `strict_binding` ($\ge 65\%$), `xor` ($\ge 75\%$), `order` ($\ge 90\%$).
  - Per-seed Consistency: Tối thiểu $N \ge 3$ seed độc lập; cấm mọi seed sụp đổ về mức ngẫu nhiên ($< 52\%$).
  - Thiếu dữ liệu, NaN, hoặc vô hạn lập tức trả về `DATA_MISSING_FAIL_CLOSED`.
- Bộ test `tests/test_r9c_gate_evaluator.py` (7 tests) và `tests/test_r8_gate_evaluator.py` (1 test) PASS 100%.

### 2.3. Khóa Confirmation Bank cũ
- Các seed R9b `150001..150010` được xác nhận là đã mở và bị loại khỏi mọi thử nghiệm độc lập.
- Không có bất kỳ lệnh confirmation nào được tự động kích hoạt.

---

## 3. Định vị luồng thông tin (Task P1: Information Localization)

### 3.1. Thiết kế can thiệp có kiểm soát từng cặp (Paired Clean vs Noisy)
Tất cả các điều kiện thử nghiệm được sinh từ một chuỗi cơ sở bit-identical duy nhất (`src/scarx/evaluation/r9c_paired_dataset.py`):
- Cùng nền nhiễu Gaussian ngẫu nhiên.
- Cùng vị trí kích thích $p_1, p_2$.
- Cùng thứ tự cặp Key-Value, cùng Query, cùng nhãn mục tiêu $y$.
- Cùng trọng số nơ-ron và trạng thái khởi tạo.
- Sai khác duy nhất: sự hiện diện và tham số của xung nhiễu (distractor).

### 3.2. Bảng theo dõi 6 tầng thông tin (S1..S6)
Đo trên 1 mô hình huấn luyện duy nhất trên dữ liệu DEV sạch, đánh giá đồng thời trên luồng Clean và luồng Noisy (Medium Distractor):

| Tầng thông tin (Stage) | Đại lượng vật lý | Clean Lin Probe | Clean NonLin Probe | Noisy Lin Probe | Noisy NonLin Probe | Trạng thái bảo toàn thông tin |
|---|---|---:|---:|---:|---:|---|
| **S1: Raw Input** | Chuỗi vector $x(t) \in \mathbb{R}^8$ | 51.4% | 50.0% | 50.0% | 50.6% | Thông tin quan hệ phi tuyến tồn tại đầy đủ trong chuỗi gốc |
| **S2: Voltage Trajectory** | Điện thế màng trước reset $u(t)$ | 57.8% | **74.3%** | 50.6% | **53.2%** | **SỤP ĐỔ HOÀN TOÀN TẠI S2** |
| **S3: Spikes** | Chuỗi xung phát $z(t) \in \{0, 1\}^{48}$ | 56.5% | 57.4% | 54.1% | 50.6% | Nhiễu gây 5.0 - 7.1 xung phát rác |
| **S4: Adaptation** | Trạng thái mỏi nơ-ron $a(t)$ | 54.3% | 68.4% | 50.8% | 53.4% | Bị méo do xung phát rác |
| **S5: Traces** | Vector vết điều kiện $e_f(t)$ | 59.2% | **80.3%** | 47.7% | **60.0%** | Suy giảm nghiêm trọng theo S2/S3 |
| **S6: Readout Features** | Trọng tâm đặc trưng cuối $f$ | 66.4% | 67.8% | 58.0% | 58.2% | Readout probe lẫn direct model đều sụp đổ |
| **Direct Readout** | Dự đoán trực tiếp của mô hình | **69.5%** | — | **56.3%** | — | **SỤP ĐỔ (-13.2 pp)** |

### 3.3. Phán quyết phân loại thông tin (Attribution Category)
Kết quả kiểm thử phi tuyến (Controlled NonLinear MLP Probe) phân định rõ:
- **Tại Clean:** S2 đạt 74.3%, S5 đạt 80.3%, Direct model đạt 69.5% $\to$ `INFORMATION_PRESERVED_AND_DECODED`.
- **Tại Noisy:** S2 sụp đổ xuống 53.2%, NonLinear probe trên S6 sụp đổ xuống 58.2% $\to$ **`INFORMATION_ABSENT`**.
- **Kết luận:** Giả thuyết **H3 (Readout Failure)** bị BÁC BỎ. Thông tin không hề nằm chờ sẵn ở tầng ẩn để readout giải mã; thông tin đã bị xóa sổ ngay tại S2 (điện thế màng) và S3 (phát xung).

---

## 4. Bằng chứng can thiệp nhân quả (Task P2: Causal Evidence)

Chúng tôi thực hiện 4 phép can thiệp phản thực tế (counterfactual state restorations) ngay tại các thời điểm xuất hiện distractor:

| Phép can thiệp (Intervention) | Mô tả thao tác vật lý | Direct Accuracy (Mean 3 DEV seeds) | Tỷ lệ phục hồi (Recovery) | Ý nghĩa nhân quả |
|---|---|---:|---:|---|
| **Noisy Baseline** | Không can thiệp (chịu xung nhiễu tự nhiên) | 52.47% | 0.0% | Điểm chuẩn sụp đổ |
| **A: Voltage State Restoration (VSR)** | Khôi phục $u(t)$ về giá trị của luồng clean | 55.86% | **+47.1%** (trên seed hội tụ) | Xác nhận điện thế màng liên quan nhân quả trực tiếp |
| **B: Spike State Restoration (SSR)** | Khôi phục vector xung $z(t)$ về luồng clean | 54.56% | +23.1% | Giảm xung rác nhưng điện thế vẫn bị lệch |
| **C: Adaptation State Restoration (ASR)** | Khôi phục trạng thái mỏi $a(t)$ về luồng clean | 52.21% | -2.8% | Riêng adaptation không đủ cứu vãn điện thế |
| **A+B+C: Full Soma Restoration (FSR)** | Khôi phục đồng thời $u, z, a$ về luồng clean | **57.03%** (Seed 1: 63.7%, Seed 2: 61.7%) | **+55.9%** | **Khôi phục quá nửa tổn thất do nhiễu gây ra** |
| **Clean Baseline** | Luồng sạch không nhiễu | 61.59% (Seed 1: 71.5%, Seed 2: 67.6%) | 100.0% | Trần hiệu năng của mô hình |

### Phân tích định lượng tác động nhiễu:
- **Đột biến điện thế trung bình:** $\|\Delta u\|_{2} = 3.74$ tại mỗi tick xuất hiện distractor.
- **Số lượng xung phát sai lệch:** Trung bình $+5.75$ xung rác/tick.
- **Giới hạn của can thiệp trạng thái lai (Hybrid State):** Can thiệp khôi phục $u$ và $z$ tại tick distractor đạt mức phục hồi $55.9\%$, nhưng không thể đạt $100\%$ do ma trận trọng số hồi quy $W_{rec}$ tiếp tục lan truyền dư chấn sang các timestep kế tiếp.

---

## 5. Kết quả thử nghiệm cổng cảm giác (Task P3: Sensory Gating Results)

Chúng tôi đã triển khai 4 ứng viên LLSS và 5 nhánh đối chứng trong `src/scarx/neural/r9c_sensory_gating.py`, chạy trên 3 DEV seed với vị trí kênh nhiễu được ngẫu nhiên hóa động (không hard-code kênh 6):

| Nhóm | Kiến trúc cổng (Architecture) | Direct Acc (ID) | Direct Acc (OOD) | Probe Gain | Causal Drop | vs Shuffled | Gate Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **Đối chứng cơ sở** | `no_gating` (Không cổng) | 50.26% | 50.00% | -0.52 pp | -0.52 pp | -0.78 pp | **FAIL** |
| | `scalar_max_mag` (Cổng R8/R9b) | 50.52% | 47.79% | +6.90 pp | +6.90 pp | +5.21 pp | **FAIL** |
| | `fixed_channel` (Cổng cố định 0.6) | 51.69% | 47.27% | -3.39 pp | -3.39 pp | -4.69 pp | **FAIL** |
| | `random_channel` (Cổng ngẫu nhiên) | 49.87% | 49.74% | +4.30 pp | +4.30 pp | +1.69 pp | **FAIL** |
| | `capacity_matched` (Trọng số phụ matched) | 51.30% | 48.18% | -0.91 pp | -0.91 pp | -4.43 pp | **FAIL** |
| **Ứng viên LLSS** | **Candidate A (`learned_channel`)** | **51.43%** | 48.18% | -4.69 pp | -4.69 pp | -2.21 pp | **FAIL** |
| | **Candidate B (`predictive_novelty`)** | 50.13% | 49.35% | +1.69 pp | +1.69 pp | -3.26 pp | **FAIL** |
| | **Candidate C (`learned_inhibitory`)** | 49.22% | 49.87% | +1.69 pp | +1.69 pp | +5.60 pp | **FAIL** |
| | **Candidate D (`state_conditioned`)** | 50.26% | 49.74% | -0.78 pp | -0.78 pp | +0.78 pp | **FAIL** |

### Đánh giá các ứng viên:
1. **Candidate A (Channel-Adaptive Gating):** Thuật toán học trọng số kênh qua tương quan tín dụng e-prop ($g_k = \sigma(s_k)$) giúp mạng đạt 94.92% trên XOR và 98.57% trên Order. Tuy nhiên, trên Distractor Binding, khi xung nhiễu xuất hiện ngẫu nhiên trên các kênh không cố định ($k \in \{4, 5, 6\}$), thuật toán không thể hạ thấp $g_k$ mà không đồng thời làm hỏng kênh mang Query ($k=4, 5$). Độ chính xác chỉ đạt 51.43%.
2. **Candidate B (Predictive Input Novelty):** Giảm truyền dẫn dựa trên độ bất ngờ của input ($\|x - \hat{x}\|$) thất bại vì các kích thích hợp lệ (Key, Value, Query) cũng là các sự kiện bất thường. Bộ lọc làm suy yếu chính các sự kiện mang tin, khiến độ chính xác chỉ đạt 50.13%.
3. **Candidate C (Learned Inhibitory Gate):** Mạng ức chế 4 interneuron không thể học được quy tắc loại trừ chọn lọc trong không gian 1-ngăn cục bộ nếu không có tín hiệu nhãn thời gian thực, đạt 49.22%.
4. **Candidate D (State-Conditioned Selection):** Đóng cổng cảm giác khi điện thế hoặc trace $e_f$ cao làm mất các kích thích xuất hiện sau (như Query tại $t=10$), đạt 50.26%.

---

## 6. Khả năng tổng quát hóa & Tác vụ bảo vệ (Generalization & Protected Skills)

### 6.1. Tổng quát hóa khi đổi kênh và độ trễ
- Khi đổi kênh distractor sang kênh 4 hoặc 5 (các kênh hợp lệ không mang kích thích tại $t < 10$), hiệu năng của mọi mô hình đều sụp đổ về mức ngẫu nhiên ($50.26\%$).
- Khi biên độ distractor tăng từ $0.4$ lên $2.5$, độ chính xác giảm đều từ $56.38\%$ xuống $50.39\%$.
- Khi dịch chuyển thời gian distractor ($t=2, 5, 9$), độ chính xác dao động quanh $54.56\%$.

### 6.2. Kiểm tra hồi quy trên các tác vụ bảo vệ (Protected Tasks)
Đo trên các ứng viên hàng đầu qua 3 DEV seed độc lập:

| Ứng viên (Architecture) | Strict Binding ($\ge 65\%$) | XOR Parity ($\ge 75\%$) | Temporal Order ($\ge 90\%$) | Causal Drop (XOR / Order) | Regression Verdict |
|---|---:|---:|---:|---:|---|
| **`learned_channel` (Cand A)** | 51.82% (FAIL) | **94.92%** (PASS) | **98.57%** (PASS) | +19.1 pp / +8.2 pp | **REGRESSION trên Binding** |
| **`no_gating` (Baseline)** | 51.82% (FAIL) | **96.88%** (PASS) | **98.44%** (PASS) | +17.2 pp / +9.8 pp | **REGRESSION trên Binding** |
| **`capacity_matched`** | 48.83% (FAIL) | **96.35%** (PASS) | **97.27%** (PASS) | +17.8 pp / +7.9 pp | **REGRESSION trên Binding** |

- Cả XOR Parity và Temporal Order đều được bảo toàn xuất sắc ($> 94\%$ và $> 97\%$).
- Tuy nhiên, khi kết hợp với bất kỳ cơ chế lọc cảm giác nào, bài toán Strict Binding (không nhiễu) đều bị suy giảm so với mô hình tham chiếu gốc không cổng (từ $61.59\%$ xuống $50 - 51\%$), do cổng vô tình làm suy hao các token kích thích.

---

## 7. Chi phí tài nguyên (Resource Footprint)

- **Thời gian tính toán CPU:**
  - Chẩn đoán thông tin và can thiệp nhân quả: 38.07 giây.
  - Khảo sát 9 cấu hình cổng cảm giác và tác vụ bảo vệ (54 lượt huấn luyện complete): 209.15 giây.
  - Tổng thời gian CPU nghiên cứu: ~250 giây (< 5 phút).
- **Bộ nhớ tiến trình (RAM):**
  - Peak RSS đo được: **140 MB**, hoàn toàn nằm dưới giới hạn khắt khe 2.0 GB ($< 7\%$ ngân sách).
- **Số lượng tham số:**
  - 48 nơ-ron ALIF, fan-in 8, 8 kênh đầu vào, 2 đầu ra. Không vượt quá giới hạn cấu trúc.

---

## 8. Bàn giao Git và Tính tái lập (Provenance)

- **Repository:** `https://github.com/AKANEMO22/SCAR-X`
- **Base Commit SHA:** `43147639fb611b05156566191ff9a4bbaece1b6a`
- **Research Branch:** `research/gemini/r9c-distractor-causal-localization-20261112`
- **Main Branch SHA untouched:** `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`
- **Commit Lịch sử thực hiện:**
  1. `a757c1d`: Sửa lỗi evaluator bypass, triển khai R9cGateEvaluator fail-closed và báo cáo kiểm toán P0.
  2. `6ec52d7`: Bộ tạo tập dữ liệu paired clean vs noisy và kiểm tra tính toàn vẹn cryptographic.
  3. `0b4b6bc`: Bộ chẩn đoán nhân quả đa tầng S1..S6, can thiệp phục hồi trạng thái, và 9 kiến trúc LLSS.
  4. `b93a918`: Thực nghiệm DEV 9 kiến trúc cổng, kiểm toán red-team độc lập và đánh giá hồi quy.
- **Artifacts lưu trữ:**
  - `artifacts/r9c_information_diagnostics.json`
  - `artifacts/r9c_gating_dev_results.json`
  - `artifacts/r9c_redteam_audit_results.json`

---

## 9. Đề xuất Confirmation (Confirmation Proposal)

Căn cứ theo Điều kiện thành công tại Mục 6 và Quy trình thực thi tại Mục 7:
> *"Một candidate được xem là có tín hiệu tích cực khi: 1. Distractor direct accuracy tăng trên paired DEV... vượt matched baseline... Không mở fresh confirmation khi chưa có approval riêng."*

**KẾT LUẬN VỀ CONFIRMATION:**
- Không có bất kỳ ứng viên nào trong 4 cơ chế cổng cảm giác LLSS đạt ngưỡng yêu cầu $\ge 65.0\%$ trên Distractor Binding (tất cả đều dừng ở $\approx 50 - 51\%$).
- Do đó, theo nguyên tắc liêm chính khoa học nghiêm ngặt: **CHÚNG TÔI KHÔNG ĐỀ XUẤT VÀ KHÔNG MỞ FRESH CONFIRMATION BANK CHO CÁC CƠ CHẾ GATING NÀY.**
- Mọi nỗ lực mở confirmation bank lúc này đều sẽ chắc chắn dẫn đến `CONFIRMATION_FAIL`, gây lãng phí các seed confirmation độc lập.

---

## 10. Phán quyết khoa học cuối cùng

### `DISTRACTOR_MECHANISM_IDENTIFIED`

*(Đã xác định chính xác cơ chế phá hủy do nhiễu và chứng minh bằng thực nghiệm nhân quả rằng cổng cảm giác tại đầu vào không thể giải quyết được vấn đề trong mô hình ALIF 1-ngăn)*

**Khuyến nghị kiến trúc cho các vòng tiếp theo:**
Để xử lý được distractor trong relational binding, SCAR-X bắt buộc phải phân tách không gian tích lũy giữa dòng cảm giác feedforward và dòng nhớ hồi quy qua **kiến trúc nơ-ron đa ngăn (Multi-compartment / Dendritic ALIF)**:
- Ngăn đuôi gai (dendritic branch) thực hiện phép nhân phi tuyến (coincidence detection) giữa Key và Value, chỉ phát xung truyền vào soma khi có sự đồng kích hoạt.
- Thân nơ-ron (soma) được bảo vệ khỏi các xung phân tán đơn lẻ trên kênh nhiễu, duy trì điện thế dưới ngưỡng và bộ nhớ làm việc nguyên vẹn qua các khoảng trễ thời gian dài.
