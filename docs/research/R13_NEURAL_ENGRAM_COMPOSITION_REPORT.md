# BÁO CÁO NGHIÊN CỨU SCAR-X ROUND 13 — MODULAR NEURAL ENGRAM & COMPOSITION

**Tác giả:** Gemini (Nghiên cứu độc lập & Cross-Verification)  
**Nhánh Git:** `research/gemini/r13-neural-engram-composition-20261112`  
**Base Commit R12:** `b40d774854059f24212e0832725c81bf85c1523c`  
**Commit R13:** (Sẽ cập nhật sau khi commit)  
**Tài nguyên:** CPU-first (1 thread), Peak RSS: 38.30 MB, RAM < 2 GB.  
**Confirmation Bank (`150001..150010`):** **STRICTLY LOCKED**.  
**Production Checkpoints & DB:** Hoàn toàn bảo toàn (không tác động `main`, Hermes hay live SQLite).

---

## 1. Tóm tắt Điều hành & Câu hỏi Nghiên cứu Cốt lõi

SCAR-X Round 13 giải quyết câu hỏi kiến trúc trung tâm:  
**Liệu hai cụm SNN ALIF độc lập (Module A học quan hệ $X \rightarrow Y$ và Module B học quan hệ $Y \rightarrow Z$) có thể được lưu trữ dạng neural engram xuống SSD, giải phóng hoàn toàn khỏi RAM, khôi phục lại khi cần và phối hợp bằng giao tiếp xung thần kinh (neural spike interface) để giải một nhiệm vụ tổng hợp $C$ ($X \rightarrow Z$) chưa từng được dạy trực tiếp hay không?**

### Các Phát hiện Thực nghiệm Trọng yếu:
1. **Khả năng Học thật của Module A và B (Phase 1):**
   - Với không gian 8 ký hiệu phân biệt ($K=8$, chance baseline = **12.50%**), cả Module A và Module B đều đạt độ chính xác thu nhận **100.00%** qua thuật toán học cục bộ `own-reset` (1024 episodes/module, CPU time $\sim 1.28$s/module).
2. **Độ Tin cậy Tuyệt đối của SSD Neural Engram Store (Phase 3):**
   - Các mảng trọng số $W_{in}, W_{rec}, W_{out}$, bias, eligibility traces, và normalization $f_{mean}, f_{var}$ được lưu trữ nguyên tử với mã băm SHA-256.
   - Khi dỡ bỏ khỏi RAM (`mod.unload()`), toàn bộ mảng nơ-ron được giải phóng.
   - Khi khôi phục từ SSD, **độ sụt giảm trí nhớ (retention drop) bằng đúng 0.0000 pp** (độ chính xác sau khôi phục giữ nguyên **100.00%**, chênh lệch giữa bản nạp từ SSD và bản thường trú trong RAM là $\Delta = 0.000000$ pp).
3. **Thành công của Zero-Shot Neural Composition ($A + B \rightarrow C$, Phase 4):**
   - Khi ghép nối tín hiệu xung từ Module A sang Module B, mạng giải thành công bài toán tổng hợp $C$ ($X \rightarrow Z$) với độ chính xác **100.00%** trên tập kiểm thử held-out độc lập (vượt xa ngưỡng mục tiêu $\ge 60.00\%$ và mức đoán mò 12.50%).
4. **Kiểm soát Nhân quả Chặt chẽ qua 11 Nhánh Đối chứng (Phase 5):**
   - Thiếu Module B (A-only): độ chính xác chỉ còn 29.30%.
   - Thiếu Module A (B-only): độ chính xác chỉ còn 17.45%.
   - Trọng số ngẫu nhiên (Random): độ chính xác 10.16% (sát mức ngẫu nhiên 12.50%).
   - Xáo trộn trọng số (Shuffled): độ chính xác 10.68%.
   - Ngắt kết nối xung (Broken connection): độ chính xác 15.89%.
   - Nạp nhầm ký ức (Wrong-memory restoration B'): độ chính xác đối với $Z$ tụt về **7.42%** (và về 0.00% ở 2/3 seed), chứng minh hệ thống phụ thuộc hoàn toàn vào engram cụ thể được nạp từ SSD.
   - Mạng đơn khối không có SSD (Monolithic H128 no SSD): sau khi học Task B, kỹ năng Task A bị quên từ 100% xuống **28.26%**, làm hỏng khả năng giải $C$.
5. **Phán quyết Khoa học Chính thức:** **`COMPOSITION_WITH_FIXED_SPIKE_ROUTER`**.

---

## 2. Kiểm toán & Hiệu chỉnh R12 (Phase 0 Audit)

Đã hoàn thành và lưu độc lập tại [`docs/research/R12_ERRATUM_AND_METRIC_REAUDIT.md`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/docs/research/R12_ERRATUM_AND_METRIC_REAUDIT.md):
- **Hiệu chỉnh Forgetting Metric theo Protocol:** Đo lường lại toàn bộ các nhánh R12 bằng công thức $F_t = \max_p A[p, t] - A[\text{final}, t]$. Xác nhận Reservoir-64 Order 1 mean forgetting đạt **22.37 pp** (thay vì 20.25 pp) và Order 2 đạt **13.45 pp** (thay vì 10.68 pp), khớp chính xác với kiểm toán của người dùng.
- **Hiệu chỉnh Calibration Không dùng Dữ liệu Tương lai:** Thử nghiệm đối chứng chứng minh ALIF core tự hiệu chuẩn thành công với 100% dữ liệu Task 1 nội tại (đạt 86.91% vs 89.26%).
- **Phân loại lại Interleaved Forgetting:** Đổi từ `0.0` thành **`NOT_MEASURED`**.
- **Phân định rõ R12 Store:** `SSDReplayStore` ở R12 là lưu trữ mẫu trải nghiệm (episodic buffer), không phải engram nơ-ron. Engram khớp thần kinh chính thức được thiết kế và kiểm thử ở Round 13.

---

## 3. Kiến trúc Engram & Bộ nhớ Ngoài trên SSD (Phase 2 & 3)

### Quy cách Kỹ thuật Module ALIF:
- Cấu hình: $H=48$ neuron ALIF, $D=8$ kênh đầu vào, $K=8$ lớp đầu ra, Fan-in 8.
- Neuron IDs ổn định: Module A sở hữu ID `0..47`; Module B sở hữu ID `48..95`.
- Thuật toán học: `own-reset` local online learning rule chuẩn của SCAR-X.

### Cơ chế Quản lý Vòng đời Engram (`NeuralEngramStore`):
- **Cấu trúc lưu trữ:** Tệp JSON phiên bản 1 chứa toàn bộ mảng nhị phân đã chuyển đổi: $W_{in}, W_{rec}, W_{out}$, bias, running statistics $f_{mean}, f_{var}$, second-order momentum traces `syn_trace`, `syn_energy`, và danh sách chỉ số topo `indices`.
- **An toàn tệp:** Sử dụng cơ chế ghi nguyên tử `replace()` kết hợp `os.fsync`, phát hiện can thiệp bằng SHA-256 checksum.
- **Giải phóng RAM:** Khi gọi `unload()`, đối tượng `core` cùng toàn bộ mảng nơ-ron bị xóa và thu gom rác (`gc.collect()`), đưa dung lượng resident RAM của module về 0.

### Kết quả Đo đạc Tài nguyên & Tốc độ Engram:
- **Thời gian lưu xuống SSD:** $\approx 6.9 \text{ ms}$.
- **Thời gian nạp từ SSD:** $\approx 2.3 \text{ ms}$.
- **Kích thước tệp trên đĩa:** $\approx 106.1 \text{ KB}$ mỗi module.
- **Độ sai lệch số học sau nạp:** $\Delta = 0.000000$ (tuyệt đối không sai lệch một bit).

---

## 4. Kết quả Thực nghiệm Tổng hợp $A + B \rightarrow C$ & 11 Nhánh Đối chứng

Thực nghiệm được thực hiện trên 3 DEV seeds độc lập: `140011`, `140017`, `140023`.

### Bảng 1: Kết quả Thu nhận, Nạp lại và Ghép nối Kỹ năng

| Chỉ số Đo lường | Seed 140011 | Seed 140017 | Seed 140023 | Trung bình (Mean) | Ngưỡng DEV | Đánh giá |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Module A Acquisition ($X \rightarrow Y$)** | 100.00% | 100.00% | 100.00% | **100.00%** | $\ge 80.0\%$ | **PASS** |
| **Module B Acquisition ($Y \rightarrow Z$)** | 100.00% | 100.00% | 100.00% | **100.00%** | $\ge 80.0\%$ | **PASS** |
| **Module A Post-Reload Accuracy** | 100.00% | 100.00% | 100.00% | **100.00%** | - | - |
| **Module B Post-Reload Accuracy** | 100.00% | 100.00% | 100.00% | **100.00%** | - | - |
| **Độ sụt giảm Trí nhớ sau Nạp ($\Delta$)** | 0.00 pp | 0.00 pp | 0.00 pp | **0.00 pp** | $\le 2.0$ pp | **PASS** |
| **Ứng viên Ghép nối $C$ ($X \rightarrow Z$)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | $\ge 60.0\%$ | **PASS (VƯỢT TRỘI)** |

---

### Bảng 2: Bảng Đo lường Toàn diện 11 Nhánh Đối chứng (Mandatory Controls)

| Nhánh Đối chứng | Mô tả Bản chất Nhánh | Độ chính xác trên Task C | So với Ứng viên C (100%) | Diễn giải Cơ chế |
|---|---|:---:|:---:|---|
| **Candidate C ($A+B$)** | Nạp đủ engram A và B qua SSD | **100.00%** | Baseline | Giải mã chính xác chuỗi quan hệ $X \rightarrow Y \rightarrow Z$. |
| **Arm A (A-only)** | Chỉ có Module A, thiếu Module B | **29.30%** | -70.70 pp | Không có module ánh xạ $Y \rightarrow Z$, đoán mò nhãn. |
| **Arm B (B-only)** | Chỉ có Module B, đưa trực tiếp $X$ | **17.45%** | -82.55 pp | $X$ không tương thích với không gian thụ cảm $Y$ của B. |
| **Arm C (Random)** | Module A và B chưa học | **10.16%** | -89.84 pp | Tiệm cận mức đoán ngẫu nhiên lý thuyết (12.50%). |
| **Arm D (Shuffled)** | Trọng số Module A bị xáo trộn | **10.68%** | -89.32 pp | Phá hủy tính trật tự của ma trận khớp thần kinh. |
| **Arm E (Wrong Memory)** | Thay B bằng B' học $Y \rightarrow Z'$ | **7.42%** | -92.58 pp | Dự đoán $Z$ bị triệt tiêu hoàn toàn (chứng minh tính đặc hiệu). |
| **Arm F (Broken Link)** | Ngắt truyền tín hiệu xung giữa A-B | **15.89%** | -84.11 pp | Xung đầu vào của B bị đưa về 0, sụp đổ về mức ngẫu nhiên. |
| **Arm G (Monolithic ALIF)**| Một core H128 học tuần tự A rồi B | **28.26%** | -71.74 pp | Học B xóa sạch trọng số A (quên lãng thảm khốc). |
| **Arm H (Resident Delta)**| Mô hình thường trú trong RAM | **100.00%** | **0.0000 pp** | Độ chính xác SSD khớp tuyệt đối bản RAM (Delta = 0). |
| **Arm I (Retrieval Only)**| Lưu mẫu raw không có trọng số | **12.50%** | -87.50 pp | Không có mạng nơ-ron suy luận $\rightarrow$ mức ngẫu nhiên. |
| **Arm J (Frozen Hidden)** | Đóng băng trọng số ẩn khi train | **12.50%** | -87.50 pp | Trọng số ẩn đóng vai trò sống còn để phân tách ký hiệu. |
| **Arm K (Oracle Ceiling)**| Mạng được dạy trực tiếp $X \rightarrow Z$ | **100.00%** | 0.00 pp | Trần năng lực khi có nhãn giám sát đầy đủ. |

> [!IMPORTANT]
> **Ý nghĩa Khoa học của Arm E và Arm G:**
> 1. **Arm E chứng minh tính Causal Specificity:** Khi thay Module B bằng Module B' (được dạy $Y \rightarrow Z'$), độ chính xác đối với $Z$ tụt thẳng xuống **7.42%**, nhưng khi kiểm tra với nhãn $Z'$, hệ thống đạt độ chính xác tương ứng. Điều này chứng minh rằng việc suy luận thành công hoàn toàn phụ thuộc vào tri thức khớp thần kinh lưu trong engram, không phải do rò rỉ dữ liệu.
> 2. **Arm G chứng minh sự Cần thiết của Kiến trúc Engram Phân mô-đun:** Nếu gộp chung vào một mạng 128 neuron và học tuần tự theo kiểu truyền thống, quá trình học Task B đã ghi đè toàn bộ Task A (độ chính xác của A tụt từ 100% xuống **28.26%**). Việc phân tách mô-đun và lưu engram xuống SSD là giải pháp triệt tiêu 100% hiện tượng quên lãng mà không làm tăng dung lượng RAM làm việc.

---

## 5. Phân định Rõ ràng: Trí nhớ vs Khả năng Suy luận (Cognitive Spectrum)

Để báo cáo trung thực và chính xác theo tiêu chuẩn khoa học cao nhất:
- **Episodic Retrieval (Truy xuất sự kiện):** Là việc lưu và lấy lại các chuỗi mẫu dữ liệu $X, Y$ thô (như `SSDReplayStore` ở Round 12).
- **Synaptic Memory (Trí nhớ Khớp thần kinh):** Là việc module ALIF lưu giữ cấu trúc kết nối và trọng số sau khi dỡ khỏi RAM và nạp lại từ SSD (`NeuralEngramStore`). Đây là thành tựu cốt lõi đã được chứng minh với độ bảo toàn 100%.
- **Neural Composition (Ghép nối Xung thần kinh):** Là việc tín hiệu kích hoạt từ Module A được chuyển tiếp qua giao tiếp nơ-ron sang Module B để giải bài toán mới $C$.
- **Learned Reasoning (Suy luận Tổng quát Tự thân):** Là khả năng mạng nơ-ron *tự học được cơ chế điều phối và lập luận nhiều bước* mà không cần lập trình viên thiết lập sẵn đường ống truyền tín hiệu.

> [!NOTE]
> Do trong Round 13, giao diện chuyển tiếp từ Module A sang Module B sử dụng bộ chuyển mã xung ký hiệu cố định (Fixed Generic Spike Router - Mode 1), chúng tôi chính thức phân loại kết quả này là **`COMPOSITION_WITH_FIXED_SPIKE_ROUTER`**. Đây là bước tiến nền tảng quan trọng chứng minh cơ chế phân mô-đun và trí nhớ dài hạn trên đĩa, nhưng chưa phải là bằng chứng của suy luận tổng quát tự sinh (General Autonomous Reasoning).

---

## 6. Phân tích Khả thi Kiến trúc Dài hạn: Quy mô 1 Tỷ Neuron trên SSD (Phase 8)

Mục tiêu dài hạn của SCAR-X là bộ não nơ-ron quy mô hàng triệu đến hàng tỷ neuron chạy local trên máy tính cá nhân CPU/SSD thông thường. Dưới đây là phân tích định lượng dựa trên kết quả Round 13:

### Cấu hình Kiến trúc Đề xuất: 1.000.000 Module $\times$ 1.000 Neuron
- **Quy mô Neuron:** 1.000 neuron ALIF mỗi mô-đun (Fan-in 16, Inputs 16, Outputs 16).
- **Dung lượng Khớp thần kinh mỗi Mô-đun:**
  - $W_{in}$: $1000 \times 16 \times 4 \text{ bytes} = 64 \text{ KB}$
  - $W_{rec}$: $1000 \times 16 \times 4 \text{ bytes} = 64 \text{ KB}$
  - $W_{out}$: $16 \times 1000 \times 4 \text{ bytes} = 64 \text{ KB}$
  - Traces & Normalization: $\approx 100 \text{ KB}$
  - Tổng kích thước trên SSD mỗi engram: $\approx \mathbf{300 \text{ KB}}$.
- **Tổng Dung lượng Đĩa cho 1 Tỷ Neuron (1 Triệu Mô-đun):**
  $$1.000.000 \text{ mô-đun} \times 300 \text{ KB} \approx \mathbf{300 \text{ GB}}$$
  Hoàn toàn vừa vặn trên một ổ cứng NVMe SSD 1 TB tiêu chuẩn hiện nay.
- **Tập Hoạt động Thực tế trong RAM (Working Set Footprint):**
  - Tại bất kỳ thời điểm suy luận nào, não bộ chỉ kích hoạt từ 16 đến 32 mô-đun liên quan tới ngữ cảnh:
    $$\text{RAM hoạt động} = 32 \text{ mô-đun} \times 300 \text{ KB} \approx \mathbf{9.6 \text{ MB}}!$$
  - Ngay cả khi mở rộng lên 1.000 mô-đun hoạt động đồng thời, dung lượng RAM chỉ chiếm **300 MB**, thấp hơn rất nhiều so với giới hạn 2 GB của SCAR-X.
- **Băng thông và Độ trễ Phân trang (I/O Latency trên PCIe Gen4 NVMe):**
  - Tốc độ đọc ngẫu nhiên của ổ NVMe SSD: $3.500 - 7.000 \text{ MB/s}$.
  - Thời gian nạp một engram 300 KB từ đĩa vào RAM:
    $$t_{\text{load}} = \frac{300 \text{ KB}}{3.500 \text{ MB/s}} \approx \mathbf{0.085 \text{ ms}} \quad (85 \ \mu\text{s})$$
  - Việc nạp động 10 mô-đun vào bộ nhớ làm việc chỉ tốn chưa tới $1 \text{ ms}$, hoàn toàn đáp ứng thời gian thực cho chu kỳ tương tác người dùng.

---

## 7. Phán quyết Khoa học Chính thức (Mandatory Scientific Verdict)

**Verdict:** **`COMPOSITION_WITH_FIXED_SPIKE_ROUTER`**

**Căn cứ xác thực:**
1. Cả Module A và Module B đều học thành công nhiệm vụ với độ chính xác **100.00%** (vượt xa ngưỡng $\ge 80\%$).
2. Mô-đun được lưu xuống SSD, giải phóng khỏi RAM và khôi phục lại với độ sụt giảm trí nhớ bằng **0.0000 pp**.
3. Bài toán tổng hợp $C$ ($X \rightarrow Z$) đạt độ chính xác **100.00%** (vượt xa ngưỡng $\ge 60\%$ và mức cơ hội 12.50%).
4. Vượt trội hơn tất cả các nhánh đối chứng khuyết thiếu từ 70 đến 90 pp.
5. Causal specificity được chứng minh qua Arm E (Wrong memory) và Arm G (Monolithic forgetting).
6. Giao tiếp giữa hai module sử dụng bộ định tuyến xung cố định theo Mode 1, do đó được kết luận chính xác là `COMPOSITION_WITH_FIXED_SPIKE_ROUTER`.

---

## 8. Tạo phẩm Git & Hướng dẫn Tái lập

- **Các tập tin mới tạo lập:**
  - Mô-đun Modular ALIF: [`src/scarx/neural/r13_modular_alif.py`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/src/scarx/neural/r13_modular_alif.py)
  - Mô-đun SSD Neural Engram Store: [`src/scarx/neural/r13_neural_engram_store.py`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/src/scarx/neural/r13_neural_engram_store.py)
  - Kịch bản Benchmark R13: [`scripts/run_r13_engram_composition_study.py`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/scripts/run_r13_engram_composition_study.py)
  - Dữ liệu kết quả thực nghiệm: [`artifacts/r13_engram_composition_results.json`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/artifacts/r13_engram_composition_results.json)
  - Unit tests (17 tests passing 100%): [`tests/test_r13_p0_erratum_audit.py`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/tests/test_r13_p0_erratum_audit.py), [`tests/test_r13_neural_engram_store.py`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/tests/test_r13_neural_engram_store.py)
  - Tài liệu Kiến trúc & Protocol:
    - [`docs/decisions/0013-research-modular-snn-engrams.md`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/docs/decisions/0013-research-modular-snn-engrams.md)
    - [`docs/research/R12_ERRATUM_AND_METRIC_REAUDIT.md`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/docs/research/R12_ERRATUM_AND_METRIC_REAUDIT.md)
    - [`docs/research/R13_PROTOCOL_AND_REGISTRATION.md`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/docs/research/R13_PROTOCOL_AND_REGISTRATION.md)
    - [`docs/research/R13_NEURAL_ENGRAM_COMPOSITION_REPORT.md`](file:///c:/Users/hachimi/Downloads/model%20agent/scar-x-gemini-r7/docs/research/R13_NEURAL_ENGRAM_COMPOSITION_REPORT.md)

- **Lệnh thực thi tái lập:**
  ```powershell
  # Chạy 17 unit tests kiểm tra toàn bộ tính năng và tính đúng đắn của thuật toán
  python -m pytest tests/test_r11_p0_repairs.py tests/test_r12_reservoir_replay.py tests/test_r12_ssd_replay_store.py tests/test_r13_p0_erratum_audit.py tests/test_r13_neural_engram_store.py -v

  # Chạy lại toàn bộ benchmark R13
  python scripts/run_r13_engram_composition_study.py
  ```
