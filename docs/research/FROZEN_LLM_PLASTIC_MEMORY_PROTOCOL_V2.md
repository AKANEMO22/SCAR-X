# SCAR-X: FROZEN LLM + PLASTIC MEMORY + AUTONOMOUS IDLE LEARNING
## Pre-Registered Scientific Research Protocol V2 (Audited & Repaired)

**Author:** Gemini (Lead AI Research Engineer, Continual Learning Researcher & Independent Scientific Auditor)  
**Date:** 2026-11-13  
**Status:** PROTOCOL V2 LOCKED (Post-Audit Scientific Repair)  
**Branch:** `research/gemini/frozen-llm-plastic-memory-a-f`  
**Repository:** `https://github.com/AKANEMO22/SCAR-X`  
**Prior Reference Commit (Audited Flaws):** `997436215473f00bed583767c36d8f4782d46b77`

---

## 1. BỐI CẢNH KIỂM TOÁN VÀ CÁC LỖI KHOA HỌC ĐÃ PHÁT HIỆN Ở COMMIT 99743621

Trong phiên kiểm toán độc lập đối với commit `99743621`, năm khiếm khuyết phương pháp luận nghiêm trọng (P0 Flaws) đã được phát hiện và ghi nhận vào hồ sơ lưu trữ (`artifacts/archive_r14_initial/`):

1. **P0-A (Rò rỉ đáp án vào Tool Call - Answer Leakage):**
   Trong file `variants.py` cũ, phương thức `_parse_tool_call` chứa đoạn mã fallback nguy hiểm:
   ```python
   if "params" not in result or not result["params"]:
       params = dict(task_params)  # task_params chứa trực tiếp ground truth reference solution!
   ```
   Khi mô hình nhỏ (Qwen2.5-0.5B) không sinh đúng định dạng JSON cho tham số, runner đã âm thầm bơm tham số đáp án từ metadata vào công cụ thực thi, dẫn đến độ chính xác thực thi bị thổi phồng giả tạo (~80-90%).

2. **P0-B (Nhiệm vụ Compositional mang tính đơn điệu, giả tạo):**
   Tất cả 40 nhiệm vụ compositional trong benchmark V1 chỉ xoay quanh một thao tác duy nhất: đảo ngược danh sách (`reverse_list`). Điều này không phản ánh năng lực tổng quát hóa thuật toán thực sự.

3. **P0-C (Bộ đánh giá độc lập phụ thuộc cờ tự báo của Runner - Rubber-stamping):**
   File `evaluate_plastic_llm_independently.py` cũ chỉ đơn thuần tính tổng `sum(1 for r in learn if r["passed"])` từ tệp JSON do chính runner ghi ra, không hề chấm lại từ log thực thi thực tế.

4. **P0-D (Đo đạc RAM chưa chặt chẽ và nhầm lẫn đơn vị):**
   Báo cáo cũ chỉ đo trọng số tĩnh (~988 MB) và cờ `rss_mb < 2000` tại một thời điểm, chưa theo dõi `peak_rss_bytes` trong suốt quá trình chạy trên CPU và phân biệt giữa $2 \times 10^9$ bytes (decimal 2.0 GB) và $2^{31}$ bytes (binary 2.0 GiB).

5. **P0-E (Thực thi mã không an toàn trong tiến trình chính):**
   File `skill_registry.py` sử dụng hàm `exec()` trực tiếp trong tiến trình Python chính, gây nguy cơ rò rỉ bộ nhớ, treo tiến trình (hang) nếu gặp vòng lặp vô hạn, và ô nhiễm không gian tên toàn cục.

6. **Thiếu Baseline thống kê ngữ cảnh công bằng (Fair Contextual Baseline):**
   Việc chỉ so sánh Biến thể C (Plastic Memory) với Biến thể F (EMA tần suất toàn cục không phân biệt ngữ cảnh) chưa trả lời được câu hỏi: Liệu eligibility traces có ưu thế hơn một mô hình phân loại tuyến tính ngữ cảnh trực tuyến (Contextual Linear / Softmax Policy Gradient) dùng cùng tập đặc trưng hay không?

---

## 2. NĂM NGUYÊN TẮC SỬA CHỮA BẮT BUỘC (P0 REPAIR MANDATES)

### P0-A: Tách rời tuyệt đối PublicTaskInput và PrivateEvaluationMetadata
- Tạo module `src/plastic_llm/task_types.py`:
  - `PublicTaskInput`: Chỉ chứa `task_id`, `prompt`, `category`. Hoàn toàn KHÔNG có thuộc tính `ground_truth`, `reference_solution` hay `test_fixture`.
  - `PrivateEvaluationMetadata`: Chứa `task_id`, `category`, `expected_skill`, `ground_truth`, `reference_solution`, `test_assertions`. Chỉ được cung cấp cho Oracle đánh giá độc lập.
- Tác nhân (Agent) chỉ được nhận `PublicTaskInput`. Mọi tham số thực thi công cụ bắt buộc phải do LLM tự sinh hoặc trích xuất từ văn bản câu lệnh công khai. Tuyệt đối không fallback về metadata ẩn.

### P0-B: Tái cấu trúc Compositional Benchmark với 6 cấu trúc thuật toán thực sự
Bộ dữ liệu 40 nhiệm vụ compositional được xây dựng trên 6 khuôn mẫu thuật toán độc lập:
1. Giai thừa đệ quy (`factorial computation`).
2. Lọc số chẵn (`even filtering`).
3. Kiểm tra chuỗi đối xứng (`palindrome check`).
4. Tìm kiếm chỉ số phần tử (`list index lookup / binary find`).
5. Tính vết ma trận chéo (`matrix diagonal trace`).
6. Đảo ngược từ điển key-value (`dictionary inversion`).

### P0-C: Oracle Đánh giá Độc lập và Tách bạch Kỹ năng
- Tạo module `src/plastic_llm/evaluation_oracle.py`:
  - Hàm `evaluate_execution_trace(meta, trace)` chấm điểm trực tiếp từ vết thực thi thực tế (`selected_skill`, `params`, `exec_result`, `raw_output`).
  - Tách bạch 4 chỉ số độc lập:
    1. **Tool Routing Accuracy:** Chọn đúng công cụ cần thiết cho bài toán.
    2. **Execution Success Rate:** Mã thực thi thành công không lỗi cú pháp/ngoại lệ trong sandbox.
    3. **Semantic Correctness Rate:** Kết quả thực tế khớp chính xác với ground truth của bài toán.
    4. **End-to-End Pass Rate:** Đồng thời thỏa mãn cả Routing, Execution Success và Semantic Pass.
  - Tuyệt đối không đánh đồng việc chọn đúng công cụ với việc giải đúng bài toán.

### P0-D: Đo đạc RAM Gate nghiêm ngặt ($\le 2.000.000.000$ bytes)
- Đo đạc `peak_rss_bytes` trong suốt quá trình chạy.
- Báo cáo rõ ràng cả hai đơn vị:
  - Decimal MB: $\text{bytes} / 10^6$ (Giới hạn: 2,000.0 MB).
  - Binary MiB: $\text{bytes} / (1024 \times 1024)$ (Giới hạn: 1,907.3 MiB).
- Nếu tổng tiến trình CPU inference vượt quá 2,000,000,000 bytes, bắt buộc ghi nhận H5 là **FAIL**, đồng thời làm rõ phần đóng góp của trọng số unquantized BF16 (988 MB) và phần đệm runtime PyTorch C++ DLL.

### P0-E: Subprocess Sandbox cô lập hoàn toàn
- Tạo module `src/plastic_llm/sandbox_worker.py` (`SandboxedPythonWorker`):
  - Chạy mã trong tiến trình con riêng biệt (`subprocess.Popen`).
  - Giới hạn thời gian cứng (`timeout = 4.0` giây). Ép hủy tiến trình (`kill()`) nếu quá thời gian.
  - Thư mục làm việc tạm thời độc lập, tự dọn dẹp sau khi kết thúc.
  - Giới hạn kích thước bộ đệm stdout/stderr (tối đa 10,000 ký tự).

---

## 3. DANH MỤC 7 BIẾN THỂ NGHIÊN CỨU (A–G)

| Biến thể | Định danh | Cấu hình kỹ thuật | Mục đích đối chuẩn |
| :--- | :--- | :--- | :--- |
| **A** | `VariantA_FrozenOnly` | Frozen Qwen2.5-0.5B + Basic Prompts | Baseline khởi điểm (Zero Memory, Zero Learning) |
| **B** | `VariantB_StaticRAG` | Frozen Qwen2.5-0.5B + SQLite FTS5 BM25 | Baseline RAG tĩnh cổ điển (Không điều chỉnh trọng số) |
| **C** | `VariantC_PlasticMemory` | Frozen Qwen2.5-0.5B + SQLite + Plastic Associative Matrix ($W \cdot \mathbf{x}$, Eligibility Traces) | Kiểm chứng khả năng học chọn lọc công cụ có tính dẻo |
| **D** | `VariantD_PlasticIdle` | Biến thể C + Autonomous Idle Engine (Weakness Clustering, Targeted Practice, Sandbox Validation, Rollback) | Kiểm chứng khả năng tự củng cố và phục hồi khi rảnh |
| **E** | `VariantE_RAGIdle` | Biến thể B + Autonomous Idle Engine | Kiểm chứng đóng góp riêng của Idle khi không có ma trận dẻo |
| **F** | `VariantF_FeedbackRanking`| Frozen Qwen2.5-0.5B + SQLite + EMA Success Rate Ranking (Không ngữ cảnh) | Đối chứng với cơ chế thống kê tần suất toàn cục |
| **G** | `VariantG_ContextualLinear`| Frozen Qwen2.5-0.5B + SQLite + Online Softmax Policy Gradient Classifier (Cùng đặc trưng bi-gram $D=128$) | Đối chứng khoa học công bằng với thuật toán học máy ngữ cảnh tiêu chuẩn |

---

## 4. CÁC TIÊU CHÍ BÁC BỎ VÀ XÁC NHẬN GIẢ THUYẾT (HYPOTHESIS LEDGER)

- **H1 (Tool Routing & Task Performance):**
  - Giả thuyết: Biến thể C đạt độ chính xác điều hướng công cụ (Tool Routing) và tỷ lệ giải quyết bài toán End-to-End cao hơn có ý nghĩa thống kê so với Biến thể A.
  - Tiêu chí Bác bỏ: Nếu Tool Routing của C $\le$ A hoặc End-to-End của C $\le$ A $\rightarrow$ **REFUTED**.
- **H2 (Rapid Mistake Adaptation):**
  - Giả thuyết: Khi gặp thất bại ở lượt 1, phản hồi trực tuyến giúp Biến thể C phục hồi ở lượt 2 tốt hơn Biến thể B (vốn không có cơ chế học từ lỗi).
  - Tiêu chí Bác bỏ: Tỷ lệ phục hồi của C $\le$ B $\rightarrow$ **REFUTED**.
- **H3 (Autonomous Idle Learning):**
  - Giả thuyết: Biến thể D (Plastic + Idle) đạt độ chính xác tổng quát hóa trên tập bài toán mới (Unseen Tasks) cao hơn Biến thể C (Plastic Only).
  - Tiêu chí Bác bỏ: Unseen Acc của D $\le$ C $\rightarrow$ **REFUTED**.
- **H4 (Catastrophic Forgetting & Stability):**
  - Giả thuyết: Độ suy giảm tri thức cũ sau khi học các nhiệm vụ mới (Forgetting Rate) $\le 10\%$, tỷ lệ ổn định (Retention Stability) $\ge 90\%$.
  - Tiêu chí Bác bỏ: Forgetting $> 10\%$ hoặc Stability $< 90\%$ $\rightarrow$ **REFUTED**.
- **H5 (Strict RAM Limit $\le 2.0\text{ GB}$):**
  - Giả thuyết: Tổng tài nguyên RAM của tiến trình LLM inference không vượt quá 2,000,000,000 bytes.
  - Tiêu chí Bác bỏ: Peak RSS $> 2.000.000.000$ bytes $\rightarrow$ **REFUTED/FAIL**.
- **Biến thể C vs Biến thể G (Plastic vs Contextual Linear Baseline):**
  - So sánh trực tiếp giữa Eligibility Traces và Online Logistic Softmax Gradient trên cùng không gian đặc trưng ngữ cảnh $D=128$.
  - Nếu $G \ge C$ về độ chính xác chọn công cụ với tốc độ hội tụ tương đương hoặc nhanh hơn $\rightarrow$ Ghi nhận trung thực rằng ma trận liên kết dẻo sinh học chưa vượt qua được bộ phân loại tuyến tính chuẩn.
