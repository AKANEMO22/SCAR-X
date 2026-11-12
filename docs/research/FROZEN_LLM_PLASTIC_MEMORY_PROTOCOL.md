# SCAR-X: FROZEN LLM + PLASTIC MEMORY + AUTONOMOUS IDLE LEARNING
## Pre-Registered Scientific Research Protocol (H1–H5)

**Author:** Gemini (Lead AI Research Engineer, Continual Learning Researcher & Independent Auditor)  
**Date:** 2026-11-13  
**Status:** PRE-REGISTERED & LOCKED PRIOR TO CONFIRMATION EVALUATION  
**Branch:** `research/gemini/frozen-llm-plastic-memory-a-f`  
**Repository:** `https://github.com/AKANEMO22/SCAR-X`  

---

## 1. GIẢ THUYẾT NGHIÊN CỨU CỐT LÕI (CORE HYPOTHESES)

Nghiên cứu này kiểm chứng giả thuyết trung tâm:
> **Một LLM nhỏ, giữ nguyên 100% trọng số pretrained (frozen), có thể cải thiện đáng kể năng lực giải quyết nhiệm vụ qua trải nghiệm thực tế nhờ hệ thống trí nhớ có tính dẻo (Plastic Memory), cơ chế củng cố và học khi rảnh (Autonomous Idle Learning), mà không cần bất kỳ fine-tuning, LoRA hay sửa đổi trọng số transformer nào.**

### Các giả thuyết chi tiết (H1–H5):
- **H1 (Plastic Memory vs Static RAG):** Biến thể C (Plastic Memory) đạt tỷ lệ thành công trên nhiệm vụ mới (Unseen Task Success) và độ chính xác chọn công cụ cao hơn có ý nghĩa thống kê so với Biến thể B (Static RAG/SQLite với truy xuất cố định).  
  *Tiêu chí PASS:* Cải thiện $\ge 5.0$ điểm phần trăm trên Unseen Tasks với $p < 0.05$ qua paired bootstrap test.
- **H2 (Plastic Memory vs Simple Feedback Ranking):** Biến thể C (Plastic Memory với eligibility traces và trọng số liên kết dẻo) vượt trội so với Biến thể F (RAG + Simple Success-Rate/EMA Bandit Ranking).  
  *Tiêu chí PASS:* Nếu F $\ge$ C, kết luận hệ thống liên kết lấy cảm hứng từ thần kinh chưa chứng minh được giá trị bổ sung so với thống kê tần suất đơn giản.
- **H3 (Autonomous Idle Learning Effect):** Biến thể D (Plastic Memory + Idle) vượt trội Biến thể C, và Biến thể E (RAG + Idle) vượt trội Biến thể B trên tập nhiệm vụ kiểm tra mới mà không cần thêm tương tác người dùng.  
  *Tiêu chí PASS:* Tỷ lệ hoàn thành nhiệm vụ mới tăng sau chu trình idle consolidation có kiểm chứng trong sandbox.
- **H4 (Plasticity + Idle Synergy):** Sự kết hợp giữa Plastic Memory và Idle Learning (Biến thể D) tạo ra hiệu ứng cộng hưởng vượt trên tổng đóng góp rời rạc ($D - B > (C - B) + (E - B)$).
- **H5 (Continual Learning & Behavioral Forgetting):** Mặc dù trọng số LLM bị đóng băng (không bị catastrophic weight forgetting), hệ thống có thể gặp "quên ở tầng truy xuất" (retrieval collision/interference). Hệ thống đạt H5 nếu duy trì tỷ lệ lưu giữ kiến thức cũ (Retention Rate) $\ge 90\%$ xuyên suốt các đợt tác vụ tuần tự T1 $\rightarrow$ T2 $\rightarrow$ T3.

---

## 2. KHÓA CẤU HÌNH LLM & GIỚI HẠN TÀI NGUYÊN (PHASE 1)

### 2.1. Lựa chọn Model & Trạng thái Khả dụng
- **Ứng viên đề xuất ban đầu:** `Qwen/Qwen3.5-0.8B`.
  - *Trạng thái kiểm tra kỹ thuật:* **BLOCKED_ARCH_UNRECOGNIZED**. Checkpoint HuggingFace sử dụng kiến trúc mới `qwen3_5` chưa được hỗ trợ trong phiên bản Transformers ổn định hiện tại (`Transformers does not recognize this architecture`).
- **Ứng viên chính thức được khóa (Locked Official Model):**
  - **Tên mô hình:** `Qwen/Qwen2.5-0.5B-Instruct`
  - **Định dạng trọng số:** `bfloat16` (16-bit unquantized nguyên bản).
  - **Số lượng tham số:** 494,032,768 (~0.49B parameters).
  - **Kích thước trọng số bộ nhớ:** 988,065,536 bytes (~942.3 MB).
  - **Ràng buộc bắt buộc:**
    - Tuyệt đối không lượng tử hóa (Không INT8, INT4, Q4_K_M, GPTQ, AWQ).
    - Không pruning, không distillation.
    - Không fine-tuning, không adapter, không LoRA.
    - Cùng một checkpoint, tokenizer, chat template và decoding (Greedy: `temperature=0.0, top_p=1.0, do_sample=False`) xuyên suốt toàn bộ các biến thể A–F.

### 2.2. Ngân sách RAM nghiêm ngặt (RAM Gate)
- **Giới hạn RAM cho tiến trình LLM inference:** **$\le 2.000.000.000$ bytes (2.0 GB)**.
- **Phân bổ thực tế:**
  - Trọng số mô hình (BF16): ~988 MB.
  - Bộ đệm KV Cache (Context 2,048 tokens): ~176 MB.
  - Runtime buffer & Activation: ~200 MB.
  - Tổng Resident Set Size (RSS) dự kiến: ~1.36 GB $\le 2.0$ GB.
- **Chế độ benchmark mặc định:** CPU-only. (GPU benchmark được đo và báo cáo riêng biệt nếu có).

---

## 3. KIẾN TRÚC HỆ THỐNG TRÍ NHỚ CÓ TÍNH DẺO (PHASE 2)

Hệ thống bao gồm 5 cấu phần phân tầng độc lập:

1. **Frozen LLM Core:** Tiếp nhận prompt, suy luận, lập kế hoạch, sinh lệnh gọi công cụ (tool call), đọc output và đưa ra kết luận. LLM không bao giờ tự nhận hoàn thành nếu thiếu bằng chứng thực thi.
2. **Skill Registry:** Lưu trữ các kỹ năng khả thi:
   - `python_debug`: Sửa lỗi cú pháp và logic mã nguồn.
   - `code_execution`: Chạy mã Python trong sandbox cô lập và đọc stdout/stderr.
   - `unit_test`: Sinh và chạy bài kiểm thử pytest.
   - `file_search`: Tìm kiếm tệp trong workspace.
   - `document_read`: Đọc tài liệu và trích xuất thông tin.
   - `data_analysis`: Phân tích thống kê dữ liệu dạng bảng/CSV.
   - `workflow_replay`: Tái hiện một chuỗi thao tác đã kiểm chứng thành công.
3. **Episodic & Semantic Memory (SQLite FTS5):**
   - Episodic: Lưu vết tương tác, chuỗi tool calls, lỗi, và phản hồi thực nghiệm.
   - Semantic: Lưu quy tắc, định nghĩa, kinh nghiệm đã kiểm chứng, độ tin cậy và nguồn gốc.
   - Hỗ trợ FTS5/BM25 lexical retrieval, deduplication và cách ly (quarantine) dữ liệu chưa kiểm chứng.
4. **Plastic Memory (Hệ liên kết học trực tuyến online):**
   - Không gian đặc trưng ngữ cảnh nhiệm vụ $\mathbf{x} \in \mathbb{R}^{D_{\text{feat}}}$ (trích xuất đặc trưng thưa xác định: từ khóa, loại bài toán, kiểu đầu vào).
   - Ma trận liên kết dẻo $W \in \mathbb{R}^{N_{\text{skill}} \times D_{\text{feat}}}$ điều chỉnh trọng số ưu tiên lựa chọn kỹ năng và ký ức.
   - Vết đủ điều kiện (Eligibility Trace):
     $$e(t) = \lambda e(t-1) + \mathbf{x}(t)$$
   - Cập nhật trọng số dựa trên phản hồi đã kiểm chứng (Verified Reward $r \in \{-1, +1\}$):
     $$W(t+1) = \text{clip}\left(W(t) + \eta \cdot r \cdot e(t), -W_{\max}, W_{\max}\right)$$
   - Giảm dần ảnh hưởng của kinh nghiệm cũ (Decay & Normalization).
5. **Autonomous Idle Learning Engine:**
   - Kích hoạt khi không có yêu cầu từ người dùng và hệ thống nhàn rỗi.
   - Xác định khoảng trống kỹ năng $\rightarrow$ Tự sinh tác vụ giả định $\rightarrow$ Thử nghiệm trong sandbox $\rightarrow$ Đánh giá bằng Oracle/Unit test $\rightarrow$ Củng cố ký ức thành công $\rightarrow$ Rollback nếu thất bại.

---

## 4. ĐỊNH NGHĨA SÁU BIẾN THỂ ĐỐI CHUẨN A–F (PHASE 3)

| Biến thể | Cấu hình thành phần | Cơ chế lựa chọn kỹ năng / ký ức | Cơ chế học khi rảnh (Idle) |
| :--- | :--- | :--- | :--- |
| **A (Baseline)** | Frozen LLM thuần túy | Prompt mặc định + Tool schema cơ bản | Không |
| **B (Static RAG)** | Frozen LLM + SQLite Memory | Truy xuất cố định FTS5 / BM25 (không học trọng số) | Không |
| **C (Plastic Memory)** | Frozen LLM + SQLite + Plastic Matrix | Xếp hạng kết hợp: BM25 score + $W \cdot \mathbf{x}$ | Không |
| **D (Full System)** | Frozen LLM + SQLite + Plastic + Idle | Xếp hạng dẻo ($W \cdot \mathbf{x}$) | Có (Sandbox Practice & Consolidation) |
| **E (RAG + Idle)** | Frozen LLM + SQLite + Idle | Truy xuất cố định FTS5 / BM25 | Có (Cùng ngân sách với D, nhưng không có Plastic Matrix) |
| **F (Feedback Ranking)**| Frozen LLM + SQLite + EMA Bandit | Xếp hạng theo tỷ lệ thành công lũy tích (EMA Success Rate) | Không |

---

## 5. BỘ DỮ LIỆU ĐÁNH GIÁ (BENCHMARK DATASET) (PHASE 4)

Tổng cộng **200 nhiệm vụ độc lập** thuộc 5 nhóm năng lực:
1. **Nhóm 1 — New Knowledge Retrieval (40 tasks):** Truy xuất thông tin mới, quan hệ mới, định dạng mới được nạp vào bộ nhớ.
2. **Nhóm 2 — Python Debug & Tool Execution (40 tasks):** Sửa lỗi mã Python thực tế, chạy trong sandbox, phân tích ngoại lệ.
3. **Nhóm 3 — Workflow Selection & Execution (40 tasks):** Lựa chọn đúng chuỗi công cụ cho bài toán nhiều bước.
4. **Nhóm 4 — Compositional Generalization (40 tasks):** Kết hợp hai kỹ năng học rời rạc để giải quyết bài toán hỗn hợp chưa từng gặp.
5. **Nhóm 5 — Edge Cases & Adversarial Inputs (40 tasks):** Dữ liệu mâu thuẫn, thông tin không đầy đủ, yêu cầu từ chối suy đoán khi thiếu dữ kiện.

### Phân chia tập:
- **Learning / Experience Set:** 100 nhiệm vụ.
- **Unseen Generalization Set:** 60 nhiệm vụ (giữ nguyên quy tắc nhưng đổi ngữ cảnh/dữ kiện mới).
- **Retention / Retest Set:** 40 nhiệm vụ (đo lường độ quên sau khi học các đợt mới).
- **Seeds:** 3 DEV seeds (`140011`, `140017`, `140023`) và 1 Held-out seed (`140099`).

---

## 6. QUY TRÌNH NĂM THÍ NGHIỆM CHÍNH (PHASE 5)

- **EXP-01: Knowledge Acquisition:** Đo Recall, Retrieval Precision, và Tỷ lệ áp dụng đúng (Correct Application Rate).
- **EXP-02: Learning from Mistakes:** Cung cấp phản hồi đúng/sai sau lần thử đầu; đo mức độ giảm lỗi lặp lại (Repeat Error Rate Reduction) trên các tác vụ tương tự.
- **EXP-03: Compositional Generalization:** Huấn luyện kỹ năng $A$ và kỹ năng $B$ riêng lẻ; kiểm tra tỷ lệ thành công khi kết hợp $A + B \rightarrow C$.
- **EXP-04: Continual Learning & Forgetting:** Huấn luyện tuần tự Batch T1 $\rightarrow$ Batch T2 $\rightarrow$ Batch T3; đo lường độ suy giảm điểm của T1 sau khi học T2 và T3.
- **EXP-05: Autonomous Idle Learning:** Đo lường mức tăng điểm Unseen Task của Biến thể D (so với C) và Biến thể E (so với B) sau 1 chu trình tự củng cố có kiểm soát trong sandbox.

---

## 7. KIỂM TOÁN ĐỘC LẬP & CHỐNG RÒ RỈ DỮ LIỆU (PHASE 10)

1. Mọi kết quả đánh giá phải lưu ở cấp độ **từng mẫu (sample-level JSON)**: câu hỏi, công cụ gọi, phản hồi, kết quả thực thi, nhãn ground truth.
2. Script kiểm toán độc lập `evaluate_plastic_llm_independently.py` tính toán lại toàn bộ chỉ số trực tiếp từ kết quả thực thi, không đọc trường `correct` có sẵn.
3. Kiểm tra tự động tính cách ly: Biến thể B, C, F sử dụng database SQLite riêng biệt.
4. Kiểm tra rò rỉ: Prompt không bao giờ chứa token đáp án của tập kiểm tra.
