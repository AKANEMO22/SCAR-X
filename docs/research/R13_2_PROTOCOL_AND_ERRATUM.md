# SCAR-X ROUND 13.2 — PRE-REGISTERED PROTOCOL & R13.1 ERRATUM AUDIT

**Author:** Gemini (Independent Scientific Auditor & SCAR-X Researcher)  
**Date:** 2026-11-13  
**Status:** PRE-REGISTERED & LOCKED PRIOR TO TEST EVALUATION  
**Branch:** `research/gemini/r13-2-independent-composition-fair-llm`  
**Base Commit:** `ffddbcaea23c1d7b22ffdd0b5715f3939541fec1`  

---

## 1. ERRATUM R13.1 — BẢNG KIỂM TOÁN TẤT CẢ CÁC SAI SÓT ĐÃ XÁC NHẬN

Dưới đây là bảng phân tích toàn diện 7 sai sót kỹ thuật và phương pháp luận trong Round 13.1:

| ID | Lỗi / Khiếm khuyết | Tệp & Dòng mã R13.1 | Mức độ nghiêm trọng | Tác động thực nghiệm trong R13.1 | Bản sửa chữa bắt buộc trong R13.2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ERR-01** | **Giám sát gián tiếp bài C (Supervised C-leakage trong SNN)** | `scripts/run_r13_1_comprehensive_benchmark.py`: lines 320-332 | **CRITICAL (Nghiêm trọng nhất)** | Module B được train bằng dòng xung $mod\_a(X)$ với nhãn đích $Z$. Đây là supervised composition, **không phải zero-shot composition độc lập**. Báo cáo 100% C của R13.1 là không hợp lệ. | **Tách biệt hoàn toàn:** Module B chỉ được train bằng mẫu $Y$ và nhãn $Z$. Không bao giờ nhìn thấy $X$ hay nhãn $Z$ đi cùng với $X$. |
| **ERR-02** | **Rò rỉ đáp án đích vào Prompt của SmolLM2** | `src/scarx/evaluation/r13_1_llm_benchmarks.py`: line 119 | **CRITICAL** | `Answer (only output the symbol like Z{target_z}): Z` đã chèn trực tiếp đáp án đúng `target_z` vào câu lệnh hướng dẫn của prompt. | Loại bỏ hoàn toàn `target_z` khỏi prompt. Prompt chuẩn: `Answer: Z` hoặc chỉ thị định dạng chung không chứa token đáp án. |
| **ERR-03** | **Chấm điểm bằng substring match lỏng lẻo** | `src/scarx/evaluation/r13_1_llm_benchmarks.py`: line 136 | **HIGH** | `str(target_z) in gen_text` ghi nhận đúng nếu chữ số xuất hiện ở bất cứ đâu, gây sai lệch thống kê. | Sử dụng bộ parser chính xác trích xuất đúng token ký hiệu đầu tiên sau nhãn Answer (Exact Token Match). |
| **ERR-04** | **Bất công với Scratch Transformer 20M** | `scripts/run_r13_1_comprehensive_benchmark.py`: lines 225-240 | **HIGH** | Model được học $X \rightarrow Y$ và $Y \rightarrow Z$, nhưng bị bắt trả lời $X \rightarrow Z$ trong 1 bước duy nhất (1-hop), dẫn đến 0%. | Triển khai cơ chế sinh 2 bước công bằng: $X \rightarrow$ sinh $\hat{Y}$, sau đó lấy $\hat{Y} \rightarrow$ sinh $\hat{Z}$ (Two-step Autoregressive Composition). |
| **ERR-05** | **Cắt cụt dữ kiện prompt của LLM ở $K=32, 64$** | `scripts/run_r13_1_comprehensive_benchmark.py`: line 430 | **HIGH** | Prompt chỉ đưa tối đa 16 quan hệ (`max_symbols_in_prompt=min(k_syms, 16)`), nhưng lại hỏi các ký hiệu ngoài 16 ký hiệu này. | Đưa toàn bộ $K$ quan hệ vào prompt (nếu nằm trong context window) hoặc chỉ đánh giá các truy vấn có dữ kiện đầy đủ trong context. |
| **ERR-06** | **Ghi nhận 0% cho Transformer 20M ở 3 bước mà không đo** | `scripts/run_r13_1_comprehensive_benchmark.py`: line 618 | **MEDIUM** | Ghi cứng `c_3step_acc: 0.0` thay vì chạy quy trình sinh 3 bước thực tế ($X \rightarrow \hat{Y} \rightarrow \hat{Z} \rightarrow \hat{W}$). | Chạy thực tế benchmark 3-step autoregressive generation cho Transformer 20M. |
| **ERR-07** | **Ghi cứng số liệu RAM và tính sai số tham số SNN** | `scripts/run_r13_1_comprehensive_benchmark.py`: lines 600, 606 | **MEDIUM** | RAM ghi cứng 38.3 MB; tham số ghi cứng 4,640 trong khi kích thước ma trận thực tế của $H=48, D=8, K=8, O=8$ là $1,160$ tham số/module (2,320 cho 2 module). | Tính toán động toàn bộ tham số từ shape thực tế (`w_in`, `w_rec`, `w_out`, `bias`), đếm nonzero synapses, và đo RSS sống qua `psutil`. |

---

## 2. GIAO THỨC ĐĂNG KÝ TRƯỚC (PRE-REGISTERED PROTOCOL) CHO R13.2

### A. Định nghĩa bài toán
- **Task A:** Ánh xạ từ không gian tín hiệu $X$ sang nhãn $Y \in \{0, \dots, K-1\}$.
- **Task B:** Ánh xạ từ không gian tín hiệu $Y$ sang nhãn $Z \in \{0, \dots, K-1\}$.
- **Task C (Bài toán kết hợp):** Nhận tín hiệu $X$, tìm đúng nhãn $Z = B(A(X))$ mà **chưa từng được huấn luyện trực tiếp trên cặp $(X, Z)$**.

### B. Quy tắc huấn luyện Module A và Module B độc lập
1. **Module A:** Khởi tạo $Mod_A$. Huấn luyện trên tập dữ liệu $\{(X_i, Y_i)\}$. Lưu toàn bộ engram $Mod_A$ xuống SSD (`engram_a.engram.json`).
2. **Module B:** Khởi tạo $Mod_B$. Huấn luyện trên tập dữ liệu $\{(Y_j, Z_j)\}$. Lưu toàn bộ engram $Mod_B$ xuống SSD (`engram_b.engram.json`).
3. **Cấm tuyệt đối:** Không cho $Mod_B$ thấy bất kỳ mẫu $X$ nào; không tính loss giữa đầu ra $Mod_B$ và nhãn $Z$ khi đầu vào là $X$.

### C. Hai phương án ghép nối Module trong Task C
Vì $Mod_A$ phát ra chuỗi xung ẩn $s_A(t) \in \{0, 1\}^{H_A}$ trong khi $Mod_B$ được huấn luyện trên mẫu tín hiệu $Y \in \mathbb{R}^{T \times D_{in}}$, ta khảo sát hai cơ chế ghép nối:
1. **Phương án 1 — Fixed Direct Spike Router:**
   - Dùng ma trận cố định $W_{\text{inter}} \in \mathbb{R}^{D_{in} \times H_A}$ chiếu trực tiếp xung $s_A(t)$ thành dòng điện vào $Mod_B$.
   - **Tiêu chuẩn khoa học:** Nếu phương án này thất bại vì $Mod_B$ chưa từng thấy dạng xung $W_{\text{inter}} s_A(t)$, ta phải ghi nhận trung thực là **FAIL** (không có zero-shot spike compatibility tự nhiên).
2. **Phương án 2 — Learned Neural Interface (Huấn luyện căn chỉnh biểu diễn $Y$ thuần túy):**
   - Một giao diện thần kinh nhỏ $I_{\text{align}}$ được huấn luyện để biến đổi chuỗi xung $s_A(t)$ (khi A nhận $X$) thành biểu diễn tương thích với đầu vào của $B$ cho ký hiệu $Y$.
   - **RÀNG BUỘC CỐT LÕI:** $I_{\text{align}}$ CHỈ được huấn luyện trên mục tiêu tái tạo biểu diễn $Y$ (Representation Alignment). **TUYỆT ĐỐI KHÔNG DÙNG NHÃN $Z$ HOẶC LOSS CỦA BÀI C ĐỂ HUẤN LUYỆN $I_{\text{align}}$**.
3. **Phương án 3 — Symbolic Router (Trần đối chứng Oracle):**
   - Decode nhãn $\hat{Y} = \arg\max(Mod_A(X))$, lấy mẫu biểu diễn chuẩn của $\hat{Y}$ đưa vào $Mod_B \rightarrow \hat{Z}$. Đây là trần lý thuyết của việc kết hợp hai module độc lập.

### D. Tiêu chí phán quyết (Verdict Criteria)
- Nếu Phương án 1 (Fixed Spike) hoặc Phương án 2 (Learned Y-alignment) đạt độ chính xác $C > 80\%$ mà không dùng nhãn $Z$ trong quá trình ghép:
  $\rightarrow$ `DIRECT_SPIKE_COMPOSITION_DEV_PASS` hoặc `LEARNED_ROUTING_COMPOSITION_DEV_PASS`.
- Nếu chỉ có Symbolic Router hoạt động được, còn các cơ chế spike thuần thất bại:
  $\rightarrow$ `SYMBOLIC_COMPOSITION_ONLY`.
- Nếu cả ba đều không thể hiện được tính độc lập khi hoán đổi tri thức (Phép thử Causal Decisive Test):
  $\rightarrow$ `NO_INDEPENDENT_COMPOSITION`.

### E. Quy tắc Phép thử Quyết định (Causal Decisive Test)
1. Dạy $A: X_1 \rightarrow Y_3$, $B: Y_3 \rightarrow Z_5$. Hệ thống trả lời $Z_5$.
2. Đổi riêng $A$: Dạy $A: X_1 \rightarrow Y_7$. Giữ nguyên $B$ (đã biết $Y_7 \rightarrow Z_2$).
3. Kiểm tra hệ thống có tự động đổi câu trả lời thành $Z_2$ hay không mà **không sửa $B$ và không sửa router**.
4. Tiếp tục đổi riêng $B$: Dạy $B: Y_7 \rightarrow Z_9$. Kiểm tra hệ thống có tự động đổi sang $Z_9$ hay không.
