# SCAR-X ROUND 9d — TỔNG HỢP VÀ KẾ THỪA TÀI LIỆU KHOA HỌC (LITERATURE ACCELERATION)

**Tác giả:** Gemini — Principal SNN Research Scientist & Multi-Agent Coordinator  
**Ngày:** 2026-11-12  
**Nhánh:** `research/gemini/r9c-repair-literature-acceleration-20261112`  

---

## 1. Phương pháp 1: Eligibility Propagation (E-prop)
- **Tài liệu tham chiếu:** Bellec et al., *"A solution to the learning dilemma for recurrent networks of spiking neurons"*, Nature Communications, 2020.
- **Repository:** `https://github.com/IGITUGraz/eligibility_propagation` (Giấy phép: Apache 2.0).
- **Nguyên lý cốt lõi:**
  - E-prop phân tách gradient mất mát toàn cục thành tích của hai thành phần độc lập:
    $$\frac{\partial L}{\partial \theta} = \sum_{j} L_j \cdot e_j^\theta$$
    trong đó $L_j$ là tín hiệu tín dụng (learning signal / credit) được chiếu ngược từ tầng readout tới nơ-ron $j$, và $e_j^\theta$ là vết điều kiện cục bộ (eligibility trace) tích lũy xuôi dòng thời gian tại synapse hoặc tham số $\theta$.
- **Kế thừa vào SCAR-X Round 9d:**
  - Sửa chữa Bug A của Round 9c: Thay vì nhân với tổng vô nghĩa $\sum \text{err} = 0$, ta tính vết điều kiện của từng nơ-ron đối với logit cổng kênh $s_k$:
    $$e_{gate, jk}[t] = \gamma_j e_{gate, jk}[t-1] + (1 - \gamma_j) \psi_j[t] \cdot w_{in, jk} \cdot x_k[t] \cdot \sigma'(s_k)$$
  - Tín hiệu tín dụng chiếu ngược từ readout: $\delta_j = (W_{out}^T \text{err})_j \cdot \text{scale}_j^{-1}$.
  - Cập nhật chính xác cho cổng kênh:
    $$\Delta s_k = \eta_{gate} \sum_{j=1}^H \delta_j \cdot e_{gate, jk}[T]$$
  - Đảm bảo kênh nhiễu (kích hoạt nơ-ron có $\delta_j < 0$) bị đóng cổng ($\Delta s_k < 0$), trong khi kênh kích thích ($\delta_j > 0$) được mở cổng ($\Delta s_k > 0$).
- **Đánh giá bản quyền:** Triển khai thuật toán từ đầu bằng NumPy, không sao chép trực tiếp mã nguồn.

---

## 2. Phương pháp 2: Dynamic Gated Neuron (DGN)
- **Tài liệu tham chiếu:** Qianyi Bai, Haiteng Wang, Qiang Yu, *"A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks"*, arXiv:2509.03281, tháng 9/2025.
- **Nguyên lý cốt lõi:**
  - Nơ-ron LIF truyền thống có điện dẫn rò (leak conductance) cố định, khiến mạng dễ bị xung nhiễu biên độ lớn gây khử cực sai lệch.
  - DGN đưa vào điện dẫn màng động (dynamic conductance) $g_{dyn}[t]$, đóng vai trò như một cơ chế cổng lọc tự nhiên:
    $$\frac{dV}{dt} = -\left( g_l + \sum_i C_i D_i[t] \right) V + \sum_i W_i D_i[t]$$
  - Khi có xung nhiễu đột ngột xuất hiện, điện dẫn động tăng vọt, làm giảm điện trở màng (shunting effect) và tăng tốc độ tiêu tán điện thế ($\tau_{eff}$ ngắn lại), ngăn không cho điện thế màng chạm ngưỡng phát xung rác.
- **Kế thừa vào SCAR-X Round 9d (Candidate B):**
  - Mở rộng phương trình màng của ALIF H48:
    $$g_{dyn, j}[t] = \alpha_g g_{dyn, j}[t-1] + (1 - \alpha_g) c_g \cdot \|x(t)\|_1$$
    $$u_j[t] = (\text{leak} - g_{dyn, j}[t]) u_j[t-1] + I_{syn, j}[t]$$
  - Giữ nguyên cấu trúc H48, fan-in 8, không tăng thêm nơ-ron.
- **Rủi ro khoa học:** Cần cân chỉnh hệ số $c_g$ để shunt triệt tiêu xung nhiễu cô lập nhưng không làm triệt tiêu xung kích thích (Key, Value, Query).

---

## 3. Phương pháp 3: DECOLLE & BrainTrace
- **Tài liệu tham chiếu:**
  - DECOLLE: Kaiser et al., Frontiers in Neuroscience, 2020 (`https://github.com/nmi-lab/decolle-public`, GPL-3.0).
  - BrainTrace: Chao et al., 2024 (`https://github.com/chaobrain/braintrace`).
- **Bài học kế thừa:**
  - DECOLLE cung cấp cơ sở cho việc sử dụng hàm mất mát cục bộ theo từng tầng để huấn luyện nơ-ron mà không cần BPTT toàn mạng.
  - BrainTrace cung cấp phương pháp quản lý bộ nhớ vết điều kiện (online traces) hiệu quả với chi phí RAM cực thấp (< 50 MB).
- **Ràng buộc pháp lý:** DECOLLE sử dụng giấy phép GPL-3.0. SCAR-X tuyệt đối không import hay sao chép mã nguồn GPL vào repository; chỉ kế thừa nguyên lý toán học độc lập.
