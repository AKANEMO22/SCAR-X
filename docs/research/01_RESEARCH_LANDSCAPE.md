# 01 — Tổng quan nghiên cứu và tính mới

Ngày rà soát: 2026-11-08. Phân biệt bằng chứng trong paper với số đo trên máy này. Chưa tái lập các LLM bên ngoài; không chuyển số FLOPs hay năng lượng neuromorphic thành tốc độ CPU.

| Công trình | Liên quan và giới hạn | Code/giấy phép kiểm tra | Ý nghĩa với SCAR-X |
|---|---|---|---|
| [Mamba-2](https://arxiv.org/abs/2405.21060) | State-space recurrent, trạng thái suy luận hữu hạn; huấn luyện không phải STDP | [Mamba](https://github.com/state-spaces/mamba), Apache-2.0 ở repo | Đối chứng recurrence, chưa chứng minh CPU/local learning |
| [RWKV-7](https://arxiv.org/abs/2503.14456) | Mô hình ngôn ngữ recurrent có state; không bảo đảm nhớ chính xác vô hạn | [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), Apache-2.0 ở repo; license checkpoint cần kiểm riêng | Baseline ngôn ngữ tương lai, không nhập làm core học cục bộ |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | Gating và delta update cho memory; có các cấu hình hybrid | [NVlabs](https://github.com/NVlabs/GatedDeltaNet), NVIDIA Source Code License-NC | Không coi code là permissive; chưa tích hợp |
| [Titans](https://research.google/pubs/titans-learning-to-memorize-at-test-time-2/) | Neural memory thích nghi tại test time; khác học STDP cục bộ | Paper đã kiểm; chưa xác nhận code/checkpoint/license đủ để tái lập | Trùng ý tưởng memory thích nghi, không chứng minh yêu cầu sinh học |
| [AHN](https://proceedings.mlr.press/v306/fang26w.html) | Attention cửa sổ và recurrent compressed memory; base model đóng băng khi học module | Proceedings có đường dẫn software; chưa audit license/checkpoint | Trùng ý tưởng phân tầng. Giảm cache không đồng nghĩa giảm tương ứng tổng RAM |
| [SpikeGPT](https://arxiv.org/abs/2302.13939) | Ngôn ngữ với kích hoạt spike; có huấn luyện backprop | [Repo](https://github.com/ridgerchu/SpikeGPT), BSD-2-Clause; archived 2026-06-02 | Là bằng chứng SNN có thể làm language modeling, không phải bằng chứng học cục bộ |
| [SpikeLM](https://arxiv.org/abs/2406.03287) | Spike hai chiều và mã hóa phong phú hơn nhị phân | Paper dẫn code; license chưa audit | Nhắc rằng mã hóa spike có thể làm mất ngữ nghĩa |
| [SDLLM](https://arxiv.org/abs/2604.16475) | Mã hóa spike nhằm giảm chi phí inference; hướng phần cứng sự kiện | Preprint đã kiểm; chưa audit checkpoint/license | Không suy ra CPU 2 GB hoặc học liên tục từ kết quả inference |
| [E-prop](https://www.nature.com/articles/s41467-020-17236-y) | Eligibility trace kết hợp learning signal để xấp xỉ gradient online | Bài nghiên cứu; chưa dùng code bên ngoài | Không BPTT chưa đủ để gọi mọi đường truyền tín hiệu là thuần cục bộ |
| [pp-prop](https://www.nature.com/articles/s41467-026-68453-w) | Phân rã pre/post traces để giảm memory phụ của online learning | Nội dung lập chỉ mục đọc được, truy cập toàn trang lỗi; chưa tái lập | Đối tượng rà soát vòng sau, chưa khẳng định tuân thủ no-backprop |
| [BitNet](https://arxiv.org/abs/2504.12285) | Native low-bit language modeling, có CPU inference | [Runtime](https://github.com/microsoft/BitNet), MIT; checkpoint kiểm riêng nếu tải | Baseline ít RAM tương lai; không phải SNN học cục bộ |
| [Mamba-3](https://arxiv.org/abs/2603.15569) | Nghiên cứu 2026 cải thiện state tracking và hiệu quả inference | Repo Mamba có liệt kê | Cập nhật landscape, tránh chỉ so với thế hệ 2024 |

Repo RWKV cũng đã có [ghi chú RWKV-8](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md); chưa đánh giá độc lập hiệu quả của phiên bản này.

## Đánh giá novelty

Chưa có cơ sở gọi việc ghép spike, memory và replay là kiến trúc mới. Đóng góp có thể kiểm chứng là một quy tắc local plasticity với ngân sách memory xác định, runtime CPU tái lập và lợi thế giữ kiến thức so với đối chứng. Prototype hiện chưa chứng minh lợi thế đó. Không có code bên thứ ba từ các repo trên được sao chép vào neural core.

## Mức bằng chứng

- **Đã có bằng chứng công bố trong phạm vi riêng:** recurrent state hữu hạn; language modeling dùng spike; CPU low-bit inference.
- **Bằng chứng thực nghiệm của dự án:** liên kết nhân tạo học được, checkpoint tái lập; xem báo cáo 06.
- **Giả thuyết:** local STDP tăng năng lực ngôn ngữ, vừa học vừa giữ kỹ năng, lợi ích khi tăng quy mô.
- **Chưa chứng minh khả thi:** ngang AI thương mại mạnh với học cục bộ hoàn toàn và 2 GB.
