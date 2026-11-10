# Tăng tải kiến thức và đối chiếu Qwen4B

## Kết quả chính

| Cơ chế | Bài đã nạp/order | Đúng nguyên cách hỏi | Đúng cách hỏi mới | Quên kỹ năng cũ TB | Xấu nhất |
|---|---:|---:|---:|---:|---:|
| frozen | [128] | 0.00% | 0.00% | 0.00pp | 0.00pp |
| full | [16] | 14.58% | 0.00% | 25.00pp | 100.00pp |
| readout | [16] | 20.83% | 0.00% | 25.00pp | 100.00pp |
| full_replay | [16] | 31.25% | 0.00% | 25.00pp | 100.00pp |
| guarded | [16] | 0.00% | 0.00% | 8.33pp | 50.00pp |

Qwen3-4B-Instruct-2507 Q4_K_M: 160 câu, exact 100.00%; server đã tắt: True.
Cùng160 chuỗi câu hỏi số học trong phép đối chiếu phụ; không suy rộng thành toàn bộ kiến thức4B. Qwen có pretrained weights và RAM cao hơn; không là matched-memory comparison.

## Nhớ lâu và ablation

Không có ví dụ mới được học hữu ích theo cách hỏi chưa gặp, nên primary retention không xác định. Bài ghi nhớ nguyên văn dưới đây là phân tích thăm dò thêm sau kết quả, không thay tiêu chí thành công:

| Cơ chế | Nhớ nguyên văn trước nhiễu | Giữ sau8bài nhiễu | Giữ khi replay |
|---|---:|---:|---:|
| frozen | 0 | 0 | 0 |
| full | 7 | 0 | 0 |
| readout | 10 | 0 | 0 |
| full_replay | 15 | 1 | 3 |
| guarded | 0 | 0 | 0 |

Cold process restart khớp;10kquiet ticks không kiểm chứng nhớ ngày/tháng. Các nhóm có3orderseed từ1initialization; CI chỉ phản ánh biến thiên thứ tự. Hidden/readout đều fail paraphrase, chưa chứng minh representation.

## Audit và giới hạn

Original old_mean_forgetting_pp was NET overall drop. This audit averages per-family forgetting from maximum past performance; thresholds unchanged, no stopping decisions affected.
V1 signed arithmetic abort do leading minus bị decoder mask; v1r1 chỉ đáp án không âm. Raw runs/protocol giữ nguyên. Core live và private weights không thay; không mở pretraining/model lớn.

**Quyết định: PIVOT_NO_SCALE.** Sửa credit assignment/representation/readout stability trước khi tăng lượng data. Không chọn candidate để thay model người dùng.
