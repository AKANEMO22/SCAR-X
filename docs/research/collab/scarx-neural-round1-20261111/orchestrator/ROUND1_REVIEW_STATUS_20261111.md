# SCAR-X — chốt review các nhánh ngày 11/11/2026

Đã đối chiếu HEAD local với các nhánh đã push. Tất cả lượt chạy dưới đây đã kết thúc và có báo cáo; “đã chạy xong” chưa đồng nghĩa “được đưa learner vào G1”. `ponytail` không có trong danh sách tool phiên này; không giả lập plugin.

| Nhánh | Commit đã đọc | Quyết định Codex | Gửi AI việc gì tiếp |
|---|---|---|---|
| Grok slow trace | `4b93593` (kết quả sửa ở `d509878`) | Duyệt lưu nghiên cứu kèm patch loader đã review; **NO-GO** cho áp dụng learner | Nhánh nghiên cứu gốc đã xong; không cần train thêm. [Review](rereview-grok-4b93593.md) |
| Antigravity Gemini DC-ALIF | `81641f7` | **REQUEST_CHANGES**, có tín hiệu representation đáng nghiên cứu, chưa đạt registered GO/retention | Sửa cách diễn giải protocol/ablation, dẫn chứng không tồn tại và tách BPTT control. [Review](review-antigravity-gemini-81641f7.md) |
| DeepSeek pair-binding | `02be6e9` | **REQUEST_CHANGES** sau bản trả lời Grok; screen vẫn âm tính | Bỏ kết luận bất khả thi đặt cứng, xuất summary sửa riêng, đưa diagnostic vào Git, sửa nghĩa frozen. [Review](rereview-deepseek-02be6e9.md) |
| MTB mang nhãn antigravity, do phiên DeepSeek viết | `5e39c8f` | **REQUEST_CHANGES**; screen PIVOT | Sửa mất chế độ neuron khi load, nhãn learner và cross-review Grok dùng kết quả cũ/sai ngưỡng. [Review](review-mtb-5e39c8f.md) |

MTB và DeepSeek pair-binding do cùng phiên DSH thực hiện. Chúng không phải hai phiếu xác nhận độc lập; nhánh Gemini mới là Antigravity riêng.

## Kết quả quan trọng

- Grok: chưa qua ngưỡng đóng góp hidden đã đăng ký. Đã tái lập bản sửa trong lượt review trước; không cần huấn luyện lại lần này.
- Gemini: strict_binding **98,78%** so hidden frozen/head học **75,22%**; refit probe tăng **24,19 điểm phần trăm**. Đây là bằng chứng thăm dò có ích về representation trên bài tổng hợp, dưới sai khác protocol đã công bố. Bài order lại giảm **18,48 điểm**, P2 quên trung bình **21,89 điểm**, nên chưa đạt học liên tục.
- DeepSeek: pair-binding **48,96%**, readout-only **48,57%**; chưa chứng minh hidden có lợi. Bản sửa đã nhận đúng lỗi nhãn scalar/proposal30 và phạm vi sampler nhưng vẫn còn các finding trong review.
- MTB: lợi thế hidden so hidden frozen chỉ **+0,43 điểm**, CI95% **[−0,91; +1,87]** trên phép gộp đã công bố. Chưa chứng minh lợi thế; checkpoint deterministic còn bị chuyển sang stochastic khi load.

## Đã kiểm chứng trực tiếp

Run chứng cứ: `runs/20261111T013650Z-round1-handoff-review-v2-cb9d2d`.
Codex tái hiện lỗi checkpoint MTB, tính lại summary DeepSeek, tính lại mean/contrast từ raw rows Gemini và chạy **27/27 test kỹ thuật** của Gemini trong detached checkout riêng. Có thao tác cập nhật nhỏ trong fixture kiểm thử; không chạy lại campaign huấn luyện, không mở seed mới, không chỉnh learner hoặc ngưỡng thành công. Bộ kiểm tra mới ở dưới 2 GB theo lấy mẫu process tree.

Không nhánh nào được duyệt thay learner đang phục vụ. Main chưa merge/push; G1/Hermes và trọng số live không bị sửa. Các kết quả và protocol lịch sử giữ nguyên. Bước tiếp theo cho các AI là **sửa review có giới hạn rồi gửi commit**, không tự mở thêm lượt train.
