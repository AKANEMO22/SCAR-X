# Hybrid RC2 — ưu tiên sản phẩm theo yêu cầu mới

User: “bỏ cái tets 8h đi tập trung xử lý nhanh ra sản phẩm đi”.
Không mở lại lượt dài, model, dataset hoặc cơ chế học mới.

## Hủy lượt 8 giờ, giữ kết quả

`scripts/cancel_hybrid_endurance_for_release.py` xác minh PID/create-time/
cmdpath và 36 source hashes, ghi yêu cầu rồi graceful shutdown đúng gateway
32012 dưới maintenance lease. Runner nhận owner exit, tự chạy finally, ghi
result và cleanup. Không force kill hoặc sửa protocol/runner đang chạy.

Kết quả `32cd13`: 11.268,281 giây, 38 chu kỳ, 7 checkpoint, worker 46940 thực
sự terminal, không private worker sót; source còn khớp trước activation.
Result gốc giữ `ENDURANCE_INTERRUPTED_OR_FAILED`. Cancellation audit thêm
`USER_CANCELLED_NOT_EIGHT_HOUR_PASS`; không gọi là test thành công hoặc đạt
8 giờ. Base-profile 8 giờ cũ và các nghiên cứu khác giữ nguyên.

## Phạm vi RC2

Giữ nguyên 48 payload files và hai candidate P0. Chỉ đổi installer/docs để
chấp nhận yêu cầu hủy đã có custody đầy đủ, vẫn chặn worker sống, source sai,
receipt/hash/identity sai và mọi cách biến cancellation thành endurance PASS.
31 safety/P0 checks qua; regression trên payload RC1 giữ làm bằng chứng của
code đã kiểm, RC2 cần đối chiếu hash/các kiểm tra sau activation.

Checklist tiếp theo: build/verify RC2 → backup/CAS/activation → kiểm startup/
CUDA/state → nghiệm thu chat Hermes, workspace/tool/compaction/search/media.
Ghi rõ lỗi hoặc khả năng chưa đo; đây là RC, không phải lifelong hay production PASS.
