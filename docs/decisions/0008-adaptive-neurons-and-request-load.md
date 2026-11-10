# ADR0008 — Neuron linh động, tải đầu vào từng lượt, vòng cải thiện nhỏ

2026-11-09. Người dùng yêu cầu cải thiện năng lực, hiển thị thông tin đang nạp và
số neuron; nhắc rõ **48 neuron không cố định, có thể cấp nhiều nếu task cần**.
Đây là phê duyệt triển khai prototype capacity theo tài liệu đã audit. Thay hạn
chế sản phẩm của ADR0006; không sửa baseline/protocol nghiên cứu lịch sử.

## Đã triển khai, còn thử nghiệm

- Một core và ID G1; khởi đầu48, cấp thêm trong trần thử nghiệm128. Topology và
  tensor mở rộng tại chỗ; giữ IDneuron và weights/indices/state cũ. Newreadout
  columns khởi tạo0: cấp thêm chưa tạo ra kỹ năng. Newneuron cần học cục bộ.
- Kernel chỉ chạy hàng input/recurrent và cột readout đang chọn. Neuron ngủ giữ
  state, lazydecay theo neural ticks khi quay lại, không bắn lại spike cũ.
  Weights vẫn resident; không gọi đây là giảmRAM hoặc tối ưuCPU đã chứng nhận.
- Budget mặc định dựa lượng input/modality, có override48–128. Đây là heuristic
  workload, **không** là bộ đo độ khó/thiếu capacity đã xác nhận. Không grow vì
  trả lời sai.128 là trần kiểm thử hiện tại, không là định nghĩa trí thông minh.
- Foreground bảo thủ giữ toàn bộ neuron có đường ra đã kết nối; không tắt bớt
  kiến thức mới chỉ vì input ngắn. Đây không phải usefulness/importance học được;
  chưa có pruning hoặc khả năng phát hiện neuron hiếm hữu ích bằng can thiệp.
- Save/restore gồm capacity RNG, active/allocated/max, neuronclocks/IDs và events.
  Growth tạo checkpoint; hồi phục có thể quay lại topology cũ, giữ lịch sử.
- Lời dạy rõ ràng được thêm outputtokens trong giới hạn256. Token IDs cũ, input
  encoding/readoutdigit offsets và recurrent topology được chuyển đúng. Không
  auto-học từ câu tự sinh; nội dung lời sửa vẫn chưa xác minh độc lập.
- Hermes hiển thị input chars/UTF8bytes/images, vectorchannels/bytes/nonzero,
  neuron đang dùng/đã cấp/trần. Backend telemetry phải khớp requestID. Lịch sử,
  retrieved episodes và tools đều0trên wire; không giả context/memorycapacity.

## Kiểm chứng và giới hạn

Engineering tests kiểm oldweights, inactivecompute/state, growth/rollback,
newtokenresume, requestload và bảo vệ readout đã nối. Bài1+1→2 là trainingpoint
trên clone, không là toán tổng quát hoặc được đưa thành đáp ántra khi inference.
Timing48vs128 chỉ là phép đo kernel nhỏ với overheadPython, không suy điện năng.

Capability screen riêng giữ48,3seed,3condition,60epoch; selection bằngdev, final
confirmation mở sau. Đổi encoder là cải thiện frontendfeature giả thuyết, không
được gán là hiddenrepresentation. Không tăng lớn để che learningrule yếu.
MasterP1/P2, forgetting≤3mean/≤5worst,2GB,CPUfirst,noBPTT vẫn giữ nguyên.
Rewiring, pruning, usefulnessimportance và tự tổ chức chưa triển khai.
