# 06 — Kết luận thực nghiệm

Ngày: 2026-11-08. **NO-GO cho tăng quy mô với quy tắc hiện tại; PIVOT cho nghiên cứu quy tắc học và giữ kiến thức.** Đây không phải kết luận mọi mạng phát xung đều bất khả thi.

## Bằng chứng

Run chính: [phase2-v1 xác nhận](../../runs/20261108T152637Z-phase2-v1-7a0790/report.md). Run đầu: [phase2-v1 ban đầu](../../runs/20261108T151927Z-phase2-v1-5e495f/report.md). Hai run dùng cùng seed và cho cùng kết quả chất lượng; không tính thành sáu seed độc lập.

| Seed | Frozen recall | Full recall | Full delayed | Full forgetting trung bình |
|---|---:|---:|---:|---:|
| 17 | 28,125% | 100% | 76,5625% | 20,78125 điểm % |
| 29 | 25,78125% | 100% | 100% | 23,90625 điểm % |
| 43 | 25% | 100% | 75% | 22,1875 điểm % |

Full − frozen: +73,70 điểm phần trăm trung bình, CI95% thăm dò [71,875;75]. Full − readout-only: 0 điểm ở cả ba seed. Các synapse tầng ẩn có thay đổi thật, nhưng chưa thấy lợi ích trên bài recall đang có hiệu ứng trần.

Mức forgetting cho phép là trung bình <=3 điểm %, worst<=5 điểm %. Cả ba seed full đều không đạt. Replay128 mẫu và EMA consolidation hiện tại không đủ giữ kiến thức qua20 context.

[Backprop control](../../runs/20261108T152334Z-bptt-control-v1-98ab19/report.md) đạt100% recall và delayed ở cả3 seed; chỉ là đối chứng, không được gọi là local learning. Chưa chạy continual20 chặng cho control này nên không so retention của nó với local rule.

## Tính đúng kỹ thuật

- Save/load qua process mới giữ từng kết quả dự đoán; unit tests kiểm cả tiếp tục học với state đã lưu.
- API learning có development guard và rollback khi suy giảm hoặc khi có exception. Benchmark forgetting cố ý không rollback để đo rule nguyên bản.
- C++ forward/local update đã build và kiểm parity với NumPy ở bốn mode.
- [Runtime smoke](../../runs/20261108T152504Z-runtime-smoke-v1-7fc8b8/report.md): NumPy nhanh hơn native/ctypes ở model64 neuron; tiếp tục giữ NumPy mặc định.
- Lookup identifier:28/28 trên1.000.000 byte ASCII lưu SQLite. Không phải bằng chứng hiểu1M subword context.
- Smoke learn/checkpoint15 giây hoàn tất6546 vòng. **Chưa chạy8 giờ.**

## RAM

Run phase2 ban đầu quan sát RSS process-tree tối đa khoảng79,9 MB và commit khoảng1,332 GB, bao gồm process reload checkpoint. Runtime smoke RSS khoảng48,2 MB, commit khoảng672 MB. Không vượt2 GB ở các lần đo này.

Đây là network nhỏ, không suy ra model ngôn ngữ lớn cũng dưới2 GB. Monitor chưa áp hard cap, chưa đo iGPU hoặc paging chính xác; chưa cấp chứng nhận mọi workload. Số liệu run xác nhận được lưu riêng trong result.json.

## Phán quyết theo giả thuyết

| Giả thuyết | Kết luận |
|---|---|
| Neural state không tăng theo chiều dài lịch sử | Đạt về cấu trúc và kiểm thử bounded state; không bảo đảm giữ nội dung |
| Synapse tự cập nhật và lưu qua restart | Đạt trên prototype nhỏ |
| Học cục bộ giúp liên kết đơn giản | Đạt trên bài synthetic này, gồm contribution của readout |
| Hidden STDP giúp hơn readout-only | Chưa có bằng chứng; bài hiện tại có hiệu ứng trần |
| Học liên tục mà giữ kỹ năng trong ngưỡng | Không đạt với cấu hình/rule này |
| CPU hoạt động dưới2 GB | Quan sát đạt trên workload nhỏ; chưa hard-cap chứng nhận |
| iGPU hoặc native nhanh hơn | iGPU chưa đo; native/ctypes chậm hơn ở smoke |
| Tiếng Việt/Anh đa năng hoặc ngang thương mại | Chưa huấn luyện/đánh giá; không có kết luận |

## Hành động theo kế hoạch đã duyệt

Đã dừng trước huấn luyện ngôn ngữ10M token, model10M–100M tham số và corpus lớn vì gate2 thất bại. Không tải LLM lớn hoặc dùng cloud để che khoảng cách. Bộ mã, kết quả và tài liệu là nền tảng nghiên cứu; chưa có sản phẩm AI giỏi như đích cuối.

Vòng sau cần bài phát triển mới yêu cầu học representation, khảo sát credit assignment và retention với ngân sách cố định, rồi mới đăng ký lại test độc lập. Chưa chạy vòng điều chỉnh thứ hai; không thay ngưỡng sau khi thấy kết quả.
