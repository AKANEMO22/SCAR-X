# 04 — Chỉ số và quy trình chấm

Cập nhật docs 2026-11-09. [Master plan](../RESEARCH_PLAN.md) là nguồn roadmap và quyết định. Các ngưỡng/protocol đã chạy giữ nguyên; các phép đo mới dưới đây là đề xuất cần đăng ký trước Phase2.2. Không phát sinh benchmark mới.

**Product-first:** ngưỡng nghiên cứu ở đây không chặn MVP. Acceptance UI/learning/persistence được ghi riêng trong [MVP protocol](../../experiments/mvp-v01-acceptance.md); before/after có thể đo trực tiếp trong app, nhưng không được gọi là đã đạt representation/retentiongate. Lần tích hợp V0.1 tạo demo sản phẩm mới, không chấm lại số liệu Phase2/2.1.

## Bộ đo hiện hành cho lifelong learning

| Nhóm | Cách đo | Điều kiện diễn giải |
|---|---|---|
| Learning speed | Learning curve, số mẫu/update/CPU seconds tới ngưỡng | Không đạt ngưỡng là censored/fail, không thay bằng thời gian0 |
| Acquisition và final quality | Điểm ngay sau học và cuối stream, từng kỹ năng/seed | Phải đủ chất lượng; không dùng quên ít của model yếu làm thắng lợi |
| Retention | Max accuracy sau khi học trừ final; mean/worst mỗi seed | Gate mean≤3 điểm phần trăm, worst≤5; không nới sau kết quả |
| Backward transfer | Final accuracy kỹ năng cũ trừ ngay sau lần học nó | Có dấu; không phải cùng công thức với forgetting |
| Forward transfer | Lợi ích ở kỹ năng mới so fresh model cùng budget | Tách zero-shot/ít mẫu và hiệu ứng đã thấy cùng ví dụ |
| Hidden representation | Freeze hidden, fit head mới cùng budget; so initial/readout-only/random modulation | Báo cả end-to-end; weight drift không đủ |
| Generalization | Tổ hợp/tình huống giữ kín, gồm near/far transfer | Chống shortcut; noise mới của mẫu cũ chưa đủ |
| Plasticity reserve | Samples-to-threshold sau tuổi stream so fresh; feature rank hỗ trợ | Nếu fresh không đạt ngưỡng thì tỷ lệ không xác định; một probe không chứng minh học vô hạn |
| Memory reconstruction | Raw episodic bytes/source/hash và neural recall chấm riêng | Khi gọi là kỹ năng neural phải retrieval-off, reset/restart |
| Consolidation benefit | Before/after so cùng compute từ repeat-new-data, replay và protection controls | Không gán thêm compute hoặc SSD lookup cho học synapse |
| Feedback quality | Nhãn chấp nhận đúng/sai, noise retained, false rejection, provenance | `verified=true` không thay verifier và bằng chứng |
| Efficiency | Update/time/RAM/I/O để học và giữ kỹ năng đúng | Báo component và process-tree peak, gồm checkpoint/rollback |

### Gate Phase 2.1 kế thừa, không chấm lại lịch sử

Theo [protocol gốc](../../experiments/phase21-v1.md): representation lower CI95% gain≥5 điểm phần trăm trên ít nhất2/3 bài hợp lệ, end-to-end≥65%, lợi ích so readout-only dương và kiểm định hiệu chỉnh đạt; acquisition/final phải đủ chất lượng; retention mean≤3/worst≤5 trên mọi seed xác nhận. Cải thiện tương đối cần lower CI95% giảm forgetting≥5 điểm và acquisition không kém quá3 điểm trong cùng budget. Chi tiết dùng đúng endpoint từng protocol, không tự gộp thành một điểm.

Kết quả Phase2.1 **chưa đạt** các gate này. Binding cũ có shortcut, strict binding audit là run riêng; không dùng điểm cũ làm bằng chứng skill.

### Endpoint mới đề xuất cho Phase2.2

Transfer/generalization gain lower CI95%≥5 điểm so matched baseline; plasticity samples-to-threshold≤1,25×fresh nếu fresh đạt ngưỡng. Đây là **tiêu chí dự kiến chưa đăng ký để chạy**; phải khóa cùng dữ liệu/seed/contrast trước experiment. Không dùng chúng để đổi phán quyết Phase2/2.1.

Bootstrap theo seed độc lập, không theo từng sample hoặc từng condition ghép cặp. Dùng tối thiểu8 seed xác nhận mới, primary endpoints và Holm hoặc quy trình correction cụ thể khóa trước test. Guard phát triển không phải test; rollback acceptance rate phải công bố. So cả equal-update và equal-CPU-time, cùng buffer bytes; báo khác biệt forward/stochasticity khi có.

## Điều kiện ưu tiên

Tối thiểu hóa RAM với chất lượng và latency đạt ngưỡng. Báo vector chất lượng theo nhóm trước khi tính một điểm trung bình. Không coi số tham số, recall, số spike hoặc tốc độ byte là trí thông minh.

ACI đề xuất gốc là `0.30T+0.25L+0.20R+0.15M+0.10A`. Các thành phần chưa có bộ đo đầy đủ; **không công bố ACI số cho prototype này**. Các nhóm có thể tương quan, trọng số có tính chủ quan, điểm tổng hợp có thể che tiếng Việt hoặc suy luận yếu.

CEI gốc `ACI*sqrt(E_ref/E_test)*sqrt(T_ref/T_test)` không trực tiếp phản ánh RAM và có thể đếm ảnh hưởng thời gian hai lần vì năng lượng phụ thuộc thời gian. Chỉ là chỉ số phụ sau quality gate, cùng phần cứng/workload. **Không có joule đo được → CEI để null.**

## Benchmark Phase 2 — định nghĩa lịch sử

- Accuracy từng episode, grouped theo seed và mode; báo noise test riêng và delay dài hơn riêng.
- Gain = full − frozen; hidden contribution = full − readout-only. Paired seed-cluster bootstrap 5000 mẫu, seed thống kê 919.
- Forgetting task j = max accuracy task j sau khi đã học − final accuracy task j. Báo mean và worst; frozen không thể thắng chỉ nhờ không quên.
- Gate: cận dưới95 gain >=10pp, hidden gain >0, delay >chance, mean forgetting <=3pp và worst<=5pp mọi seed, persistence đạt, RAM quan sát trong trần.
- Restart: load trong process mới, reset trạng thái, không retrieval, so từng correct flag với trước khi save. Unit test còn kiểm continuation của toàn bộ weights/traces.
- Chỉ 3 seed nên CI yếu. Run lặp cùng seed là kiểm tái lập, không phải 3 seed mới.

## RAM/thời gian

Đơn vị byte decimal, trần 2e9. Báo sampled process-tree RSS/commit (có thể đếm trùng shared pages), high-water working set và pagefile-commit của process chính. Không diễn giải pagefile-commit là bytes đã swap. Monitoring 10 ms có thể bỏ sót spike ngắn ở process con, không phải enforcement.

Text smoke báo byte tokens/s, ký tự hiển thị/s, p95 prefill+first byte và p50 toàn lượt. Model ngẫu nhiên, output UTF-8 sai được đánh dấu; metric này chỉ phục vụ overhead/runtime. iGPU, điện năng, 8h chưa xác nhận.

## Đánh giá đa năng — mục tiêu dài hạn chưa mở

Giữ riêng 6 nhóm Việt/Anh/chỉ dẫn/kiến thức/toán/code, tối thiểu100 bài/nhóm và nguồn/giấy phép/split có hash. Test code bằng execution trong môi trường tách biệt; văn bản chấm mù theo rubric; không dùng một LLM judge làm chân lý duy nhất. Không dựng 600 câu tự sinh dễ rồi gọi là benchmark mạnh.

Thương mại: cùng bài, công cụ, ngân sách và phiên bản; yêu cầu cận dưới chênh lệch >=−5pp ở từng nhóm. Điểm công bố không thay matched evaluation. Chưa thu thập đầu ra tham chiếu, chưa có kết luận ngang thương mại.

Các mục này không mở phê duyệt language training hoặc tăng model. Trong hướng hiện tại, năng lực thích nghi/giữ kỹ năng/representation là trọng tâm; context length và ACI/CEI không thay thế chúng. Xem [roadmap và resource budget](../RESEARCH_PLAN.md#7-experiment-roadmap).
