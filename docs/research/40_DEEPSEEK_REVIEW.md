# Đối chiếu các cập nhật mới của DeepSeek

2026-11-10. **Hoàn tất đối chiếu và chẩn đoán trên bản sao; PIVOT cơ chế học,
không tăng quy mô.** [Master plan](../RESEARCH_PLAN.md) là kế hoạch duy nhất.
[Protocol đăng ký trước phép đo](../../experiments/deepseek-product-review-v1.md).

## Phạm vi và nguồn bằng chứng

Đọc thay đổi từ Git baseline `8337d60` tới `27b6c10`, báo cáo sản phẩm A–I,
STATUS vòng 20–24, audit 39 và bản đề xuất Readout Protection mới. Trong lúc
review, một phiên khác commit tiếp thành `a836755` / `f2cb499`, bao gồm một phần
bản sửa đang làm. Review không init Git, commit, push hoặc hoàn tác thay đổi
của phiên đó. Run đo độc lập ghi HEAD thực tế `f2cb499` và có source snapshot.

- [Receipt công khai](../../artifacts/deepseek-product-review.json).
- [24 can thiệp và đầu ra thật](../../runs/20261110T124259Z-deepseek-product-review-v1-f302b6/causal-details.json).
- [Cùng đầu ra, hai metric](../../runs/20261110T124259Z-deepseek-product-review-v1-f302b6/metric-comparison.json).
- Kiểm tra acceptance riêng dùng bản sao SQLite nhất quán của phiên hiện tại;
  DB, raw session và fingerprint riêng tư ở `local/deepseek-review/`, không đóng gói.

Không dạy hoặc restore phiên thật. Không đổi learning rule/LR, neuron ALIF,
topology, đường tín hiệu học hoặc ngưỡng guard/P1/P2. Không chạy prototype
Readout Protection. Qwen không tham gia serving hay cung cấp nhãn học.

## Những cập nhật có ích được giữ

1. Guard kiểm tra cả thao tác dạy thủ công và học tự động; before-output đo
   trên clone, không vô tình làm thay đổi recurrent state của model thật.
2. Tách đúng nội dung với hoàn tất câu; có thể sửa `Hello` thành `Hello.` mà
   không mặc định coi dấu chấm là kỹ năng mới. Update mất nội dung cũ bị chặn.
3. Kiểm tra cô lập đã chứng minh rollback khôi phục đầy đủ trọng số, stable
   arrays, eligibility, điện thế, RNG, topology và trạng thái controller.
   Nhận định “rollback hỏng” ở báo cáo A4 cũ bị phép đo mới thay thế: quan sát
   live có idle/generate xen vào không là phép thử rollback thuần túy.
4. Dừng lặp lại proposal bị reject giống hệt, đóng câu hỏi đã xử lý và giảm
   nhãn học sai từ lượt người dùng. Không biến `verified` thành chứng cứ đúng.
5. Repetition guard trong decoder là heuristic có ích để chặn vòng lặp; nó
   không phải bằng chứng synapse đã học ngôn ngữ hoặc representation mới.

## Các lỗi kỹ thuật đã sửa

| Vấn đề | Bản sửa và giới hạn |
|---|---|
| Intervention v2 luôn chép `w_out` AFTER, nhưng các nhánh bias/hidden không thật sự chép nhóm AFTER tương ứng | Helper `parameter_interventions` chỉ ghép nhóm được yêu cầu, kiểm shape/dtype/finite/topology/vocabulary trước mutation; nhóm khác giữ BEFORE. Những kết luận phân rã cũ từ script lỗi không dùng làm bằng chứng. |
| Clone lỗi thì fallback gọi `generate()` lên model live | Abort và rollback; fault injection chứng minh không suy luận/học live trong nhánh lỗi. |
| `copytree` SQLite cùng WAL khi nguồn đang ghi | Dùng SQLite online backup vào DB mới, không chép lock; kiểm thử snapshot khi nguồn tiếp tục commit WAL. |
| Chấm substring chấp nhận phủ định, đáp án thừa từ hoặc lặp | Metric `product-canonical-v2` dùng đáp án đầy đủ đã chuẩn hóa và kết thúc câu. Giữ `legacy-keywords-v1` để tái lập lịch sử. |
| ACCEPTED bị tính là acquisition | Ghi before/after/final, phân biệt guard chấp nhận, nội dung mới đúng, câu mới hoàn tất và câu vốn đã đúng. |
| Sáu paraphrase cũ được gọi là held-out | Ghi rõ development đã xem; generalization trên final test chưa thấy để `null`. |
| RAM chỉ nhìn RSS, thiếu số đo lại coi là pass | Tính cả private commit và peak, thiếu bằng chứng để chưa kết luận. Không đo iGPU/điện năng. |
| Giữ event reject nhưng rollback cursor về thấp hơn event ID | Giữ high-watermark của event reject còn lại, lưu manual reject sau khôi phục. Lỗi persist thông thường vẫn rollback toàn bộ discarded ticks; không thay neuron state để sửa trace. |

Metric v2 là thước đo **đáp án canonical khép kín**, không chấm mọi cách diễn
đạt đúng. Ví dụ `Rất vui được giúp bạn.` có thể phù hợp với lời cảm ơn nhưng
không đạt target `Không có gì.`. Không lấy sai khác này làm bằng chứng model
không hiểu ngữ nghĩa. `_pair_verdict` của production guard và ngưỡng nghiên cứu
không bị đổi theo metric mới.

## Kết quả đo độc lập

Ba trạng thái khởi đầu: phiên hiện tại, checkpoint khác trọng số cũ nhất còn
cùng vocabulary/topology, và public release. Đây **không phải ba seed độc lập**.
Mỗi trạng thái: 4 bài công khai × full-local/readout-only × 6 bước học; sau đó
ghép 8 nhóm BEFORE/AFTER. Đo canonical trên clone, cố ý bỏ guard trong nhánh
chẩn đoán để quan sát lỗi; đây không phải các update production được chấp nhận.

| Trường hợp | Mất nội dung đã đúng | Bước mất đầu | Chỉ `w_out` AFTER | Chỉ hidden AFTER | Chỉ bias AFTER |
|---|---:|---:|---:|---:|---:|
| Current, `hi → Hello.`, full local | 1 | 5 | 1 | 0 | 0 |
| Current, cùng bài, readout only | 1 | 5 | 1 | 0 | 0 |
| 22 trường hợp còn lại | 0 | — | — | — | — |

Mọi nhóm ghép đều khớp chính xác selected/unselected arrays. Nhóm NONE tái tạo
baseline; nhóm ALL tái tạo model đã cập nhật. Model nguồn giữ nguyên weights
và states. `w_out` đủ tái tạo **lỗi mất một cặp trong snapshot này**. Điều này
ủng hộ giả thuyết readout overwrite ở trường hợp cụ thể; không chứng minh mọi
forgetting do readout, hidden có contribution bằng 0 hay đã học representation.
Bước 5 của rerun cũng không được thay vào số bước 4 của run lịch sử khác state.

Cùng 24 đầu ra thực chấm v1/v2 đều 14 pass. Chặn false positive được chứng
minh bằng regression tests, không phải bằng cách nhận v2 là model tốt hơn.
Chạy 40,00 giây, một numerical thread; sampled peak RSS 52.715.520 byte,
private commit 38.162.432; Windows process peak RSS 59.162.624, commit
46.067.712 byte. Cap 2.000.000.000 byte được áp dụng cho process; số monitor
`hard_cap_enforced=false` mô tả monitor chỉ quan sát, receipt `cap.enforced=true`
mô tả cơ chế cap riêng. Đây là run nhỏ, không chứng minh hiệu quả ở quy mô lớn.

## Acceptance sau sửa trên bản sao

- 5/8 đáp án canonical hoàn chỉnh; 3/6 paraphrase development đã xem đạt target.
- 6 giao dịch dạy, 2 được guard chấp nhận; 1/3 target vốn chưa đúng chuyển thành
  đúng nội dung/câu hoàn chỉnh và vẫn đúng sau chuỗi dạy. Một giao dịch ACCEPTED
  còn lại vốn đã đúng, nên không tính acquisition.
- Trường hợp cải thiện: `bạn khỏe không` từ `Tôi đang hoạt động hoạt` thành
  `Tôi đang hoạt động.`. Đây là sửa đáp án cụ thể, chưa là kỹ năng tổng quát.
- Không mất cặp nào trong 8 probe cũ của acceptance; retention trên tập hữu hạn
  này là 1,0. Đây không là kết quả 20-context Phase2.1 hay bằng chứng giải quyết
  catastrophic forgetting. Guard vẫn chặn 4/6 giao dịch, có đánh đổi plasticity.
- Restart giữ nguyên trọng số và canonical outputs trên bản sao. Peak RSS
  42.434.560 byte, private commit 28.266.496 byte, dưới 2 GB cho lượt này.

## Giới hạn của audit 39

Giữ nguyên số đo lịch sử và bổ sung caution ở báo cáo 39. Permutation/sign-flip
của gradient trước Adam không mặc định khớp applied delta sau per-synapse
variance, momentum và clipping. Một bước cập nhật trên copy vẫn là optimizer
step; “không autograd/BPTT” không có nghĩa không dùng xấp xỉ gradient cục bộ.
Kết quả one-step loss không chứng minh retention, ngôn ngữ, hay một nguyên nhân
thất bại duy nhất. Không chạy lại audit, đổi thuật toán hoặc sửa ngưỡng dựa vào
kết quả này; muốn kiểm định mạnh hơn phải đăng ký control applied-delta phù hợp.

## Quyết định và việc tiếp theo

**GO cho các sửa tính đúng đắn/quan sát trên; PIVOT cho cơ chế học; không scale.**
Hidden representation có ích, credit assignment đa bước, tích lũy kỹ năng và
retention dài hạn vẫn chưa được chứng minh. Việc lớp đọc có thể gây mất kiến
thức không đủ để chỉ tối ưu readout rồi bỏ qua neural representation.

Ưu tiên kế tiếp là sửa thiết kế Readout Protection dưới đây trước khi trình
duyệt: test mới chưa xem, tách chọn hyperparameter với đánh giá cuối, comparator
khớp applied-update budget, acquisition/retention cùng lúc và đối chứng hidden.
Không triển khai sweep, learner mới hoặc đưa approved30 vào Hermes trong review.

## Nhận xét về bản thiết kế Readout Protection mới

`experiments/readout-protection-v1.md` hiện ghi CHƯA DUYỆT. Review này không
triển khai prototype, đổi LR hoặc chạy sweep của nó. Hướng bảo vệ readout phù
hợp ưu tiên P2, nhưng bản thiết kế cần được làm rõ trước khi duyệt:

- Sáu câu `HELDOUT` đã được chạy và xem nhiều lần. Chúng là development hiện có,
  không phải tập test cuối chưa thấy; cần bộ test mới và tách bước chọn β.
- Ba checkpoint được chọn vì còn trả lời đúng không phải ba seed độc lập.
  Cần tránh CI/p-value theo giả định độc lập và công bố ảnh hưởng của selection.
- `EMA(|error × feature|)` là độ lớn tín hiệu cập nhật, chưa phải importance
  nhân quả. Với EMA, importance cũ cũng giảm nếu không được kích hoạt; mẫu mới
  khó có thể được bảo vệ mạnh hơn kỹ năng cũ. Cần ablation với importance tráo.
- G1 hiện dùng feature lọc spike trong [0,1] và softmax error trong [-1,1].
  Nếu importance khởi tạo 0 thì EMA nằm trong [0,1]; với β≤4, `g≥0,2`.
  Điều kiện STOP `g<0,1` không thể kích hoạt ở cấu hình này. Phải xét effective
  LR toàn bộ, cả nhánh LR/4; EMA bị chặn không tự tăng vô hạn theo số update.
- Cùng phân bố g chưa bảo đảm cùng bước cập nhật sau feature weighting và
  clipping. Cần báo applied ΔW/budget, tách “chọn đúng synapse” khỏi học chậm.
- 30 trial và điều kiện A+C2 chưa được mô tả thành một bảng job duy nhất rõ
  ràng: β nào được chọn, lúc nào mở validation, Aβ=0 có trùng baseline không.
- Phase2.1 đã có metaplasticity/importance và SI-readout control. Cần chỉ ra
  khác biệt bằng phương trình và đối chứng, không chỉ đổi tên thành protection.

Giữ nguyên ngưỡng forgetting P2; không coi giảm quên nhưng acquisition thấp là
thành công. Các điểm trên là phản biện thiết kế, chưa phải kết quả thực nghiệm.
