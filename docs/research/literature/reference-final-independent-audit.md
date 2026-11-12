# Kiểm toán độc lập reference DEV — 12/11/2026

**Integrity PASS; kết luận chất lượng `REFERENCE_MIXED` được giữ. Không P1/P2
PASS, không chấp thuận đưa learner vào main/Hermes.**

Reviewer thực: subagent `/root/reference_audit_resume`, tách khỏi người viết
runner/report. Kiểm trực tiếp source/protocol/raw; không nhận metric agent như
chân lý. Chỉ sửa hai test kỹ thuật và tài liệu được giao trước khi source khóa;
trong final audit không sửa source, không học dataset, không mở seed hoặc tác
vụ mới. Ponytail không xuất hiện trong tool inventory.

Run `20261112T080533Z-literature-reference-dev-v1-839e07`, source
`a3b79ea66cb6676f8b03a22b0cb907d158e5b2a2`; protocol
`experiments/literature-reference-dev-v1.md`. Main trước/sau còn
`e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`.

## Tính đầy đủ, nguồn và không leakage

Đủ **132 cell đúng grid**: 72 primary =6methods×4tasks×3seeds, 12 nominee
signal-shuffle và48 equal-CPU. Không cell trùng, thiếu, thay task/seed hoặc thay
condition. Seeds140011/140017/140023 là exploratory DEV đã từng dùng; không gọi
chúng là fresh confirmation và không đọc bank confirmation cũ để chọn learner.

15 source files khớp hash manifest. Peer source body và HEAD đã pin còn đúng;
hai upstream clone đúng HEAD, tracked-clean và5 relevant file bodies/license
khớp. Pin e-prop `efd02e6879c01cda3fa9a7838e8e2fd08163c16e`, BrainTrace
`fafa25753b743a2e2d5b47209c4ee947abc02cfb`. 12 dataset hashes khớp, các split
không có episode byte-identical giao nhau. Oracle sự kiện viết độc lập kiểm
**27.648 nhãn** TRAIN/refit/ID/OOD, không có sai lệch.

Kiểm **120 cặp initial state** trên cùng(task,seed), qua methods và budgets:
toàn bộ core arrays, normalizer, momentum và energy đều exact. Variant filter
eligibility được phép khác theo protocol; không đánh đồng nó với init weights
khác. Readout calibration chỉ dùng32TRAIN; probe chỉ fit/chuẩn hóa từ refit.
Task metadata chỉ phục vụ evaluator, không truyền task ID vào learner.

## Kiểm số học và dữ liệu thô

Reviewer dùng chương trình inline độc lập, không gọi hàm tính metric/selection
của runner, và không ghi đè bằng chứng:

| Kiểm tra thực hiện | Số lượng | Kết quả |
|---|---:|---|
| Scalar rescore ID/OOD và3hidden interventions |675.840prediction entries|Exact |
| Checkpoint +initial hash, final byte round-trip |132models|Exact |
| Literal hard-ALIF recurrence, không gọi model.step/features |2.112direct predictions|Exact |
| Ridge probe kiểm bằng SVD độc lập trên nominee |144probe predictions|Exact |
| Full-revert refit probe khôi phục readout-only |132cells, cả equalCPU/shuffle|Exact |
| Source-history/counters/RNG, loss/norm/CPU receipt |132histories|Hợp lệ |
| Scalar paired contrasts: effects,27resamples,8sign flips |27contrasts|Exact |
| Stress prediction entries chấm lại |41.472entries|Exact |
| Direct replay từ transform stress tự dựng |1.296predictions|Exact |

Raw primary đều đúng1.024updates. Readout-only không đổi hidden weights,
hidden_updates=0 và actual hidden norm=0. Các arm học có hidden counter bằng
số episode. EqualCPU dùng process CPU của learning, không gồm thời gian eval;
đạt0,5giây ở episode boundary, không vượt4096updates. Các histories thực tế
dừng theo quy tắc này. Số lượt khác nhau giữa methods là đặc tính của equalCPU,
không được coi là matched-update evidence.

Stress36files đủ own/stop/nominee×4tasks×3seeds. Reviewer tự dựng delay12,
nhân/swap nonfinal channels6/7, event-amplitude và query-flip; giữ final marker.
Nhãn query-flip bằng đáp án cho key còn lại trên cùng writes, paired flip
fraction tính đúng. Đây là counterfactual, không phải một unseen query class.
Noise stress còn biến đổi distractor pulses; tài liệu đã nêu rõ, không gán
mọi tác động cho background noise riêng.

P0 trước DEV đã kiểm actual pinned e-prop AST bằng NumPy primitives, scalar
recurrence, detached reset Jacobian, cold-process resume và strict loader.
Reference/math/snapshot subset reviewer chạy **42 test PASS**; không tuyên bố
đã chạy TF1.15/BPTT numerical suites gốc. E-prop và BrainTrace ở đây là adapter
phương trình cho SCAR-X, không phải execution toàn bộ framework/paper gốc.

## Nomination, gates và kết quả âm tính

Tái tính ranking đúng protocol trên primary isolated tasks, một nominee toàn
cục, không chọn theo từng task:

| Candidate | Worst mean direct/probe margin vs better endpoint | Mean direct margin |
|---|---:|---:|
| eprop_matched |−6,83594đ|−1,62760đ |
| eprop_aligned |−16,21094đ|−5,75087đ |
| braintrace |−16,79688đ|−6,85764đ |

Vì vậy `eprop_matched` được chọn đúng rule, nhưng "được chọn" không có nghĩa
"đạt". Direct Binding78,58% thấp hơn own85,42% **6,84đ**, Binding OOD direct
giảm **4,30đ** so own. Direct distractor60,03% cải thiện khoảng3,06đ so endpoint
tốt hơn nhưng vẫn dưới65%; XOR98,24% thấp hơn stopped99,35%. Protected order
giữ được nhưng bài gần trần không được tính như representation breakthrough.

Probe gain vs readout và causal direct drop có thật ở mức DEV, đặc biệt Binding/
XOR. Signal shuffle làm điểm giảm và full-revert khôi phục frozen probe; không
thấy bằng chứng model chỉ đổi readout mà toàn bộ hidden đứng yên. Tuy nhiên,
những điều này không miễn direct65%, OOD protection hoặc endpoint preservation.
Không family nào đạt trọn điều kiện cải thiện direct kèm ngưỡng và causality;
quyết định `REFERENCE_MIXED` phù hợp. Không phát hiện chọn bỏ seed xấu.

Ba seed chỉ đủ mô tả khám phá. Exact sign-test hai phía nhỏ nhất0,25; bootstrap
27tổ hợp không tạo thêm seed độc lập. Các khoảng95% mô tả sau global selection
không phải confirmation CI đã hiệu chỉnh selection. Không có significance/P1
claim được hỗ trợ ở đây. Ngưỡng P1/P2 lịch sử không đổi.

EqualCPU có kết quả yếu cần giữ cùng báo cáo, không chỉ bảng1.024updates:

| Method | Binding direct | XOR direct | Distractor direct | Order direct |
|---|---:|---:|---:|---:|
| own |58,33%|67,97%|51,95%|96,94% |
| stop |56,25%|86,20%|52,80%|95,31% |
| eprop_matched |54,43%|86,13%|51,17%|98,18% |
| readout_only |57,10%|73,05%|52,02%|87,30% |

Đây là average3seed; số episode thực tế min–max: own159–212, stop136–171,
eprop_matched156–215, readout-only161–214. Các số này là phạm vi từng cell,
không phải số update được chọn bằng accuracy.
Không có bằng chứng lợi thế CPU đồng đều cho nominee. Raw actual update norms
đã ghi nhưng reference study **chưa có norm-matched causal control**, vì vậy
không kết luận lợi ích độc lập với amplitude hoặc optimizer trajectory.

## Tài nguyên và phạm vi kết luận

Worker hoàn tất khoảng766,97s, dưới budget50phút; audit của parent8,80s dưới
10phút riêng. RSS owned tree133.709.824B, committed175.263.744B, dưới2GB. Cap
receipt ghi Windows per-process limit enforced; sampler field
`hard_cap_enforced=false` chỉ mô tả sampler quan sát, chưa đồng bộ với receipt.
Báo cáo đã giải thích hai trường, không coi committed bytes là lượng swap.
Các PID tích lũy còn gồm git subprocess đã thoát, không có nghĩa tất cả đồng
thời sống. Không có iGPU/energy đo được để suy ra lợi thế điện năng.

Report `LITERATURE_REFERENCE_RESULTS_20261112.md` giữ các regression và giới hạn,
không gọi transformed adapter là bản gốc thắng/thua literature nói chung.
Reviewer yêu cầu hiển thị thêm equalCPU negative table trong bản đọc chính;
raw/per-seed CSV đã bảo toàn toàn bộ cell. Không thấy blocker integrity mới.

**Phán quyết:** nhận code/tests/evidence vào nhánh nghiên cứu; chưa nhận learner
vào production. Credit timing/filter của literature là cơ sở hữu ích đã tái
lập, nhưng tradeoff Binding/distractor còn tồn tại. Chưa chứng minh lifelong
retention, một checkpoint tích lũy nhiều skill, chat hoặc vision tổng quát.
Không suy ra ACR bị loại từ run này: ACR còn partial, cần hoàn tất đối chiếu
hợp lệ hoặc báo dừng rõ ràng; không đổi gate để cứu bất kỳ phương pháp nào.

Số cập nhật học được tạo trong final audit: **0**. Không fresh confirmation,
P2, main/Hermes/SQLite migration hoặc can thiệp process/port.

## Closeout bảng v2 — không thay điểm hoặc quyết định

Reviewer đã đối chiếu `audit-v1.json`/`audit-v2.json`: mọi nội dung giống nhau
ngoại trừ thời gian chạy audit. Toàn bộ132CSVrows v1/v2 cũng khớp, số giá trị
probe_gain thay đổi bằng0. Lookup readout trong CSV đã được làm rõ là **cùng
budget** với candidate. V1 dùng readout primary cho hàng equalCPU nhưng không
làm sai trị số ở run này:12cặp readout-only có hidden weights cố định và probe
refit ID/OOD giống chính xác giữa hai budgets, dù direct readout có thể khác.
Đây là sửa độ rõ và độ bền reporting, không phải sửa learner hoặc cứu kết quả.
Không huấn luyện lại; `REFERENCE_MIXED` và mọi primary contrast giữ nguyên.
