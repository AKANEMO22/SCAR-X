# Ma trận prior art và ràng buộc từ kết quả âm

Ngày15/11/2026. Đi kèm [đặc tả chuyển hành vi/reuse P](SCOPE_ALIF_CAPACITY_RELEASE_ANALYSIS_20261115.md).
Chỉ phân tích, không train hoặc chọn kiến trúc bằng TEST. Tính chất toán,
claims tác giả và kết quả SCAR-X được giữ riêng. Không có phát minh mới được xác nhận.

## 1. Nguồn chính và phạm vi đọc

| Nguồn/version | Phần đã đọc trong lượt này | Chưa làm |
|---|---|---|
| SCOPE-ALIF v2 tại baseb3a48983 | Toàn bộ candidate và amendment§10, equations/masks/persistence/claim boundaries; đối chiếu source core/learner | Không thực hiện reviewerloop mới, không nhận hai PASS như model evidence |
| [Progressive Networks v4](https://arxiv.org/html/1606.04671v4) | Paper body, limitations, transfer analysis, experimental setup và supplementaryA–E | Không chạy tác giảcode hoặc tái lập game scores |
| [Progress & Compress v2](https://arxiv.org/html/1805.06370v2) | Main methods/objective/onlineEWC, results/discussion và appendicesA–C | Không tái lập distillation hoặc nhận định Bayesian guarantee như chứng nhận mạng nonlinear |
| [HLOP-SNN v1](https://arxiv.org/html/2402.11984v1) | Body và appendicesA–G, gồm training assumptions, rank growth, limitations và sharedhead setting | Không port PyTorch/HLOP vào ALIF hoặc chứng nhận biological locality của mọi learner |
| ROW commit e846633e1d1a5cfa60e8c83646f68175e810a1ba | Toàn README; paperdraft§7/7.5/7.6/9 và scope addenda chọn lọc; O6plan; summary/protocolJSON O2C/O6/E6.2; source routines chọn lọc | Không tuyên bố đọc mọi file/984commits, fullraw forensic audit, paperdraft toàn bộ hoặc tái lập results |

Full HTML paper và các nguồn ROW chọn lọc có hash trong
[manifest](literature/sources/scope-alif-capacity-release-20261115.json).
Nguồn được lưu private để đọc; không copy code vào core hoặc chạy scripts.
ROWcurrentmaster được kiểm vẫn trùng pin. Paperdraft và README có đoạn status
cũ chưa đồng bộ với bảng O6/O11 mới; dùng protocol/result cụ thể khi trích
claim, không gộp mọi nhánh thành một verdict toàn repo.

## 2. Ma trận cơ chế, giới hạn và chuyển giao

| Hướng | Cơ chế gốc cần phân biệt | Khi Plastic hết chỗ | Khác biệt cần kiểm trong adaptation ALIF |
|---|---|---|---|
| Progressive Networks | Frozen columns, lateral features, column mới; learner dùng gradient | Tiếp tục thêm capacity; không có release fixedbudget trong thiết kế cơ bản | SCOPE giữ topology cố định nên không kế thừa cách mở thêm cột; freeze không chứng minh reuse hữu ích |
| Progress & Compress | Activecolumn học rồi distill vào knowledgebase, onlineEWC, có thể reinitialize activecolumn | Chuyển chức năng rồi dùng lại active storage; phụ thuộc KB chứa được union | Đây là prior art **gần nhất** của đề xuất. Thay gradient bằng local eligibility và dùng evidence/functional checks là adaptation, không novelty đã chứng minh |
| HLOP-SNN | Lateral Hebbian activity-subspace projection, learner chính tách riêng; các setting có single/multihead | Bảo vệ hướng ghi; không tự giải union capacity hoặc tái tạo vùng đã dùng | Không nhập projector lý tưởng sau clipping; rank hữu hạn có saturation. Không đưa làm thành phần bắt buộc khi chưa biết projection giúp transfer hay chặn nó |
| ROW | Synthetic operator libraries, shared/task state, route search và consolidation; neural updates dùng backward | Tùy family: reuse/lifecycle có thể có ích hoặc không; không là một giải pháp ALIF fixedbudget | Áp dụng bài học adequacy/functional substitution/cost, không nhập router/teacherprimitives/extra information hoặc countint8 thành lợi thế nativeFP32 |
| SCOPE v2 hiện tại | Two populations cùng ALIF, S→P nhưng không P→S; oldhead frozen | Có isolation theorem, chưa correction/release | Hai PASS là consistency. Chưa có S đã học theo partition hoặc migration từ mixedH128 |
| Adaptation trong đặc tả | StudentS local relearning từ verified outcomes/teacherbehavior, atomic acceptance, reset cùngP | Chỉ release nếu S-only giữ cả old/new chức năng; fail giữ teacher | Không thêm cột/head theo task; không giảm dtype/neuron/edge; phải tách hidden gain khỏi readoutfold/replay |

## 3. Những khác biệt paper làm thay đổi quyết định

### Progressive Networks

[§2/limitations](https://arxiv.org/html/1606.04671v4#S2) đặt vấn đề tăng tham số
và chọn column bằng tasklabel. AppendixB phân tích mức dùng features, chưa
chứng minh một learner cùng fixedbudget có thể tự giải phóng column. §5.1
dùng top3/25jobs khi tổng hợp một số kết quả; không nhập cách chọn đó vào
SCAR-X vốn phải giữ đầy đủ seed đã khóa.

### Progress & Compress

[§2/4 và discussion](https://arxiv.org/html/1805.06370v2#S2) đã có distillation
KL và onlineEWC; không có lý do gọi chuyển P→S/resetP là ý tưởng kiến trúc mới.
Paper nhận biết distribution changes để chuyển pha; không đã giải task-free
trigger. AppendixA cho thấy đánh đổi bảo vệ cũ/học mới. Một số phép đo revisit
task nhiều lần, nên không chuyển thành cam kết một trải nghiệm đủ tạo skill.

### HLOP-SNN

[AppendicesC/E/F/G](https://arxiv.org/html/2402.11984v1#A6) cho thấy các subspace
mới được nối thêm sau task và rank là hyperparameter. BNstats/threshold có
phần được khóa để tránh drift; backbone không luôn ALIF/local learner như
SCAR-X. Domain5datasets sharedhead báo BWT−9,94%, khác một lời hứa mọi setting
“gần0forget”. BWT với điểm sau task khác max-past forgetting3/5pp của dự án.

### ROW

[E6.2report](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/reports/e6_2_compiled.json)
báo compilation không đạt matchedcapacity và không có lợi theo economics đã
đăng ký. Không dùng ít calls thay cho chi phí đầy đủ. [O6plan](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/O6_SEALED_REROUTE_CONFIRMATION_PLAN.md)
và [result](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/reports/o6_sealed_reroute.json)
ghi re-route+sleep45/45cells, nhưng dùng64retainedexamples/task, short routes
và8.192extra updates. [Code consolidation](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/src/row/experiments/o2c_consolidation.py#L96)
gọi backward; không là localALIF result hoặc constanttotalreplay. Đây là source
claim đã đọc, chưa independent replication. Không chọn chỉ kết quả âm hoặc
chỉ kết quả dương để suy mọi consolidation sẽ thất bại/thành công.

## 4. SCAR-X: kết quả âm nào ràng buộc đề xuất này?

| Artifact đã có | Kết quả và phạm vi | Điều không được hứa trong release proposal |
|---|---|---|
| [Localcredit/retention15/11](../hybrid/LOCAL_CREDIT_AND_RETENTION_20261115.md) | Signed hidden commonhead DEV78,33vsfrozen62,50%, rawfeatures79,72%; importance/anchor giảm meanforget8,06→4,72pp, chỉ1/3seedqua jointgate | Không giả định eligibility đủ mạnh hoặc EMA đã chuyển skill thành công; không lấy DEVgain thành TESTgain |
| Cùng report, slowweights | Stable-in forward cuối thấp hơn fastweights trên ba DEVprobes | Weightaveraging không thay output-consolidation; không tự promote slowmodel |
| [ACR results](ACR_DEV_RESULTS_20261112.md), [credit limits](ACR_CREDIT_SPACE_LIMITS_20261112.md) | Specializedgain, Distractor chưa65%; jointcheckpoint chỉXOR qua, scalartrace mixture bỏ cross-neuron paths | Transfer bằng localcredit là hypothesis; learner bị chặn chưa chứng minh capacity đã hết |
| [CapacityH64/H128](CAPACITY_H64_H128_RESULTS_20261112.md) | CAPACITY_ONLY_SIGNAL; hiddengain có điều kiện, growthcontrast có ceiling, chưa20stage lifetime | RAM nhỏ/tăngH không chứng minh tìm được representation đủ để release |
| R13.2report và [saved recount manifest](literature/sources/operator-lattice-review-20261115.json) | Fixedspike19,92%, Yaligner33,40%, standardizedsymbolic98,05% theo savedpredictions; chưa fullcausalLLMaudit | Có S→P edge không bảo đảm reuse; consumer tương thích phải kiểm, không xem bất kỳ alignment là đủ |
| [Information-path audit](../release/evidence/CURRENT_CREDIT_INPUT_RC13_20261115.json) | Ba outcome khác nhau từng thành cùng8×64input | Reset/relearn không tạo lại thông tin encoder đã bỏ |
| [ActionbankSTOP](../hybrid/P1B_MULTI_API_STOP_20261115.md) | Oracle bằng constantaction trên outcomesDEV đã có | Không train selector hoặc đổi architecture để vượt một cận vốn không có headroom |
| [Behavior spec](ALIF_READOUT_BEHAVIOR_PRESERVATION_SPEC_20261115.md) | Feature-nullspace chưa là reachable weight update; behavior safe-ray có thể không đơn điệu | Không project hoặc binarysearch rồi coi bảo toàn toàn chức năng |

Các số đã công bố là bối cảnh ràng buộc, **không được rescore TEST hoặc dùng
TEST chọn cấu hình**. Lượt này không mở lại raw prediction TEST. Với report
Gemini, các cụm “miễn nhiễm quên nhờSSD/đột phá” không được nhận vào kết luận:
save/load đúng không là học lifelong; report tốc độ/LLM cần scopeaudit riêng.

## 5. Quyết định reuse thay vì tái đặt tên

- **REUSE:** ALIFforward, localeligibility đã có; source/teacher/evidence
  versioning; kiểu lifecycle P&C; nguyên tắc adequacy/matchedbudget.
- **ADAPT_ON_PAPER:** output-directed local relearning trong shadow S, bounded
  data và transactional reset. Chưa code hoặc permission thử.
- **VERIFY_GAP:** representability union trongS; learner có tới được nghiệm;
  hidden contribution vs readout; chi phí coverage và correction/lifetime.
- **DEFER:** HLOP thêmrank, recurrent rewiring, operatorcompiler, learnedrouter,
  headpertaskgrowth, change neuron hoặc precision. Chưa chứng minh cần chúng.
- **DROP_DUPLICATE_CLAIM:** gọi freeze+lateral+distill/reset là phát minh mới;
  lấy hai reviewerPASS thành empiricalvalidation; gọi resetarrays là tăng
  intelligence hoặc lấy nhỏRAM thành chất lượng.

## 6. Điều kiện để có đóng góp ngoài một adaptation

Phải có bằng chứng rõ một localALIF learner chuyển được skill có ích với
fixedtotalbudget, retainedold/correctedknowledge, removal of P dependency và
nextcycle learning, hơn controls cùng replay/compute. Không chỉ công bố lại
proof cách ly, loss xuống hoặc head học tốt. Nếu không có lợi thế, đóng hồ
sơ bằng kết quả âm; không thay tên hoặc tăng số module để giữ một claim mới.

Ma trận không chọn learner bằng mức điểm paper; các setup/data/training/
hyperparameter selection khác nhau. Không công trình nào ở trên tự cấp bằng
chứng SCAR-X đạt2GB lifelongintelligence hay ngangmodel thương mại.
