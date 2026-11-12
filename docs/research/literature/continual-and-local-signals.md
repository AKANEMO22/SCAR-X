# Literature gap audit — local signals và duy trì kỹ năng

Ngày12/11/2026. Research-only, không train/P2/confirmation, không đổi main,
Hermes hoặc live state. Ponytail không có trong tool inventory. Pin/hash và
bản sao license: [manifest](sources/upstream-pins-20261112.json). External clones
nằm trong `scar-x/local/references`, ngoài source production. Bản đọc repo không
được xem như đã tái lập kết quả paper.

## Ưu tiên theo giá trị kiểm chứng / chi phí

| Nút thắt SCAR-X | Phần có thể kế thừa | Tương thích H48 hiện tại | Chi phí / phép phân biệt cần làm | Trạng thái |
|---|---|---|---|---|
| Temporal credit/reset | e-prop: phân tích graph, trace recurrence, phép đối chiếu autodiff | Cao sau khi chuyển phương trình đúng reset ALIF của SCAR-X | Adapter NumPy; exact forward, oracle trace và reset Jacobian; không gọi adapter là original implementation | `PARTIALLY_SOLVED`; adapter/DEV do nhánh reference bàn giao |
| Approximate gradient | BrainTrace D-RTRL: block Jacobian theo state của từng neuron, trace theo tham số | Cao cho block local; full recurrent oracle chỉ là control | Ba state sensitivities/neuron; so cosine với surrogate oracle và readout filter thực | `PARTIALLY_SOLVED`; không cần nhập compiler/framework vào app |
| Readout/hidden signal | DECOLLE: loss phụ cục bộ và projection cố định | Ý tưởng có thể áp dụng; original nhiều tầng LIF khác core recurrent ALIF | Phải phân biệt tín hiệu mới với thay trace/optimizer; giấy phép code hiện có vướng mắc | Paper-level `ALREADY_SOLVED_IN_LITERATURE`; lợi ích SCAR-X `UNRESOLVED` |
| Distractor phá state | DGN: conductance phụ thuộc input và memory decay thích ứng | Thấp nếu buộc giữ nguyên forward; DGN thật thay neuron dynamics | Chỉ secondary architecture arm, matched capacity/compute và local credit riêng | `UNRESOLVED` trong primary; không nhập theorem ổn định sang input gate |
| Forgetting/readout overwrite | SI: importance từ unregularized loss-gradient × actual weight delta | Cao về bộ nhớ; ý nghĩa importance phụ thuộc chất lượng credit | Helper toán đã chạy; cần acquisition/plasticity lẫn retention trước P2 | Toán `REPRODUCED_IN_SCARX`; chống quên SCAR-X `UNRESOLVED` |
| Bảo vệ tham số | EWC / online EWC: Fisher và quadratic penalty | Bộ nhớ O(P) được, hidden Fisher dùng local credit chỉ là proxy | Cần định nghĩa model-Fisher/empirical-Fisher, estimator và boundary policy trước | `UNRESOLVED`; chưa triển khai hoặc kết luận thất bại |
| Phát lại | Reservoir replay với RAM/update budget cố định | Cao cho task-agnostic sampling | Tính cả rehearsal updates/CPU, SSD I/O; giữ replay RNG/checkpoint | Giải pháp phổ biến, chưa kiểm trong protocol mới |

Hai hàng đầu cần xử lý trước: học được representation hữu ích rồi mới bảo vệ
nó. SI/EWC không thể tạo kỹ năng chỉ bằng cách làm trọng số ít thay đổi. Không
có kết quả ở đây chứng minh H48 đã chạm trần năng lực; muốn kết luận giới hạn
kiến trúc phải tách lỗi estimator, optimizer/readout, thiếu train budget và lỗi
dữ liệu bằng đối chứng đã khóa.

## SI: kiểm nguồn và helper có thể chạy

Pin pathint `86c5e07c5603d6a44c2f53585c19ee70773ef15c`, MIT. Paper
[Zenke–Poole–Ganguli, ICML2017](https://proceedings.mlr.press/v70/zenke17a.html).
Source quan trọng là
[protocols.py:27–43](https://github.com/ganguli-lab/pathint/blob/86c5e07c5603d6a44c2f53585c19ee70773ef15c/pathint/protocols.py#L27),
[optimizers.py:118–126](https://github.com/ganguli-lab/pathint/blob/86c5e07c5603d6a44c2f53585c19ee70773ef15c/pathint/optimizers.py#L118),
[utils.py:27–38](https://github.com/ganguli-lab/pathint/blob/86c5e07c5603d6a44c2f53585c19ee70773ef15c/pathint/utils.py#L27).

Với `g=dL_unreg/dw` trước cập nhật và delta vật lý sau optimizer/clip:

```text
work <- work - g * (w_after - w_before)
contribution = work / ((w_current - anchor)^2 + xi)
Omega <- max(0, combine(Omega, contribution))
combine = sum, hoặc decay*old + (1-decay)*new theo cấu hình upstream
penalty = lambda * sum(Omega * (w-anchor)^2)
```

Rectification sau combine mới đúng code pin; không âm thầm bỏ từng đóng góp
âm trước khi cộng. Upstream có `task_updates` và cập nhật anchor/reset work ở
ranh giới do caller cung cấp. Không có task-ID argument vẫn chưa đủ để gọi
task-agnostic nếu controller bí mật gọi consolidate đúng ranh giới.

Helper `evaluation/literature_si.py` chỉ triển khai toán này bằng float64,
không kết nối learner.10test qua bằng lệnh sau, gồm Taylor trên quadratic,
physical clipping delta, sum/EMA, finite-difference penalty, state resume và
fail-closed validation:

```powershell
python -X utf8 -m pytest -q -p no:cacheprovider tests/test_literature_si.py
```

Bốn mảng anchor/previous/work/importance tốn32P byte; P=866 của hidden768 và
readout/bias98 tương ứng27.712byte. Đây là array bytes; process RAM và buffer
phải đo riêng khi tích hợp. SI helper chưa có bounded file codec/atomic rollback.
`state_dict/from_state` được kiểm dữ liệu/copy, chưa là tương thích checkpoint
production. Không có retention score SI trong vòng này.

Tín hiệu `W_out.T@error * eligibility` của SCAR-X là xấp xỉ hướng tăng likelihood,
không mặc nhiên là loss-gradient g. Nếu thay g bằng âm tín hiệu đó, phải gọi
`local-credit SI adaptation`, kiểm sign/scale trước optimizer và không nhận
nguyên diễn giải loss-attribution của SI gốc. Đặc biệt không dùng bước đã
momentum/RMS-normalize như g rồi nhân lại delta; điều này làm sai importance.

## EWC: phần repo thực sự có

Pin ContinualAI `4900462fa58d7e0ab03cec7c6c0d9918b553095a`, MIT. File
[experiments/permuted_mnist/ewc.py:17–51](https://github.com/ContinualAI/continual-learning-baselines/blob/4900462fa58d7e0ab03cec7c6c0d9918b553095a/experiments/permuted_mnist/ewc.py#L17)
gọi Avalanche/PyTorch; mặc định `ewc_mode='separate'`, vòng train theo experience.
Đây là wrapper, chưa phải implementation Fisher/online EWC được pin độc lập.
Test upstream là benchmark accuracy PermutedMNIST, không phải numerical proof
cho ALIF. Không cài/chạy nó trong app ở audit này.

[EWC gốc](https://arxiv.org/abs/1612.00796) bảo vệ weights bằng Fisher-weighted
quadratic penalty. Online EWC có thể giữ một consolidated importance/anchor
thay vì một bộ cho mỗi task; xem
[Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html).
Việc tái sử dụng estimator vẫn phải công bố gradient, data/labels sampling và
consolidation schedule; bình phương local update không tự trở thành Fisher.
Forward task-agnostic không xóa được việc training dùng task boundaries.

## DECOLLE: local error và giấy phép có khác mô tả ban đầu

Paper [DECOLLE](https://arxiv.org/abs/1811.10766) sử dụng loss cục bộ qua readout
projection và presynaptic traces; không yêu cầu backward xuyên các tầng/thời
gian của toàn network. Bài học có ích là tách tín hiệu học của hidden khỏi
readout chính. Với một quần thể ALIF, cần kiểm liệu auxiliary local signal tạo
representation tốt hơn, không chỉ thêm một readout có nhãn dày hơn.

Code pin `7d72cb105891fe71a4d6a8b7f9c8e95303219078`: fixed readouts ở
[lenet_decolle_model.py:140–151](https://github.com/nmi-lab/decolle-public/blob/7d72cb105891fe71a4d6a8b7f9c8e95303219078/decolle/lenet_decolle_model.py#L140),
state detach ở
[base_model.py:180–189](https://github.com/nmi-lab/decolle-public/blob/7d72cb105891fe71a4d6a8b7f9c8e95303219078/decolle/base_model.py#L180).
Upstream có local autograd/backward; việc thấy `.backward()` không đủ chứng minh
backward toàn mạng, phải kiểm chỗ detach. SCAR-X chưa import/chạy code này.

**License finding:** README/setup còn ghi GPLv3, nhưng commit pin ngày13/10/2026
đổi sang commercial terms. Chính
[LICENSE.txt](https://github.com/nmi-lab/decolle-public/blob/7d72cb105891fe71a4d6a8b7f9c8e95303219078/LICENSE.txt)
có non-commercial reservation trước BSD text. Vì vậy không gọi pin hiện tại là
plain GPLv3/BSD. Không copy/import/distribute implementation DECOLLE vào SCAR-X;
bản license được lưu làm bằng chứng, chỉ kế thừa ý tưởng từ paper ở bước này.

## DGN: không phải chỉ một cổng input đặt trước ALIF

[Paper arXiv v1, eq5–8](https://arxiv.org/html/2509.03281v1#S3.SS1) thêm trạng thái
presynaptic D và hệ số leak rho(t) phụ thuộc input, với delayed reset. Appendix
A.1.3 dùng BPTT. Đây là thay đổi dynamics; không tương đương nhân input bằng
gate rồi giữ nguyên ALIF. Kết quả chống nhiễu/theorem của DGN không tự áp dụng
cho SCAR-X. Venue được kiểm ở
[ICLR2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc92064236f06645e0b5ec5a8a16565d-Abstract-Conference.html).
Chưa xác minh được official code URL/SHA/license từ các primary source đã đọc;
ghi unknown, không suy đoán repo. Chưa triển khai DGN quality arm.

## Đối chiếu SCAR-X P1 và Gemini R9b

P1 Codex sourcebc686500:12seed, only XOR đủ gate; binding/OOD regression và
distractor direct63,64% vẫn chặn P1. Đây không phải bằng chứng rằng e-prop gốc
hoặc SI đã thất bại. Baseline của dự án gồm heterogeneous filtering/gating,
readout-normalization và local optimizer riêng; từng khác biệt phải đối chiếu.

R9b đọc tracked source/report/artifacts ở commit
`e65dbbdb735cc37834f0c2de365d317ee7c1563a`; giữ nguyên file untracked của tác giả.
[Arithmetic check](sources/r9b-readonly-metric-check-20261112.json) tái tính15
skill-matrix records, không train. `full` BF-Cascade acquisition64,91%, final
58,14%, mean forgetting10,76pp; baseline67,84%/61,07%/10,03pp. Các số khớp
artifact. Worst forgetting35,55pp ở một seed full; selective có seed43,75pp,
không được dùng mean-of-worst19,40pp thay gate từng seed5pp. Toàn bộ vẫn FAIL.
Đây là kiểm arithmetic/source gap, chưa phải audit mọi prediction R9b.

288factorial rows có3seed độc lập,72config-seed rows/task. Binding thắng shuffle
72/72rows là mô tả, không phải72independentseeds. Script analysis chỉ đếm wins;
không có cơ sở từ đó gọi p<1e-20 độc lập. Clipping từng coordinate không khớp
actual update norm; vì vậy chưa chứng minh reset gain chỉ đến từ direction.
R9b hữu ích để đặt failure hypotheses, chưa là code sẵn sàng nhập main.

## Lộ trình continual learning khi được duyệt P2 riêng

1. Khóa learner có acquisition/direct/representation hợp lệ trước. Ghi toàn bộ
   readout, normalizer, hidden traces, importance, counters, replay/RNG vào codec;
   kiểm cold resume/next update/rollback và peak RAM trong consolidation.
2. Baseline boundary-assisted: SI gốc/online EWC với boundaries được khai báo,
   giữ riêng với strict stream. Không so nó như thể không được trợ giúp.
3. Strict task-agnostic: một learner/head chung, không task ID; SI adaptation
   dùng fixed update-count windows/decay được khóa trước, không reset/consolidate
   theo true boundary. Nếu có detector, đo riêng lỗi phát hiện và false alarm.
4. Replay-allowed: reservoir theo thời gian, budget mẫu/bytes cố định; mọi replay
   tính vào tổng updates/CPU. Dữ liệu cũ nguyên bản trên SSD không được gọi là
   synapse đã học. Có arm no-replay và matched-budget replay.
5. Đo acquisition>=65%, final>=65%, mean forgetting<=3pp và worst<=5pp theo seed;
   giữ ma trận kỹ năng đầy đủ, plasticity trở lại, nhiễu label, abrupt/gradual
   drift và task quay lại. Chưa có quyền chạy P2 từ tài liệu này.

`POTENTIAL_NEW_CONTRIBUTION` chỉ nên dùng cho lợi thế còn lại sau các reference
và controls hợp lệ. ACR hiện là giả thuyết mới chưa chứng minh; không bảo vệ nó
bằng cách bỏ baseline literature hay mở search lớn. Không có lợi thế product
hoặc lifelong retention được chứng nhận bởi literature audit này.
