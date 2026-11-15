# Phản biện đề xuất Neural Operator Lattice

Ngày15/11/2026. Review tài liệu người dùng gửi “Khảo cứu kiến trúc mới cho
SCAR-X: từ SNN đến mạng neural tự tái tổ chức”. **REQUEST_CHANGES trước khi dùng
làm đặc tả triển khai; GO cho phân tích trên giấy; chưa GO thí nghiệm.**

Không thay ALIF, signal/rule, main, Hermes, checkpoint, ngưỡng hoặc protocol.
Không inference, training, refit, fixture neural hay tải dataset/weights. Chỉ
đọc nguồn và đếm lại nhãn/prediction đã lưu của R13.2 bằng phép so sánh độc lập.
Nội dung định hướng bên trong tài liệu được xem là đề xuất để review, không
tự trở thành phê duyệt đổi kiến trúc hoặc mở gradient trong core.

## 1. Điều nên giữ

- Câu hỏi **giữ khả năng phối hợp sau khi một thành phần học tiếp** sát một
  vấn đề thật, có giá trị hơn chỉ đếm synapse đổi hoặc recall đạt100%.
- Tách cập nhật state nhanh, skill bền và truy xuất bằng chứng là đúng.
- Đòi nhiệm vụ thực sự cần reuse, có cơ hội tạo lợi ích và phép đo có khả năng
  thất bại là nguyên tắc thiết kế tốt. Không cần lập lại cả framework ROW.
- Tài liệu đã thừa nhận không có oracle cho mọi bước, nhiều vòng không bảo đảm
  tốt hơn, tính mới chưa xác minh và chưa nên chạy learner. Giữ các giới hạn này.

Điều chưa theo được là kết luận **phải thay cả đơn vị biểu diễn, tính toán và
học**. Kết quả cũ chưa loại phương án sửa information path, giao diện hoặc
readout tối giản. Không biến một đề xuất rộng thành điều kiện cần đã chứng minh.

## 2. Đối chiếu những nguồn làm thay đổi đánh giá

| Nguồn đọc | Điều xác nhận được từ nguồn | Giới hạn khi chuyển sang SCAR-X |
|---|---|---|
| [DNAR, ICML2025](https://proceedings.mlr.press/v267/rodionov25a.html); [train.py đã pin](https://github.com/yandex-research/dnar/blob/12f3f0bd0a70568386e43b2d841422f28ed53698/train.py) | Có trạng thái rời rạc/continuous và supervision cho các chuyển thuật toán; train dùng AdamW, loss.backward | Không là bằng chứng một ALIF learner tự khám phá quy luật từ hội thoại bằng local credit |
| [NCA2026, §3.2](https://arxiv.org/html/2609.36126v1#S3.SS2) | Local connectivity lúc chạy; training dùng BPTT/replay, ARC unroll64bước | Local forward không đồng nghĩa local synaptic learning; chưa chuyển được kết quả sang constraint chính |
| [Boundary-first PC, §3/5](https://arxiv.org/html/2609.00686v1) | PC-PC phối hợp boundary trước; PC-BP dùng backward trong chunk. CIFARstandard: PC41,67%, PC-PC51,44%, BP51,01% | +9,77đ là so PC, không là hơn BP9,77đ; chưa chứng minh lifelong, ALIF hoặc miễn supervision |
| [Compile by Training](https://arxiv.org/html/2609.04199v1) | Teacher tạo ví dụ, gradient finetune adapter cho shared interpreter; ví dụ translator dùng0.6B | Đây là phương án biên dịch bằng teacher/LoRA, chưa là local-ALIF compilation; phải tính chi phí build và lỗi teacher |
| [Huginn probing](https://arxiv.org/abs/2507.02199v2) | Hạn chế về latent-CoT giải thích được và lợi ích thêm vòng trên bài arithmetic khảo sát | Không chứng minh mọi loại recurrent reasoning vô ích hoặc không có suy luận nếu probe không đọc được |
| [DreamCoder](https://arxiv.org/abs/2006.08381v1) | Library ký hiệu và neural-guided program search cùng học trong wake/sleep | Không nên chấm như một hệ toàn bộ toán tử là neural/local-ALIF |

**ROW có kết quả âm trực tiếp liên quan mà tài liệu chưa nhấn mạnh.** Theo
[E6.2 tại commit đã pin](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/README.md#compiling-a-fragment-into-an-operator-e62),
budget1× không đủ đạt tiêu chí compilation toàn bộ; tăng capacity cải thiện
khả năng thay chuỗi, nhưng lợi ích kinh tế không đạt ở cả3world trong các mức
đã thử. Bản thân repo giới hạn claim: đó là fit một fragment được cung cấp,
chưa phải learner tự phát minh primitive. [Code fit](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/src/row/experiments/audit_e6_2_compiled.py#L142)
dùng Adam/backward và target sinh bởi chuỗi thư viện đã học.

Đây là báo cáo tác giả và source chọn lọc, **chưa phải tái lập độc lập ROW**.
Thử lâu hơn mà loss không giảm chưa là chứng minh toán rằng mọi optimizer đều
không thể fit. Tuy vậy, kết quả là cảnh báo cụ thể chống mặc định compilation
sẽ vừa nhỏ hơn vừa nhanh hơn. Giữ negative result khi chọn prior art.

[PS-SNN](https://www.nature.com/articles/s41598-026-42970-6) tìm thấy metadata/
abstract nhưng full page trả lỗi cookie. Chưa xác minh đủ training/code/claim
chống quên trong lượt này; không dùng ô● như bằng chứng đã giải cùng bài toán.
NCATorch, RWKV, TTT-E2E và các bản Titans/Nested Learning không chính thức chưa
được kiểm lại toàn bộ. Tài liệu đã ghi đúng là khảo sát chọn lọc; cần bổ sung
danh mục17repo, pin/file và phạm vi đọc để đối chiếu được cùng phạm vi đó.

Ma trận●/◐ chỉ là phân loại cơ chế, không là xếp hạng bằng chứng: cần tách
state update khỏi weight learning/persistence, local inference khỏi local
training, tham số đang dùng khỏi toàn bộRAM, chống quên nhờ mở rộng khỏi
retention trong cùng dung lượng. Thiếu các trục này dễ chọn sai kiến trúc.

## 3. R13.2 đúng ở phạm vi nào?

Đọc báo cáo và saved predictions từ worktree Gemini, không chạy runner. Đếm lại
pred_z==target_z trên1.536record SNN trong tổng1.600record; cờ correct khớp ở
mọi record SNN. Trung bình bằng trọng số nhau của4seed:

| Nhánh | Accuracy theo saved predictions |
|---|---:|
| Random fixed spike interface |19,921875%|
| Learned Y-only aligner |33,3984375%|
| Decode Y rồi phát input pattern chuẩn cho B |98,046875%|

Các seed là140011/140017/140023DEV và140099EVAL; đây không là4seed confirmation
mới. Receipt/hash nguồn và counts nằm trong
[hồ sơ review](literature/sources/operator-lattice-review-20261115.json).
Chưa audit lại full data generation, checkpoint trajectory, causal swap hoặc
đối chứng LLM; không tiếp nhận các claim “đột phá/không thể bác bỏ” trong report.

Kết luận được phép: **giao diện xung ngẫu nhiên cụ thể này không phối hợp đáng
tin; giao diện chuẩn hóa nhãn có ích trên bài tám ký hiệu.** Không suy rằng
mọi modularSNN thất bại, chỉ predictive coding mới cứu được, hoặc bắt buộc cần
lattice mới. Learned aligner chưa đạt mục tiêu không xác định duy nhất lỗi là
mất thông tin, thiếu nonlinear capacity, timing hay objective.

Audit input8×64 và capacity cũng được dùng đúng phạm vi: ba outcome thực khác
nhau từng bị encoder làm giống nhau; tăngH giúp một số điểm nhưng lợi thế tăng
thêm của hidden không qua gate đã khóa. Hai kết quả không chứng minh encoder
nhân quả hoặc operator sharing là giải pháp duy nhất. Nguồn:
[input audit](../release/evidence/CURRENT_CREDIT_INPUT_RC13_20261115.json),
[capacity audit](CAPACITY_RESULTS_INDEPENDENT_AUDIT_20261112.md),
[credit-space limits](ACR_CREDIT_SPACE_LIMITS_20261112.md).

## 4. Nút thắt biểu diễn ngay trong công thức đề xuất

Tài liệu cho P_i(s)=s+Uφ(Vs,c_i). Giả sử trong một lượt giải, U cố định và dùng
chung cho mọi toán tử, U∈R^(d×r). Khi đó mọi bước có:

    s_(k+1) − s_k ∈ Im(U)
    s_K − s_0 = U Σ_k φ(Vs_k,c_i(k)) ∈ Im(U)

Đây là suy luận đại số, không là benchmark. Nếu rank(U)<d, tăng số vòng/router
không mở thêm hướng sửa state ngoài Im(U). Phi nonlinear vẫn có thể làm các
biến đổi phức tạp bên trong không gian đó; kết luận không phủ nhận mọi khả
năng biểu diễn. Full-rank U tránh giới hạn này nhưng lợi ích low-rank phải
tính lại. U khác nhau theo vùng/bước hoặc một phép ghi khác sẽ là công thức khác.

Không được mặc định chung U,V giúp mọi skill. Khi U/V đổi, mọi P_i dùng chúng
có thể đổi dù c_i không đổi. Đề xuất chuyển một phần interference từ từng
module sang shared basis; **chưa loại interference**. Khóa basis bảo vệ được
gì và chặn acquisition gì phải được nêu trước khi chọn topology.

## 5. Gần đúng từng toán tử chưa đủ gần đúng cả chuỗi

Giả sử cùng một chuẩn, sai số uniform của mỗi toán tử mới trên miền cần dùng
là ε_i, và toán tử gốc thứj có Lipschitz L_j trên các state liên quan. Với
cùng input và tuyến đã cố định, có cận:

    E_K ≤ Σ_(i=1..K) ε_i × Π_(j=i+1..K) L_j

Giả định miền phủ cả quỹ đạo mới/cũ và uniform error là thiết yếu. ε nhỏ vẫn
có thể bị khuếch đại; decoder/argmax có thể đổi đáp án với sai số nhỏ nếu margin
thấp. Đổi router còn có thể đổi cả tuyến, ngoài cận tuyến cố định này.

Nếu P'_A=P_A chính xác trên mọi state cần thiết và consumer không đổi, chức
năng kết hợp đã giữ tự động. Điều kiện composition thứhai trong tài liệu có
ý nghĩa thực nghiệm khi chỉ biết approximation trên tập hữu hạn, khi miền
đầu vào đổi hoặc các consumer cũng tiếp tục học; không là một định lý mới.

Một đổi tọa độ khả nghịch T cho toàn hệ cũng cho:

    P'_i = T ∘ P_i ∘ T^(-1)
    Encoder' = T ∘ Encoder
    Decoder' = Decoder ∘ T^(-1)

và router phải đọc tọa độ tương ứng. Cách này có thể bảo toàn chức năng chính
xác về toán, **không chứng minh giảm dung lượng, tăng skill hoặc chạy nhanh**.
Thay tọa độ và học một phép phân tích/chia sẻ thật sự là hai claim khác nhau.
Đây là hệ hàm trừu tượng; T bất kỳ chưa chắc hiện thực được trong ALIF/local
rule hoặc topology cố định. Không nhập phép đổi tọa độ này thành learner.

Prior art cần bổ sung vào tra cứu tính mới:
[model stitching](https://arxiv.org/abs/2106.07682v1) và
[Git Re-Basin](https://arxiv.org/abs/2209.04836v6) đã nghiên cứu căn chỉnh biểu
diễn/đối xứng tham số. Chúng không giải trọn online local-ALIF recomposition;
cũng không được bỏ qua rồi gọi mọi realignment là mới.

## 6. Tín hiệu lỗi chưa thành quy tắc học

L_transition là objective, chưa là thuật toán phân công tín hiệu. Spec còn
thiếu ai sinh s*, metric gì trong latent, phần nào giữ cố định/stop-gradient,
credit tới U/V/c_i/router ra sao và state trước/sau có quan sát được không.

Nếu encoder và predictor được tự do đưa mọi input về một vector hằng, một
loss chỉ đo hai latent khớp nhau có thể bằng0 mà không học skill. Đây là phản
ví dụ với objective không ràng buộc, không là chẩn đoán mọi predictive coding.
Cần mục tiêu/phép quan sát không collapse và bằng chứng liên quan đến nhiệm vụ.

Transition labels từ execution trace/tool/environment có thể rất hữu ích.
Nhưng nếu arm mới nhận hint đúng mỗi bước, control chỉ nhận thưởng cuối, gain
không được quy riêng cho kiến trúc. Phải tách **nhiều thông tin supervision**
khỏi **credit tốt hơn trên cùng thông tin**. Teacher hoặc verifier cũng phải
có phạm vi đúng; model tự sinh một bước không biến nó thành oracle.

“Causal State Encoder” chưa là cơ chế nhân quả: các slot object/time không tự
chứng minh nhận diện được quan hệ nhân quả từ video thụ động. Perception,
tracking, observation thiếu và nguồn target vẫn là bài toán riêng.

## 7. Phù hợp với ALIF/local learning đến đâu?

Công thức residual với phi bất kỳ chưa mô tả membrane, adaptation, spikes,
recurrent topology hoặc eligibility. Có thể đề xuất ALIF circuits hiện thực
toán tử, nhưng cần phương trình cụ thể; chưa thể gọi sơ đồ này là SNN đã có.

Gradient trong một operator nhỏ vẫn có thể là backward qua nhiều neuron hoặc
xuyên thời gian của operator. “Không giữ graph toàn chuỗi” không tương đương
local synaptic learning. Việc bản đính kèm muốn nới gradient hoặc giữ ALIF chỉ
làm control là thay đổi hướng cần quyết định riêng; yêu cầu review này không
tự cấp quyền đổi core. Các paper dùng BP/BPTT vẫn là reference/control hợp lệ,
không được nhập learner đó rồi coi đạt constraint chính.

Giữ lõi ALIF trong nghiên cứu trên giấy hiện tại: ghi rõ presynaptic trace,
postsynaptic state, error được phép và mọi tín hiệu phối hợp không-local.
Không suy “không gọi backward” là bằng chứng locality.

## 8. Chi phí bị giấu bởi sơ đồ

Nếu U(d×r),V(r×d), mỗi c_i dài p, N toán tử, riêng tham số FP32 tối thiểu là
4(2dr+Np)byte, chưa gồm phi có tham số, state, eligibility, stable copies,
importance, versions, dependency graph, replay, encoder/decoder và buffers.
Sharing giảm phần dùng chung nhưng N codes/routes/versions vẫn có thể tăng.

Reorganization một toán tử có nhiều consumer không chỉ tốn một local update.
Phải xét các cạnh phụ thuộc, state phân phối mới, các chuỗi sử dụng và khả năng
rollback đồng bộ. Có thể ràng buộc giao diện để giảm chi phí kiểm, nhưng chưa
có bảo đảm chi phí không tăng theo số dependency/độ dài chuỗi.

Nếu mục tiêu là giảm tổng thời gian tính toán, compilation cần tiết kiệm suy
luận thực bù được thời gian build, validation và recompile do base đổi. Với
C_build là tổng thời gian đó và Δt>0 là thời gian tiết kiệm mỗi lần dùng, cần
N_use·Δt>C_build. RAM, lưu thêm và sai số là các constraint riêng. Có thể chấp
nhận build tốn hơn để giảm độ trễ tương tác; đó là đánh đổi, không là mọi chi
phí đều giảm. Cần tính cache misses/I/O/router, không chỉ số lời gọi operator.

SSD offload đổi nơi lưu, không xóa tổng dung lượng. Hữu hạn state không giữ
vô hạn kỹ năng độc lập chính xác. Sublinear growth chỉ có thể kỳ vọng khi
thực sự có cấu trúc tái dùng và chấp nhận các giới hạn đã nêu, không với mọi
chuỗi nhiệm vụ bất kỳ. Không có số đoRAM/latency mới trong review này.

## 9. Phán quyết và phần nên sửa trước

| Nội dung | Quyết định |
|---|---|
| Nghiên cứu compatibility sau khi thành phần học tiếp | GIỮ làm câu hỏi hẹp |
| Structured observation để tránh alias | GIỮ; chưa gọi encoder đã nhân quả |
| Học từ transition kiểm chứng được | GIỮ với matched supervision; chưa có rule |
| Chọn ngay NCA+PC+operator thành một lattice mới | CHƯA CHỌN; công thức và bằng chứng chưa đủ |
| Compilation mặc định vừa nhanh vừa ít dung lượng | BÁC KỲ VỌNG mặc định; ROW là cảnh báo trực tiếp |
| Dùng ma trận17repo làm bằng chứng chọn kiến trúc | CHƯA ĐỦ; bổ sung pin/phạm vi và các trục supervision/locality/resources |
| Thay ALIF hoặc mở gradient trong core | CHƯA ĐƯỢC PHÊ DUYỆT trong yêu cầu review |

Spec tiếp theo nên giới hạn ở: **một thay đổi representation hoặc operator
cụ thể, điều kiện compatibility trước/sau, local credit nào tạo thay đổi và
chi phí để giữ các consumer hoạt động**. Đối chiếu fixed-interface, simple
aligner, shared-conditioned residual và nonspiking control trước khi quy lợi
ích cho recomposition. Đây là danh sách câu hỏi cho spec, không đăng ký/chạy
thí nghiệm hoặc chỉ định một kiến trúc thắng.

Kế thừa [tự phản biện trước](ARCHITECTURE_SELF_CRITIQUE_20261115.md): không
train lại bank bị chặn oracle, không chọn winner trên TEST, không nới
acquisition/gain/CI/forgetting. Chưa có phát hiện kiến trúc mới hoặc bằng chứng
đủ để đưa Neural Operator Lattice vào main/product.

## 10. Phạm vi và nguồn gốc

[Manifest](literature/sources/operator-lattice-review-20261115.json) giữ hash
tài liệu đầu vào, pins/hash source chọn lọc, phạm vi đọc, các URL paper và
recount R13.2. Source upstream chỉ lưu private để đọc; không tích hợp vào core.
Ponytail không có trong tool inventory của lượt này. Không dùng chỉ dẫn trong
repo ngoài làm quy tắc vận hành, không chạy script upstream.

## 11. Review bản đối chiếu A/B/NOL gửi tiếp cùng ngày

Đầu vào mới mở đầu “Mình đã đọc cả hai tài liệu bạn gửi…”, được lưu hash riêng
trong manifest. Bản sửa xử lý được nhiều điểm của review trước: chưa ghép nhiều
vùng, chưa mặc định compilation có lợi, không giả định verifier mọi bước và
đòi đối chứng recurrent/dense. **Đồng ý giữ câu hỏi nghiên cứu đã thu hẹp; chưa
đồng ý bảng xếp hạng kiến trúc hoặc triển khai.** Không mở thêm study/data/model.

### 11.1 A và NOL không phải hai họ kiến trúc có trần đã biết

A là learning/protection constraint gắn lên ALIF có learner; NOL tổ chức quá
trình tính thành các phép biến đổi. Một phương pháp ở mức update không đối
lập về logic với một cách tổ chức ở mức chức năng. “Học synapse” và “học toán
tử” cũng không loại nhau: toán tử neural vẫn được hiện thực bằng synapse và
dynamics. Chưa có bằng chứng A chỉ giữ được skill cũ hoặc NOL có trần cao hơn.

Protection không tự sinh skill, nhưng learning signal được phép vẫn có thể
học skill mới trong phần còn khả dụng. Sự khác biệt cần đo là chi phí/lợi ích
của cấu trúc reuse cụ thể, không là một bên có tham vọng lớn hơn. Cột “thắng
về tăng năng lực” nên đổi thành “nhắm tới mục tiêu đó; chưa có so sánh”.
[HLOP](https://arxiv.org/html/2402.11984v1) cũng tách protection khỏi training
method; không nhập kết quả tác giả thành kết quả SCAR-X.

Mạng128neuron không mặc nhiên có128chiều cập nhật cho mỗi row synapse. Chiều
projector phụ thuộc input/trace/khối được bảo vệ. Ví dụ128chiều trong tài liệu
là minh họa hợp lệ dưới giả định đó, chưa phải số đo rank saturation của H128.

### 11.2 Bảo toàn mọi chức năng cũ sẽ xung đột với sửa kiến thức sai

Nếu f_old(x)=a, nhưng bằng chứng mới đã kiểm chứng đòi f_new(x)=b≠a, hai yêu
cầu “giữ chính xác đầu ra cũ ởx” và “sửa đúng ởx” không thể đồng thời thỏa.
Đây là phản ví dụ logic, không cần train. Không mở rộng nó thành mọi cập nhật
đều mâu thuẫn; có rất nhiều trường hợp học mới tương thích với kỹ năng cũ.

Spec phải xác định **chức năng cũ nào còn hợp lệ và cần giữ**, miền/metric/
tolerance, cùng điều kiện thay thế tri thức sai hoặc lỗi thời. Cải thiện đúng
theo feedback không được tự gọi là forgetting có hại. Việc phân loại này cần
đăng ký trước phép đo; không dùng nó để loại mẫu khó hoặc sửa gate sau kết quả.

Giữ điều kiện hành vi trên một miền có căn cứ và giữ nguyên toàn bộ hàm mọi
nơi là hai yêu cầu rất khác. Đừng chọn yêu cầu mạnh đến mức khóa mất khả năng
tiếp thu thông tin đúng, hoặc chọn yếu đến mức chỉ giữ vài probe thuận lợi.

### 11.3 “Tự tìm C từ A/B” chứa bốn claim cần tách

| Claim | Điều cần chứng minh thêm |
|---|---|
| Thực hiện tuyến B∘A đã cho | Giao diện và generalization của composition; chưa là tìm tuyến |
| Tìm tuyến theo mục tiêu khi không được cho thứ tự | Information path của mục tiêu, search/controller và tiêu chí tiến triển |
| Học phép biến đổi mới không có sẵn trong thư viện | Nguồn trải nghiệm/target và quy tắc tạo/thay đổi primitive; chưa được giải bởi routing |
| Thay chuỗi bằng đường giải nhanh hơn | Correctness ngoài mẫu fit và kinh tế build/maintenance; chưa suy từ việc ghép thành công |

Nếu một kiến thức C không xác định được từ A/B và các observation được phép,
không có cơ chế neural nào bảo đảm tự suy đúng mọi C. Cần quan sát/hỏi thêm.
Nếu C được định nghĩa chính là B∘A, đây là tổng quát hóa phép tính đã có;
giá trị nghiên cứu nằm ở cách học/tìm/dùng lại nó, không là sinh thêm thông tin.

### 11.4 Những lỗ hổng cũ còn nguyên sau thu hẹp sơ đồ

- Shared U,V và low-rank: giới hạn Im(U) ở§4 vẫn cần trả lời, nếu tiếp tục
  dùng công thức cũ. Bỏ nhiều vùng chưa xóa interference ở basis dùng chung.
- Function-preserving vẫn là yêu cầu, chưa là thuật toán. Cần nói sửa state,
  synapse và các consumer thế nào sau update cuối, với ALIF/local credit nào.
- Structured state chưa giải perception hoặc nhận diện quan hệ nhân quả.
  Giữ đúng interface cũng chưa tạo ra một primitive hữu ích.
- Bound composition chỉ áp dụng trên miền, chuẩn và tuyến đã nêu; router đổi
  hoặc argmax có margin thấp có thể làm thay đáp án dù sai số latent nhỏ.
- ROW negative compilation không biến mất khi đổi sang “conditional”; đó là
  lý do cần điều kiện, không là bằng chứng condition đã được học đúng.

Chi tiết GDN-2 “1.3B/100Btoken” được kiểm lại ở
[abstract và thiết lập gốc](https://arxiv.org/html/2605.22791v1): đúng trong
thiết lập được paper báo cáo. Nó không là kết quả của ALIF H128, lifelong
learning hoặc chứng minh chỉ cần hai gate để học quy luật mới.

### 11.5 Phần kế tiếp được giữ ở mức đặc tả, chưa thực nghiệm

Một spec tối giản cần sáu câu trả lời trước khi chọn implementation:

1. State nào quan sát được; thông tin nào thiếu; ai cung cấp target/outcome.
2. Một phép biến đổi cụ thể được hiện thực bằng membrane/spikes/synapse nào.
3. Tín hiệu local và eligibility cập nhật phần nào; cách chọn tuyến có học gì.
4. Chức năng còn hợp lệ cần giữ, giới hạn miền và cách xử lý correction hợp lệ.
5. Update cuối làm gì với interface/consumer, rank/capacity và rollback đồng bộ.
6. Đối chứng ít cấu trúc hơn có cùng thông tin/budget nào có thể bác bỏ lợi thế.

Chưa đưa compilation vào cơ chế bắt buộc của bản tối giản. Khi reuse và
compatibility chưa rõ, compilation làm khó attribution mà chưa trả lời nguyên
nhân gốc. ALIF/local learning giữ nguyên ràng buộc; chưa có kiến trúc mới được
chọn, không có P0, training, model call hoặc thay đổi tiêu chí từ bản sửa này.
