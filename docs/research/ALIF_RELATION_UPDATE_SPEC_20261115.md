# Đặc tả trên giấy: ALIF học phép cập nhật quan hệ vị trí

Ngày15/11/2026. Đầu vào người dùng mở đầu “Mình đã đọc §11 tại commit
ae584c48…”. Giữ hướng một phép biến đổi, chưa xây NOL/router/compiler.
**CONTRACT_DRAFT_ONLY**: không code encoder/decoder/guard, tạo dataset,
forward, fixture, train, refit, benchmark hoặc đổi main/Hermes/ALIF.

Đặc tả này cụ thể hóa phép toán đề xuất; không là protocol đã duyệt hoặc
bằng chứng đã học được. Master duy nhất: [RESEARCH_PLAN](../RESEARCH_PLAN.md).
Điều kiện bảo toàn kế thừa [ALIF interface preservation](ALIF_INTERFACE_PRESERVATION_SPEC_20261115.md),
không lùi về công thức margin đối xứng thiếu xử lý tie.

## 1. Phân biệt ba thứ đang thay đổi

| Đại lượng | Vai trò | Khi gặp một sự kiện đã biết |
|---|---|---|
| State của thế giới s | Đối tượng đang ở đâu theo bằng chứng | Cần cập nhật theo sự kiện |
| State làm việc neural h=(v,a,z,f) | Mang thông tin trong đoạn xử lý | Thay đổi bằng forward ALIF |
| Synaptic parameters Θ | Kỹ năng xử lý một loại sự kiện | Không cần sửa sau mọi lần vật di chuyển |

Model cần học **quy luật cập nhật**, rồi áp dụng nó với Θ cố định cho trải
nghiệm mới. Đổi trọng số để ghi từng vị trí cụ thể không tự chứng minh đã học
quy luật. Bảo vệ kỹ năng MOVE cũng không có nghĩa đóng băng một fact “vật ởA”
khi một sự kiện mới hợp lệ cho biết vật đã tớiB.

Học Θ chỉ xét từ lỗi/outcome đã có căn cứ, sau đoạn forward; không sửa giữa
một lượt trả lời. Không biến độ tự tin hoặc đầu ra tự sinh thành nhãn đúng.

## 2. Ngữ nghĩa tối giản của phép biến đổi

Miền phân tích hữu hạn, không là cấu hình benchmark đã chọn:

- Hai object slots O={o0,o1}; hai vị trí L={L0,L1}.
- State s:O→L đầy đủ, đúng một vị trí cho mỗi object.
- Event e=MOVE(o,l) nghĩa là **đã quan sát/xác minh di chuyển xong**, không
  phải ý định, câu lệnh chưa chạy hoặc dự đoán vật có thể di chuyển.

\[
P^*(s,\operatorname{MOVE}(o,l))(p)=
\begin{cases}l,&p=o,\\s(p),&p\ne o.\end{cases}
\]

Hai thành phần của rule đều cần: sửa đúng object và giữ object không liên
quan. Idempotence: MOVE(o,l) hai lần có cùng finalstate như một lần. Hai MOVE
trên object khác nhau giao hoán; trên cùng object tới hai vị trí khác nhau
thì thứ tự quan trọng. Đây là đặc tính của target, không là kết quả model học.

Unknown location, mất dấu, nhiều vật trùng định danh, MOVE thất bại, quan sát
mâu thuẫn và timestamp không có thứ tự nằm ngoài miền đầy đủ này. Không tự
mặc định một vị trí để đạt accuracy. Khi mở miền, phải sửa contract và target
policy trước; không tuyên bố bản tối giản đã giải các tình huống đó hoặc vision.

## 3. Interface đề xuất, chưa hiện thực

### 3.1 Chuỗi input giữ thứ tự, không tính sẵn đáp án

Một encoding trên giấy có cùng width64 với Hybrid:

| Channels | Nội dung |
|---|---|
| 0..2 | One-hot loại INIT/MOVE/END |
| 3..4 | One-hot object slot cho INIT/MOVE |
| 5..6 | One-hot location cho INIT/MOVE |
| 7..63 | Zero/reserved; không nhét target hay task-ID vào đây |

END không mang object/location. Input một đoạn là hai INIT facts cho s0,
sau đó các MOVE theo thứ tự thực và một END. Encoder chỉ serialize từng bản
ghi, không fold events thành finalstate, không đưa nhãn/quỹ đạo tương lai hoặc
đáp án intermediate vào input. Canonicalizer cung cấp object IDs; lợi ích
nhận diện/binding đến từ đó phải được ghi, không quy cho ALIF học từ pixel.

Gọi E=(e1,…,en). Target cuối là F*(s0,E)=fold(P*,s0,E). Trường hợp n=1 là
P_update ban đầu. Candidate ALIF cần xử lý INIT/E theo thời gian rồi sinh
finalstate; không được runtime áp dụng P* thay cho model trong candidate arm.

### 3.2 Đầu ra đủ để consumer đọc ổn định

Finalstate hai object có bốn giá trị. Đề xuất một decoder bốn lớp, mapping
cố định theo thứ tự(s(o0),s(o1)): (L0,L0),(L0,L1),(L1,L0),(L1,L1). Consumer
đọc tuple đã decode để trả lời vị trí object được hỏi; việc index tuple là
phần mềm cố định, không là skill neural mới hoặc learned composition.

Decoder/mapping, readout weights/bias và normalization phải được khóa trong
phạm vi bảo toàn. Cách khởi tạo/chọn readout có cho phép acquisition hay không
chưa chứng minh. Không giả định một readout ngẫu nhiên cố định luôn đủ tốt.

### 3.3 Những phần chưa có trong code hiện tại

[features()](../../src/scarx/hybrid/learner.py#L24) hiện hash unigram/bigram rồi
lặp cùng vector8tick; nó không phải event codec vừa mô tả. Head action có6
nhãn, head relevance có2; [masked readout](../../src/scarx/hybrid/learner.py#L173)
chưa có bốn nhãn joint-location. Không được relabel sáu action live thành vị
trí hoặc nói encoder/readout đang chạy đã hỗ trợ contract này.

Một adapter nghiên cứu với codec/head khác là thay đổi interface cần đặc tả/
duyệt riêng trước khi code. Physical outputs8 không tự tạo mapping đúng.
Spec chỉ cho thấy có thể diễn tả target hữu hạn, chưa chứng minh khả năng
biểu diễn/optimization/local learning trên trọng số ALIF cụ thể.

Forward hiện reset state mỗi đoạn và giới hạn256rows. Việc cung cấp cả lịch
sử từ INIT là episode inference, không phải continuous stream không giới hạn.
Feed finalstate đã decode sang INIT lượt sau là một chế độ khác, có external
state và lỗi tích lũy; không trộn hai chế độ khi báo memory/reuse. Không mở
chế độ streaming mới hoặc tăng sequence limit trong lượt đặc tả này.

## 4. Những phản ví dụ chứng minh task cần giữ gì

Các ví dụ trên giấy, không là dataset đã tạo hoặc model predictions:

1. Cùng s0=(L0,L0), hai lịch sử:
   - MOVE(o1,L1), MOVE(o0,L1) →(L1,L1).
   - MOVE(o1,L0), MOVE(o0,L1) →(L1,L0).
   Finalevent giống nhau nhưng finalstate khác: chỉ chép destination của
   event cuối không đủ giữ cả hai object.
2. MOVE(o0,L0),MOVE(o0,L1) và thứ tự ngược lại có cùng multiset events nhưng
   vị trí o0 cuối khác nhau. Encoding hoàn toàn bỏ thứ tự không thể giải cả hai.
3. Đổi object của event từo0 sango1 trong cùng prestate có thể đổi phần tuple
   cần cập nhật. Chỉ nhìn location mà bỏ identity không đủ.

Các phản ví dụ loại shortcut cụ thể, chưa chứng minh bắt buộc hiddenlearning
hoặc một architecture mới. Readout của random recurrent features có thể đủ
cho miền nhỏ; phải tách khi có protocol sau này. Renaming trong hai slot không
là generalization sang số object vô hạn hoặc tự nhận diện object mới từ video.

## 5. Vì sao đây chưa đủ là một campaign kiến trúc

Một finite-state machine giữ một bit location cho mỗi object và cập nhật bit
được chỉ định giải chính xác toàn bộ miền target trên giấy: bốn worldstates,
hai bit trạng thái ngữ nghĩa. Đây không phải dự toán tổng RAM của một chương
trình; còn code/runtime/input và metadata. Không chạy baseline trong lượt này.

Do đó, bài vị trí hữu ích để phân tích interface/binding/thời gian, không tự
chứng minh khả năng suy luận sâu, lifelong learning hay lợi thế hơn hệ nhớ
có cấu trúc. Lời giải bằng state update đơn giản cũng là hướng kỹ thuật cần
đối chiếu, không được bỏ đi để làm neural trông cần thiết hơn.

Không đăng ký mục tiêu “hơn5pp” so một oracle có accuracy100%; đó là cận không
thể vượt, tương tự toolbank đã đóng. Oracle/target definition và đối chứng
học cùng thông tin có vai trò khác nhau, phải ghi rõ lợi thế prior knowledge.
Nếu study tương lai chỉ giải lại đúng rule máy trạng thái đã có mà không phân
biệt cơ chế còn mở, không mở campaign mới. Không nới gate để biến diagnostic
toy thành qualification toàn dự án.

Task QA tổng hợp để tách các kỹ năng có tiền lệ
[bAbI](https://arxiv.org/abs/1502.05698v10). Nguồn này được đọc abstract/metadata
để xác định loại benchmark, chưa tái lập hoặc audit task data/code. P_update
trên không là một nhiệm vụ mới học thuật chỉ vì nay viết bằng ALIF.

## 6. Cách học được xét và những gì phải giữ nguyên

Ứng viên signal×eligibility kế thừa đã có; không thêm learning rule trong
lượt này. Chỉ phân tích nhánh W_in-only; freeze W_rec/topology/readout/
normalization/config/encoder cho claim bảo toàn. [LocalALIF.learn](../../src/scarx/hybrid/learner.py#L220)
live sửa nhiều phần hơn; không áp spec này trực tiếp cho nó. [CreditALIF hidden](../../src/scarx/evaluation/semantic_credit_alif.py#L57)
gần phạm vi W_in-only nhưng codec sáu semantic intents khác bài này, không là
implementation của relation operator.

Nếu có target lớp y đúng ở cuối đoạn, local readout error là δ=y_onehot−p;
signal về hidden phụ thuộc readout cố định và đường normalization đã công bố.
Candidate synapse direction dạng m_jE_ji đi qua smoothing/normalization/
clipping nào phải được khai báo nguyên vẹn. Target intermediate từng bước,
nếu được cấp, là thêm supervision; mọi control cần được tiếp cận công bằng.
Không mặc định loss đúng từng bước hoặc encoder biết sẵn finalstate.

Học được một vài transition đúng chưa chứng minh operator reuse. Câu hỏi
attribution giữ lại cho thiết kế sau: learned input synapses so frozen hidden,
readout-only, cùng encoder/head phù hợp và same-budget nonspiking/memory
controls; revert/shuffle phải ảnh hưởng đúng phần muốn kiểm. Chưa có seed,
runner, tập train/dev/test hoặc ngân sách thực thi được đăng ký ở đây.

## 7. Đối tượng bảo vệ và sửa sai

| Trường hợp | Hành động về mặt contract |
|---|---|
| Kỹ năng cũ và old trajectory còn đúng, model đã thực hiện đúng | Có thể đưa vào tập bảo vệ với provenance/phiên bản trước khi đo |
| Old model trả lời sai trên input có nhãn đúng | Không khóa spike sai chỉ vì nhãn được verified |
| Có event mới MOVE tới vị trí khác | Đổi worldstate theo input mới là hành vi đúng, không là forgetting |
| Bằng chứng cũ bị sửa, sai thời điểm/phạm vi | Version/correction policy xác định chức năng còn hiệu lực; chưa tự sửa gate |
| Outcome chưa rõ hoặc mâu thuẫn | Giữ trạng thái unresolved; không tạo training label bằng tự tin |

Quan sát text/video không tự là evidence transition đã thực hiện. Bản tối giản
dùng semantics đã biết để định nghĩa oracle, chưa cung cấp verifier cho mọi
chuyển động thực tế. Chất lượng vision/ASR và object tracking là câu hỏi riêng.

Điều kiện(C) trong spec giữ giao diện dùng h+Δwᵀr≥0 cho oldspike1 và<0 cho
oldspike0. Như vậy xử lý được tie do comparator>=. Chỉ bảo vệ old sequences
và consumer frozen, không giữ vị trí trước MOVE khi input đã thay đổi. Tính
tổng physical displacement so reference, không kiểm từng increment riêng.
Chưa có FP32/CUDA certificate hoặc guard được triển khai.

## 8. Những claim phải tách và điểm chưa giải quyết

- State tracking trong một đoạn ≠ synapse đã học skill mới.
- Nhớ rule sau save/load ≠ nhớ mọi sự kiện video; khi context bị xóa, replay
  input hoặc worldstate SSD là nguồn thông tin phải tính riêng.
- Consumer đọc cùng tuple contract ≠ Functional Recomposition: bản này chưa
  đổi representation rồi tự căn chỉnh các consumer.
- Giữ oldspikes trên S ≠ giữ mọi futureinput hoặc mọi skill cũ.
- Hướng local bị chặn ≠ hết capacity; sign constraints khác nullspace giữ drive.
- Đạt100% trên bốn trạng thái ≠ vượt RAG/LLM hoặc có intelligence đa năng.

Các blocker trước bất kỳ implementation/study nào: adequate discriminating
question; codec/head contract tách khỏi action live; frozen readout feasibility;
nguồn feedback và matched supervision; numerical guard/persistence accounting;
coverage còn hợp lệ và chi phí chấp nhận/từ chối. Chưa giải các blocker bằng
một experiment mới hoặc đổi ngưỡng đã đăng ký.

## 9. Quyết định

**ACCEPT làm đối tượng đặc tả hẹp; chưa ACCEPT làm benchmark quyết định kiến
trúc hoặc permission chạy.** Contract đã cụ thể hơn một sơ đồ NOL. Chưa có
bằng chứng acquisition, retention, useful representation hoặc khả năng
Functional Recomposition. Phần implementation vẫn chưa có, main/live nguyên vẹn.

[Manifest](literature/sources/alif-relation-update-20261115.json) giữ hash đầu
vào, source/ranges và phạm vi primary-source reading. Không chạy code upstream,
tạo dữ liệu hoặc model call; các ví dụ và finite-state bound là suy luận.
