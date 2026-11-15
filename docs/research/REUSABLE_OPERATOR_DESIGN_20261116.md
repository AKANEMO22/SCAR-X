# Reusable neural operators: thiết kế pilot 16/11/2026

**ROOT_REVIEWED_D0_ONLY; chưa GO neural/profile/fit.** Root đã duyệt thiết kế
để implement D0 thuần dữ liệu. Chỉ execute một lần sau root commit source/
protocol; chưa tạo dataset, model hoặc run tại thời điểm đổi status này.
Protocol đi kèm: [rc-operator-pilot-v1](../../experiments/rc-operator-pilot-v1.md).

## 1. Nguồn, quyền và câu hỏi hẹp

- Evidence base bất biến: `1ea065be194d44000ba795138c5917f3d13adda0`, worktree
  `scarx-scope-alif-design-20261115`; mọi kết quả cũ giữ nguyên.
- Worktree mới: `local/worktrees/rc-1116`, branch
  `research/autonomous-reusable-20261116`. HEAD khi đọc lại là
  `bee9b68c530d174f1bbc28d42cee5e11aebc81a9`; đây là worktree dùng chung với
  nhóm forensic, không phải nguồn thay thế evidence base.
- Directive UTF-8 attachment `6c52fa15-ef88-4b5b-8cf6-811e07eb38d1`, SHA256
  `57f636019eba5e5301a5b6b5d45771c911cee6f707df43073a83909d3b006b25`.
  User cho phép pilot sau protocol/preflight hợp lệ. Root đang giữ gate review
  trước mọi neural call; tài liệu này không tự mở gate đó.
- Ponytail không có trong enabled tool inventory. Không bịa rules/context.
- Không sửa main/Hermes/core/live/checkpoint cũ; không push/merge/xóa. Hermes
  đã được root tắt theo yêu cầu người dùng; pilot không tự bật lại.

Câu hỏi: cùng một model có học hai chuyển trạng thái MOVE/SWAP từ các ví dụ
đơn bước, rồi thực hiện tuyến chưa được train trên trạng thái mới bằng cùng
trọng số không? Đối tượng đo là **primitive generalization và error propagation
qua một state interface đã biết**, chưa là tự tìm tuyến, phát minh primitive,
compilation, continual learning hoặc kiến trúc mới.

## 2. Duplicate check và bằng chứng tái sử dụng

Đã đọc README, PROJECT_MAP, STATUS, AGENTS; source core/LocalALIF/
ReadoutEligibility/semantic-credit/P1b; relation-update, NOL review, prior-art
và manifest R13.2. CLI catalog được đọc trước khi dùng; searches MOVE/R13/
operator/relation không tìm thấy pilot hai toán tử fullstate đã kiểm chứng.
Catalog không đầy đủ bằng các tài liệu gần đây: kết quả search rỗng không là
bằng chứng chưa ai làm. Không mở TEST cũ hoặc chạy lại learner.

| Tiền lệ | Bằng chứng được dùng | Điều không được nhập thành claim mới |
|---|---|---|
| [R13.2 recount](literature/sources/operator-lattice-review-20261115.json) | Saved predictions: random fixed bridge 19.921875%, learned Y-only aligner 33.3984375%, decode/re-encode 98.046875%; 1,536 SNN records, bốn seed gồm một EVAL lịch sử | Không là fresh confirmation, full provenance/causal audit hoặc learned reasoning. Tám ký hiệu và label bridge không chứng minh unseen-state rule learning |
| [Relation-update draft](ALIF_RELATION_UPDATE_SPEC_20261115.md) | Hai object, hai location, MOVE; phân biệt worldstate/workingstate/weights; explicit shortcut counterexamples | Chỉ spec, 0 acquisition evidence; không chạy lại bốn-state toy hoặc gọi rule FSM là neural skill |
| [NOL review](NEURAL_OPERATOR_LATTICE_REVIEW_20261115.md) | Shared basis vẫn interference; low-rank write bottleneck; gần đúng từng bước chưa bảo đảm cả chuỗi; phải tính chi phí compilation | Không cần xây lattice/router/compiler để trả lời câu hỏi primitive/interface |
| [P1b closed result](SCOPE_P1B_TEMPORAL_TRANSFER_RESULT_20261116.md) | G2 FAIL, sign-sensitive readout, teacher/shuffle chưa khác robust; CPU nhanh hơn CUDA trên workload đó | Không đổi FAIL, không chuyển benchmark này thành rescue P1b, không suy CPU luôn tốt hơn |
| [Capacity prior art](SCOPE_ALIF_CAPACITY_PRIOR_ART_20261115.md) | Chia protection/acquisition/capacity và resource cost | Không suy H128 đủ biểu diễn MOVE/SWAP chỉ vì nhỏ RAM |

Điểm mới đối với evidence SCAR-X hiện có: năm object/bốn location; cả MOVE và
SWAP; TRAIN không có composition; fullstate input/output; TRAIN pre/poststate
tách khỏi true DEV pre/prefixstate; đo intermediate, same-data GRU và retrieval.
Đây là một adaptation diagnostic, không phải tuyên bố tính mới học thuật.

## 3. Primary literature và author repositories

Đọc chọn lọc ngày 16/11; không chạy code upstream, không tải weights/dataset,
không tái lập external results. Cột cuối giới hạn việc suy từ nguồn.

| Nguồn gốc và phạm vi đọc | Bài học dùng | Phạm vi không có |
|---|---|---|
| [DNAR, ICML2025 publisher abstract](https://proceedings.mlr.press/v267/rodionov25a.html); author repo [train.py](https://github.com/yandex-research/dnar/blob/12f3f0bd0a70568386e43b2d841422f28ed53698/train.py), [models.py](https://github.com/yandex-research/dnar/blob/12f3f0bd0a70568386e43b2d841422f28ed53698/models.py), toàn hai entrypoints ngắn | Discrete state trajectory giúp kiểm từng bước; train.py dùng AdamW/backward, model có teacher-forcing training path | Không chuyển kết quả DNAR thành local-ALIF success hoặc import training/test-selection policy của repo |
| [SCAN paper](https://proceedings.mlr.press/v80/lake18a.html), abstract và task/split description; author [README](https://github.com/brendenlake/SCAN/blob/c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f/README.md), toàn README | Random heldout gần TRAIN khác heldout primitive/template/length; báo riêng state, sequence và shift | Không chạy SCAN hoặc suy task state-update này là language generalization |
| [Neural Programmer-Interpreters](https://arxiv.org/html/1511.06279v3), abstract, §3–3.1 và supervision context | Shared recurrent core, program arguments và rich execution supervision là prior art; phải công bố ai cấp route/state labels | Không là evidence cho local learning; pilot không học controller/NPI stack. Chưa xác minh author code nên không sử dụng code NPI |
| [E-prop paper abstract](https://arxiv.org/abs/1901.09049v2); [official author README](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/README.md), repo overview/numerical-verification description | Forward eligibility cộng learning signal có tiền lệ; official repo phân biệt hardcoded và autodiff implementations | Không gọi SCAR-X approximation là exact e-prop/BPTT gradient; không port TensorFlow/autodiff vào ALIF |
| ROW [README E6.2/status](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/README.md), [E6.2 report](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/reports/e6_2_compiled.json), [adequacy](https://github.com/doeixd/reusable-operator-worlds/blob/e846633e1d1a5cfa60e8c83646f68175e810a1ba/DESIGN_ADEQUACY.md) đọc bảng negative và necessity/opportunity/discrimination | Compilation dùng capacity lớn hơn vẫn không đạt kinh tế ở các world đã thử; phải kiểm cheap impostors, measurable headroom và null trước train | Author claims/source review, chưa independent reproduction; không nhập mọi ROW result hoặc dùng nhiều operator calls như proxy tổng compute |

SCAN/e-prop HEAD được đọc bằng `git ls-remote`, pin ở URL trên. DNAR/ROW pin
kế thừa manifest 15/11 và source được đọc lại. Không dùng blog, Reddit hoặc
search snippet làm bằng chứng kỹ thuật.

## 4. State, operators và neutral interface

Worldstate `s=(s0,...,s4)`, mỗi `si∈{0,1,2,3}`; có `4^5=1,024` fullstates.
MOVE(i,l) đặt object i ở l, giữ bốn object khác. SWAP(i,j), i<j, đổi location
hai object, giữ ba object khác. Hai vật có thể cùng location; oracle không
áp đặt occupancy constraint hoặc thực hiện physics. Đây là verified-event
semantics, không perception/action execution ngoài thế giới.

Codec 64 channels, giống nhau cho cả bốn neural arms:

| Channels | Nội dung |
|---|---|
| 0–19 | Năm nhóm location one-hot, mỗi nhóm bốn channels, full prestate |
| 20–21 | MOVE/SWAP one-hot |
| 22–26 | object argument thứ nhất |
| 27–31 | object argument thứ hai; zero cho MOVE |
| 32–35 | destination location; zero cho SWAP |
| 36–63 | zero |

Lặp cùng input tám tick để hiện thực một primitive, reset neural workingstate
trước mỗi primitive. Không input tương lai, sequence ID, target state, operator
result hoặc hint. Core forward ALIF nguyên phương trình hiện có.

Head là **một affine map ra 20 fullstate logits**, reshape `(5,4)`. Năm softmax
chỉ dùng cho loss; quyết định bằng argmax từng nhóm, tie chọn location nhỏ
nhất. Không residual copy/attention/crossbar/binding mask/chọn affected rows
từ arguments. Mọi object output đều do model dự đoán.

Feedback `encode_state(decode_state(logits))` chỉ serialize chính tuple model
đã dự đoán. Hàm feedback không nhận op/args hoặc oracle. Không copy nguyên
những slots được xem là unaffected. Route AB/BA/... do evaluator cấp công khai;
không gọi oracle trong candidate forward/decoder/handoff.

Oracle dùng làm control, tạo TRAIN single-transition labels và chấm DEV.
DEV intermediate labels chỉ evaluator đọc; không update hay chọn checkpoint.

## 5. Điều interface này chứng minh được và không chứng minh được

Nếu `D(E(s))=s` và mỗi primitive đúng trên mọi state thật cần đi qua, induction
cho cả chuỗi đúng. Nếu prefix prediction đúng thì input model ở bước kế tiếp
trùng input oracle, vì workingstate được reset. Với deterministic inference,
free-run và oracle-input predictions khi đó phải bằng nhau.

Vì vậy **không tồn tại một learned composition mechanism riêng được nhận diện
bởi pilot này**. Cấu trúc fold và route là software được công bố. H3 ở đây chỉ
là engineering compatibility cộng zero-additional-training generalization.
H2 được chẩn đoán theo primitive errors trên trajectory distribution hoặc lỗi
codec/harness; không tự gán composition failure cho ALIF representation.

Fullstate prediction ngăn shortcut chỉ đo destination/last-event, nhưng ba hoặc
bốn slots thường được giữ. Per-object accuracy có thể cao khi exactstate bằng
0; chỉ exact fullstate/path là primary. Cặp AB/BA có cùng state/arguments nhưng
khác kết quả để phát hiện bỏ order. No-op/cancellation báo riêng.

## 6. Bốn arms, cùng dữ liệu và giới hạn attribution

Tối đa ba init/order seeds `261181,261187,261191`; một DATA seed `261180` cố
định cho mọi arm/seed. Bốn arms × ba seeds = **12 trajectories tối đa**.

| Arm | Phần học | Vai trò |
|---|---|---|
| CURRENT_ALIF | Current reward-modulated STDP math + trained affine head, sparse recurrent weights được học theo rule gốc | Baseline current core/rule với interface adapter nghiên cứu; không gọi đây là product head không đổi |
| FROZEN_ALIF_READOUT | Freeze input/recurrent/topology; head update giống candidate | Tách gain từ readout/random features khỏi hidden learning |
| SHARED_LOCAL_ALIF | Một H128 ALIF dùng chung hai operators, W_in-only own-neuron forward eligibility × pre-update readout-error feedback; train head, fixed W_rec | Candidate; không autograd/BPTT hoặc temporal/cross-neuron reverse credit |
| SHARED_GRU | Một GRU H36 và affine head, BP chỉ trong isolated control | Codec/task/harness competence control ở gần cùng số tham số |

Không thêm allocation control, teacher-supervised composition arm, architecture
sweep, H/lr/seed cứu. ALIF representation witness H128 **NOT_PROVEN**. GRU đạt
mà ALIF không đạt sẽ xác định một learner/inductive-bias gap trong budget này,
không chứng minh mọi ALIF không biểu diễn được rule.

Candidate-vs-frozen là contrast matched để quy hidden contribution. CURRENT
ALIF khác optimizer/update coverage/recurrent plasticity; GRU khác neuron,
learner, trainable fraction. So sánh chúng là utility control, không pure
causal ablation của một yếu tố. Hyperparameters khóa trong protocol.

## 7. Data và phép đo để không ghi nhớ cả bảng

Protocol chọn 512 TRAIN examples (256 mỗi operator), hai epoch cố định:
1,024 updates/trajectory, tối đa 12,288 neural optimizer/update calls cả pilot.
TRAIN input và target fullstates đều thuộc TRAIN partition; mọi true DEV
initial/intermediate/prestate ở ngoài partition đó. Giữ cả input/target
intersection receipts. Nếu không đủ pool/coverage thì STOP_DATA, không nới.

IID và occupancy-shift báo riêng. Primary SHIFT lấy exactly bốn object cùng
location; bốn all-five states không có changed SWAP nên không được chọn làm
initial evaluation/composition states. Terminal/prefix có thể all-five nếu
đáp ứng path constraints và phải được tag; riêng all-five initialstate evaluation
NOT_MEASURED. Toàn occupancy≥4 vẫn bị loại khỏi TRAIN. Shift là distribution
của initialstate; sau MOVE có thể đổi stratum, mỗi bước phải tag.
Tất cả chuỗi dài 2/4 đều held out vì TRAIN chỉ một primitive. Không TRAIN bằng
composition target, oracle intermediate, free-run self-label hoặc replay DEV.

Exact lookup/whole-tuple 1NN không thể emit fullstate ngoài các targets đã lưu;
trên split này nó yếu theo thiết kế. Không dùng nó làm bằng chứng thắng memory.
Thêm **k=3 command-matched kNN, per-slot vote** từ cùng TRAIN examples; vote có
thể sinh tuple mới, không hardcode MOVE/SWAP/diff/patch. Báo storage và latency,
không tăng k/tune sau DEV. Symbolic oracle vẫn 100% ceiling theo định nghĩa.

Metrics: exactstate từng operator và step; all-prefix/path exact; final exact;
affected-slot correctness; unaffected preservation; noop; argmax margins/NLL;
free-run vs oracle-input evaluation; AB/BA order twins; frozen/reverted W_in
diagnostic sau candidate training với cùng head. Revert chỉ đánh giá, không
fit và không tạo thêm trajectory.

## 8. Falsifiers và thứ tự STOP

1. Codec alias/roundtrip lỗi, candidate gọi oracle, leakage/split overlap hoặc
   inadequate shortcut gap ⇒ INVALID/STOP trước training.
2. Không đủ resource/profile/updates trong envelope ⇒ BUDGET_STOP; không đổi
   backend âm thầm hoặc cắt dữ liệu rồi vẫn PASS.
3. Candidate một operator không đạt 80% exactstate IID ở bất kỳ seed nào ⇒
   PRIMITIVE_GATE_FAIL; không chạy composition candidate, không thêm epoch/H/lr.
4. Cả GRU lẫn candidate không acquire ⇒ chưa phân biệt task/budget/learner;
   không kết luận ALIF thiếu capacity. GRU acquire nhưng candidate không ⇒
   narrow local-learner gap, không architecture impossibility.
5. Candidate acquire nhưng oracle-input trajectory primitives sai ⇒ primitive
   distribution gap; free-run đúng-prefix khác oracle-input ⇒ harness/interface
   defect. Không gọi đây là learned-composition failure đã xác định.
6. Candidate không hơn matched frozen ở exactstate hoặc revert không giảm ⇒
   hidden contribution chưa đo được; nhiều synapse đổi không thay kết luận.
7. Không hơn GRU/kNN ở quality hoặc resource frontier ⇒ diagnostic only,
   không architectural advantage. Compilation/lifelong/H5 vẫn NOT_MEASURED.

Ba seed chỉ hỗ trợ pilot descriptive; không thể claim population p<.05 từ
paired sign/permutation test với n=3. Báo từng seed, mean/range, counts và
state-cluster bootstrap có nhãn conditional-on-this-benchmark. Không gọi 12
trajectories hoặc hàng nghìn prediction là 12/hàng nghìn independent replications.

## 9. Chi phí phải đếm

ALIF D64/H128/fan8/K20 có `128*64 + 128*8 + 20*128 + 20 = 11,796`
coefficients = 47,184 FP32 weight bytes. Candidate trainable W_in+head =
10,772, fixed recurrent = 1,024; frozen trainable head = 2,580. GRU H36
PyTorch layout có `3hD+3h²+6h+20h+20 = 11,756` trainable coefficients,
nhỏ hơn tổng ALIF 40 (0.34%); không matched trainable fraction/FLOPs.

Một ALIF forward primitive tám tick khoảng `8*(HD+Hfan)+KH = 76,288` MAC,
chưa tính unused native probability calls, spikes, eligibility/optimizer,
Python/runtime/serialization. GRU H36 khoảng `8*3h*(D+h)+20h = 87,120`
affine MAC, chưa gates/backward. Đây là static arithmetic estimates, chưa
measured latency/energy. Code phải báo cả actual calls nếu native step tính
head thêm, không bỏ nó khỏi receipt.

Tính riêng arrays/state/eligibility/momentum/energy/stable copies, full process
tree RSS/private commit, CUDA context/cache/VRAM, profiler/evaluator/controller,
checkpoint/rollback, training examples, logs và disk. kNN raw compact records
512×14 uint8 khoảng 7,168 bytes trước container/index/runtime; không đồng nhất
con số này với process RAM. Neural không được mặc định tiết kiệm hơn memory.

Cap mới: 300s total active (sum controlled-process active wall, không được nhân
ba nhờ parallel), experiment tree RAM <1,000,000,000 bytes, tối đa ba workers;
global research RSS <2,000,000,000 bytes khi forensic còn chạy. Backend sau
matched representative profile; CPU thắng thì không dùng GPU cho arm đó.
Không reserve dummy VRAM, không chiếm/kết thúc phiên live.

## 10. Handoff review

Root cần kiểm protocol/data-source/hyperparameters và cấp GO no-learning
preflight trước; GO đó chưa là kết quả gate. Neural profile rồi training chỉ
sau prerequisites được lưu/hashes hợp lệ. Mọi sửa sau outcome là protocol mới,
không chỉnh v1 để cứu FAIL. Hiện trạng: **0 new model calls, 0 fits, 0 updates,
0 dataset generation; design/protocol only**. H5 acquisition acceleration,
retention/continual stages, novel operators và useful advantage over small
LLM+RAG+skill library đều **NOT_MEASURED**.
