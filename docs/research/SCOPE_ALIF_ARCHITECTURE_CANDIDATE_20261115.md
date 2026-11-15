# SCAR-X: SCOPE-ALIF — kiến trúc ứng viên có hợp đồng giao diện và vùng học dẻo

**Reviewer amendment v2, 15/11/2026:** phần §10 bên dưới sửa các chỗ chưa đủ
chi tiết ở §4/§7 và giới hạn claim ở §3/§6. Original v1 được giữ byte-exact
ngoài repo trong log review. Tất cả vẫn là đặc tả chưa triển khai/chạy.
Trạng thái: RESEARCH CANDIDATE / REQUEST CRITICAL REVIEW. Không phải phát minh đã xác minh, không phải model đã hoạt động.

Nguồn khóa: SCAR-X commit b3a48983e148024c75b093d0413140c255c78073, đặc tả docs/research/ALIF_READOUT_BEHAVIOR_PRESERVATION_SPEC_20261115.md; kế thừa ALIF_INTERFACE_PRESERVATION_SPEC_20261115.md, ALIF_RELATION_UPDATE_SPEC_20261115.md, các negative verdict R13.2, ACR và ROW E6.2. Giữ nguyên main, Hermes, model live, kết quả và ngưỡng cũ. Chỉ thiết kế trên giấy; không có quyền tự mở experiment.
0. Tại sao phải thay giả thuyết?
Bản core–readout đã chứng minh với readout/normalization cố định: giữ spike => giữ feature => giữ probability => giữ decision. Giữ decision ngược lại không yêu cầu giữ spike. Tuy nhiên, tập thay đổi weight cho cùng decision không lồi/không đơn điệu, và không có sound bound rẻ từ physical Delta W_in tới các spike flips. Null(A_o) của feature liên tục không phải không gian update reachable của mạng hard-spike. Guard replay trên 100.000 ticks còn có chi phí lớn. Không được gọi project-to-nullspace hay binary-search bước học là lời giải.

Mục tiêu mới, có điều kiện: tách bằng cấu trúc nhân quả những ảnh hưởng của phần đang học khỏi đường đọc một kỹ năng đã xác minh. Sau đó hỏi liệu có thể học một phép biến đổi mới và tái sử dụng kỹ năng cũ mà không cần đọc lại toàn bộ lịch sử chỉ để bảo vệ một interface đã được cô lập. Chúng ta không giải được toàn bộ continual learning bằng việc cô lập.
1. Quyết định ứng viên
SCOPE-ALIF = Stable Contracted Output + Plastic Execution ALIF. Hai quần thể neuron cùng một core ALIF, tổng neuron cố định H=H_s+H_p. Kết nối hồi quy có định hướng: quần thể ổn định S gửi thông tin sang quần thể dẻo P, nhưng P không gửi trở lại S trong giai đoạn học an toàn. Kỹ năng cũ được đọc chỉ từ S qua head/normalizer bất biến. Kỹ năng mới có thể dùng S lẫn P qua một head nghiên cứu khác.

Đây là kiến trúc ứng viên có tiền lệ gần với progressive networks / Progress & Compress, modular continual learning và chức năng cách ly; không tuyên bố novelty. Điểm nghiên cứu mở là đường học ALIF-local, compatibility và việc thoát khỏi vùng dẻo hữu hạn. Nếu không vượt đối chứng mạnh cùng ngân sách thì NO-GO.
2. Phương trình forward, không có ký hiệu toán tử mơ hồ
Đặt h_t^S=(v_t^S,a_t^S,z_t^S,f_t^S), h_t^P tương tự. Cho cùng event encoder x_t. Trong mỗi quần thể dùng neuron ALIF thật của SCAR-X với ngưỡng thích nghi, subtractive reset:

u_t^S = lambdav_(t-1)^S + W_in^Sx_t + R_SS*z_(t-1)^S

theta_t^S = theta_0 + beta*a_(t-1)^S

z_t^S = 1[u_t^S >= theta_t^S]

v_t^S = u_t^S - z_t^Stheta_t^S; a_t^S = rhoa_(t-1)^S + z_t^S

u_t^P = lambdav_(t-1)^P + W_in^Px_t + R_PSz_(t-1)^S + R_PPz_(t-1)^P

theta_t^P = theta_0 + beta*a_(t-1)^P

z_t^P = 1[u_t^P >= theta_t^P]

v_t^P = u_t^P - z_t^Ptheta_t^P; a_t^P = rhoa_(t-1)^P + z_t^P

f_t^{S,P} = gamma*f_(t-1)^{S,P} + (1-gamma)*z_t^{S,P}.

Đồ thị bắt buộc: R_SP=0 (hàng đích S, cột nguồn P). Sparse fan-in phải thực sự không có cạnh P->S, không chỉ đặt hệ số nhỏ. Không có shared layer-norm hay clock/threshold được điều chỉnh từ activity của P. Không được cập nhật W_in^S, R_SS, readout cũ hoặc normalization của S trong giai đoạn cách ly.

Head còn hiệu lực: ell_old = W_old^S * q^S + b_old với q^S=clip((f^S-mu^S)/scale^S,-5,5). Mọi cột từ P đến head cũ bằng 0 theo topology. Nếu consumer cần probabilities, p_old=softmax(ell_old) phải giữ; nếu consumer đọc trạng thái, điện thế P hay new head thì định lý không áp dụng.

Head ứng viên: ell_new = U_Sq^S + U_Pq^P + b_new. Vừa nhận tri thức S, vừa thử học quy luật mới trong P. Head mới không được tự đổi lựa chọn output/head cũ qua một router không kiểm chứng.
3. Mệnh đề bảo toàn có điều kiện (không phải kết quả thực nghiệm)
Với cùng x_1..x_T, cùng initial state S, clock và parameters S, cùng topology R_SP=0, cùng normalization/head cũ; chỉ sửa W_in^P, R_PS, R_PP, head mới và state P. Bằng quy nạp trên t, toàn bộ (u^S,z^S,a^S,v^S,f^S) không đổi, nên q^S, ell_old và p_old không đổi trong số thực, cho mọi chuỗi được mô hình hóa bởi cùng contract, không chỉ tập replay hữu hạn. Không cần giữ spike P hoặc replay toàn bộ kỹ năng cũ để chứng minh bất biến của head S.

Đây là noninterference by construction, không chứng minh kỹ năng S đúng, không chứng minh làm tốt tác vụ mới và không bảo vệ controller ghép hai head. Với FP32/CUDA, deterministic S path cùng backend/order thường có thể cho parity, nhưng không có chứng nhận bitwise khi backend/precision/encoder thay đổi. Không được gộp với guarantee cho toàn Hermes.

Giới hạn nghiêm trọng: Nếu mạng ban đầu là LocalALIF H128 với tất cả 128 neuron góp phần tạo head cũ và các cạnh recurrent trộn, không thể đơn giản tách S/P mà giữ output cũ. Cần một core nghiên cứu riêng được học/khởi tạo theo topology mới; không có migration proof từ checkpoint live.
4. Quy tắc học thực, không dùng backprop xuyên thời gian
Giai đoạn thử lý thuyết: chỉ các synapse có hàng đích P được cập nhật; cấu trúc R_SP=0 bất biến. Eligibility E^P_(j,i) theo forward local ALIF e_v,e_a,e_f trong src/scarx/hybrid/learner.py, không gọi BPTT/RTRL. Tín hiệu đầu ra mới delta=one_hot(y_verified)-softmax(ell_new), m_j= [(U_P)^T delta]j * g_j (g_j là hệ số normalization/clip cụ thể như code). Ứng viên direction d(j,i)=m_j E^P_(j,i); áp dụng smoothing/syn_energy/clipping/weightbound giống local learner H128 đã khóa, chỉ mask hàng P và edges được phép. Readout của new head có thể học bằng supervised local delta rule trên q^S,q^P; head old đóng băng.

Đây chỉ là đường cập nhật ứng viên tương thích locality trong nghĩa e-prop-like (feedback W_out^T không phải thuần một synapse). Nó không bảo đảm acquisition; label cuối chuỗi không có target trung gian, temporal credit yếu và gating m_j có thể cho signal zero. Cần ghi rõ nguồn nhãn, stop-gradient, clipping và normalization nếu sau này viết spec triển khai. Không được lén cấp intermediate state oracle cho candidate mà baseline không nhận.

Lợi ích về reuse chưa được chứng minh: Vì R_PS lấy spike S làm input, P chỉ có đường truy cập tín hiệu cũ, chưa được bảo đảm giải mã hoặc dùng được nó. Đây chính là cảnh báo từ R13.2 (ghép spike trực tiếp yếu hơn cầu nối đã chuẩn hóa nhãn trên toy 8 ký hiệu). Phải kiểm causal ablation R_PS=0 cùng budget; đối chiếu với interface chuẩn hóa được quy định rõ; không cho cầu nối nhãn/decoder oracle vào candidate rồi quy thắng lợi cho learner. Nếu không có cơ chế P sử dụng S một cách đúng, kiến trúc chỉ là hai nhóm neuron cách ly chứ không phải skill composition.
5. Cập nhật sự kiện và operator được học
Khởi điểm giữ contract INIT -> MOVE -> END: x_t mã hóa event type, object slot, location và thứ tự; input codec không fold ra final state. World state thay đổi khi MOVE xảy ra, không có nghĩa synapse cập nhật ở từng MOVE. Core P cần học quy luật transform khi có verified outcome và áp dụng với Theta cố định trên episode mới. Mọi module/slot được cấp từ codec không được giấu như khả năng bind của neural learner.

Miền hai object hai position có finite-state oracle chính xác với hai bit thông tin; bài này chỉ kiểm khả năng operator, không được lấy 100% làm chứng minh SNN tốt hơn FSM hay tăng trần suy luận. Cần một family compositional không được dạy sẵn trong codec trước khi claim reuse; không mở dataset/thí nghiệm tự động.
6. Sửa sai và tái tổ chức: phần chưa giải, không che giấu
Chia S/P cho phép học trong P mà bảo toàn head S; song nếu head S từng sai hoặc outdated, điều kiện cách ly sẽ giữ lỗi. Correction cần contract provenance, version và quy trình nâng cấp có quyền sửa S; lúc đó định lý noninterference không còn áp dụng. Nếu muốn hợp nhất P thành S để giải phóng dung lượng, phải thẩm định lại old and new consumers trên miền đã công bố, gồm probability/decision/tie/threshold và controller. Không có bằng chứng cost thấp hoặc guarantee global; tuyệt đối không gọi là 'functional recomposition solved'.

H_p hữu hạn sẽ bão hòa. Shared operator và việc cho phép update S sau một controlled release là giả thuyết; đây chính là blocker quan trọng nhất. Nếu chỉ có lợi bằng freeze old rồi dành neurons mới cho task khác, đó là capacity reservation (prior art), không phải AI tự mở rộng năng lực. Tác vụ mới cần liên quan S để đánh giá reuse, nhưng không được ưu ái candidate bằng thông tin/giám sát bổ sung.
7. Chi phí và công bằng
H_s+H_p = 128 cho candidate minh họa, không tự động đề xuất chia 96/32. Total params phải gồm W_in, sparse recurrence, every head, traces, readout/normalizer, versions, controller, verifier và I/O. Với 64 input, k=8 sparse fan-in, chính neuron core khoảng H*(64+8) synapse, không gồm head. Comparator dense recurrent H128, same-encoder & same-head, frozen hidden/readout only, original H128 local learner, optional nonspiking control. Một candidate có head cũ + head mới không được so với baseline thiếu head mới. Match dữ liệu, supervision, train updates, RAM, active compute, storage và retention coverage.

Cách ly S tránh replay để chứng minh stability của phần chỉ đọc S, không tránh chi phí học/verify mới, không bảo vệ controller ngoài S, không bảo vệ probability của head kết hợp S/P. Phải đo/thẩm định trên giấy trước: per-head old behavior, new rule acquisition, R_PS causal reuse, loss-of-plasticity khi P đầy, correction path, và transition to new representations. Không dùng old TEST làm development, không nới các threshold lịch sử.
8. Checklist bác bỏ ứng viên trước khi làm code
1. Nếu readout/normalizer/threshold/controller của old output thực tế vẫn phụ thuộc P -> proof sai -> REJECT/REVISE.
2. Nếu không thể làm rõ ALIF delta, clipping, eligibility và mask R_SP trên sparse topology -> REJECT/REVISE.
3. Nếu H128 budget chứa ít hơn baseline một cách bất công hoặc cần pretrained hidden khác -> REJECT/REVISE.
4. Nếu chỉ làm tốt INIT/MOVE toy nhưng không chứng minh reuse trên bài chưa gặp -> NO CLAIM.
5. Nếu P bắt buộc tăng vô hạn/không có con đường correction + compatible release -> NOT A LONG-LIFE ARCHITECTURE.
6. Nếu khả năng mới đến từ encoder/verified oracle/head thay vì plastic synapses -> NO SYNAPSE-LEARNING CLAIM.
7. Nếu chỉ tăng retention nhờ đóng băng S mà không cải thiện acquisition -> NO-GO so với SCAR-X mục tiêu.
8. Nếu reviewer phát hiện một phản ví dụ mà không sửa được giả định, ghi NO_GO thay vì ép một thiết kế thành PASS.
9. Trạng thái giao việc
Đây là một ứng viên để Codex phản biện, không phải quyết định thay lõi. Mục tiêu vòng nghiên cứu là một SPEC_REVIEWED (giả định, phương trình, novelty, failure modes nhất quán), không phải MODEL_VALIDATED. Chỉ khi có tiêu chí học, kiểm chứng và ngân sách được người dùng duyệt mới xem xét code/forward/train. Không tạo PR, merge, deploy, hay sửa main/Hermes.

## 10. Amendment v2: physical masks, persistence và phạm vi noninterference

### 10.1 Đường nguồn và phạm vi chứng minh

Source pin b3a48983, [core step](../../src/scarx/neural/core.py#L94),
[local eligibility](../../src/scarx/hybrid/learner.py#L132),
[readout](../../src/scarx/hybrid/learner.py#L173) và
[learn](../../src/scarx/hybrid/learner.py#L220). Code hiện chưa có partition
S/P hoặc head/normalizer riêng như candidate; không gọi thẳng learn() hiện
tại là đã triển khai SCOPE. Đây là adaptation đề xuất với destination masks,
không là migration hoặc permission chạy.

Mệnh đề §3 là **open-loop, input-conditional**: cùng x_t và cùng initial S.
Nếu new head chọn action khác làm thay môi trường rồi encoder nhận x'_t khác,
S có thể cho đầu ra khác dù không có cạnh neural P→S. Đây không là vi phạm
proof; nó nằm ngoài điều kiện cùng input. Resource contention, bỏ tick,
clock, input preprocessing và controller đều phải nằm trong contract thực.
Không suy đồng nhất behavior closed-loop của Hermes.

Induction: từ S state bằng nhau, current S chỉ dùng x_t và old S spikes;
threshold/reset/adaptation/filter cố định cho S state tiếp theo bằng nhau.
P và mọi update có đích P không xuất hiện trong biểu thức này. Head S và
normalizer cố định suy cùng logits/probabilities trong số thực. FP32 parity
cần backend/order/stream synchronization được kiểm về sau; chưa chứng nhận.

### 10.2 Topology và masks bắt buộc

Chọn H_s,H_p trước study, tổng128; chưa chọn split. Sparse fan-in8 chỉ hợp
lệ nếu mỗi row có đủ nguồn cho các kết nối được cấp; row S chỉ có indices
thuộc S, nên H_s≥8 theo sampling hiện tại. Row P có explicit indices trong
S∪P theo allocation đã khóa, không tự cộng8edge S và8edge P rồi báo fan-in8.

Mask M_in(j,i)=1 iff j∈P. Với từng recurrent slot(j,r), mask M_rec=1 iff
j∈P và nguồn indices[j,r] là cạnh PS/PP được cấp. Không tồn tại slot SP.
Mask topology khác trainability: các cạnh SS tồn tại nhưng frozen, còn các
slot cấm không được tồn tại/dùng cho current. Old head chỉ nhận S, không có
cột P được quyền ghi; S/P/head arrays không alias bộ nhớ mutable.

### 10.3 Quy tắc candidate cụ thể, chưa thực hiện

Forward eligibility của từng row P dùng pre=[x_t,z_(t−1)[indices[j,:]]].
e_v/e_a/e_f cập nhật cùng recurrence local ở learner.py, với topology/source
đúng; không lấy trace S hoặc target tương lai. Cross-neuron Jacobian vẫn bị
bỏ, như rule reference; không gọi đây là exact gradient hoặc BPTT/RTRL.

S normalizer μ_S,var_S giữ nguyên. P normalizer dùng scale=max(sqrt(var_P),
0.05), raw_P=(f_P−μ_P)/scale và q_P=clip(raw_P,−5,5). Mọi signal/gradient
dưới đây dùng weights/statistics **trước** cập nhật, sau forward đã hoàn tất:

    δ = reward × (one_hot(y_verified) − softmax(ell_new))
    m_P = (U_P.T @ δ) × 1[abs(raw_P)<5] / scale_P
    D_jr = m_j × e_f[j,r]                  (chỉ row P và slot được cấp)
    trace_jr' = 0.9 trace_jr + 0.1 D_jr
    energy_jr' = 0.99 energy_jr + 0.01 D_jr²
    d_jr = clip(0.008 trace_jr'/(sqrt(energy_jr')+1e−4),−0.01,0.01)
    w_jr' = clip(w_jr+d_jr,−3,3)           (chỉ M_in/M_rec=1)

Reward/label phải đến feedback có nguồn với miền hợp lệ như rule hiện tại;
reward không thay target thành oracle trung gian. S/cạnh cấm không bị ghi cả
trace/energy/weights, kể cả normalization hoặc decay toàn mảng. Hidden row P
dùng inputweights và các recurrent slots PS/PP cùng layout trace đã khóa.

New head:[U_S U_P] và b_new học output-local:

    U' = clip(U + clip(0.04 δ [q_S;q_P].T,−0.025,0.025),−3,3)
    b_new' = clip(b_new + clip(0.04 δ,−0.025,0.025),−3,3)
    μ_P' = 0.99 μ_P + 0.01 f_P
    var_P' = 0.99 var_P + 0.01 (f_P−μ_P')²

Không update μ_S,var_S,W_old,b_old,SS,W_in^S. New-head U_S được học nhưng
không là old readout hoặc một đường feedback ghi vào S. Không có anchor/
consolidation mới trong isolated phase; nếu sau này thêm, phải chặn mọi ghi
S/cạnh cấm và phân tích lại contract. Stop-gradient qua tín hiệu feedback
và state chỉ là tính tường minh rule, không nối graph reverse vào forward.

Masks là invariant trên initializer, update/clipping, serialization và load.
Illegal edge/mask hoặc changed S phải bị từ chối, không âm thầm zero một
checkpoint lỗi rồi báo parity. Nếu implementation dùng storage zero cho
forbidden edges, các giá trị zero đó và việc không có contribution phải được
kiểm; frozen SS không bị zero cùng forbidden SP. Đây là điều kiện thiết kế,
chưa có code/test chứng minh.

### 10.4 Persistence, old skill và kết thúc cách ly

Checkpoint phải có format/config/codec/version, S/P stable IDs, sparse source
indices, allocation/learning masks và old-head support, reference S weights,
neuronparams/normalizer/head hashes. Lưu cả state v/a/z/f, eligibility,
trace/energy/importance/stable copies nếu có, all heads/stats, RNG, replay,
pending credit/controller và journal liên quan. Load kiểm shape/dtype/bounds,
partition/no-SP, head support và fixed-S identity; không infer partition từ
weights đang bằng0. Reject incompatible state, không load vào live engine.

Rollback khôi phục cùng checkpoint contract và toàn learning/control state;
reject không là synapse học được skill. Logging giữ riêng; nếu external tool
đã có side effect, model rollback không giả định đảo được môi trường.

S cần được học/khởi tạo và xác minh trên chính topology partition trước khi
được gọi là có old skill; chưa có protocol hay bằng chứng đó. Không chuyển
score H128mixed, không xem random reserved S là competent. Chưa có migration
proof từ checkpoint live.

Correction S, consolidation/release hoặc gộpP→S thoát khỏi theorem isolated.
Chưa có thuật toán/guarantee hoặc approval làm các bước này. FiniteP **có
thể** hạn chế acquisition; không biết khi nào đầy hoặc mất plasticity, không
khẳng định mọi taskstream chắc bão hòa ở một số chặng cụ thể.

### 10.5 Reuse attribution, resources và prior art

Hai đường reuse khác nhau là R_PS (lateral recurrent) và U_S (new readout đọc
S trực tiếp). Ablate riêng R_PS chỉ đo đường lateral; để tách toàn bộ reuse
cần kiểm cả U_S, tránh gọi còn đường kia là “không reuse”. Chưa chạy ablation
hoặc chọn benchmark/seed. Không cấp thêm target/decoderoracle cho candidate.

Dense recurrent H128 có H² recurrent weights, trong khi fan-in8 là8H;
không mặc định parameter-matched. Primary reference phải khớp topology budget
hoặc tổng parameters/compute bằng cách đã đăng ký; dense control là separate
reference nếu khác budget. Same heads gồm cả old/new và normalizer/state,
không giấu cost. FrozenS stability và useful newP acquisition báo riêng.

[Progressive Networks](https://arxiv.org/abs/1606.04671v4) đã dùng frozen prior
features/lateral transfer; [Progress & Compress](https://arxiv.org/abs/1805.06370v2)
đã có knowledge base/active column cùng consolidation. Đọc abstract/metadata
trong timebox, không audit fullrepo hoặc tự claim khác biệt đủ mới. SCOPE hiện
gần một adaptation cách ly + reserved capacity; chưa giải bài toán release,
lifelong capacity, local reuse hoặc corrected knowledge.

**Verdict scope:** SPEC consistency review only; no acquisition/retention/
novelty/performance or independent empirical validation. Không code/forward/
train/modelchanges/PR/merge/deploy; các ngưỡng và raw/protocol cũ nguyên vẹn.
