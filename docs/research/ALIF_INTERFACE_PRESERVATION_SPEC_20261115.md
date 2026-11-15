# Đặc tả trên giấy: bảo toàn giao diện spike khi cập nhật ALIF

Ngày15/11/2026. Theo tài liệu người dùng gửi sau review b698702a: chỉ đặc tả
cập nhật synapse có ràng buộc chức năng và phân tích bế tắc; không xây NOL,
router hay compiler. **MATHEMATICAL_DRAFT, chưa learner hoặc chứng nhận runtime.**
Master duy nhất là [RESEARCH_PLAN](../RESEARCH_PLAN.md).

Không chạy model/fixture/P0, training, refit hoặc benchmark; không thay code,
main, Hermes, weights, neuron, thiết bị hay ngưỡng. Các ví dụ là đại số trên
giấy. Chưa đăng ký protocol/seeds/budget thực thi hoặc cấp quyền mở lượt mới.

## 1. Kết luận review đầu vào

Lập luận giữ spike bằng margin trong tài liệu là đúng **dưới giả định đầy đủ**
và là một hướng đặc tả hẹp có ích. Cần bổ sung: quy ước phát xung tại bằng
ngưỡng, bảo vệ toàn dependency cần thiết, normalization/readout, trạng thái
đầu, tích lũy nhiều update và sai số FP32. Không gọi đây là thuật toán mới
hoặc đã giải forgetting/credit assignment.

Điểm cần sửa trong diễn giải bế tắc: **ràng buộc bất đẳng thức giữ spike không
giống chiếu update vào nullspace**. Nhiều constraints hoặc rank đầy không tự
chứng minh hết hướng học. Bước của một learner bị chặn cũng chưa chứng minh
dung lượng kiến trúc đã hết.

## 2. Phương trình khớp code đã đọc

Nguồn hiện tại: [SpikingCore.step](../../src/scarx/neural/core.py#L94),
[CUDA step](../../src/scarx/hybrid/cuda_alif.py#L63),
[masked readout](../../src/scarx/hybrid/learner.py#L173).
Gọi u là điện thế **trước reset**, v là điện thế sau reset:

\[
u_{j,t}=\lambda v_{j,t-1}+w_j^\top x_t+
\sum_\ell R_{j\ell}z_{\ell,t-1},\quad
\theta_{j,t}=\theta_0+\beta a_{j,t-1},
\]
\[
z_{j,t}=\mathbf 1[u_{j,t}\ge\theta_{j,t}],\quad
v_{j,t}=u_{j,t}-z_{j,t}\theta_{j,t},\quad
a_{j,t}=\rho a_{j,t-1}+z_{j,t},
\]
\[
f_{j,t}=\gamma f_{j,t-1}+(1-\gamma)z_{j,t}.
\]

w_j là row W_in; R viết dạng dense để ký hiệu, code dùng indices/fan-in thưa.
Đây là subtractive adaptive reset, không là hard reset v=0. Core có nhánh
rate khác nhưng đặc tả này chỉ xét neuron alif. Lambda/rho/gamma là các hằng
forward trong config; không nhầm gamma với mảng decay của eligibility wrapper.

## 3. Phạm vi được bảo vệ và giả định

Cho tập hữu hạn S các chuỗi đã chọn cùng trạng thái đầu, và W0 là inputweights
tham chiếu. Chỉ xét candidate W0+ΔW, bất biến trong mỗi đoạn forward.

| Thành phần | Giả định của định lý |
|---|---|
| Input/encoder | Cùng x_t, normalization input, thứ tự, độ dài và observation |
| Recurrence | R và indices không đổi |
| Neuron | Cùng leak, threshold, adaptation/reset, feature filter và lịch tick; xác định |
| Trạng thái đầu | Cùng v0,a0,z0,f0; không ngầm reset một nhánh mà nhánh kia streaming |
| Weights học | Chỉ W_in; không update giữa đoạn |
| Readout/consumer | Nếu cần giữ đầu ra, freeze mọi head, bias, f_mean/f_var, masks, router và code đọc liên quan |
| Tập neuron/thời gian | Mọi neuron/bước của mạng đang ảnh hưởng tới giao diện, hoặc dependency closure đã chứng minh |
| Số học của chứng minh | Số thực chính xác; FP32 được xử lý riêng ở§10 |

Không chỉ kiểm một vài neuron giao diện rồi dùng công thức dưới đây cho phần
còn lại. Spike chưa được bảo vệ có thể đổi dòng recurrent tới neuron giao
diện. Có thể thu hẹp bằng ancestor closure trên graph cố định; chưa chứng
minh closure nhỏ cho topology H128 hiện tại, nên spec an toàn xét toàn mạng.

S chỉ giữ các chức năng cũ còn hợp lệ theo bằng chứng/provenance đã khóa.
Tri thức sai/lỗi thời cần cơ chế correction riêng; không đóng băng mọi đáp án
cũ và không hậu nghiệm bỏ constraint vì kết quả không thuận.
Không bảo vệ spike của một mẫu model đang trả lời sai chỉ vì mẫu có nhãn
đúng: nguồn nhãn đúng chưa chứng minh hành vi cũ của model là đúng.

## 4. Mệnh đề giữ quỹ đạo spike

Tính trên quỹ đạo tham chiếu:

\[
h_{j,t}=u_{j,t}-\theta_{j,t},\qquad
r_0=0,\quad r_t=\lambda r_{t-1}+x_t.
\]

r là filtered input phục vụ phân tích guard, không phải e_f/eligibility dùng
để tạo candidate; không tráo hai đại lượng để giữ nguyên một proof sai.

Khi mọi spike ở các bước trước giữ nguyên, adaptation/threshold cũng giữ
nguyên và dòng recurrent không đổi. Vì δv0=0, ta có:

\[
\delta u_{j,t}=\Delta w_j^\top r_t
=\sum_{k=1}^t\lambda^{t-k}\Delta w_j^\top x_k,
\qquad \delta v_{j,t}=\delta u_{j,t}.
\]

Do code dùng >=, điều kiện dấu **một phía** cần giữ là:

\[
\begin{cases}
h_{j,t}+\Delta w_j^\top r_t\ge0 & z_{j,t}=1,\\
h_{j,t}+\Delta w_j^\top r_t<0 & z_{j,t}=0.
\end{cases}
\tag{C}
\]

Nếu(C) thỏa cho mọi row/time/sequence thuộc phạm vi trên, toàn bộ chuỗi spike
tham chiếu được giữ nguyên. Trong số thực với các giả định này,(C) cũng mô
tả đúng cell trọng số cho cùng spike pattern trên S.

**Chứng minh quy nạp.** Ởt=1, initial state và recurrent input giống nhau;
δu=Δwᵀx1. (C) giữ quyết định threshold, nên z1 giống nhau. Adaptation giống
nhau và subtractive reset trừ cùng zθ, vì vậy δv1=δu1. Giả sử đúng tới t−1:
mọi recurrent spike và adaptation giống nhau. Khi đó δu_t=λδv_(t−1)+Δwᵀx_t
=Δwᵀr_t; (C) lại giữ z_t, reset/adaptation giữ giả thiết. Lặp tới cuối đoạn.

### 4.1 Điều kiện margin đối xứng của tài liệu

Với m=|h|>0, |Δwᵀr|<m là điều kiện đủ dễ viết nhưng chặt hơn cần thiết.
Nếu h=0, code phát spike: cần Δwᵀr≥0. Điều kiện |Δwᵀr|<0 từ chối cả update0,
nên không được dùng nó làm phép kiểm tổng quát cho tie. Spike với h>0 cũng
không cần cấm một dịch chuyển dương lớn chỉ để giữ dấu hiện tại.

## 5. Giữ spike bảo vệ được consumer nào?

Nếu f0/filter cố định và z_t giống nhau, f_t giống nhau. Khi core readout
W_out/bias cố định, probabilities của SpikingCore trên S được giữ. Hybrid
còn chuẩn hóa bằng f_mean/f_var và clip, chọn head/mask; các phần này cũng
phải cố định để suy cùng đầu ra. Consumer đọc voltage, eligibility hoặc state
khác không được bảo đảm chỉ bởi spike equality.
Đầu ra ở đây là core/head xác định dưới các giả định; không phải toàn bộ câu
trả lời VLM/Hermes hoặc kết quả tool trong môi trường thay đổi.

**Khác biệt với code live:** LocalALIF.learn hiện sửa W_in,W_rec,W_out,bias
và chạy f_mean/f_var updates. Vì vậy không thể gắn định lý W_in-only vào live
learner rồi tuyên bố đã giữ chức năng. Research CreditALIF hidden là phạm vi
gần hơn vì chỉ sửa W_in; chưa có guard này trong code đó. Không sửa hai learner.

Giữ spike không giữ nguyên membrane voltage hoặc eligibility: surrogate slope
có thể đổi khi u−θ đổi, dù threshold decision không đổi. Định lý bảo vệ forward
interface trên S, không bảo toàn toàn bộ learning trajectory tương lai.

## 6. Hình học của không gian còn lại

Với từng rowj, (C) là giao hữu hạn các halfspace mở/đóng trong Δw_j. Thêm box
trọng số tạo một miền lồi bị chặn; do dấu<, không mặc nhiên là polytope đóng.
Không áp dụng thuật toán tối ưu/solver trong lượt này.

Giả sử mọi h khác0, mọi r hữu hạn và W0 ở trong box. Với chuẩn bất kỳ và
dualnorm tương ứng, mỗi row có bán kính đủ:

\[
\|\Delta w_j\|<\varepsilon_j,
\quad \varepsilon_j=\min_{(s,t):\|r_t\|_*>0}
\frac{|h_{j,t}|}{\|r_t\|_*},
\]

đồng thời nhỏ hơn khoảng cách tới box. Nếu không có r khác0 thì inputweight
row không ảnh hưởng các điểm đó, không dùng min trên tập rỗng. Với tập hữu
hạn, các margin dương cho một lân cận khôngzero trong số thực. Rank của tập
r có thể đầy mà miền này vẫn có nội thất. Ví dụ một chiều: h=1,r=1 chỉ đòi
Δw≥−1 cho spike, không ép Δw=0.

Ngược lại, yêu cầu **giữ điện thế chính xác**, Δwᵀr=0, mới tạo nullspace có
dimension d−rank{r}. Khi rank=d, update đó buộc bằng0. Không chuyển kết luận
rank của equalities/projection sang các sign inequalities giữ spike.

Nhiều constraints có thể thu nhỏ margin/bước hữu ích; tie, boxboundary và
FP32 có thể làm một số hướng không dùng được. Lân cận khác0 không chứng minh
học task mới: thay đổi quá nhỏ hoặc sai hướng có thể không tạo gain nào.

## 7. Một hướng cập nhật: cận bước trên giấy

Gọi G là **delta cuối của candidate** sau anchor, optimizer, per-coordinate
clip và weightbound. Xét W(α)=Wcurrent+αG, 0≤α≤1, không clip theo tọa độ thêm
sau đó trong bài toán số thực. Nếu hai đầu ở trong box, đoạn nội suy cũng ở
trong box. Đây là guard bổ sung giả thuyết, chưa được triển khai hay duyệt chạy.

Với reference margin hiện tại h và d_jt=G_jᵀr_t:

| Trường hợp | Điều kiện bước giữ spike |
|---|---|
| z=1,d<0 | α≤h/(−d) |
| z=0,d>0 | α<(−h)/d |
| d không đẩy qua boundary | Không thêm cận từ điểm đó |
| h=0,z=1,d<0 | Chỉ α=0 theo hướng này |

Lấy giao với[0,1] trên mọi constraint. Cận strict là supremum, không nhất
thiết một maximum được phép. Không chọn đúng boundary silent. Safety factor
có thể tránh equality trong số thực, chưa thay cho numerical-error certificate.

“Giảm learningrate rồi chạy lại optimizer” không nhất thiết bằng scale delta
vật lý cuối: anchor/clips/optimizer state có thể tạo hướng khác. Cast FP32 sau
nội suy cũng phải được phân tích. Không gọi một raw gradient an toàn là một
physical update an toàn nếu writer còn biến đổi nó.

### 7.1 Không đặt lại ngân sách margin sau mỗi update

Nếu dùng W0,h0,r đã lưu, phải kiểm **tổng** Δ=Wnew−W0 trong(C). Ví dụ h0=−0,1,
r=1; hai update+0,06 mỗi cái đều nhỏ hơn |h0|, nhưng tổng+0,12 làm h thành+0,02
và đổi spike. Đây là phản ví dụ trên giấy, không là run mới.

Có thể cập nhật margin cache theo tổng delta khi điều kiện quỹ đạo còn đúng,
hoặc chuyển reference sang Wcurrent với state/margin đã xác minh tương ứng.
Không tái dùng margin cũ cho từng increment riêng rồi gọi bảo toàn dài hạn.

## 8. Phân biệt bốn kiểu bế tắc

| Quan sát/điều kiện | Có thể kết luận | Chưa thể kết luận |
|---|---|---|
| G của learner bị chặn hoặc α quá nhỏ | Hướng cập nhật này khó dùng dưới guard | Mọi hướng đều bị chặn hoặc model hết capacity |
| Nullspace equality hết rank | Không còn update giữ drive chính xác theo ràng buộc đó | Sign-cell bảo vệ spike cũng chỉ còn0 |
| Miền guard còn hướng nhưng task mới không cải thiện | Chưa tìm được update hữu ích; objective/credit/inputs cần phân biệt | Bắt buộc bỏ ALIF hoặc tăng neuron |
| Nhãn mới mâu thuẫn với chức năng cũ tại cùng điều kiện | Hai mục tiêu đó không thể giữ chính xác đồng thời | Mọi kiến thức cũ đều cần xóa |

Không gian chứng nhận nhỏ có thể do yêu cầu quá mạnh (giữ mọi hidden spike dù
output cho phép đổi), coverage không đại diện, margin nhỏ, hướng learner yếu,
precision hoặc giới hạn biểu diễn thật. Phải tách các nguyên nhân; một guard
thất bại chưa xác định riêng nguyên nhân “kiến trúc đã bế tắc”.

Để nói không còn hướng hữu ích về mặt toán cần xét giao giữa miền được giữ
và các update làm giảm mục tiêu mới dưới cùng thông tin. Rank hoặc α của một
hướng không giải được bài toán đó. Không có một proof khả năng học mới hoặc
phân tích đầy đủ dung lượng từ các kết quả DEV cũ.

## 9. Trạng thái, locality và chi phí cần công khai

Candidate generator vẫn dùng quy tắc local đã được phép; checker dùng các
quỹ đạo cũ là **guard hệ thống có memory**, không là synapse tự biết tầm quan
trọng. Đánh giá tương lai phải tách lợi ích local learner khỏi replay/guard,
có control mạnh cùng thông tin và compute. Không dùng checker nhãn/TEST cấm
cho learner. Việc chọn S không được miễn gate forgetting trên tập đánh giá.

Nếu sau này triển khai, trial phải dùng state riêng; reject/rollback khôi
phục weights, traces/optimizer, importance/stablecopies, stats, RNG và pending
controller/replay đúng hợp đồng. Journal có thể ghi reject nhưng không gọi nó
là học được skill. Accepted cũng không mặc nhiên có gain; α=0 không là update
synapse hữu ích. Checkpoint phải pin S/encoder/config/reference và guard state.

Với M tổng tick của S, H neuron, d input, một cách giữ r và h bằng FP32 tốn
xấp xỉ4Md+4MH byte, thêm spikebits≈MH/8byte, initialstates/provenance và model.
Kiểm G_jᵀr_t trực tiếp tốn O(MHd) phép toán; đừng gọi miễn phí chỉ vì không
reroll ALIF. Có thể phân lô/stream, nhưng tính cả I/O, cache, scratch, snapshot/
rollbackcopy, CPU/GPUallocations và toàn runtime. Đây là dự toán ký hiệu,
không là số đoRAM, tốc độ hoặc kết quả trong2GB. GPU/CPUprofile hiện tại không đổi.

## 10. Số thực không phải chứng nhận FP32

Numpy/CuPy dùng FP32 arrays; nhân/cộng, reduction và fused operations có thể
khác biểu thức số thực và khác cách lấy hiệu hai trajectory. Định lý không
cho bitwise parity hoặc một tolerance an toàn miễn phí. Một implementation
certificate cần bound sai số của reference margin và candidate evaluation,
hoặc interval/sound verification đúng backend/graph. Chưa làm phần đó.

Thêm một epsilon bất kỳ không thành proof. Nếu bound sai số lớn hơn margin,
phải báo chưa chứng nhận, không tự sửa threshold hoặc bỏ constraint. Khi bước
nhỏ hơn độ phân giải FP32 của weights, α>0 vẫn có thể ghi delta0; phải tính
delta vật lý thật. Không đề xuất đổi precision/quantization để né vấn đề.

Với streaming không reset, voltage sau đoạn có thể khác dù spikes giống nhau.
Không coi hai endpoint state là identical để ghép chứng minh đoạn sau. Phải
giữ lịch sử/độ lệch state trong proof, hoặc cùng initialstate đã xác định cho
toàn phần tiếp diễn. Stochastic neurons/noise/encoderdrift nằm ngoài theorem.

## 11. Prior art và câu hỏi còn mở

- [HLOP-SNN](https://arxiv.org/html/2402.11984v1): prior art về bảo vệ hướng hoạt
  động; không gọi các sign constraints hoặc guard bộ nhớ thành firstprojection.
- [Functional regularisation, NeurIPS2020](https://papers.neurips.cc/paper_files/paper/2020/hash/2f3bbb9730639e9ea48f309d9a79ff01-Abstract.html): bảo vệ hành vi trên ví dụ nhớ đã có tiền lệ; không nhập score/thuật toán đó thành local-ALIF result.
- [Geometry of robustness in SNNs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9307274/): lead về hình học/synaptic perturbation; mô hình và mục tiêu khác, chưa có audit chứng minh novelty hay chuyển trực tiếp sang core này.

Ứng dụng quy nạp vào forward đã biết là phân tích kỹ thuật có điều kiện,
không đủ claim học thuật mới. [Manifest](literature/sources/alif-interface-preservation-20261115.json)
giữ source hashes/ranges và phạm vi nguồn đã đọc.

Các câu hỏi giữ lại: guard yếu hơn chỉ giữ output có giúp plasticity không;
local credit nào tìm được hướng hữu ích trong miền hợp lệ; budget guard nào
còn đáng dùng; khi nào cần thay đổi representation thay vì chỉ đổi guard.
Chưa có đủ bằng chứng trả lời bằng một kiến trúc mới.

## 12. Quyết định sau đặc tả

**GO cho câu hỏi bảo toàn giao diện có điều kiện; chưa GO code/chạy/deploy.**
Đã có điều kiện spike đúng theo >=, proof và hình học của miền giữ chức năng.
Chưa có numerical certificate, thuật toán learning đủ tốt, phép đo acquisition/
retention, lợi thế hơn replay/readout/RAG hoặc giải pháp Functional Recomposition.
Không nới ngưỡng đã đăng ký. Main/Hermes/liveALIF và raw/protocol lịch sử giữ nguyên.

**Cụ thể hóa tiếp theo, chưa thực thi:** [relation-update contract](ALIF_RELATION_UPDATE_SPEC_20261115.md)
xét một phép đổi vị trí theo event. Phân biệt kỹ năng neural với worldstate
và phần interface cung cấp sẵn; không biến proof này thành acquisition claim.
