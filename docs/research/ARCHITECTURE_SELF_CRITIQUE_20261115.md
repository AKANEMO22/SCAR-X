# Tự phản biện ma trận kiến trúc: điều suy ra được và điều còn phải kiểm

Ngày 15/11/2026. Đọc lại [khảo sát trước](ARCHITECTURE_DESIGN_SPACE_20261115.md),
các source đã khóa và báo cáo kiểm toán đã có. **Không có thí nghiệm, inference,
training, P0, dữ liệu kiểm thử hoặc số đo neural mới trong lượt này.** Những ví
dụ đại số dưới đây là phản ví dụ trên giấy, không là kết quả chạy model.

## 1. Những kết luận trước cần sửa mức chắc chắn

| Nhận định trước | Tự phản biện | Mức kết luận sửa lại |
|---|---|---|
| Ưu tiên ngay core chia vùng + residual + protection | Mới chỉ thấy phạm vi nó muốn giải rộng; chưa có rule hoàn chỉnh, bằng chứng các phần phối hợp tốt hoặc lợi ích thêm so với phương án ít thành phần | B là ứng viên tổng hợp để phản biện; chưa ưu tiên triển khai B |
| Importance/EMA chưa đủ, nên xét hướng hoạt động | Kết quả cũ cho biết gói protection chưa qua gate, chưa chứng minh nguyên nhân là bỏ mất tương quan input | Projection là giả thuyết có cơ sở từ literature; chưa là nguyên nhân gốc đã xác minh |
| Protection tốt hơn norm-matched nên importance hữu ích | Protected arm đồng thời có giảm tốc theo importance và kéo về anchor; control chỉ khớp L2, có trajectory khác | Có lợi ích của **gói cập nhật** trên DEV đó; chưa tách được từng thành phần hoặc chứng minh một nguyên nhân |
| Có gain common-head nên hidden đã giải được representation | Gain là trong bank sáu intent với encoder pretrained và nhãn có sẵn, DEV đã dùng; chưa hơn raw-feature control | Hidden học biểu diễn phân biệt nhãn hữu ích trong phạm vi đã đo; chưa tổng quát hoặc vượt encoder mạnh |
| HLOP dùng backward nên khó dùng cho local SCAR-X | Lateral projection của HLOP và learner chính là hai phần tách được; source learner dùng backward không làm mọi thành phần của HLOP thành không-local | Có thể nghiên cứu chuyển riêng protection, giữ local credit; không được nhập nguyên score/đường training |
| Có nhiều memory sẽ giảm forgetting | Memory, router, shared encoder và readout vẫn có thể cùng trôi; chia cùng số neuron thành nhiều vùng còn giảm dung lượng mỗi vùng | Có khả năng giảm một loại can thiệp; chưa bảo đảm giảm forgetting toàn hệ |
| Nghiên cứu loss-of-plasticity giải thích SCAR-X yếu | SCAR-X đã đo forgetting; chưa có phép đo chặt về tốc độ học skill mới sau lịch sử dài so với fresh learner | Loss of plasticity là giả thuyết riêng, chưa là chẩn đoán đã xác minh cho SCAR-X |

Không thay đổi bất kỳ raw result, protocol, seed, nhãn, metric hoặc verdict cũ.
Đây là hiệu chỉnh cách suy luận từ bằng chứng đó.

## 2. Phân loại suy luận

- **D — Suy ra theo điều kiện toán/logic:** đúng nếu giả định ghi rõ đúng.
- **O — Quan sát đã kiểm trong phạm vi cũ:** dữ liệu nói được điều đó, không
  được tự mở rộng sang task/model khác.
- **H — Dự đoán có cơ sở:** literature/cơ chế gợi ý; chưa phải kết quả SCAR-X.
- **U — Chưa biết:** cần đặc tả hoặc thí nghiệm phân biệt được sau khi duyệt.

Không gán các phần trăm xác suất chủ quan như “80% sẽ thành công”. Một CI trên
DEV cũng không là xác suất một kiến trúc mới sẽ hoạt động.

## 3. Điều đã đủ cơ sở để không mở thêm một chiến dịch

### D1. Oracle của tập action không hơn baseline thì selector không có đường thắng

Với bank hữu hạn X và tập action A đã khóa:

\[
S(\pi)=|X|^{-1}\sum_x u(x,\pi(x))
\leq |X|^{-1}\sum_x\max_{a\in A}u(x,a)=S_{oracle}.
\]

Nếu oracle chỉ bằng RAG/constant action, tăng neuron, đổi STDP, thêm replay
hoặc train thêm seed không tạo gain dương trên **chính bank và action contract
đó**. V1 TEST oracle bằng RAG; v2 DEV oracle bằng constant inspect_status.
Không cần train một selector mới để xác nhận cận đã tính từ outcomes.

Giới hạn: outcome phải được chấm đúng, action table phải chứa đầy đủ hành vi
được phép. Một policy được quyền tạo chuỗi action mới không còn thuộc cận của
single-macro bank cũ. Không suy thành “SNN không thể hơn RAG ở mọi bài”.
Nguồn: [audit và quyết định toolbank](../hybrid/P1B_MULTI_API_STOP_20261115.md).

### D2. Cùng toàn bộ thông tin nhưng cần hai đáp án khác nhau là bài bất khả định

Nếu input, state trước quyết định và mọi observation được phép đều giống nhau
nhưng nhãn phụ thuộc một context bị giấu, một policy xác định phải cho cùng
output. Với policy ngẫu nhiên, phân phối output cũng giống nhau; nó không thể
biết context ẩn chỉ bằng tính lâu hơn.

Không cần benchmark học dài để biết lỗi alias như vậy cần sửa information path.
Nếu có lịch sử/action/outcome khác nhau mà encoder làm mất chúng, cần bảo toàn
khác biệt đó trước. Nếu lịch sử thật sự không chứa thông tin, cần hỏi/quan sát
thêm. **Input khác nhau vẫn chưa đủ bảo đảm representation tốt.**

### O1. Không cần lặp prototype hoặc EMA như thể chưa biết thất bại cũ

Prototype TEST62,64% đã kém BM25 78,47% và ridge83,33%. Gói protection giảm
quên nhưng chỉ1/3 seed qua joint gate; sử dụng slow inputweights cuối còn kém
fastweights trên ba DEV probes đã đo. Những biến thể giống hệt về cơ chế và
điều kiện không cần một lượt đổi tên để “tìm lại thành công”.

Kết quả đó không phủ nhận mọi prototype/cascade/EMA trên mọi task. Thay giả
thuyết thực sự, tín hiệu hoặc điều kiện thì phải nêu rõ khác biệt trước khi
xin chạy; không âm thầm mở thêm sweep learning rate.

### D3. Không đổi synapse mà không quên không chứng minh học liên tục

Freeze có thể bảo toàn những gì model vốn làm được. Nó không cung cấp bằng
chứng acquisition hay chuyển trải nghiệm thành skill. Không cần chạy thêm
20 chặng frozen rồi gọi forgetting0 là thành công. Frozen vẫn cần làm đối
chứng trong một phép đo nhân quả mới, với ngân sách tối thiểu phù hợp.

### D4. State/SSD offload và checkpoint giải bài toán khác với học kỹ năng

Khôi phục đúng dữ liệu có thể khôi phục cả một câu trả lời sai. Streaming chỉ
đổi nơi giữ weights, không tạo thêm tín hiệu học hoặc nguồn sự thật. Không cần
thí nghiệm SNN để suy ra quan hệ này; compatibility và chất lượng của runtime
mới vẫn cần kiểm riêng nếu sau này quyết định dùng nó.

## 4. Các phản ví dụ phá vỡ suy luận quá nhanh về kiến trúc mới

### D5. Projection trước, clip từng phần tử sau có thể phá protection

Chọn hướng cũ q=(1,2), candidate update d=(2,-1). Ta có qᵀd=0.
Nếu clip từng thành phần d vào[-1,1], thu được d'=(1,-1), nhưng qᵀd'=-1.
Update đã quay vào hướng cần bảo vệ. Không cần chạy model để thấy điều này.

Do đó proof cho Δw=ηmPe chỉ áp dụng nếu **update vật lý cuối cùng** vẫn thỏa
điều kiện chiếu. Momentum, coordinate clipping, weight-bound clipping, decay,
anchor và chuẩn hóa sau đó đều phải được đưa vào phân tích. Không tự sửa thứ
tự phép toán trong learner hiện tại: đây là yêu cầu cho spec tương lai.

Giảm bằng một scalar chung có thể giữ hướng của một update đã nằm trong
nullspace, nhưng không giải mọi ràng buộc trọng số/finite precision. Muốn có
cả box constraint và protection phải đặc tả bài toán giao ràng buộc; không
được đổi dấu/shape hay nới tolerance để giữ một lời hứa toán sai.

### D6. Giữ hidden hoặc trọng số cũ chưa giữ quyết định cũ

Giả sử một mẫu cũ có điểm các lớp A/B là(2,1), model chọn A. Khi thêm lớp C
có điểm3, A/B và hidden vẫn nguyên nhưng đáp án chuyển sang C.

Tương tự, hai kho nhớ không đổi nội dung nhưng router chuyển từ kho đúng sang
kho sai vẫn làm hỏng đáp án. “Không ghi đè synapse nguồn” chỉ bảo vệ một phần
đường tính. Đây là lý do cần xem readout/router và cạnh tranh lớp, không là
bằng chứng một kiểu bảo vệ readout cụ thể đã đúng.

### D7. Projection hết không gian sẽ chặn plasticity trong mô hình tuyến tính lý tưởng

Với Q trực chuẩn và P=I−QQᵀ, rank(P)=d−rank(Q). Khi rank(Q)=d thì P=0;
mọi update dạng Pe bằng0. Thêm rank bảo vệ đến hết không gian không thể đồng
thời giữ mọi hướng cũ và học hướng mới độc lập trong cùng d chiều.

Điều kiện Q trực chuẩn là thiết yếu: với các hàng H học Hebbian xấp xỉ,
I−HᵀH chưa mặc nhiên là projector trực giao. Cần chỉ rõ thuật toán duy trì
basis và sai số được bảo vệ, không nhập proof của Q lý tưởng cho H bất kỳ.

Trong SNN, encoder/context/gates có thể đổi để tạo cách biểu diễn khác; khi
đó điều kiện bảo vệ cũ cần kiểm lại. Điều này dự đoán một đánh đổi dung lượng,
không dự đoán số chặng hay accuracy tại đó SCAR-X sẽ thất bại.

### D8. Học phương sai hoạt động lớn chưa đồng nghĩa bảo vệ kiến thức hữu ích nhất

Một chiều input có phương sai lớn có thể chỉ là nhiễu; một chiều phương sai
nhỏ có thể quyết định nhãn. Vì vậy một subspace mô tả activity tốt không mặc
nhiên xếp đúng mức quan trọng với task. HLOP cung cấp cơ chế có căn cứ và kết
quả của tác giả; điều đó không biến PCA thành task-importance oracle.

Với SCAR-X, cần phân biệt “hướng hoạt động đã thấy”, “hướng dùng cho đáp án
đúng” và “hướng cần giữ cho skill hiếm”. Hiện chưa xác minh đây là nguyên nhân
gói importance cũ thất bại. Đọc/đặc tả cách chọn hướng có ích phải đi trước
việc dựng một hệ router/memory lớn.

### D9/H1. EMA chậm tự tạo độ trễ; retention tốt là câu hỏi riêng

Với S_t=(1−α)S_{t−1}+αW_t và W_t−W_{t−1}=v không đổi, đặt D_t=W_t−S_t:

\[
D_t=(1-\alpha)(D_{t-1}+v),\qquad
D_\infty=\frac{1-\alpha}{\alpha}v.
\]

Đây là trường hợp lý tưởng, không là mô hình đầy đủ của lượt train cũ. Nó
giải thích vì sao slowweights có thể tụt lại khi fastweights đang cải thiện;
không thể dự đoán accuracy từ công thức này. Khi W ổn định, EMA lại hội tụ về
W. Vì vậy không được suy “EMA chắc chắn kém” hoặc “chậm chắc chắn giữ skill”.

## 5. Những dự đoán có cơ sở nhưng chưa thể coi là kết quả

| ID | Dự đoán | Cơ sở | Điều có thể làm dự đoán sai |
|---|---|---|---|
| H2 | Chia vùng có thể giảm interference khi ghi | Một số kết nối không bị cập nhật bởi trải nghiệm không liên quan | Router sai/trôi, sharedhead trôi, dung lượng mỗi vùng nhỏ đi |
| H3 | Protection theo hướng có thể tốt hơn chỉimportance từng weight | Giữ được tương quan giữa input/trace trong mô hình phù hợp | Subspace nhầm, bão hòa rank, clip phá chiếu, recurrence làm trace đổi |
| H4 | Tách erase/write có thể giúp sửa association cụ thể | GDN-2 có prior art và ablation trong mô hình của tác giả | Gate học sai, không đủ supervision, memory finite, biến thể ALIF không tương đương |
| H5 | Nhiều timescale có thể giữ tín hiệu qua nhiều độ trễ hơn một EMA | Benna–Fusi có tương tác nhanh/chậm hai chiều | Chỉ bảo tồn nhiễu/lỗi, tiêu tốn state, credit vẫn sai |
| H6 | Đổi unit ít hữu ích có thể duy trì khả năng học thêm | Nghiên cứu plasticity và CBP | Unit ít hoạt động hiện tại giữ skill hiếm; chưa xác minh SCAR-X mất plasticity |
| H7 | Biểu diễn giữ thứ tự/quan hệ thời gian đáng ưu tiên cho câu hỏi video phụ thuộc thứ tự | Bài semantic cũ không học recurrence/quan hệ sự kiện; gộp bất biến theo thứ tự làm mất thông tin cần thiết cho bài đó | Không bắt buộc một kiểu event encoder cụ thể; frame/OCR/ASR sai hoặc local credit vẫn có thể làm phương án thất bại |

Không suy từ bảng này rằng ghép H2–H7 sẽ cộng mọi lợi ích. Các cơ chế có thể
triệt tiêu nhau: protection chặn feature mới, routing làm sai credit, slowmemory
giữ lỗi, thay unit làm mất một hướng đã bảo vệ. “Có bài riêng hỗ trợ từng phần”
không là bằng chứng về cả tổ hợp.

## 6. Những nghiên cứu nên bỏ, hoãn hoặc thu hẹp

| Phép nghiên cứu dự định | Quyết định sơ bộ | Lý do và phần tối thiểu còn cần |
|---|---|---|
| Train thêm selector trên toolbank cũ | BỎ | Cận oracle đã chặn gain; chỉ audit lại nếu contract/outcomechecker thực sự thay đổi |
| Rerun nguyên prototype/EMA family chỉ để có thêm số | BỎ | Kết quả đã có; cần giả thuyết phân biệt mới mới mở lại |
| Dùng frozen0forget để xác nhận lifelong | BỎ CLAIM | Frozen chỉ là control, không đáp ứng acquisition |
| Chạy benchmark chỉ chứng minh có trace/weightdelta | BỎ CLAIM | Đã biết có update; chưa nói representation/skill hữu ích |
| Tăng neuron chỉ vì RAM đang trống | HOÃN | Capacity từng có gain có điều kiện; chưa phân biệt lỗi information/credit/head với thiếu dung lượng |
| Dựng ngay kiến trúc B nhiều kho/fastslow/projection/router | HOÃN | Chưa có rule đủ rõ hoặc bằng chứng từng phần cần thiết; khó attribution |
| Tái lập cả HLOP/LOCO/MIRAS trên fullbenchmark | HOÃN | Khác signal/boundary/compute; đọc source và phân tích chuyển cơ chế trước |
| Chỉ xem đạo hàm/trace tương tự paper để kết luận quality | BỎ CLAIM | Correctness và estimator alignment là điều kiện kỹ thuật; chất lượng còn phải đo |
| Sweep α/β/LR lớn để tìm một seed đẹp | BỎ | DEV selection và nhiều thử không tạo freshconfirmation; không nới gate |
| Formalize update dưới clip/bounds/anchor | LÀM TRÊN GIẤY | Có phản ví dụ cụ thể; chưa cần dữ liệu hay model mới |
| Phân tách contribution của input/hidden/head/router | GIỮ CÂU HỎI | Common-head và control mạnh bắt buộc; thí nghiệm mới phải có protocol riêng |
| Task-free fixed-rank protection với sharedoutput | GIỮ CÂU HỎI | Literature/source chưa xác nhận giải pháp phù hợp constraint SCAR-X |
| Recurrent temporalcredit và causalreuse | GIỮ CÂU HỎI | Điểm classification/lookup hiện tại không trả lời được |

“BỎ” ở đây là tránh thí nghiệm không cung cấp thông tin mới trong điều kiện
cũ; không là phán quyết mọi công trình cùng họ đều vô ích. Các kiểm tra tối
thiểu để chuyển phần mềm có thể vẫn cần về sau, khi được duyệt, và không được
đặt tên là correctness để lén chạy một study chất lượng mới.

## 7. Cách diễn giải số liệu cũ phải được giữ chặt

1. Hidden học từ lỗi +15,83pp so với frozen trong common-head DEV; so với raw
   features là−1,39pp, CI95%[−6,11;3,33]. Có tín hiệu task-discriminative learning,
   chưa SNNsuperiority hoặc general-language learning.
   Arm hidden cập nhật input synapses; recurrent weights và readout được giữ
   cố định trong lúc học arm này. Input là cùng vector semantic lặp8tick,
   không phải chuỗi sự kiện video. Vì vậy không suy thành đã học temporal
   representation hoặc recurrent skill mới.
2. Protection−normmatched +8,61pp trên actualhead DEV là kết quả gói cập nhật.
   Cùng common-head, contrast chỉ+2,50pp, CI[−3,33;8,33]. Hai phép đo trả lời
   câu hỏi khác nhau; không chọn phép thuận lợi rồi quy toàn gain cho hidden.
3. Protection−unprotected actualhead accuracy−1,94pp, CI[−7,78;3,61]. Chưa chứng
   minh cải thiện accuracy; giảm quên có đánh đổi acquisition trên bộ đã đo.
4. Importance và EMA được đổi cùng nhau. Chưa có importance-only/anchor-only
   ablation; kết quả không xác định riêng cơ chế thắng/thua.
   Importance hiện là EMA của bình phương direction cập nhật cục bộ; không
   mặc nhiên là Fisher, causal importance hay độ đúng của kiến thức được giữ.
5. Không có probe đủ dài so tốc độ học task mới với freshlearner để xác nhận
   loss-of-plasticity. Không dùng forgetting hay neuron im lặng làm thay thế.

Nguồn số liệu: [audit common-head3seed](../../runs/20261115T102212Z-semantic-three-seed-data-audit-v1-bcf16c/result.json),
[audit protection](../../runs/20261115T101809Z-semantic-metaplasticity-data-audit-v1-fa2c42/result.json).
Các đường link ở trên trỏ tới bộ raw đã chấm, không tạo lượt đánh giá mới.

## 8. Điều chỉnh ưu tiên thiết kế

Ưu tiên trước mắt là **xác định phép ghi hợp lệ và đối tượng cần giữ**, rồi mới
quyết định cần bao nhiêu kho/module. Cần một spec tối giản nêu được:

- Input/history/outcome nào phân biệt được tình huống; đâu là thông tin thiếu.
- Update cuối sau mọi clamp/anchor/normalization còn giữ tính chất nào.
- Sharedreadout/router có được thay đổi không; nếu có thì giữ quyết định cũ
  bằng điều kiện gì, và khi dữ kiện cũ sai/lỗi thời thì được sửa thế nào.
- Rank/dung lượng còn lại và khi nào phải từ chối ghi, gộp hoặc báo chưa biết.
- Learning signal nào thực sự local; đâu là approximation và đâu là external
  verifiedfeedback. Không giả định controller biết sẵn task đúng.

A được giữ làm **bài toán đặc tả toán có tiền lệ**, không gọi là learner đã tốt
hơn. B được hạ xuống **ứng viên tổng hợp, chưa chọn ưu tiên triển khai**. C vẫn
để đối chiếu vì đổi dynamics/signal. Chưa có bằng chứng chọn một kiến trúc thay
SCAR-X ở bước này. Điều chỉnh này sửa đánh giá của tài liệu trước, không tạo
masterplan thứ hai hoặc approval thí nghiệm.

## 9. Nguồn đối chiếu và giới hạn

- [HLOP AppendixF/G](https://arxiv.org/html/2402.11984v1): cần phân biệt lateral
  protection với trainingalgorithm và singlehead/domain setting; không nói
  mọi nhánh HLOP đều cần task-ID ở inference. Source đã khóa trong intake cũ.
- [MoM AppendixG](https://arxiv.org/html/2502.13685v3): equalactiveparameter và
  equalmemory là hai comparison khác nhau; active nhỏ không có nghĩa allocated
  state/weights nhỏ. Chưa chuyển scores tác giả sang laptop.
- [SPIRE equations11–13](https://arxiv.org/html/2611.14840v1): fixedmotor ring và
  learnedactor là các phần khác; causalmotorablation không tự chứng minh hidden
  weights học representation. Chưa có code/rawaudit độc lập.
- [Loss-of-plasticity source](https://github.com/shibhansh/loss-of-plasticity):
  đọc lại source/paper đã có; Nature trang hiện tại trả lỗi cookie, không giả
  có một fullpaperread mới. Không chạy CBP hoặc reset neuron.

Các suy luận D là theo giả định đã nêu; các H/U không phải dữ liệu mới. Không
dùng kết quả này để hạ các ngưỡng65%, gain5pp/CI, forgetting3/5pp, hoặc công bố
SNN mạnh hơn mọi hệ RAG. RAG là cách truy xuất, SNN là lớp kiến trúc/tính toán;
chúng có thể cùng tồn tại. So sánh có nghĩa phải khóa hệ, task, thông tin và
ngân sách, thay vì đặt một tuyên bố ưu thế tuyệt đối.
