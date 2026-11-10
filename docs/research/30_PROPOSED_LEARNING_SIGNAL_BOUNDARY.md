# Đề xuất cần chốt ranh giới tín hiệu học

10/11/2026. **Người dùng đã DUYỆT NGHIÊN CỨU đề xuất30:** “Duyệt nghiên cứu đề xuất 30”.
Chỉ mở prototype trong phạm vi dưới đây; chưa duyệt triển khaiHermes, tăng quy
mô hoặc coi cơ chế đã hiệu quả. [Protocol35](../../experiments/direct-readout-eligibility-v1.md)
khóa phương trình và phép đo đầu tiên. Phần lý do xin duyệt bên dưới được giữ;
các báo cáo32–34 ghi pending đúng tại thời điểm lịch sử, không sửa kết quả cũ.
Không thay mainSCAR-X. Không cho phép Transformer/pretrainedLLM/LLMproxy,
BPTT hoặc tăng model/RAM. Đây là quyết định về nguồn tínhiệuhọc hidden,
không phải yêu cầu đổi gốc neuronphátxung LIF/ALIF.

## Vì sao cần quyết định riêng

Master§3.3 ghi **“Không truyền vector sai số qua `W_out` về hidden.”** Đây là
mô tả đườnghọc hiện tại; tôi không tự suy ra từ quyền nghiên cứu rằng có thể
bỏ ranh giới này. Master§11 yêu cầu hỏi người dùng nếu đổi tínhiệu học cho phép.
Các thử nghiệm scalar-score và observedassociation chưa chứng minh language
representation hữu ích; BPTTcontrol27 là đối chứng, tuyệt đối không nhập vào main.
Failure không chứng minh mọi purelocalrule bất khả thi hoặc bắt buộc đổi ranhgiới.

## Phạm vi xin duyệt cụ thể

Cho phép **một nhánh nghiên cứu riêng**, chưa phục vụHermes, trong đó một neuron
nhận tínhiệusai số trực tiếp từ readout đã học, thay vì mọi hiddenneurons cùng
nhận một scalarreward hoặc classifierphụ fixedrandom đãSTOP. Giữ cùng ALIF
forward/topology/CPU2GB; synapse vẫn cập nhật từ presynaptictrace, hoạtđộng
postsynaptic và tínhiệu tại neuron. Không reverse-time tape/autograd/RTRL hoặc
gradient truyền qua các neuron/recurrentedges; không khởi tạo bằng LLMweights.

Với `err[k,t]=one_hot(target)[k]-p[k,t]`, đề xuất tínhiệu tại neuron j:

`L[j,t] = Σ_k W_out[k,j] * err[k,t]`

`delta_w[j,i] = clip(eta * L[j,t] * e[j,i,t], -update_clip, update_clip)`

`e[j,i,t]` là eligibility tính xuôi theo state của synapse/neuron, gồm leak,
reset, thresholdadaptation và lọcspike. Bỏ các đạo hàm lan qua neuron khác;
do đó đây là xấp xỉ, không exactwhole-networkgradient. Cần đăng ký phươngtrình
và kiểmfinitedifference fixedpresynaptictapes trước training; chưa chọn implementation.

**Điều không được che giấu:** phép `W_out.T @ err` là truyền sai số một bước
ngược qua readout, dùng thông tin từ đầu ra và độ mạnh kết nối đã học. Nó
không phải chỉpre/post+rewardscalar hoặc hoàn toàn local theo nghĩa sinh học.
Không gọi “khôngbackward()” là bằng chứng không có gradient/errortransport.
Dù khôngBPTT/toànmạngbackward, đây là MỞ RỘNG đường tínhiệu hiện cho phép,
nên cần duyệt riêng (nay đã duyệt nhánh nghiên cứu). Chưa khẳng định cơ chế
này sẽ nói/nhìn tốt hoặc giải forgetting.

Nguồn đối chiếu: [Bellec và cộng sự2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/)
và [repository tác giả](https://github.com/IGITUGraz/eligibility_propagation).
E-prop kết hợp eligibility với learningsignal, có các lựa chọn feedback;
không đồng nhất mọi phiên bản với localityconstraint củaSCAR-X. Đề xuất này
phải tự đo/audit, không lấy chất lượng/điện năng của công trình làm kết quảdự án.

## Kiểm chứng bắt buộc nếu được duyệt nhánh nghiên cứu

1. Prototype/codec/signaldiagram riêng; outputerror chỉ tới từng neuron một
   bước, không truyền error giữa hiddenneurons hoặc ngược thời gian. Test
   ngắt feedback vẫn giữ inferenceforward; no-target-input khi sinh câu/probe.
2. Baselinefrozen, hiddenlearnsignal đúng/sai/zero và đối chứngBPTT côlập.
   Cùngforward/input/head/data/updates; đo thêm time/RAM, không giả matchedcompute.
3. Freshseeds/splits/budget và ngưỡng phải đăng ký trước. Dùng cả refitprompt-only
   probe, freeutterance, temporal/relationaltests; không nhận readoutgain/ceiling
   recall hoặc lấyQwenoutput làm chân lý. Tối đa2registeredrounds/giảthuyết.
4. P1representation/P2retention/2GBgates giữ nguyên; chưa có quyền scale10Mtoken/
   tham số hoặc deploymain. Mainweights/history chỉ đổi sau acceptance riêng.

## Hai lựa chọn giữ rõ hệ quả

**Bổ sung bằng chứng ngày2026-11-10, không thay phạm vi xin duyệt:**
[Đối chứng33](33_VISUAL_WORD_LEARNABILITY_CONTROL.md) chỉ đạt25,67%dev và
48,43%train trênjointvisualwordtask. [Audit34](34_VISUAL_INFORMATION_PATH.md)
không thấy exactfullpathcollapse/clipbottleneck; fixedsemanticprobe trên
neuralfeatures chỉ8–12%dev. Chưa giải được representationgeneralization,
không có bằng chứng rằng directreadouterror sẽ tự khắc phục vấn đề này.
Vì vậy duyệt30 chỉ mở một thí nghiệm giả thuyết, không là duyệt lời giải,
đổi forwardALIF, tăng quy mô hoặc nạp candidate vàoHermes. Giữ ngưỡng cũ và
đăngký protocol/engineeringparity/locality/eligibility trước mọi thử nghiệm.
Nếu giữ ranh giới cũ, không tự triển khai learner mới hoặc mở lại các family
đãSTOP bằng đổi tên; cần một đề xuất khác trong tínhiệu đã được phép.

- **Duyệt nhánh nghiên cứu này:** cho phép directreadout-errortransport nêu
  trên trong prototype; vẫn cấm Transformer/LLMproxy/BPTT/toànmạngbackward.
  Chưa duyệt triển khai hay đánh dấu thành công.
- **Giữ ranh giới cũ:** không triển khai nhánh này; tiếp tục chỉ nghiên cứu
  cơ chế pre/post/observation/scalar-modulation trong giới hạn hiện có. Không
  kết luận purelocalbất khả thi; thừa nhận chưa có lộ trình được chứng minh
  tới normalchat/vision và tránh tuning các familyđãSTOP vô hạn.
