# Đối chứng learnability của bài sinh từ

10/11/2026. **Đã hoàn tất6đối chứng, không đổi mainSCAR-X.**
[Protocol](../../experiments/g1-word-learnability-control-v1.md).

Vòng26 thất bại chưa cho biết bao nhiêu lỗi do khả năng biểu diễn/đầu ra và
bao nhiêu do creditassignment. Đối chứng này giữ forwardALIF, đầu vào, readout,
sampleorder và lịchupdate, chỉ cho nhánhcôlập dùng taskerror xuyên neuron/time
qua surrogategradient. Không đổi gốc SCAR-X sang Transformer hoặc đưa backward
vào maincore. Quyền dùng BPTT đối chứng đã nằm trong master; weights đối chứng
không được nhập vào Hermes hoặc gọi là bằng chứng học cục bộ thành công.

NumPycore vẫn tạo xung thật, graphcontrol replay cùngspiketape để tính tín hiệu
gradient. Surrogate derivative là giả định học có bias, không là exactgradient
qua randomBernoulli. Dùng cùng per-pathNLL và per-synapseAdam/blocktiming như
vòng26; outputhead tiếp tục localdelta. Tính thêm8graphtrajectories và4backward
passes mỗiword phải báo riêng, không tuyên bố matchedcompute. Cách dùngsurrogate
tham khảo khung [Neftci và cộng sự2019](https://arxiv.org/abs/1901.09948), không
tái lập số đo hoặc công thức cụ thể của họ.

Six fullcontrols=2encoders đã chỉ định×3pairedseeds; frozenbaseline6lượt thật
từvòng26 chỉ tái dùng sauparity. Cùng1024responses, không searchLR/epochs hoặc
scale. Preflight64trainingresponses đo tốc độ/RAM, không chấm dev/prior; chỉ
mở formalrun nếu dự toán dưới6000s,900s/trial và2GB. Một Torchchild/CPU1thread,
không đểTorch nằm trong tiến trình app hoặcpytest đoRAMmodel.

Nếu control học được bài này, kết quả chỉ giúp định vị nút thắt tín hiệu học.
Nếu không, chưa thể quy mọi thất bại cho localrule; cần kiểm tra representation,
decoder và objective. Cả hai kết quả đều không giải quyết lifelongretention,
ngôn ngữ đa năng hoặc vision bình thường; không mở lạiword-scorefamily đãSTOP.

## Preflight và sự cố kỹ thuật được giữ lại

[Preflight](../../runs/20261109T191356Z-g1-word-control-preflight-v1-15ca2f/report.md)
kiểm6frozenconfigurations×64responses: loss và fullstatebytes khớp NumPy.
64response timing/encoder ởseed29011 mất11,55/11,21giây; graphstateerror lớn
nhất1,91×10⁻⁶, featureerror0. PeakRSS560,48MB/commit981,33MB. Dự toán sáu
formalcontrols có25%headroom≈1511,39giây, dưới6000s. Không đo dev/prior ởbước này.

Lượt đầu`20261109T192017Z-g1-word-learnability-control-v1-446c16` dừng do
PermissionError/WinError5 tại os.replace journal, không do neuralquality.
Worker đã đến448response, chưa finaldev/checkpoint. Giữ nguyên partiallogs và
[incidentreceipt](../../artifacts/g1-word-control-io-failure.json), không chọn
checkpoint theo loss hoặc nối giả phần học đã mất.

[Kiểm traWindows thật](../../runs/20261109T192742Z-g1-observer-atomic-recovery-v1-a5de0a/report.md)
giữ reader mở để tái hiện cản trở atomicreplace; boundedretry xuất bản payload
đầy đủ sau khi reader đóng. Oldfile giữ nguyên khi chờ; lỗi permission vĩnh
viễn vẫn bị trả lại. Sửa chỉIO/coordinator, math/protocolhash giữ nguyên.

Chạy lại từ initialization cũ ở
`20261109T192850Z-g1-word-learnability-control-v1-d74480`, cùngsampler/noise,
không giữ bất kỳ devresult nào từlượt lỗi. Deadlinegốc vẫn6000s/round và900s/
firsttrial: tính cả513,21giây từstartlượt cũ qua thời gian sửa tới lúc chạy lại.
155tests qua, không gọi softwarepass là mainlearning thành công.

## Kết quả learnability đã khóa trước đo

[Run](../../runs/20261109T192850Z-g1-word-learnability-control-v1-d74480/report.md).
Đây là cùng stochasticALIF/canonicalprefixforward của prototypeScoreWordv2,
không phải trạng thái continuous/hardspike của phiênUnifiedCore đangdùng.
Haiencoder và3pairedmodelseeds được chỉ định trước; frozen/localreference tái
dùng kết quảv2 sauparity, không chọn lại hoặc huấn luyện thêm để tăng điểm.

| Input | Phương pháp | Train16 tự sinh | Dev31 tự sinh | Prior48 tự sinh |
|---|---|---:|---:|---:|
| Legacy | Frozenhidden |16,67%|3,23%|6,25%|
| Legacy | Localscorev2 đãSTOP |10,42%|6,45%|4,17%|
| Legacy | SurrogateBPTT đối chứng |93,75%|36,56%|72,92%|
| Densebits128 | Frozenhidden |25,00%|8,60%|13,89%|
| Densebits128 | Localscorev2 đãSTOP |12,50%|7,53%|9,72%|
| Densebits128 | SurrogateBPTT đối chứng |91,67%|44,09%|77,78%|

Densecontrol vs frozen devgain+35,48đ, CI95%25,81..41,94; vs localfull+36,56đ,
CI95%29,03..41,94. Legacy vs frozen+33,33đ, CI95%19,35..41,94; vs localfull
+30,11đ, CI95%16,13..38,71. Chỉ3seedbootstrap thăm dò, dev/prior đã được xem
trong nghiên cứu trước; không là confirmation8freshseeds hoặc freshfinaltest.

Interpretation đăng ký: dense=`CONTROL_LIMITED_LEARNABILITY_MAIN_UNPROVEN`,
legacy=`CONTROL_FITS_TRAIN_GENERALIZATION_UNRESOLVED` vì meanlegacydev<40%.
Train16 chỉ16mẫu diagnostic đãhọc, không đại diện toàn181train hoặc năng lực
mở. Seedđầu dense100%train16 **không** được dùng kết luận thành công.

Kết quả cho thấy neuroncount/input/decoder hiện tại không ngăn tuyệt đối việc
học8stylemappings. Tín hiệu học hidden khác đi giúp rõ trong đốichứng, nên cần
xử lý creditassignment/hiệu quả học trong main. Nhưng cả phương pháp gradient
và compute thêm đều khác; chưa tách được mọi nguyên nhân hoặc chứng minh một
quy tắc local thay thế. Dev44,09% còn thấp, kể cả đối chứng; representation/
generalization vẫn chưa giải quyết. Không suy ra ngangLLM4B, hiểu ngôn ngữ
đa năng, học liên tục hoặc nhìn ảnh bình thường.

**Mainword-scorev1/v2 vẫnSTOP. Không nhập controlweights/graph/BPTT vào main.**
SCAR-X tổng thể tiếp tụcPIVOT trong phạm vi LIF/ALIF+locallearning/CPU2GB.
Không mở modelscale hoặc tuning tiếp các family đã đóng. Bước tiếp theo cần
thiết kế tín hiệu học cục bộ có tính đặc hiệu hơn và phân biệt compute, chất
lượng representation, acquisition và retention, đăng ký trước mọi phép đo.

## Chi phí, tính toàn vẹn và dữ liệu

Sixcontrols có tổng6.056.688tickNumPy thật, thêm6.056.688tickgraph và135.728
backwardpasses; cùng33.932wordtargetupdates với baseline tương ứng. Không
tuyên bố matchedcompute. DatasetAGY/authored thật đã dùng6.144responseupdates
(lặp theo sampler), không phải6.144kiến thức mới độc lập. Không thêm teacher
labels, publicdev/prior vào train hoặc sửa guard private.

6/6coldprocesscontinuations khớp loss/nextreply/fullbyteshash. Analogstateerror
lớn nhất2,86×10⁻⁶, featureerror0. Controlwrapperformat riêng bị productionloader
từ chối. Staticimportcheck không thấyTorch/control trong neural/app/runtime;
fingerprintweights,topology,vocabulary,encoding và updatecounter241363 của
phiên cá nhân khớp checkpoint trước lượt nghiên cứu. Chỉ lưu booleanproof ở
artifact công khai; fingerprintprivate chỉ ởlocal.156tests qua, gồm nativeparity.

Lượt chạy lại1485,36s; tính cảlượt lỗi vàgap sửa theo deadlinegốc1998,72s, dưới
6000s. Firsttrial hiệu lực747,92s<900s. PeakprocesstreeRSS623,37MB/commit1014,57MB;
không iGPU/energyclaim. Windowsper-processcap và sampledtreememory báo riêng.

## Qwen reference và xem đủ đầu ra thật

[Protocol](../../experiments/qwen-word-control-reference-v1.md),
[run](../../runs/20261109T195432Z-qwen-word-control-reference-v1-e3cc5f/report.md).
Sau khiTorchcontrols đã dừng, Qwen4B pinnedhash được chạyCPU riêng trên47prompt
còn thiếu (31dev+16train); không lặp lại48prior, không chọn theo lỗi/điểm.
Singleusertext,context1024,temp0,seed919,max_tokens64 như bản đo cũ.47/47requests
có output,24bịcắt; cộng bảncũ95reference có41lengthtruncations. RawQwen không
là nhãn đúng hoặc nguồn trọng số. Phép đo khoảng257,03s, peakRSS3,236GB/
commit3,281GB với referencecap4,5GB riêng, **không** thuộc claimSCAR-X dưới2GB.
Ownedllama-server đã tắt; không bật lạiQwen9B hoặc đổiHermesprovider.

[Toàn bộ đầu ra đã lưu](http://127.0.0.1:8765/#answer-archive) tự hiện95case
của tệp gần nhất, cả sai/đúngstyle, không chọn model/bấm chạy. CờISOLATED BPTT
CONTROL tách rõ khỏi main; mốc thời gian là lúc lưu record, không giả lượt
inference mới. Qwen ghép95/95đúngpublicprompt, nguồn/finish_reason giữ riêng.
Backendreader chỉ đọc fileđãđo và publicpair đãreview/hash; không train/rerun
để điền trang hoặc trả privateguard/weights. Đây là quan sát, không năng lực mới.
