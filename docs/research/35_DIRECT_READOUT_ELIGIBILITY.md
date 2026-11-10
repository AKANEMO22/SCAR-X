# Nhánh30 được duyệt: directreadout signal × eligibility cục bộ

2026-11-10. **Người dùng đã duyệt nghiên cứu; v1 đã chạy45trial, chưa qua P1.**
[Protocol35](../../experiments/direct-readout-eligibility-v1.md).

Giữ neuronALIF và sparse recurrenttopology. Chỉ prototype riêng nhận tín hiệu
lỗi một bước từ readout đã học; mỗi synapse giữ sensitivity của chính neuron
theo voltage/reset/adaptation/filteredspike. Không truyền đạo hàm qua neuron
khác hoặc ngược thời gian, không autograd/RTRL trong ứng viên. Không gọi đây
là hoàn toàn local về sinh học hoặc gradient chính xác của cả mạng.

P0 kiểm forwardparity, anchoredsurrogate finite-difference, no-target-in-input,
signalablation, codec/RNG/moments và RAM trướctraining. Sau đó screen45trials
3freshseed×3temporaltask×correct/shuffled/zero/frozen/BPTTcontrolcôlập. Rule và
ngưỡng khóa trước. Positive cần8freshconfirmation; 100%recallđơn không là thành công.
P2retention/normalchat/vision chưa chứng minh. Chưa triển khai vàoHermes hoặc
tăng quy mô. Đây là bước kiểm cơ chế trước word/visionbridge, không mục tiêu cuối.

Nguồn ý tưởng eligibility × learning signal: [Bellec và cộng sự2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/).
Phương trình cụ thể trong protocol là xấp xỉ riêng cần tự kiểm chứng; không
nhập code/weights hoặc mượn kết quả của bài báo làm bằng chứng của SCAR-X.

## Kết quả vòng đầu

[P0 engineering](../../runs/20261110T051354Z-direct-eligibility-engineering-v1-005d49/report.md)
qua:64sensitivities trên4neuron khớp finite-difference anchoredsurrogate,
maxerror2,59e−8; forward khớp NumPy chuẩn. Partialsegment→checkpoint→endpoint→
Adam/RNG continuation khớp; zero/frozen cùng neural/headweights. Đây không
phải finite-difference của hardspikes hay gradient chính xác toàn mạng.

[Screen hoàn tất](../../runs/20261110T053510Z-direct-readout-eligibility-v1-9e7f21/report.md):
3freshseed35113/35119/35141 ×3task ×5method,1024experiences/trial. Toàn bộ
45checkpoint có coldprocess continuation exact; không chọn seed theo điểm.

| Bài | Correct accuracy | Frozen accuracy | Correct probe gain | BPTTcontrol accuracy |
|---|---:|---:|---:|---:|
|Order / delayedrecency|98,44%|94,92%|+2,34đ|98,57%|
|XOR|91,80%|81,25%|+10,94đ|93,10%|
|Strictbinding|63,41%|62,24%|−1,95đ|63,54%|

XOR có tín hiệu sơ bộ: correct−frozen accuracy+10,55đ (CI95%3,52..18,75),
probe gain+10,94đ (CI95%2,34..23,05), correct−shuffled probe+16,02đ (CI95%
12,11..21,09). Shuffled vẫn nhận đúng nhãn ở readout, chỉ phá phân bổ tín
hiệu về neuron; vì thế ablation này có khả năng phân biệt hiddencredit.
Đây là3seed thăm dò; chưa qua yêu cầu8freshconfirmation hoặc lowerCIgain5đ.

Order có nền frozen đã mạnh, gain dưới5đ; không lấy gần100%delayedrecall làm
thành công. Strictbinding chưa đạt direct65%, probe còn giảm (CI95%−3,91..
−0,78đ). BPTTcontrol cũng chỉ63,54% ở binding: chưa thể quy nguyên nhân chỉ
cho việc bỏ recurrentcredit. Bagbaseline khoảng51–52%, resetstate đưa probe
về khoảng50%. Timepermutation cũng đổi vị trí query/marker, không tự chứng
minh hiểu thứ tự ngữ nghĩa. Không chứng nhận giải được relations tổng quát.

**STOP_DIRECT_READOUT_ELIGIBILITY_V1:** chỉ1/3task qua screen, không mở8seed
xác nhận, không tăng dữ liệu/model/LR hoặc hạgate. P1/P2 và normalchat/vision
vẫn chưa đạt. Có bằng chứng ban đầu cho khả năng học feature trên XOR bằng
cơ chế vừa được duyệt; chưa có bằng chứng đủ cho kiến trúc đa năng/lifelong.

## Khôi phục lỗi I/O, giữ nguyên toán học

Run đầu`20261110T052542Z-direct-readout-eligibility-v1-fc6452` dừng do Windows
PermissionError khi đọc heartbeat.36NumPytrials và2BPTTcontrols đã hoàn tất,
có checkpoint/coldproof;38trial này được giữ nguyên byte, không học lại.
Trial BPTTorder35141 dở có heartbeat832examples; số chính xác trước khi dừng
không biết và chưa có checkpoint hoàn tất. Chỉ7trial còn thiếu được chạy.

Thêm bounded PermissionError retry cho đường đọc, kiểm bằng filehandle Windows
thực sự deny sharing; persistentfailure vẫn báo lỗi. Không sửa eligibility,
optimizer, forward, data hoặc gate; đoạn chọn/xác nhận gate khớp source cũ.
Giữ failure/log/source gốc và [failure receipt](../../artifacts/direct-eligibility-read-io-failure.json).

Originalactive57,04s; recoveryactive56,17s. Roundclock tính cả sửa I/O624,82s
vẫn dưới1800s. Toàn bộ57,04s cũ còn được tính bảo thủ vào cap180s của trial
khởi động lại đầu tiên. Không coi partial832+ là formaltrial mới hoặc kiến
thức mới; ghi rõ công việc mất do lỗi, không bịa exactcount đã không lưu.

RecoverytreepeakRSS621,75MB/commit1055,83MB; peak câyprocess lần đầu/losttrial
không lưu đủ nên chưa có fullprogrampeak. Windowscap2GB/process được áp;
khôngiGPU/offloadweights/energyclaim. Candidatearrays29666byte, tính đủ
traces/bank/moments/core/head; runtime/graph/checkpointcopy vẫn nằm trong số đo.
184softwaretests qua (gồm nativeparity/I/O/codec); không thay qualitygate.

## Phạm vi và hành động tiếp theo

46080formalexperiences/headupdates,27648hiddenupdates trên correct/shuffled/
BPTT,552960NumPytrainingticks và110592BPTTgraphforwardticks thêm. Cùng9stream
train được lặp qua5method, không46080kiếnthức mới. Timingcopies, coldproof
discardedcopies và partialattempt tách khỏi formal. Không dùng Antigravity/
Qwen để đặt nhãn cho tensor số; các dữ liệu/so sánh văn bản thật trước đó còn
nguyên, không giả normalchat hoặc Qwenvision. Stage này chưa huấn luyện từ.

Runtime trong Hermes giữ nguyên mộtALIFcore/weights/topology/vocab; không có
Transformer, LLMproxy, controlweights hoặc feedback35 được nhập vào model đang
dùng. Localhost tự hiển thị actualloss/ΔW/neuron và đủ256case của tệp trial
gần nhất, cả câu sai; scope ghi rõ numericresearch/BPTTcontrol, không chat.

**PIVOT trong nghiên cứu, chưa scale/deploy.** Trước v2 phải tách khó khăn
representation/forward/readout/budget của binding vì ngay matchedcontrol
chưa học tốt. Không tự coi XOR là đủ cho word/vision, không retune familyv1
bằng đổi tên. Nhánh đã được duyệt còn có thể nghiên cứu tiếp trong ranh giới
tín hiệu đã chốt, với giả thuyết/protocol cụ thể và tối đa2vòng đăng ký.
