# P1: sai số classifier cục bộ và eligibility

Ngày 09/11/2026. Một adaptation thủ công theo P1, không tái lập đầy đủ DECOLLE,
không đưa vào core live hoặc thay framework. Hai vòng local-target-rate/main
random-feedback trước đó vẫn dừng; trial này ghi rõ objective và đường tín hiệu
khác, không sửa kết quả hoặc tiếp tục tune nhóm đã NO-GO dưới cùng objective.

## Cơ chế được thử

Nguồn đối chiếu: [DECOLLE](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full),
[repository chính thức](https://github.com/nmi-lab/decolle-public).
Giữ ALIF 48 neuron, K=8, R=12 và topology cũ. Một ma trận auxiliary classifier
cố định G đọc filtered spikes; sai số supervised của chính classifier đó được
chiếu về tín hiệu tại neuron, nhân forward EPSP/eligibility của synapse.

G không lấy từ task readout, không học từ task W_out; không truyền sai số qua
neuron khác hoặc ngược thời gian. Có pseudo-derivative và teacher signal không
cục bộ, không gọi là unsupervised learning hoặc exact network gradient. Tests
kiểm local tangent finite difference, forward parity, độc lập với task readout
và checkpoint/RNG continuation trước khi chạy.

[Protocol khóa trước](../../experiments/p1-local-classifier-v1.md): 54 development
trials, ba fresh seed, ba bài order/XOR/strict-binding; readout-only, scalar STDP,
hai learning rate auxiliary và shuffled label control. Cùng 2.048 task updates;
auxiliary có bốn hidden updates/episode, scalar một, nên không equal-compute.
Task head dùng normalization train-only đã công bố, không ridge runtime.

## Kết quả

[Run và mọi dự đoán](../../runs/20261109T114551Z-p1-local-classifier-v1-7384dd/report.md):

| Bài | Initial hidden probe | Gain của candidate được chọn bằng dev | Direct accuracy |
|---|---:|---:|---:|
| Order | 91,80% | +2,73 điểm | 94,79% |
| XOR | 80,99% | −0,52 điểm | 78,39% |
| Strict binding | 55,60% | −0,65 điểm | 62,50% |

Không bài nào đạt yêu cầu tăng representation ít nhất 5 điểm. Không mở tám seed
confirmation và không gọi độ đúng 94,79% trên order là model hiểu ngôn ngữ.
Kết luận **PIVOT_LOCAL_CLASSIFIER_V1**, chưa deploy. Peak process-tree RSS khoảng
138 MB; CPU, không backward xuyên mạng. Chưa có matched learnability control
mới cho thiết kế này, nên failure không bác bỏ DECOLLE hoặc mọi local SNN.

## Quan sát được trong lúc chạy

Lượt này đã công bố tiến độ 0–54/54, task/condition/seed, RAM và mẫu neuron thật
qua localhost. UI đã được quan sát khi job đang sống (20/54, khoảng 133 MB) và
khi hoàn tất. Mẫu cuối là một training tick, không tự diễn giải thành suy nghĩ.
Journal giữ status cuối và lịch sử ngắn; raw run giữ đầy đủ evidence.

Thêm quan sát không thay gate nghiên cứu. Năng lực hội thoại/vision bình thường,
useful hidden learning và lifelong retention vẫn chưa đạt; không tăng quy mô.
