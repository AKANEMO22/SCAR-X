# P1: score eligibility, phương sai và vòng paired-score cuối

Ngày 09/11/2026. **GO_P2_RESEARCH_ONLY:** đã qua gate representation đã đăng ký
trên order/XOR với 8 seed xác nhận mới. Binding thất bại; chưa kiểm retention
của quy tắc mới và chưa có năng lực chat/vision bình thường. Không deploy.
Đây là vòng cuối điều chỉnh họ score gốc củaPhase2.1, không mở lại local-classifier
hoặc target-rate/random-feedback đãNO-GO. Model live/publicrelease chưa đổi.

## Audit trước khi huấn luyện

[Protocol audit](../../experiments/p1-score-integrity-audit-v1.md) và
[run](../../runs/20261109T131136Z-p1-score-integrity-audit-v1-845ace/report.md)
liệt kê4096đường phát xung trênmỗi fixture2neurons/6steps, không huấn luyện.
ForwardALIFstochastic, xác suất đường đi và expectedreturn được tínhfloat64;
so weightedlocal-score expectation với finite-difference trên6incomingweights.

| Fixture | Sai số tuyệt đối lớn nhất | L2bias do traceclip10 |
|---|---:|---:|
| Moderate | 3,63×10⁻¹¹ | 1,07×10⁻⁶ |
| Strong input | 2,87×10⁻⁹ | 0,993 |
| Delayed cue | 4,58×10⁻¹¹ | 1,87×10⁻⁷ |

Strongfixture tănginput10lần và giảminputweights10lần để giữ operatingvoltages
nhưng tăng sensitivity. Bias có đơn vị đạo hàm, **không phải điểm accuracy**.
Nó không chứng minh clipbias phổ biến ởH48. Cả3oracle khớp tolerance đã khóa;
không tìm thấy lỗi “xóa trace cũ” trong `PlasticCore` gốc. Clipping trước đó đã
được mô tả là heuristic, không phải một implementationbug được phát hiện mới.

8192noise-pairs×3seeds/fixture: antitheticU/1-U có variancetrace khoảng0,87–0,90
lần independentpairs. Đây là giảm nhiễu trongfixture, không là bảo đảm khi48
neurons tương tác/học. Hai trajectory dùng chính presynaptichistory riêng và
cùng priorbaseline; không dùng baseline trung bình của chính hai rewards.

Cơ sở đối chiếu: [Williams1992](https://doi.org/10.1007/BF00992696) và
[reward/eligibility](https://neuronaldynamics.epfl.ch/online/Ch19.S4.html).
Không coi phép kiểm toán này là phát minh mới, bằng chứng biologicalbrain hoặc
khả năng hiểu ngôn ngữ. Idealrawestimator không đồng nghĩa clippedAdamlearner
trongfloat32 là unbiasedgradient của toàn chương trình.

## Prototype thực sự làm gì?

`neural/paired_score.py`, format riêng `scarx-paired-score-v2`; không được
servingimport. CùngALIF48/K8/R12,8input/2output trongdiagnostic. Synapse giữ
forwardpresynapticq và score `(spike-probability)/.4*q`, không taskW_out
transpose, feedbackmatrix hoặc backward qua neuron/time.

Mỗi block12rows xử lýhai rollouts, reward=-NLL từ head thực. Scalarreturn
trừ priorEMA baseline nhân eligibilitytại synapse; update saupair. Bốnblock
phủ48rows, head giữ nguyên trongcácpair rồi học1lần trênmean8features. Running
moment/secondmoment dùng signal tại chính synapse và active-rowclock; không
trộn gradient của synapse khác. Có clippingupdate/weight nhưprotocol.

Neuralworkingstate được reset ởđầu diagnosticrollout để làm counterfactual
có kiểm soát; đây không thay policy continuousstate của model trongHermes.
Một rollout là12ticks; hairollouts×bốnblocks=96ticks/example, không giả thành
96bước deepthinking. Cấu hình nghiên cứu không tăng neuron hoặc tham số lõi.

Checkpoint gồm weights/stablearrays, neuronstate,eligibility,presynaptictrace,
moments,rowclocks,headstatistics,RNG/mode,counters và baseline. Formatv1/P2
không bị ép sang codec mới. Tests kiểm fixed-uniformforward khớp stochastic
ALIFgốc, inactiveweights/moments, frozenhidden và exactpartialcontinuation.

## Thí nghiệm đã khóa

[Protocol paired-scorev2](../../experiments/p1-paired-score-v2.md),
[run nguồn](../../runs/20261109T131841Z-p1-paired-score-v2-0dc401/report.md).
108developmenttrials,3freshseeds,12configurations/task. Tách frozen,
clippedSGD/clippedAdam, unclippedAdam và shuffledsign, independent/antithetic.
Cùng2048examples,8rollouts/example,4hiddenupdateopportunities/1headupdate.
Statistics từ128unlabeledcalibration episodes, frozen trướclabels;8rollout
featuremeans dùngcommonnoise cho initial/refit/trained/evaluation.

| Developmentselected | Order | XOR | Strict binding |
|---|---:|---:|---:|
| Gain refitprobe soinitial | +18,23đ | +25,39đ | +1,04đ |
| Directaccuracy | 90,23% | 76,30% | 49,87% |

Selection đã khóa **unclippedAdam/independent,LR0,002**, không phải antithetic.
Qua màn sàng lọc 2 bài nên mở 8 seed mới theo protocol, nay đã hoàn tất.
Không chọn lại hoặc tune sau test. Binding vẫn gần chance; không dùng order/XOR để nói đã học mọi
quan hệ, có tư duy đa năng hoặc bình thường trongchat/vision.

8rollouts và all-rowcoverage qua4blocks khác Phase2.1historial1rollout/1block;
không chuyển gainthành lợi thế equal-cost sovới các kết quả cũ hoặc BPTT
deterministic. Chưa có matchedstochasticBPTTcontrolmới; negativebinding không
bác bỏ mọi SNN/locallearning. GateP1 và retentionP2 giữ nguyên.

## Quan sát và bước kế tiếp

Localhost tự nhận actualjob/progress,neuronspikes,loss/ΔW mỗi512examples và
tensorpredictions. Không đưa target vàoinput, giả outputchat/Qwen hoặc học
từ câu tự sinh chưakiểmchứng. Dữ liệuAntigravity và Qwenreference trước đó giữ
nguyên, chưa nhập vào prototype này chỉ để làm số liệu đẹp hơn.

## Kết quả xác nhận đã đóng

108 lượt phát triển và 120 lượt xác nhận đã hoàn tất, tổng khoảng 1.293 giây.
Lựa chọn giữ nguyên từ tập phát triển; mỗi task có 8 seed mới, 512 mẫu đánh giá.

| Bài | Gain hidden, điểm % | CI95% gain | Accuracy đầu ra | Holm p | Gate |
|---|---:|---:|---:|---:|---|
| Order | +18,33 | 14,65…22,53 | 91,99% | 0,0234 | Qua |
| XOR | +20,00 | 15,06…24,68 | 74,56% | 0,0234 | Qua |
| Strict binding | −1,29 | −2,37…−0,10 | 49,56% | 0,0938 | Chưa qua |

CI bootstrap theo seed, đúng protocol; không lấy số noisy examples làm số seed.
Refit probe dùng cùng dữ liệu/thuật toán sau khi freeze hidden, nên đây là
bằng chứng hidden weights tạo feature hữu ích trên hai bài đã đăng ký. Không
chỉ suy ra có học từ ΔW khác 0 hoặc readout nhớ đáp án. Các bài vẫn là bit-tasks,
có heldout noise và timing; chưa chứng minh novel-combination/language rộng.

[Audit additive](../../runs/20261109T134107Z-paired-score-technical-attribution-cac3b6/report.md)
giữ nguyên gate, không chọn model khác. So với đảo dấu tín hiệu, selected có
probe cao hơn 8,59 điểm trên order và 11,23 điểm trên XOR; CI95% lần lượt
3,34…13,38 và 5,69…16,04. Holm p=0,0469/0,0234. Đây là phân tích phụ, hỗ trợ
vai trò của reward đúng; một phần cải thiện cũng có ở shuffled control.

Không chứng minh selected vượt mọi baseline: clipped-SGD cùng compute có
order probe gain 23,93 điểm, cao hơn selected 18,33 điểm, nhưng direct thấp
hơn và dao động. Clipped-Adam có direct order 91,09%/XOR67,19%; các so sánh
selected-vs-clipped chưa cho lợi thế thống kê nhất quán. Không quy toàn bộ
gain cho bỏ clip, Adam hoặc paired sampling; independent đã thắng antithetic
trong lựa chọn phát triển. Không tune thêm sau kết quả này.

Can thiệp temporal state: refit accuracy order 92,07% giảm còn 53,98% khi đảo
thứ tự và 50,05% khi reset giữa các bước; bag-only48,73%. XOR refit73,93%,
reset50,12%, OOD timing62,89%. Order OOD timing98,63%; không chuyển con số này
thành chứng nhận reasoning. Binding gần chance với mọi kiểm tra liên quan.

24/24 checkpoint selected tiếp tục đúng hash trong process Python mới, kể cả
neuron state, traces, moments, calibration, readout và RNG. Continuation chỉ
trên bản sao bỏ đi; không sửa original checkpoint hoặc train trên test labels.
129 tests qua, gồm native baseline parity và kiểm expected derivative bằng
finite paths trên chính local core. Neural arrays của cấu hình nhỏ 17.600 byte;
peak process-tree RSS165,0MB/commit187,9MB, bao gồm parent và workers. Mọi
allocated weights ở RAM; không dùng SSD làm weight offload.

Chẩn đoán hậu kiểm trên 64 input cũ, không dùng label: tỷ lệ feature vượt clip5
của selected chỉ khoảng0,05%/0,15%/0,32% trên order/XOR/binding. Chưa có bằng
chứng clipping readout là nguyên nhân chính của binding failure trong selected.
Không gọi chẩn đoán tương quan này là lời giải catastrophic forgetting.

**Bước được phép tiếp: nghiên cứu P2 ở quy mô nhỏ**, với acquisition ≥65% cùng
forgetting mean≤3/worst≤5 điểm theo ngưỡng cũ. Giữ riêng vấn đề binding và
generalization rộng. Chưa nhập core này vào Hermes, train10Mtoken, tăng model,
hoặc tuyên bố Experience-to-Skill/normalchat/vision đã hoàn thành.
