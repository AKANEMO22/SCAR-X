# Nhiễu tín hiệu học tại cùng checkpoint

10/11/2026. **Đã đo đủ144conditionalinputs; chỉ chẩn đoán, không học thêm.**
[Protocol](../../experiments/g1-credit-noise-audit-v1.md).

Đối chứng27 học tốt hơn localscore nhưng có compute và đường tín hiệu khác.
Trước khi đề xuất localrulemới, kiểm tra độ ổn định của tín hiệu synapse tại
cùng weights/input: reward×eligibility hiện hành, cùngscore với baseline từ
nhómnoiseđộc lập và surrogatecontrol trên cùngxung. Không train, tune hoặc
nhập weights, không đổi gốcALIF sang Transformer/LLMproxy. Các familySTOP giữ.

Ninecheckpointstates (3pairedseeds×frozen/local/control) và8prompttrain×từđầu/
EOS,128noisepairs/input. Dữ liệu goldprefixởEOS chỉ là trainingdiagnostic,
không phải trả lời tự do. Đo variance, tái lập haihalfmeans và hướngtínhiệu;
surrogate là biasedcontrol, không groundtruthgradient. Baselinecrossfit chỉ
là variancecontrol, không mở lại scorelearner đãdừng hoặc chọn sau kết quả.

Thước đo giúp xác định giả thuyết kế tiếp, không chứng minh cải thiện quality,
lifelongretention hoặc normalchat/vision. Maincore/privateweights giữ nguyên.
Một Torchprocess CPU1thread,2GB,1200s; forecast từ inputđầu trước chạy toàn bộ.

## Kết quả đã đo

[Run](../../runs/20261109T211546Z-g1-credit-noise-audit-v1-941266/report.md).
3pairedmodelseeds28001/28019/28027×3checkpointstates×8trainingcases×2positions.
Mỗi conditionalinput dùng128independentnoise-pairs. Các bảng lấy median16input
trongmỗi seed rồi trung bình3seed;144input không bị tính thành144seed.

| Trạng thái checkpoint | ScoreEMA: cosinehaihalf | Crossfit-score: cosine* | Surrogate: cosinehaihalf | Crossfit/EMA noiseenergy |
|---|---:|---:|---:|---:|
| Frozenhidden đãfitreadout |0,2684|0,3155|0,9973|0,5534|
| Local-score đã học |0,2019|0,2443|0,9987|0,2274|
| Surrogatecontrol đã học |0,0520|0,1518|0,9827|0,3382|

Cosine so sánh meanhướngsynapse của hai nhóm64pairs tại cùng input/weights,
không là phần trăm accuracy. Crossfit dùng rewardmean của nhómđối diện làm
baseline. *Do đó hai mean-scorecrossfit phụ thuộc lẫn nhau; cosinecrossfit
không là phép lặp độc lập hoặc ước lượngunbiased của hướngđúng.* Noiseenergy
đo phương sai trong từng nhóm với baseline cốđịnh từnhómkia; không khẳng định
đây là baseline tối ưu hoặc đã đo qualitygain.

Ởlocalfinal, noiseenergyratio có CI95%0,1035..0,3822 theo3seedbootstrap; giá
trịtrungbình0,2274 là mean của **median từnginput/seed của cácratio**, không
phải tỷ số của hai cộtnoiseenergyaggregate. Surrogate-minus-scorecosine có
mean0,7908, CI95%0,7690..0,8267. Các seedlocal có scorecosine0,2272/0,1589/
0,2195; directionpoor không chỉ do một seed. Đây vẫn là thăm dò3seed,8styles.

Crossmeanenergy/noiseproxy củascoreEMA ởlocalfinal≈0,00245, surrogate≈0,7605.
Không clamp giátrịâm hoặc điền0 khi không xác định; rawcases giữ cả dấuâm/
missing. Chỉ là finitesampleproxy, khôngtrueSNR hoặc phép dự báo cầnbao nhiêu
trainingepisodes. Không so độ lớn noise của hai estimator khácscale thành
resource/qualitygain. Surrogate là biasedpathwisecontrol, không exactgradient
của expectedBernoullireward và không groundtruth để kết luậnscoređi sai.

## Điều gì đã rõ và chưa rõ

- **Có bằng chứng nhiễu đáng kể:** tại cùng weights/head/input, scorepairmeans
  kém tái lập hơn surrogatecontrol. Học bằng2rollouts/block có thể bị hạn chế
  sampleefficiency, phù hợp failureword-score và kết quảcontrol27. Audit chưa
  tách nhân quả giữa variance, optimizerdrift và chất lượng tínhiệuhọc.
- **Baseline hiện tại có chi phí phương sai:** conditionalcrossfit giảmnoise
  trong nhiều case. Chưa huấn luyện với baseline này, không kết luận retention/
  acquisition tốt hơn và **không mở scorev3** hoặc tune thêm baseline/rollouts.
- **Không tìm thấy lỗi đổi/xóa eligibility ở phép đo này:** directpairformula
  khớp source; weights/moments/head/calibration/rowclocks/counters/RNG được giữ.
  Tồn tại nguyên nhân thuật toán khác không bị loại trừ bởi test này.
- **Chưa có localrule mới hoặc normalchat/vision:** không chọn ứng viên từ
  dev/test, không tăng neuron/model hoặc đưaBPTT vàoSCAR-Xmain.

Điểm cần nghiên cứu tiếp là tínhiệu có ích/đặc hiệu và representationgeneralize,
không chỉ EMA/replay/consolidation. Một hướng khác thực chất có thể là học
liên kết dự đoán trải nghiệm bằng hoạt độngpre/post và ngưỡng học thích nghi,
không dùng scalarREINFORCE hoặc classifierphụ đãSTOP. Đây mới là giả thuyết,
chưa thiết kế/triển khai/kiểm chứng, không tự cấp quyền thay core/allowedsignals.
Mọi đề xuất phải ghi rõ tínhiệu, statephụ, baseline/ablation và budget trong
master trước khi chạy, giữ yêu cầu neuronphátxung/locallearning/CPU2GB.

## Tính toàn vẹn và tài nguyên

9/9filecheckpointSHA giữ nguyên;144/144input giữ weights/trạngtháihọc/RNG.
Neuron/eligibility chỉ được mô phỏng trên bản sao dùng một lần, không nói
fullworkingstate không đổi trong lúc simulation. Mỗi trạng tháiđãhọc cóhead/
weightskhác, nên so giữa trạngthái không phânlập learningrulecausality; so
estimator trong cùngstate mới giữ cùngpoint.

801.792tickNumPy, thêm801.792tickgraph và18.432backwardpasses chỉ trong
isolatedcontrol; không có hidden/head/Adamupdate. Firstconfigurationforecast
123,73s; thực159,05s vẫn dưới1200s. PeakRSS594,03MB/commit994,04MB; Windows
OSpeaks594,99MB/1011,22MB báo riêng. Không iGPU/offload/energyclaim/teacherAPI.
Source/protocol/math hash khớpsnapshot; rawsignalfirstfixture tính lại khớp
statistics từngfloat.158tests trướcaudit đãqua; sốkiểm thửhiệnhành ởSTATUS.

Nguồn lý thuyết về reward/eligibility và baseline:
[Williams1992](https://link.springer.com/article/10.1007/BF00992696),
[Neuronal Dynamics19.4](https://neuronaldynamics.epfl.ch/online/Ch19.S4.html).
Các nguồn đặt cơ sở thuật toán, không thay số đoSCAR-X hoặc chứng minh năng
lựcbrain/tiếtkiệmđiện. Mainmodel tiếp tụcPIVOT; các familySTOP và gateP1/P2 giữ.
