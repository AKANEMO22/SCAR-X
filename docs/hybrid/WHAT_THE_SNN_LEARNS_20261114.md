# SNN đã học chọn hành động; chưa chứng minh học kỹ năng hoặc tự sửa lỗi

**Kết luận:** các cập nhật là thật, có ảnh hưởng đến lựa chọn của ALIF. Tuy
nhiên nhãn của lượt8hbase chỉ dạy chọn `verify`, không dạy tạo đáp án toán hay
viết ngôn ngữ. Hidden ảnh hưởng một số lựa chọn trên state cũ; chưa có lợi thế
so với readout-only/RAG hoặc khả năng tổng quát hóa được xác nhận.

## 100 cập nhật thực sự gồm những gì?

[Protocol audit](../../experiments/hybrid-learning-target-attribution-audit-v1.md),
[70f370 ledger/result](../../runs/20261114T102423Z-learning-target-audit-v1-70f370/result.json),
[đối chiếu nguồn/nhãn](../../runs/20261114T102423Z-learning-target-audit-v1-70f370/independent-data-audit-v1.json).
Auditor nối actualdecision → máy kiểm chứng → learningevent → replay theo ID và
counter, không suy từ thời gian gần nhau. Giá trị20phép cộng được tính lại độc
lập; checkpointSHA và sourcearchive giữ nguyên. Không có forward/update mới.

| Đo trong lượt base8h554c19 | Kết quả | Ý nghĩa |
|---|---:|---|
| Synapseupdates/hiddenupdates được chấp nhận |100/100|Cập nhật thật, không là100trải nghiệm hoặc kỹ năng mới|
| Trải nghiệm mới / replay |20 /80|Chỉ20sourceIDs mới;80lần ôn cùng dữ liệu|
| Nhãn mới |20×actionverify|Label là số thứ tự hành động, không phải đáp án15/17/…|
| Quyết định nhận nhãn mới |19neuralpolicy,1runtimeguard|19lần neural chọn trong nhiều action hợp lệ; chưa có matchedbaseline để quy gain|
| Phản hồi âm / mẫu sửa tool được xác minh |0 /0|Lượt này không kiểm chứng học từ toolerror|
| VLMweightupdates |0|Qwen frozen; không huấn luyện ngôn ngữ, arithmetic hoặc LoRA|

43autonomousdecisionevents gồm21review/21verify/1ask. Có21mask chỉ cho một
actionreview; actor `neural_policy` không tự chứng minh có lựa chọn thực sự khi
chỉ một action được cho phép. Ngoài20verify có proof, còn1verify không có bộ
kiểm chứng phù hợp. `consolidate` có slot trong head nhưng không nằm trong
allowedmasks của các log này; sleep/replay được scheduler điều phối. Không gọi
đó là neural đã tự học quyết định ngủ.

## Hidden có ảnh hưởng hành vi không?

[Protocol reset trước inference](../../experiments/hybrid-action-policy-counterfactual-v1.md),
[b66e37 actualCUDAarrays/result](../../runs/20261114T103233Z-action-policy-counterfactual-v1-b66e37/result.json),
[auditor](../../runs/20261114T103233Z-action-policy-counterfactual-v1-b66e37/independent-data-audit-v1.json).

80forwardCUDA0 trên clone,0bước học,10,281s; physicalfree lấy mẫu≥2740MiB.
Đúng20decisionsequences/masks đã ghi trong checkpoint cũ, cùng seed260113 và
ALIFH128/FP32. Không nạp checkpoint cá nhân vào lab hoặc weights lab vào Hermes.

| Can thiệp trên clone | Số state cũ chọnverify |
|---|---:|
| Initial, chưa học |0/20|
| Final100updates |20/20|
| Final, resetw_in/w_rec vềinitial; giữreadout/statistics |17/20|
| Final, resetw_out/bias vềinitial; giữhidden/statistics |13/20|

Auditor tính lại initialweights từ RNG, kiểm final bằng checkpoint gốc,
exactresetarrays và probability từ standardizedfeatures/readout đã lưu. Sai
khác probability lớn nhất~1,1e-7. Fingerprint readout/hidden sau forward khớp
trước đó. Không thêm một forward nào để chấm lại hoặc chọn thêm variant.

Reset hidden làm thay đổi3lựa chọn verify trong phép can thiệp này: hidden có
ảnh hưởng đến kết quả trên các input đó. **Chưa đủ để gọi hidden học được
representation hữu ích tổng quát**, vì:

- Đây là các state đã tham gia học, chỉ một seed, target đềuverify. Một luật
  luôn chọnverify khi được phép cũng khớp20/20; số20/20 không là thành công AI.
- Finalweights dùng việc học xảy ra sau các quyết định đầu. Kết quả là fit
  hậu nghiệm, không phải accuracy online lúc quyết định đó xảy ra.
- Reset giữ normalizationstatistics của trạng thái cuối. Thay đổi có thể chứa
  mismatch giữa hidden/statistics/readout; không là completefactorial causaltest.
- Chưa huấn luyện matchedreadout-only trên cùng chuỗi này hoặc đo unseen/
  retention. [MatchedDEV/commonprobe trước](HIDDEN_BINDING_AND_COMMON_PROBE_20261114.md)
  vẫn không có advantage đáng tin cậy; không đảo kết luận đó bằng20statecũ.

## Khoảng trống học từ sai đã thấy ở đâu?

Trong logbase8h có1`verification_unavailable`: SNN chọnverify cho mục tiêu mà
bộ kiểm chứng hiện không hỗ trợ. Sourcebranch chỉ ghi sự kiện, không đưa
negativeactionfeedback vào `_guarded_learn`. Vì vậy episode này không tạo nhãn
âm để dạy “hành động này không phù hợp ở đây”. Các lỗi tool khác cũng không
tự trở thành nhãn thật chỉ nhờ reportedstatus hoặc cờverified.

Phải tách hai việc:

1. **Xác minh dữ kiện/đáp án:** hiện có checker số học hẹp và lời sửa người dùng;
   không có checker sự thật tổng quát. Tựcritique/VLMconfidence không là truth.
2. **Đánh giá ích lợi hành động:** nếu runtime có bằng chứng độc lập rằng action
   không hỗ trợ, không tạo tiến triển hoặc vượt budget, có thể nghiên cứu tín
   hiệu utility/cost riêng. Nó không tự xác minh dữ kiện hoặc biết nguyên nhân
   ngữ nghĩa. Phải có causalobservation, outcomeverifier, provenance và kiểm
   rollback/retention trước đưa vào học tự động; chưa triển khai trong lượt này.

Điểm còn thiếu là causalobservations → action có thể thực thi → outcome được
kiểm chứng → rewardđúng → giữ được kỹ năng. Repeatcounter, replay nhiều hoặc
thayweights không đủ khép vòng này. [Inputloss](INPUT_BUDGET_ROOT_CAUSE_20261114.md)
và [khả năng phục hồi đã đo](RECOVERY_ACTION_OUTCOMES_DEV_20261114.md) vẫn là các
nút thắt riêng, không được che bằng việc dạyverify trên bài cộng.

## Tác động đến hướng đi

- Giữ ALIF/localrule/H128 và việc tách máy kiểm chứng khỏi model. Không nhập
  BPTT, Transformer hoặc LoRA vào SNN để làm đẹp kết quả.
- Thử nghiệm tiếp phải có cả lúcverify hữu ích và không hữu ích, toolerror
  thực/nguồn độc lập, nhiễu/thiếu thông tin, unseenstate và đối chứng mạnh.
  Targetđồng nhất không đủ chứng minh selectivelearning hoặc hiddenadvantage.
- Rulehistory variant vẫn chờ duyệt; chưa dùng audit này để đổi công thức học.
  Ngưỡng65%acquisition,3/5ppforgetting,5ppgain/lowerCI>0 giữ nguyên.
- Base8hengineeringPASS không đổi. Smallmixed32cd13 còn chạy với source khóa,
  main/protectedR13 và model cá nhân nguyên vẹn. Các panel localhost mới chỉ
  trình bày actualaudit/resetoutcomes, tách khỏi counterlive;20Nodechecks qua.

**Quyết định:** có chứng cứ actionpolicy thay đổi và hidden có tác động trên
state đã học; **PIVOT** cho learnedrepresentation, autonomouserrorrecovery và
retention. Chưa có cơ sở nói SCAR-X mạnh hơn cùngLLM+strongRAG hoặc càng chạy
thì tự tăng kỹ năng đa năng.
