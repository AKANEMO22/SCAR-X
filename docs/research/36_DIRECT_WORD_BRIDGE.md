# Word bridge của cơ chế30: pilot sinh câu, không triển khaiHermes

2026-11-10. **Đã chạy15trial; STOP_DIRECT_WORD_BRIDGE_V1. Khôngmainimport.**
[Protocol36](../../experiments/direct-word-bridge-v1.md) thực hiện phầnwordbridge
đã nêu trongprotocol35, sau P0math/codec đã qua. V1temporal35 chưaP1; pilot
MVP nhỏ này không mởconfirmation/P2, bỏgate hoặc tăngmodel/pretrain10Mtokens.

Giữ48ALIF/sparse recurrence, reusedG1context/3tickwordinput, cùngeligibility35
vàreadout-errorstep đãduyệt. Sinh từngtoken từneuralhead, không dùng intentrouter,
classformatter/answerlookup/LLMproxy. Supervision8style vi/en, promptAGY mới
phải review toàn bộ; nguồn nhãn là canonicalstyle của dự án, không teacherfacts.

3freshseed×5method/frozen-wrong-zero-BPTTcontrol,1024response/trial. Phải đo
freeutteranceheldout, hiddenablation/resetheadprobe, actualRAM/state/coldresume.
Qwen4B trả lời cùng64DEVprompts riêng, dữ liệu không thành nhãn học. Qwen có
budgetRAM lớn hơn, không chấm nó bằng exactstyle của SCAR-X hoặc nhận là ngang.

Các giới hạn:4intent×2language là closedstylepilot, chưa conversationđa năng,
vision/lifelong hoặc benchmarkcommercial. MainALIF/Hermes/weights giữnguyên.

## Dữ liệu thật và hai sửa đổi trước đo

[AGYCLI](../../runs/20261110T060709Z-antigravity-direct-word-prompts-v1-820d4e/report.md)
đã sinh128prompt trong62,66s. Đã đọc toàn bộ:1row`vi_greet_11` trộnEnglish
vi phạmcontract, giữgốc/cáchly và gọiAGY sửa đúng1row; replacement tiếngViệt
được đọc/kiểm, không chọn theo modelscore. [36B](../../experiments/direct-word-bridge-v1-language-repair.md)
ghi phạmvi sửa trướctraining, không rewritecandidate/protocol gốc.
[Review](../../runs/20261110T061343Z-direct-word-semantic-review-v1-921842/report.md)
khóa64train/64dev theoRNG36101,6caveats,3overlaps với publicprompt lịch sử.
Do đó không gọi freshfinaltest hoặc independentcertification.

[36A](../../experiments/direct-word-bridge-v1-data-amendment.md) đổi ownedtarget
en_thanks thành “You are welcome.” trướccode/training/quality, vì tokenizer
hiện tại tách apostrophe khiến “You're” không render đúng. Không sửa tokenizer,
neuraloutputs hoặc target theo score. AGY chỉsinhprompt, chưa nhận targets;
nhãn8style do dựán địnhnghĩa không là factualtruth/knowledge mới.

## Phép học và kết quả đã khóa

[Preflight](../../runs/20261110T061535Z-direct-word-preflight-v1-6591af/report.md)
khớpG1context128/3tickwordinput, fixed64TRAIN timing, zero=frozen, BPTTgraph và
partialstate/optimizercontinuation. Khôngdevscore trongpreflight.
[Run15trials](../../runs/20261110T062226Z-direct-word-bridge-v1-a6a908/report.md)
3freshseed36313/36319/36341,1024TRAINresponses/method, no learnedembedding/
intentrouter/classformatter. Sinh từ bằngneuralhead, previousgold chỉtraining;
inference chỉpreviousneuralwords. Headlocaldelta/hiddenbankmean cuốiresponse,
eligibility35 và synapse-localAdam nguyên côngthức, không đổi learningrule.

| Method, mean3seed | Câu train | Câu tự sinh dev | Resethead probe gain | Dev sau trả hidden về initial |
|---|---:|---:|---:|---:|
|Correct feedback|99,48%|35,94%|+14,06đ|17,71%|
|Shuffled neuronfeedback|76,56%|17,19%|+6,25đ|11,46%|
|Zero feedback|92,71%|16,67%|0,00đ|16,67%|
|Frozenhidden|92,71%|16,67%|0,00đ|16,67%|
|BPTTcontrol côlập|100,00%|36,46%|+11,46đ|17,19%|

Correct−frozen freeutterancegain+19,27đ, CI95%18,75..20,31, positive cả3seed.
Resetheadfirstwordprobe gainmean14,06đ nhưngCI95%0,00..25,00, mộtseedgain0:
chưa qua lower95>0. Probe là diagnosticclassifier ngoàicore, không câu trả lời
hoặc decoder mới. Trảhidden vềinitial làmdev giảm18,23đ, làlesion ngoài phân
phối chứ không pairedtrainingbaseline riêng. Hai canthiệp gợi ýhidden cóđóng
góp trongpilot này; chưađủ P1representation8seed/toànbộtask.

Devcorrect từngseed35,94%/32,81%/39,06%; thấp hơn65%. Nhiềufamily dưới50%,
en_thanks ởseed36341 bằng0. BPTTcontrol cũng chỉ36,46%dev mặc dùtrain100%:
hạn chế tổngquáthóa không thể quy duy nhất cho việc khôngBPTT. Noisy/hashinput,
vốnword nhỏ/closedlabels/48neuron và ngân sách học có thể góp phần; chưa tách
nguyênnhân bằngcanthiệp. Traincao không là bằng chứngnó nói chuyện bìnhthường.

**STOPv1:** macrodev65/everyfamily50/probeCI chưađạt, chỉpairedfreegain qua.
KhôngtuneLR/epochs/encoder/dataset đểmở lại v1, không scale/confirmation35/
deployHermes. Không đổiP1/P2/forgettinggate. Normalchat/vision/lifelong chưađạt.

## Tái lập và tài nguyên

15/15checkpoint cófreshprocess nextlearn→freeanswer→sampler→fullhash khớp.
Zero/frozen khớp weights/head/predictions, streams/wordupdates/trainingticks
khớp5methods theo seed. Protocol36 vàeligibility35 được hash trướcformal.
15360formalresponseexposures,67110wordheadupdates,9216hiddenupdates(correct/
shuffled/BPTT);385650NumPytrainingticks,77130BPTTgraph ticks thêm. Repeated64
trainprompts không là15360kiếnthức mới. Timing/colddiscardedcopies ghi riêng.

Round68,28s, peakprocess-treeRSS627,22MB/commit1024,74MB; Windowscap2GB/process,
noGPU/iGPU/offload/energyclaim. Candidatearrays374662byte kểcảstate/traces/
bank/Adam/headcontext, runtime/rollback/checkpointcopy đo ởprocess. Khôngnhập
trọngsố vào release/phiên riêngHermes.

190tests qua/46,94s với OPENBLAS/OMP/MKL_NUM_THREADS=1, gồmnativeparity và
archive64prompt/typedmetric/sourcehop/privacychecks (trướcformal189tests qua,
sau runtime-reader hoàn thiện thêm1test và rerunfullsuite).
Hai lượt fullsuite defaultthread trướcđó có1legacyguardtest HTTP507; testđơn
qua. Chạy1thread nhưresearchbudget thìsuitequa; chưakếtluận duy nhất một
nguyênnhân allocator. Không nới productionguard/threshold hoặc xóa test fail.
Không dùngsoftwaretests thayqualitygate.

## Qwen và quan sát

Qwen4B đãđo xong cùng64DEVsingleuserprompts bằngruntime/model cósẵn,
seed36313/correctpair cốđịnh trướcscore, không chọnbestseed. Chỉrawprompt đi
vào Qwen, khôngtarget/family/SCAR-Xanswer. Qwen referenceRAM4,5GB riêng, lớn
hơntrầnSCAR-X2GB. Không chấmexactstyle/identitykhác củaQwen thànhsai, không
đưa outputQwen vào nhãn. Truncation/rawoutputs/phầnthiếu phải giữ thực.

[RunQwen](../../runs/20261110T062343Z-qwen-direct-word-reference-v1-5e4d7d/report.md)
hoàn tất64/64 trong320,99s,14output bịcắt ở64tokens; rawtext/finishreason
được giữ. PeakRSS3362,55MB/commit4046,53MB, không ngangbudgetSCAR-X. Owned
server đãtắt trongfinally, khôngQwen9B/llama-server cònchạy. Không có exactstyle
score hoặc tuyên bố Qwen luônđúng từ64mẫu. Chẳnghạn Qwen tựnhận danh tính khác
SCAR-X là hợp ngữcảnh của modelthamchiếu, khôngphảinhãnidentityđểSCAR-Xhọc.

GUI đã xác nhận bảng5methods và cặp actualwordoutput/Qwen vớinguồn/thờiđiểm
riêng; vídụ cuối theo thứtựđo, không chọncâuhay, cho thấy SCAR-X trảlời sai và
lặp trong khi Qwen cócâu farewell. Dữ liệu phản ánh khoảng cách lớn hiệncòn,
không che bằngrerender/lookup hoặc thayQwenproxy vào model đangdùng.

Localhost tựhiện actualwords/targets/loss/ΔW/neuron; archive64DEVcase từfile
hoàn tất gần nhất, không theoaccuracy, có cảcâu sai. Research vàBPTTcontrol
đượcghi riêng; GET chỉđọc, khôngtựtraining đểlấpUI. Main vẫn mộtALIFcore,
khôngTransformer/LLMproxy hoặc modelthứhai chochat/ảnh.

## Quyết định tiếp theo

**PIVOT_NO_SCALE; wordbridgev1 dừng theo đăngký.** Có pairedgain/hiddenlesion
gợi ýđónggópnhưng chưađủnăng lực; không phát hành lênHermes, không thêmepochs/
prompts/LR/encodergrid để mở lại v1. Cơ chế35 giữ nguyên nhưđãduyệt, chưaqua
P1/P2; cảwordbridgevàBPTTcontrol cònkémvềparaphrasegeneralization.

Cần tách biểu diễnngữnghĩa của inputhash khỏi học chuỗi từ, bằng giảthuyết
representation có tínhiệu/đối chứng rõ, không tăngparam hoặc nhập pretrained.
Vision vẫn là nhánh chưađạt trong32–34; nếu chuyểncơchế30 sangjointRGBword,
phải đăngký riêng tínhiệu/trace/cachederivative vàmissingmodalitycontrols,
không suyra từpilottext này. Mục tiêu cuối normalchat/vision/lifelong giữnguyên
và chưa hoàn thành, không thu nhỏ thành8styleclassification để gọi success.
