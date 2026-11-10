# Prototype câu hỏi + RGB → ALIF → từ

10/11/2026. **Đã hoàn tất9cấu hình; STOP_VISUAL_WORD_READOUT_V1. Không mainimport.**
[Protocol](../../experiments/visual-word-readout-v1.md). Độc lập với đềxuất30
đangpending; không triển khai errortransport mới, Transformer hoặc BPTTmain.

Audit31 cho thấy prompt không vào đườngvision đangdùng. Prototype này nối
questionfeature vào cùngALIF khi đọcRGBpatches và cho chính wordreadout sinh
chuỗi từ, không host chuyển nhãnclass thành câu. Hiddenweights đềuđóngbăng;
chỉ output-localdelta học. Nếu tiến bộ, đó là readoutgrounding với neural
features sẵn có, không bằng chứng hiddenrepresentation tựhọc hoặclifelong.

Giữ48ALIF/K8/R12 vàchannelbudgetG1. Câu hỏi dùng fixedhashfeature, RGB dùng
InputEvents16patches; coactive trong cùngcore. Cache48neuralvisualfeatures
hữu hạn trongworkingstate, phối hợp0,5với featurehiệntại khi sinhword. Không
thêm model/LLM, không truyxuấtảnh/đápán hoặc masktheotask/language/rendererclass.
Các tham số này khóa, không mở lại closedbyteworking/scorefamily để tune.

96train/96devsyntheticscenes,32AGYquestionwordings chia16/16,864QA mỗi split.
Dev có tổhợpmàu–hình/thứtựobjects chưa train và vịtrí/kíchthước khác. Cảnh do
renderer sởhữu, target từmanifest ngoài modelinput. Không nói geometrictoy
là normalphotoQA/OCR hoặc đạtknowledge4B. Qwen4Btext-only khôngvisionreference.

9trials=3freshseeds×full/noimage/noquestion,1536experiences/model; thêm blank/
swappedimage/question interventions trênFULLtrainedhead. Sinh từ thật, no
goldinput/templateselector; hidden không đổi. Gate chấtlượng/causalablation và
codec trướcđo, giữ mọi failure. Positive chỉfreshvalidationrequired, không
mainimport/scale/P1/P2. CPU2GB,300s/trial/2400sround; UI tựhiệnactualwork.

## Kết quả đã khóa trước đo

[Run](../../runs/20261109T222354Z-visual-word-readout-v1-a11424/report.md),
[dataset](../../runs/20261109T221711Z-visual-word-dataset-v1-f93af5/report.md).
96train/96devimages,864QA/split,16questionwordings/split từ32actualAGYquestions.
Dev có720QA thuộc tổhợpnovel,144QA seencomposition/newgeometry; khôngtrùngPNG.
Ba freshmodelseeds33113/33119/33133,1536supervisedexperiences/configuration.
Targets renderer chỉ đi vào supervisedoutputdelta, không prepare/generate.

| Trainedinputcondition | Macrodev tự sinh | Novelcomposition | Màuđơn | Hìnhđơn | Đếm | Màutrái |
|---|---:|---:|---:|---:|---:|---:|
| FullRGB+question |18,88%|16,94%|29,86%|6,60%|5,73%|33,33%|
| Noimage |17,97%|18,30%|33,33%|0,00%|9,38%|29,17%|
| Noquestion |21,50%|22,03%|34,72%|4,17%|34,38%|12,73%|

Macro=trungbình accuracy từngkind/language, không để số lượngcount/leftcases
lấn điểmshape/color. Điểm targetstyleexact khác humanrubric/generalphotoQA.
Full−noimagegain+0,91đ, CI95%−4,69..5,99; full−noquestion−2,62đ, CI95%−4,51..
−0,69. Chỉ3seedbootstrap thăm dò; không240/864independentmodelseeds.
Không qua65%macro/everykind50%/novel60% hoặc gain20đ đãđăngký.

Trên FULLtrainedhead, xóa ảnh giảm9,24đ, xóa câu hỏi12,23đ, tráo ảnh chỉ3,26đ
(CI95%2,08..4,56). Các OODinterventions cóảnhhưởng nhưng không đủjointgrounding
gate20đ; nhất là FULLkhông hơn TRAINEDmissingmodalitycontrols. Không biến
input sensitivity hoặc initialrandomhead0%→18,88% thành bằng chứngnormalvision.
Train64diagnostic chỉ mẫuđầu, không toàntrain; vài casecolortrain100% không
được dùng nhậnthànhcông. Hiddenweightsđóngbăng, khônghiddenrepresentationclaim.

## Hai lỗi reporting được giữ, không huấn luyện lại checkpoint

Lượt đầu`20261109T222030Z-visual-word-readout-v1-b774f5` đã học xong full/noimage
seed33113, nhưng `macro(emptydiagnosticsubset)` sinhNaN: train64khôngcónovelcases/
left_color nên nhóm đó khôngápdụng. JSONallow_nan=False từchối côngbố vàprocess
dừng. **KhôngNaNneuralweights**, không đổi model/protocol/gate hoặc chọn seed.
Sửa reducer trảNone choemptygroup, giữ0accuracy nếu cócase nhưng sai tấtcả.

Wrapperkhôi phục đầu`20261109T222305Z-visual-word-readout-v1-59b089` có lỗi đếm
arguments, dừng trước khi tạo/trainingmodel. Đã sửa/kiểmcontractpayload9/10args.
Giữ cảhai failedjournals/source/failure receipts. Hai complete1536checkpoints
được táichấm từfilehashcũ, **không học lại**; chỉ7cấu hìnhthiếu đượctrain mới.
Sampler/headcalibration táidựngkhớp và12learningblocks cũ giữnguồnthờiđiểmcũ.
Train_seconds của2modelrecovered ghiNone, không bịa sốthờigian đãmất.

Originalprogramactive12,44s tính cảfailedwrapper; lượt khôi phục32,91s, total
active45,58s dưới2400s. Thời gian sửa giữa cácprocess tách khỏiactivecompute.
Originalfailedattempt peakmemory không khôi phục được; không tuyên bố đã đo
đủ peak toànprogram. Recoveryprocess-treeRSS197,39MB/commit150,34MB; từng
process cóWindowscap vàtreeđượcsample riêng. KhôngiGPU/offload/energyclaim.

## Integrity và đầu ra thực

[Audit](../../runs/20261109T222632Z-visual-word-audit-v1-75fa9f/report.md):9/9cold
prepare→BOS→save/load→nextword/answer/learning/RNG continuations khớpfullhash.
Hiddenw_in/w_rec khớpinitial, source/protocol/data/192PNGhashes giữnguyên.
Hai recoveredmodel byteidentical với checkpoint gốc; sampler/wordupdates/
trainingticks khớpgiữaconditions từngseed. Mỗimodel112.844byte neural/localarrays,
cachevisual48floats, khôngmạngphụ/answerlookup. Onlywordheadweightslearn.

Nguồncorpus câu hỏiAGY thật được dùng16trainwordings, dev16wordingsriêng;
13.824experienceupdates lặp supervisedexamples, không13.824kiếnthức mới.
KhôngnewteacherAPI, Qwen3-4Btext-only vẫnkhôngvisionreference và khôngproxy.
MainALIF/privateweights/Hermes không đổi. Complete864DEVcases từtệp mới nhất
tựhiệnlocalhost vớiRGB/query/target/actualneuralwords, cảcâu sai vàsourcecondition.
Không bắtngườidùngchọn/run hoặc hostchuyểnclassID thànhcâu.

**Quyết định: STOP v1, không cachemixture/encoder/LR/epoch/gridtune hoặcmainimport.**
Đã nốiđược đườngjointinput/wordoutput trongprototype; chưahọcđượcgeneralized
visualgrounding. Normalchat/vision, usefulhiddenlearning vàlifelongretention
vẫn chưađạt. Đềxuất30 vẫnpending, chưacode/train; không coi continuation làduyệt.

Đối chứnglearnability cho chínhjointwordtask có thể là bước nghiên cứu riêng,
chỉở`evaluation/controls.py`, matchedforward/localhead và côngbốcompute thêm;
chưađăngký/chạymới ởvòng này. Không dùng control làm main hoặc mở lại v1 bằng
tuning. Quyết định về directreadouterror trongproposal30 vẫn thuộc người dùng.
