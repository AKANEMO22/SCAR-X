# Direct eligibility cho jointRGB–câu hỏi–từ

2026-11-10. **Đã hoàn tất21trial; STOP_DIRECT_VISUAL_BRIDGE_V1. Khôngmainimport.**
[Protocol37](../../experiments/direct-visual-bridge-v1.md) là phầnvisionbridge
trongscope35/36, không tăngmodel hoặc thayTransformer. Giữ48ALIF và toàn bộ
forward/cache32; chỉ ánh xạ eligibility35 vào đúng pooledfeature mà readout đọc.

Cachefeature32 là mean16filteredspikes, nên sensitivitycache phải là mean16
f_e cùngneuron; wordendpoint lấy0,5(cache_e+current_e). Lỗiđầu ra đi mộtbước về
neuron, không reverse qua patch/thờigian hoặc neuron khác. CandidatekhôngTorch/
networkBPTT; globalcontrol ởcontrols.py riêng. Đây là mở rộngresearch30 đãduyệt,
khôngmainmigration hoặc lời cam kết hữu ích về ngônngữ/ảnhthật/lifelong.

Reused192ownedPNG/32AGYwordings của32 đãreview,864QA mỗi split,
devđãxem nên exploratory. Không cần gọiAGY lại đểlặp dữliệu, Qwen4Btext-only
không làmvisionreference. Qwen64actualtextpairs36 giữ riêng và khôngthànhnhãn.

3freshseed×7condition1536experiences, fixedgates32+frozencontribution, resethead
probe, image/question/swappairablation. P0finite-difference/forwardparity/fullstate
trướctraining; GUI actualoutputs/learning, no mock/mainweightschange.

## Kết quả đã khóa trước đo

[P0](../../runs/20261110T064337Z-direct-visual-preflight-v1-927d70/report.md)
qua21paritychecks: mỗi seed/method16TRAINresponses khi tắthiddenupdate khớp
nguyên VisualWords32forward/head/calibration. Pooledsensitivity64synapse khớp
anchoredsmooth finite-difference, maxerror1,78e−7; đây không gradientexact của
hardspikes/toànmạng. Pending33inputvectors, feature/cacheeligibilitysums/cursor
được lưu để midimagecontinuationexact; không giữ T×N×NJacobian/neurontape.
197tests/nativeparity/rollback/codec/archiveguards qua trướcformaltraining,
BLAS/OMP/MKL1thread, không thay guard hoặc ngưỡng.

[Run21trials](../../runs/20261110T065124Z-direct-visual-bridge-v1-3390d3/report.md):
3freshseed37313/37321/37339, mỗi1536experiences, cùngsampler/calibration/gold
wordtargets/NumPyforwardticks theo seed. Dữliệu32reused/864train864dev và
novelcompositions đãxem, không gọi finaltest hoặc newteacher/newknowledge.

| Method, mean3seed | Câu train | Macrodev | Tổ hợp mới | Màuđơn | Hìnhđơn | Đếm | Màutrái |
|---|---:|---:|---:|---:|---:|---:|---:|
|Correctfeedback|51,53%|25,10%|25,76%|38,89%|1,74%|13,37%|46,41%|
|Shuffledfeedback|40,17%|21,32%|21,58%|25,00%|0,00%|24,05%|36,23%|
|Zerofeedback|54,18%|15,05%|14,70%|25,00%|4,17%|0,00%|31,02%|
|Frozenhidden|54,18%|15,05%|14,70%|25,00%|4,17%|0,00%|31,02%|
|Noimage/correct|42,71%|13,54%|14,58%|20,83%|4,17%|0,00%|29,17%|
|Noquestion/correct|18,84%|17,65%|17,25%|32,64%|0,00%|13,19%|24,77%|
|BPTTcontrolcôlập|53,79%|26,30%|27,05%|43,40%|5,56%|16,32%|39,93%|

Correct−frozen+10,05đ, CI95%7,81..13,32; cả3seedpositive, gatefrozencontribution
qua. Correct−shuffled+3,78đ nhưngCI95%−6,16..11,00 chứa0 và mộtseedshuffled
tốt hơncorrect. Vìvậy chưa đođược lợi thế phân bổ feedbackđúng với độ chắc
chắn cầnthiết. Không so trực tiếpkếtquả25,10% với18,88% của32 nhưpairedgain:
seeds/method khác, baselinepaired mới là15,05% trongvòng37.

Correct−trainednoimage11,56đ CI7,16..19,70; correct−trainednoquestion7,45đ
CI5,21..9,11, đều chưa20đ. Fullcorrect dev từngseed29,08%/21,88%/24,35%,
novel30,05%/23,59%/23,66%; xa65%/60%. Không kindnào đáp ứng50% ở mọi seed.

## Ảnh có ảnh hưởng nhưng chưa groundinghữuích

Blankimage giảm19,63đ (CI9,38..29,08), blankquestion12,08đ (CI6,51..15,19),
swappedimage2,18đ (CI−4,69..11,89). Zeroing là input ngoài phânphối học, nên
blankdrop lớn không đủchứngminh hiểu nội dung ảnh. Khiđổi ảnhsangcảnh có
targetkhác, tácđộng nhỏ/chưa chắc chắn; modelvẫn thườngdựa vào shortcut hoặc
phụthuộcinputkhôngđúngmức, chưa đủcausaljointgrounding đãđăngký.

Firstwordresetheadprobe trained−initial mean−9,26đ CI−11,46..−7,29, âm cả3seed;
khônghiddenrepresentationgate. Đây là diagnosticclassifier ở actual48pooled
feature/đúng1536exposurelabels, không readoutpolicy thay thế hoặc accuracycảcâu.
Vẫn cócoadaptation: trảhidden vềinitialgiữhead làmdevgiảm25,10→9,14%. Lesion
này ngoài phânphối không chứngminh featuremới tổngquáthoá tốt; không dùng nó
để phủ định kếtquảprobehoặc gọi synapseđãhọcvisualconcept.

MatchedglobalBPTTcontrol cũng26,30%dev/53,79%train; chưa đủ cơ sở kếtluận
chỉthiếu recurrentbackward gây thấtbại. Các giảthuyết input/representation/
spatialstructure/compression/headinterference/budget cònmở, chưa cócausalsplit
cho từngnguyênnhân. Không suyra ALIF hoặc mọi họclocal bất khảthi.

**STOPv1:** chỉfrozengain qua, các accuracy/novel/eachkind/missingmodality/
causal/probegate fail. Không tuneencoder/cachemixture/LR/epochs/modelsize hoặc
thaygateđể mởlại v1, không mainHermes/P1/P2/normalvision/retentionclaim.

## Integrity, tài nguyên và ảnh thực trên observer

21/21checkpoint đượcfreshprocess nextlearn→imagequestionfreeanswer→sampler→
fullstatehash khớp; sourcefiles/dataset192PNG vàalgorithms hashgiữ nguyên.
Zero/frozen neural/headweights vàpredictions khớp;7methods cùngstream/wordupdates/
NumPytrainingticks theo seed. Formal32256responseexposures,73717wordheadupdates,
23040hiddenupdates,1285599NumPytrainingticks; BPTT4608backwardpasses và183657
graphforwardticks thêm. Timing/coldproof là bảnsaoloạibỏ, khôngformalexamples.

Round330,14s, peakprocess-treeRSS643,47MB/commit1042,43MB, Windowscap2GB/process
vàtreeđượcsample riêng. Candidatearrays485832bytes gồmneuron/head/localtraces/
Adam/cache/pendinginputbuffers; checkpoint643KB cóbảnserialcore xácminhshared
alias, không phải2coreinference. Runtime/checkpoint/rollbackcopy tínhtrongRAM.
KhôngiGPU/weightsSSDoffload/energyclaim hoặcđổiOSpagefile đểgiảRAMthấp.

Localhosttựhiện đủ864DEVoutputs và96ownedimages của completedfile gầnnhất,
cảcâusai, method/noimage/noquestion/BPTTscope. Trọngsốmain/corevocabulary/
topology giữnguyên; `/v1/models` vẫn1scarx-generation1. Qwen4Btext-only không
được giả thànhvisionmodel;64actualQwentextcomparisons36 còntrongcatalog/source.
Không newAGYcallđểlặpdata vàkhông dùngmodeloutput làm nhãn.

## Quyết định tiếp theo

**PIVOT_NO_SCALE**, khôngdeployvision37. Approved30 có tínhiệu hẹp ởword36/XOR35,
nhưng vision37 khôngđủ representation/grounding. Cần giải quyết tínhhữuích của
representation và tổngquáthóa trước tối ưureplay/consolidation/growth. Hướng mới
phải có tínhiệu học/đối chứng rõ trongmaster, đăngkýgiảthuyết riêng và giữmọi
STOP; không tiếp tục cùngfamily bằng đổi tên hoặc nạpLLM/Transformer.
Normalchat/ảnhđờithật/OCR/actions vàlifelongretention vẫn chưahoànthành.
