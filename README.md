# SCAR-X · Thế hệ đầu

Một model, **một quần thểALIF hồi quy có ngân sách neuron linh động**, chung trọng số và trạng thái
cho văn bản và ảnh chữ số. Lúc rảnh cùng lõi tiếp tục xử lý neural; tin nhắn được
ưu tiên. Website hiển thị topology, spike raster và sự kiện thật.

Khởi đầu48, prototype hiện có thể cấp tới128neuron. Thanh Hermes hiển thị
**Nạp X ký tự · N neuron**, popover có active/allocated/max và inputbytes/vector.
Tự chọn theo inputvolume/modality, hoặc chọn budget ởwebsite; đây chưa là bộ đo
độ khó ngữ nghĩa. Neuron mới cần học; cấp thêm không tự tạo kỹ năng. Những
readout đã nối được giữ bảo thủ khi trả lời, chưa có causalpruning.

Checkpoint cũ vẫn đọc được; growth lưu phiên bản và có thểrestore topology cũ.
Chỉ nhóm selected được tính; weights đã allocated vẫn ởRAM. `48` trong số đo
lịch sử phía dưới là cấu hình trial, không là giới hạn vĩnh viễn của dự án.

**Bản thử nghiệm còn yếu:** một số nhóm câu đơn giản đã học; còn sai/lặp trên
cách hỏi mới. Vision chỉ phân loại chữ số8×8. Chưa có hiểu screenshot, OCR,
thao tác máy hay suy luận đa năng. Không LLMpretrained/cloud/BPTT.

## Cài và chạy

```powershell
cd 'C:\Users\hachimi\Downloads\model agent\scar-x'
python -m pip install -e '.[vision]'
python run_app.py
```

Mở **[http://127.0.0.1:8765](http://127.0.0.1:8765)**. Python3.11+, NumPy,
psutil,Pillow; trọng số tự huấn luyện kèm gói. `Ctrl+C` lưu và dừng.
`--port 8766 --data-dir local/my-session` tạo phiên riêng; `--no-idle` tạm dừng
nền. Mỗi data-dir có một process ghi. `Stop-SCARX.ps1` xác minh process checkout
trước khi force-stop; lần commit cuối được giữ. Không cần bật Qwen9B/llama.cpp.

Data mặc định `local/generation1/generation1.sqlite`. Nền lưu state mỗi30s;
chat/dạy/checkpoint/dừng bình thường lưu ngay. Crash có thể mất tối đa30s hoạt
động nền. Dữ liệu cũ `local/mvp` giữ lịch sử, không nạp hai pilot không tương
thích vào neuron mới. Thư mục dữ liệu trống được nạp từ
`artifacts/generation1-release.json`, không huấn luyện lại từ số ngẫu nhiên.

`Start-SCARX.ps1` từ chối mở lần thứ hai nếu cổng đã có process.listen.
`Stop-SCARX.ps1` chỉ dừng process của đúng checkout này. Kiểm tra một phiên tạm,
không đụng `local/generation1`:

```powershell
python scripts\smoke_release.py
```

Smoke kiểm status, câu neural chưa kiểm chứng, tick còn nguyên sau restart,
một lượt dạy được guard chấp nhận, và một lượt dạy khác bị guard từ chối. Lần
từ chối phải giữ nguyên hash của `model.zip` cùng controller, RNG ứng dụng,
dialogue và chính sách nền. Con trỏ `sequence` và log sự kiện được phép đổi.
Hai kết quả không được thay thế cho nhau. Vision hiện là chữ số, không phải ảnh
tổng quát. Chat chưa qua cửa chất lượng.

Bộ test của bản chạy, không gồm control BPTT:

```powershell
python -m pip install -e '.[vision,test]'
python -m pytest -q -rs
```

Extra `control` cài `torch` cho các bài BPTT cô lập. Không có torch thì những bài
đó bị bỏ qua. Không cài torch vào process đang phục vụ trang localhost.

## Dùng và quan sát

Mở **[Công việc đang thực hiện](http://127.0.0.1:8765/#agent-work)** để xem nhật
ký cập nhật/kiểm tra có bằng chứng, tự lên mỗi1,5giây. Nhật ký này tách khỏi
hoạt động neuron, không cần chọn hoặc bấm chạy.

**[Câu trả lời kiểm thử](http://127.0.0.1:8765/#test-answers)** và
**[Quá trình học](http://127.0.0.1:8765/#learning-observer)** để tự theo dõi
những gì Codex đang thực hiện: input, câu sinh thực, target/Qwen khi có nguồn,
loss, cập nhật synapse, ΔW và guard/rollback. Không cần chọn/bấm chạy.
**[Neuron theo task](http://127.0.0.1:8765/#capacity-lab)** tự hiện lượt đo tăng/
giảm tải gần nhất. **[Đang thử nghiệm](http://127.0.0.1:8765/#research-lab)** hiện
job, tiến độ, RAM, mẫu neural và kết quả nguồn, kể cả hướng chưa đạt.
[Hướng dẫn đầy đủ](docs/OBSERVING.md).

[Hội thoại/state/curriculum mới](docs/research/23_REAL_DIALOGUE_STATE_AND_CURRICULUM.md):
Antigravity và Qwen đã chạy thật; các candidate ôn lại/expandeddata chưa qua
acceptance, nên chưa thay vào live. Giữ rõ learning từ fixture dự án, lời dạy
người dùng và output chưa kiểm chứng; normalchat/vision vẫn chưa đạt.

[Word-score vòng cuối](docs/research/26_FIXED_INPUT_NEURAL_TRANSFER.md):27lượt
học thật không chứng minh lợi thế hidden; hướng application đãSTOP sau2vòng.
Mainmodel không đổi sang Transformer hoặc nhận candidate chưa đạt.152tests và
27coldresumes qua là bằng chứng kỹ thuật, không là chứng nhận chat/vision.

[Đối chứng learnability27](docs/research/27_ISOLATED_WORD_LEARNABILITY.md): cùng
prototypeALIF48, globalBPTTcontrol đạt44,09%dev8styles so localscore7,53%; chưa
là mainlocallearning và không nhập weights vàoHermes. Đã bổ sung47Qwenreference,
có đủ95prompt đối chiếu (41bịcắt ở64tokens). Xem
[toàn bộ đầu ra đã lưu](http://127.0.0.1:8765/#answer-archive), tự cập nhật cả câu
sai và nhãnđốichứng. Không gọi train16ceiling hoặc controlwin là goalhoànthành.

[Audit nhiễu tín hiệu28](docs/research/28_CREDIT_SIGNAL_NOISE.md) đo trên cùng
checkpoint/input: scalar-score có phương sai lớn và hướng ít tái lập hơn
surrogatecontrol. Không huấn luyện thêm; variancecontrol không là rulemới hoặc
chất lượngđãcải thiện. Giữ các STOP/gate, chưa normalchat/vision/retention.

[Observedassociation29](docs/research/29_OBSERVED_RESPONSE_ASSOCIATION.md) đã
train thật12lượt với các pha quan sát/prediction táchbiệt; pairedkhông hơn
frozen, v1STOP. Không tự nhập candidate vàoHermes. [Đề xuất tínhiệu30](docs/research/30_PROPOSED_LEARNING_SIGNAL_BOUNDARY.md)
đãđược người dùng duyệt NGHIÊN CỨU ngày10/11/2026; [protocol35](docs/research/35_DIRECT_READOUT_ELIGIBILITY.md)
mở prototypeALIF/forwardeligibility + directreadoutsignal. ChưadeployHermes,
khôngTransformer/BPTTmain. Các mốc pending trong báo cáo cũ là lịch sử.
V1đã chạy45trial: XOR91,80%direct/probegain10,94đ, nhưng chỉ1/3taskqua;
STOPv1, khôngconfirmation/scale/deploy.184tests/cold45exact không chứng minh
normalchat/vision. Localhost ghi đúng nghiên cứu tensor số, không lời đáp chat.

[Visionquestionaudit31](docs/research/31_VISION_QUESTION_PATH.md):240lần gọi thật
cho thấy question chưa vào neuralvisioncontext, chỉ trảdigitJSON.32câuAGY đã
review/12ảnhsynthetic để kiểm đường vào, chưatraininghoặcnormalvision. Trace
hiệnRGB gốc và64pixelxám cạnhoutputthật; không giảQwen4Btext-only làvision.

[Jointvisualword32](docs/research/32_JOINT_VISUAL_WORD_READOUT.md) đã thử nối
question+RGB vào cùngALIF48 và họcneuralwordreadout. Macrodev18,88%, không hơn
đối chứng đãtrainmissingmodality đủngưỡng; STOPv1, chưa thayHermes/mainweights.
Sinh từ thật và cóinputpath không đủ để gọi normalvision hoặchiddenlearning.

[Đối chứng visual33](docs/research/33_VISUAL_WORD_LEARNABILITY_CONTROL.md) giữ
cùng48ALIF/input/head; BPTT chỉ trongcontrolcôlập đểchẩnđoánlearnability,
không đổiHermes/localrule hoặc mởproposal30. Preflight3×64trainresponses khớp
fullstate; timing/graphparity qua. Formalmeantrain48,43%/dev25,67%, chưaqua
qualitygate; controlSTOP, khôngimportmain hoặcnormalvision. Trace tựhiện864dev
cóảnh/câu sai/nhãnCONTROL và actuallearning. Khôngthay gốcALIF/locallearning.

[Audit đường thông tin34](docs/research/34_VISUAL_INFORMATION_PATH.md) kiểm
checkpoint32/33, không cập nhật synapse. Probeclassifier ngoàicore giúp phân
biệt mất thông tin và headtraining; không thay câu neural hoặc mở lại tuning.
Kết quả lênlocalhost với nhãnPROBE, tách khỏi câu trả lời đầyđủ của lượt nguồn.
Đãđo: rawsemanticprobe50,22%dev so neuralfeature8–12%, chưarepresentationgain
đủchắc; khôngexactfullpathcollapse/clipbottleneck đãchứngminh. Khôngsửa
normalization hoặc dùngclassifierdiagnostic làm câu trả lời. PIVOT giữnguyên.

[Wordbridge36](docs/research/36_DIRECT_WORD_BRIDGE.md) là pilot48ALIF sinh từ
với unchangedeligibility35 đượcduyệt, actualAGY128reviewed/64train64dev. Không
LLMproxy/answerrouter/mainimport/P1waiver, chưanormalchat. Qwen4B sẽđođộc lập
cùng64DEVprompts:64/64đãđo,14lengthcut, serverđãtắt; rawreferencekhôngthànhnhãn.
Pilotcorrect35,94%dev vsfrozen16,67%, nhưngtrain99,48%/devthấpnênSTOPv1, chưa
normalchat/P1/P2. Trace tựtheodõi64actualoutputs/Qwen, cảcâusai/loss/ΔW.

[Directvisual37](docs/research/37_DIRECT_VISUAL_BRIDGE.md) chuyểnapproved30local
eligibility lênsame32ALIF/RGB/cache, cómath/parity/midimagecheckpoint trước đo.
21trials nhỏ/7conditions, khôngmodelscale/mainimport hoặcnormalvisionclaim.
Trace tựhiện actualRGB/question/wordoutputs/learning, khônggiảQwen4Btext-only làvision.
Đãđo21trial:correct25,10%dev vsfrozen15,05% nhưngprobeâm/imageswapchưađủ,
STOPv1; chưaảnhđờithật/OCR/generalvision.197tests/cold21exact không thay gate.

[Audit cáchdạy38](docs/research/38_WORD_TRAINING_OBJECTIVE_AUDIT.md) kiểm
FREE/GOLDPREFIX và tínhđúngđắn của bài8style trêncheckpoint36, khôngtraining
mới. Goldprefix/tokenaccuracy không làfreeanswer hoặc hiểu ngônngữ; localhost
ghi rõchẩnđoán/nguồn36/Qwen đãđo, khôngtrộn thành modelđangdùng.

[Reference data43](docs/research/43_ADAPTIVE_DATA_GROUNDING.md) hoàn thiện đáp án
chat có nguồn dự án và QA ảnh có pixeloracle, kèm audit giới hạn đầu vào/vốn từ.
Đây là dữ liệu development riêng, không trả lời thay neural core hoặc mở pilot.

[P0 usefulness42](docs/research/42_READOUT_USEFULNESS_P0.md) đã kiểm math/parity/
checkpoint/restart trong prototype ALIF riêng;276tests qua. Không có bằng chứng
cải thiện chat/vision/retention từ P0, không đưa prototype vào Hermes. Pilot P2
chưa được duyệt; matched-norm phải từ chối các state float32 không khớp budget.

[Chuẩn bị41](docs/research/41_ADAPTIVE_CURRICULUM_AND_P2_PROPOSAL.md): AGY128wording
đã review,127giữ;24chat đối chiếu thật SCAR-X copy/Qwen,10Qwenoutput bị cắt.
Đây là development/reference, không training hoặc cải thiện chất lượng. P2
usefulness v2 đang chờ duyệt; chưa sửa model đang dùng hoặc mở lại familySTOP.

[Review DeepSeek40](docs/research/40_DEEPSEEK_REVIEW.md) giữ các sửa guard/state có ích,
sửa lỗi intervention/metric/SQLite/trace và đo lại trên bản sao. Readout có thể
ghi đè một cặp đã biết; chưa chứng minh hidden representation hoặc lifelong
learning. Không đổi ALIF/learning rule/weights phiên thật, không chạy draft
Readout Protection chưa duyệt. Kết quả thật tự hiện trên localhost.

P2 hiện tại: [retention và cue/state diagnostic](docs/research/21_P2_RETENTION_AND_CUE_INFORMATION.md)
chưa đạt acquisition/forgetting. Replay chưa đủ; đang ưu tiên representation,
working-state và readout stability, chưa triển khai importance hoặc model mới.

P1 mới nhất: [paired-score / 8 seed xác nhận](docs/research/20_SCORE_ELIGIBILITY_AND_VARIANCE.md)
đã qua gate representation trên order/XOR bằng local eligibility × scalar reward,
không backward; binding còn thất bại. Chỉ mở nghiên cứu P2, chưa đổi model đang
dùng hoặc chứng minh normalchat/vision/retention.

[Local-classifier](docs/research/18_LOCAL_CLASSIFIER_FINAL_AND_LEARNABILITY.md)
vẫn dừng. BPTT chỉ là đối chứng cô lập, không là cơ chế học của sản phẩm.

[Audit optimizer/coverage](docs/research/19_CONTROL_OPTIMIZER_AND_COVERAGE.md)
phân lập thêm các yếu tố của đối chứng; không mở lại family hoặc đổi maincore.

- Nhắn `hi`, `chào`, `chào bạn`, `bạn là ai`. Text sinh từ neural readout,
  không tra bảng đáp án. Những prompt khác vẫn có thể sai hoặc lặp.
- Để rảnh và xem **Dòng sự kiện thực**: recurrent dynamics, probe đầu vào,
  spike/voltage và text thử **chưa kiểm chứng**. Không dùng text thử làm nhãn.
  Tạm dừng/đổi nhịp1–30lần mỗi giây; mỗi nhịp có thể gồm nhiều tick neuron.
- Topology khởi đầu384synapse recurrent, tăng theo số neuron đã cấp. Kernel
  chỉ tính nhóm đang chọn và giữ nguyên weights của nhóm ngủ;
  ít spikes chưa chứng minh tiết kiệm compute. Prototype mới tính theo active
  rows/readoutcolumns, không tạo modelthinking riêng.
- **Dạy từ lời sửa** cập nhật synapse cục bộ, có ΔW và checkpoint. Vocabulary
  được mở theo explicitlesson trong trần256targettokens, không tự học từ output. Lời dạy chưa được xác minh
  độc lập, không tự coi nội dung là factualtruth.
- Ảnh chữ sốPNG/JPEG≤500KB,≤1Mpixels. Kiểm hình dạng là heuristic chưa xác nhận
  OOD; kết quả là server định dạng phân loại neural, không mô tả ảnh tự do.

**Học xuyên hội thoại:** lõi giữ trạng thái, controller có thể hỏi khi chưa rõ;
lời sửa/đáp án rõ ràng tạo lesson. Mỗi lượt dùng vài cập nhật cục bộ có guard;
lúc rảnh xem lại trải nghiệm và ôn nhãn đã được chấp nhận. Có thể tạm dừng học
trên website. Không tự lấy output làm nhãn đúng. Human feedback là nguồn nhãn,
không tự thành chân lý; phép tính đơn giản được kiểm lại bằng parser riêng.

Trace hiển thị hoạt động thật, không dịch neuron thành độc thoại tiếng Việt.
Guard rollback nếu mất probe cũ nhưng có thể chặn cả bài học mới. Chưa chứng minh
reasoning, retention dài hạn hoặc representation hữu ích. Xem
[kết quả luồng học](artifacts/continuous-dialogue-acceptance.json) và
[dữ liệu Antigravity](docs/research/10_DIALOGUE_DATA_AND_ONLINE_LEARNING.md).

Đã thử dừng tại cuối câu để giảm lặp, nhưng điểm development trong lượt thử
giảm từ37,5% xuống25%; không bật quy tắc đó trong sản phẩm. Giữ decoderEOS
và công khai lỗi lặp. Không chấm lại test hoặc sửa câu trả lời bằng đáp án mẫu.

## Lưu và khôi phục

Một checkpoint chứa sharedweights/stableweights,voltage,adaptation,spikes,
features,eligibility,learningcounter/rewardbaseline,topology,vocabulary,RNG,
bộ đệm128input,probeđangdở,cursor và schedulerpolicy. G1 chưa có importance/
metaplasticity mới; `plasticity-v1` vẫn là nhánh nghiên cứu lịch sử riêng.

Website lưu/khôi phục và tải bundle. Restore tạo version mới, giữ lịch sử;
transaction học lỗi rollback fullstate, checksum hỏng bị từ chối.

```powershell
python run_app.py --import-checkpoint 'C:\path\SCAR-X-generation1-session.zip'
```

Không lặp flagimport các lần sau. Formatv0.1/v0.2 bị từ chối, không âm thầm
chuyển weights. Backup toàn bộ history bằng cách dừng rồi copy data-dir.
Giới hạn2000checkpoint yêu cầu sao lưu/quản lý trước khi học tiếp.

## Hermes: một lựa chọn SCAR-X

```text
Provider: scarx-local
Base URL: http://127.0.0.1:8765/v1
Transport: chat_completions
Model: scarx-generation1
API key: để trống; nếu client bắt buộc dùng no-key-required
```

Chạy `python scripts/install_hermes_stateful.py` để cài adapter Hermes có trạng
thái, sau đó `python scripts/install_generation1_provider.py`. Khởi động lại
backend/Hermes và mở phiên mới để nạp Python code mới.
Thêm tab riêng bằng `python scripts/install_hermes_learning_tab.py`: câu hỏi
SCAR-X và lời dạy hai chiều ở **Hỏi để học**. Correspondence log giữ trong
SQLite, không là neural knowledge; neural-export không chứa toàn bộ log512tin. Adapter giữ nguyên core
SCAR-X, bỏ yêu cầu64K chỉ cho provider opt-in không dùng tools; không tăng con
số khai báo để giả năng lực. Xem [chi tiết và backup](integrations/hermes/README.md).

`python scripts/install_generation1_provider.py` thay các modelSCAR-X cũ bằng
mộtID và chọn G1. Giữ providerkhác, backupprivate ở Hermes/backups; không đóng
gói secrets. Llama.cppautostart vẫn tắt. ReloadHermes/mở phiên mới nếu menu còn
cache. Lệnh: `/model custom:scarx-local:scarx-generation1`.

`/v1/models` chỉ có mộtID. ChatCompletions hỗ trợ nonstream/SSE và một ảnhdataURI.
Latestusertext là input; system/history chưa được học xử lý. SSE đóng khung sau
inference; không giả reasoning_content. Visiontransport chỉ cho ảnh chữ số;
reasoning/toolcalling false. Trace xem trên website. RuntimePython/NumPy, khôngGGUF.

**Context không còn là mục tiêu chất lượng.** 4096 là bộ đệm host; input hiện
tối đa2048ký tự/tin. Trí nhớ neural phải đo theo độ trễ, interference, giữ kỹ
năng, học mới và transfer, tách khỏi truy xuất SSD. [Bảng đo mới](docs/research/09_MEMORY_LEARNING_MEASURES.md)
hiện là protocol đề xuất, các giá trị capacity/horizon chưa được đo.

## Tài nguyên, kiểm thử và kết quả

Mục tiêu2.000.000.000byte toàn tiến trình, gồm học/save/rollback. WindowsJob áp
processcommitcap khi hỗ trợ; website báo enforcement và RSS riêng. Browser/OS
không nằm trong RAMprocess. Trace256record, inputbuffer128; không log nền vô hạn
hoặc offloadweightsSSD để giảRAMthấp. Lúc rảnh có sử dụng CPU/RAM thực.

[Jointtraining2seed](runs/20261109T000625Z-generation1-joint-bd6c92/report.md):
greetingacceptance50%/50% trên6prompt (**fail mục tiêu80%**); toàn bộ18text
33,33%/22,22%; digits88,33%/89,44% diagnostic. Release chọn seed17 bằng dev.
[Audit tie-break](runs/20261109T000625Z-generation1-joint-bd6c92/PROTOCOL_DEVIATION.md)
ghi sai lệch triển khai so với protocol; không đổi tiêu chí sau kết quả.
Researchforgetting/representationgate vẫn fail: **PIVOT_NO_SCALE**.

```powershell
python -m pip install -e '.[test,hermes-test]'
python -m pytest -q
python scripts/verify_generation1.py
python scripts/package_generation1.py
```

Gói hiện tại `artifacts/SCAR-X-Generation-1.zip`; ZIPcũ lưu ở `artifacts/retired/`.
Legacycode/tests/runreport giữ để tái lập, không chạy legacyserver làm sản phẩm.

[Masterplan](docs/RESEARCH_PLAN.md) · [Trạng thái](docs/STATUS.md) ·
[Bản đồ](docs/PROJECT_MAP.md) · [ADR0006](docs/decisions/0006-one-core-generation1.md)
