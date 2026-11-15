# SCOPE P1: dừng trước training vì tín hiệu teacher chưa phân biệt được

**PREFLIGHT_NO_GO. P1 không đăng ký/chạy; 0 learning updates.**
Quyết định: **STOP thiết kế terminal-logit-only này; PIVOT giả thuyết tín hiệu
teacher trước khi mở một P1 khác.** Không kết luận mọi teacher/SNN vô ích.

## 1. Phạm vi được duyệt và hồ sơ đã đọc

User duyệt riêng CPU-only preflight không học; nếu qua mới đăng ký rồi chạy
một P1,3seed,≤24configs,≤120s tổng, RAM<2GB. Yêu cầu chạy song song sau đó
không đổi prerequisite, số giả thuyết hoặc budget tổng. Không push/merge/
deploy, main/Hermes/checkpoint live hoặc reset P serving. Source base
df80b1b5; [protocolpreflight](../../experiments/scope-p1-teacher-preflight-20261115.md)
được commit bf648591 **trước forward**. Không có P1 protocol được mở.

Đọc master, toàn candidatev2/capacity-release/prior-art, core.py, learner.py
và readout_eligibility.py. Handoff `CODEX_SCARX_HANDOFF_P1_20261115.md` không
được đính kèm và chưa tìm thấy trong project/attachments paths đã kiểm.
Artifact của microcircuit P0 cũng chưa tìm thấy; tỷ lệ75/100% là **user-reported**,
không được ghi là Codex đã tái lập. Duplicate check dùng reported mechanism
và source, không diễn giải các tỷ lệ đó như dữ liệu mới xác nhận.

[Prior-art matrix](SCOPE_ALIF_CAPACITY_PRIOR_ART_20261115.md) và các kết quả âm
đã có được reuse. P&C đã có output distillation; source local-feedback vẫn
dùng W_out.T×error. Không lặp OR/XOR/MOVE/R13.2/forgetting/EMA/HLOP hoặc train
P0 với thêm seed. New interval inputs chỉ là **preflight thiết kế**, không
một campaign đo chất lượng mới. No oldTEST prediction access/selection.

Nguồn chính kiểm lại: [P&C](https://arxiv.org/html/1805.06370v2),
[e-prop](https://www.nature.com/articles/s41467-020-17236-y) và
[KD for SNN](https://arxiv.org/abs/2511.06902v2). Nature/bioRxiv/AAAI fullpage
bị cookie/403; reuse source/equations local đã pin, KD mới chỉ abstract đọc
được. Không claim fullpaper/codeaudit hoặc nhập KD methods đó vào core.

## 2. Một task temporal và construction đã khóa

Chuỗi10tick, input64chiều: hai pulse trên channel0. Pulse2 luôn ởindex8,
pulse1 ở8−g. Gap g∈{1,2,4,5}; labelshort1 cho1/2 vàlong0 cho4/5.
Amplitude a∈{0.96,1,1.04}. Domain phân tích có thể xét a liên tục trong cùng
khoảng; không đưa g/label vào channel phụ hoặc suy nhãn từ episode length.

Seed kỹ thuật261151/261157/261163,4constructionroles/seed: teacherS64+P64,
initialstudentS64/readout+3, initialstudentS64/readout−3, studentwitnessS64.
Các hữu ích weights là construction được khai báo, chưa learned/random-search.
Sparse topology seeded nhưng Rweights0 nên seeds không là3qualityreplications.
Không thay tổngneuron phục vụ hoặc migrate checkpoint live.

ALIF thật leak.85,threshold.65,subtractive reset/adaptation; no source edit.
Active inputweight teacherP=.4, initialstudent=.3, studentwitness=.4.
Readout coefficient±3 trong bounds; bias−1.5(1−.92).92. Normalization được
xem là identity cố định, không dùng normalizer tự học để đổi kết quả.

Đầu vào tác động lên neuron qua **weight khác0**. Trước pulse2, khi chưa có
spike, u2=w a(1+.85^g). Vì vậy delay/amplitude thực sự tác động membrane;
không là held-out distractor có weight0. Reserved train-like coordinates
gap1/5,a1, còn novelcoordinates có gap2/4 hoặc amplitude khác. **Không train
nên không có phép đo OOD generalization của một model đã học.**12inputs ở
đây đã được xem như technicalfixtures, không là future freshTEST.

## 3. Preflight thực sự đã chạy

[Run duy nhất](../../runs/20261115T165305Z-scope-teacher-preflight-v1/result.json).
144forwards/1.440ticks, không gọi learn/fit/calibrate/backward. Các seed dùng
lại12uniquecases;36teacherrecords không là36independenttestquestions.

| Gate khóa trước forward | Kết quả | Ý nghĩa đúng phạm vi |
|---|---|---|
| G1 input đủ, temporalvariable có tác động | PASS; voltage span tối thiểu≈0,117013 trên các initialstudent so gap | Không zero-weight nuisance |
| G2 teacher competent |36/36predictionrecords đúng (12uniquecases) | Teacher hand-constructed, chưa bằng chứng học teacher |
| G3 student biểu diễn được lời giải | Constructedwitness36/36 đúng | Representability, không là local acquisition |
| G4 readout-only có giới hạn | Cả hai dấu initialstudent cùng f=0; balancedceiling50% | Không fit readout hoặc gọi50% là kết quả training |
| G5 lợi ích teacher phân biệt được với labeltargets | FAIL | Targettable hai giá trị theo nhãn, không cần P để sinh cùng targets |
| G6 creditmechanism khác P0-sign đã báo cáo | FAIL theo source/proof | Chỉ thay terminaltarget, không thêm một đường credit mới |

G5 đủ để dừng; G6 là đánh giá source/logic, không là một neural measurement
hay tái lập P0. Gates và constructions không đổi sau results. P1 không mở,
không sweep topology/teacherweight/sign/gap để cứu thiết kế.

## 4. Vì sao causal attribution không đạt?

Trên domain construction, teacher chỉ có một spike ởindex8 nếu label1;
label0 không spike. Các teacherP spike khác đều0, và fP cuối là
(1−.92).92×y. Terminal probabilities **không chứa variation ngoài class**:

| Verified y | p_T(class1), FP32 đo được |
|---|---:|
|0|0.47242799401283264|
|1|0.527571976184845|

Từ source công thức và khoảng input, spikeidentity đúng trên miền đã định,
không chỉ là correlation vừa quan sát. Auditor kiểm thêm saved tapes.

Gọi c_y là target vector của bảng đó. Với cùng student/RNG/input/update rule:

    delta_teacher = c_y − p_student = delta_label_table
    m_teacher = Wout.T @ delta_teacher = m_label_table
    D_teacher = m_teacher * E = D_label_table.

Theo quy nạp qua optimizer state, trace, physical clipping và weight bounds,
hai arms nhận cùng targettable cho cùng trajectory. Một arm gọi teacherP
thật có thể được thay bằng label-derived softtarget control mà không đổi
cập nhật. Không chạy training để xác nhận một đẳng thức có điều kiện đó.

So chỉ với one-hotlabels hoặc teacherablation làm đổi targetdistribution có
thể tạo khác biệt optimization, nhưng không tách **P-specific temporalcredit**
khỏi label smoothing/supervision transform. Đây là lý do không mở pilot thiếu
đối chứng mạnh rồi gọi mọi gain là teacherP tạo representation.

Với binary student và một usefulfeature, hiddenfeedback còn có dạng
m=a_readout(r1−p1)×g. Trong initialstate của construction, class1 có error
dương cả khi dùng teacher target và dùng label; dấu feedback vẫn phụ thuộc
readout ban đầu. Softtarget không tự cung cấp một tín hiệu định hướng khác.
Chưa xác định teacher hints/feedback khác vừa không chứa futureoracle vừa
cho phép attribution công bằng; không tự đổi giả thuyết ngay sau fail này.

Đây không là định lý mọi logitdistillation vô ích. Teacher trên task khác
có thể truyền thông tin ngoài class hoặc calibration có ích. Cần một protocol
khác kiểm matched-information/budget; hiện không có quyền tự mở lượt mới.

## 5. Audit không train và tài nguyên

[Auditor](../../scripts/research/audit_scope_teacher_preflight.py) dùng stdlib,
literal scalarALIF và probability calculation riêng, không import model/
NumPy/learner. Kiểm inputtimes→labels, rawCSV/tapes, nguồn/hash, featurecollision,
targettable, zero-learning và tái dựng verdict. Cùng tác giả Codex, không
giả là reviewer độc lập hoặc empiricalreplication của agent khác.

- [audit.json](../../runs/20261115T165305Z-scope-teacher-preflight-v1/audit.json):
  PASS tính nhất quán dữ liệu; max scalar-forwarderror≈8,06e−8. Đây không là
  qualityPASS hoặc universalFP32certificate.
- Worker+auditor wall≈0,44446s, trong120s tổng execution đã duyệt. Thời gian
  đọc literature/viết code không được báo là thời gian model huấn luyện.
  Đây là timer trong runner (bắt đầu trước NumPy/model import, sau một số
  stdlib imports); toolcall launch/copylogs wall≈0,815s cũng dưới120s. Không
  dùng0,444s như một số đo Python coldstartup hoàn chỉnh.
- WorkerpeakRSS42.262.528bytes; auditorpeakRSS23.801.856bytes. Tổng các
  processpeaks là conservative upperbound66.064.384bytes, dưới2e9.
- Peakcommitsum45.203.456bytes; Windows per-processJobcap2e9 enforced.
  Không giả đó là một whole-tree hardcap: compliance toàn hai Pythonworkers
  dựa conservativepeakbound. Tool shell/desktop/Hermes không nằm trong worker
  measurement; không nói toàn máy chỉ dùng66MB.
- CPU-only,0GPUimports/calls;12technicalconfig-seedcells,0trainingconfigs,
 0learningupdates. Không cần chạy nhiều worker để tăng tốc dưới0,5s; nếu qua
preflight mới cân nhắc parallelseed/arms dưới cùng tổngcap/budget.

Code/protocol/source archive, inputs.json, raw.csv, traces.json, stdout/stderr,
process exit và audit giữ trong run. Sourceprotocol commitbf648591 trước
forward; không sửa rawresults. Không save/load một student learned checkpoint
vì không có student đã học. Restart/oldskill/20stage/CI đều NOT_MEASURED.

## 6. Trả lời trực tiếp

1. **Teacher P có lợi ích nhân quả không?** Chưa xác nhận. Chosen terminal
   signal không nhận diện được lợi ích riêng so matched labeltarget control.
2. **Hidden ALIF-local hơn readout-only không?** Chưa đo learning. ROceiling
   và constructedwitness chỉ cho thấy task có thể cần representation khác;
   không chứng minh local rule đã học representation đó.
3. **Student giữ skill khi bỏP/restart không?** Chưa kiểm một learnedstudent;
   witness tự dựng không là skill được chuyển. Không claim release thành công.
4. **Continual learning đã chứng minh chưa?** Chưa. Không20stage hoặc CI95%gate.
5. **Tiếp theo?** STOP bản P1logit-only này; PIVOT sang định nghĩa teacher
   signal có variation/đường tác động khác labeltable, với controls cùng thông
   tin/compute và không futuretarget. Không tăngseed/neuron hoặc đổithreshold.

Nếu không xây được phép thử phân biệt, tiếp tục dừng trước training. User's
parallel request không là lý do chạy đồng thời các studies không trả lời được
câu hỏi. Main/Hermes/livecheckpoint và Pserving giữ nguyên; không push/merge.
