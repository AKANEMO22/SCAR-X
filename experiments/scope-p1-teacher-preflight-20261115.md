# SCOPE P1: preflight teacher-P credit, khóa trước forward

Status: LOCKED_PREFLIGHT_ONLY. Base df80b1b5. CPU-only; no learning/model fit.
Chỉ khi mọi gate preflight đạt mới đăng ký một protocol P1 riêng rồi thực thi.
Không có learner/P1 invocation trong runner này; không giả đây là training.

## Một giả thuyết và một task

H: terminal output tín hiệu từ competent teacher P có thể giúp hiddenlocal
student vượt label-only/readout-only khi readout ban đầu có cả hai dấu, và
lợi ích không giải thích được bằng thêm supervision hoặc đổi label targets.

Task đề xuất: phân biệt khoảng cách giữa hai pulse trong chuỗi10tick. Pulse
thứ hai cố định ởindex8, thứ nhất ở8−gap. Gap1/2 làshort(label1), gap4/5 làlong
(label0). Amplitude a∈{0.96,1,1.04}; cả hai pulse cùng a. Chưa dùng task này
để train; kiểm mechanism trùng P0 dù task input có thời gian thật.

Reserve3seed 261151,261157,261163 trước forward. Bốn cấu hình kỹ thuật/seed:
teacher128(S64/P64), frozenstudent64 readout+3, frozenstudent64 readout−3,
constructedstudentwitness64. Đây không là3independentqualityseeds: các weights
hữu ích được construction cố định; random sparse topology có Rweights0.
Nếu P1 được mở phải có3seed learner độc lập theo protocol riêng.

## Analytic construction trước kết quả

Giữ SpikingCore ALIFFP32/config leak.85,threshold.65,feature_decay.92,fan8,
resetstate0; no change to source. Chỉ inputchannel0 có pulse, inputweight
teacherPactive=.4, initialstudentactive=.3, witnessstudentactive=.4.
Các hidden khác không góp output; không giảm dtype/prune servingmodel.
TeacherrowS indices chỉS; rowP indices allocatedwithinS/P, Rweights0.

Trước pulse2, u2=w a (1+.85^gap); chưa có spike1. Teacher construction có
spike ởindex8 iffshort; feature cuối F=(1−.92).92. Binarylogits làell0=0,
ell1=3 f_active−1.5F. Initialstudent cả hai dấu có f=0, readout-only chỉ thấy
một featurevector cho hai nhãn cân bằng, ceiling50%. Witnessweight.4 với
readout+3 biểu diễn target; đây là representability, không là học.

Held-out gap2/4 và amplitude .96/1.04 đi qua **cùng nonzero inputweight** vào
membrane; không là distractor zero-weight. Reserved training-like cases chỉ
gap1/5,a1; mọi tổ hợp còn lại là preflight stress/novel-coordinate checks,
không gọi là OOD chất lượng hoặc dùng như một finalTEST đã đo modelhọc.

## Các gate preregistered

G1 input đủ và thứ tự/gap thật tác động u2; zero gap-sensitivity ⇒NO_GO.
G2 teacher decisions đúng12/12constructioncases mỗi seed; labels không đi
vào forward. Không chọn teachercheckpoint sau khi xem scores.
G3 constructed S-only witness đúng12/12 và source/configbounds giữ nguyên.
G4 frozenstudent featurecollision giữa nhãn khác và ceiling50% cho cả hai
readout signs. Không fit readout để tìm trần.
G5 **có khả năng attribution:** teacher-terminal targets không được chỉ là
một classconditional constant có thể tạo từ verifiedlabel với cùng budget.
Nếu p_T(x)=c_y cho toàn miền contract, một label-derived softtarget control
cho cùng δ,signal,eligibility,physical updates từng bước. P1 không phân biệt
teacherP khỏi target smoothing ⇒PREFLIGHT_NO_GO trước training.
G6 không lặp cơ chế P0: chỉ thêm mult tick hoặc seeds không tạo đường credit
mới; binary scalar feedback vẫn có dấu của readout khi error giữ dấu.

G5/G6 dự kiến là rào cản từ source, không phải criteria tạo sau kết quả.
Nếu fail, không mở P1, không đổi sang teacher voltage/hints/intermediate
oracle hoặc sweep weight/task/seed. Báo đúng không xác nhận lợi ích P.

## Resource, evidence và audit

Tổng worker+auditwall≤120s, CPU-only,3seedreserved,4configs/seed=12technical
cells (<24configs), RAMwhole-owned-tree<2e9decimalbytes. Không gọi GPU,
autograd, learn/calibrate/finish hoặc sửa source/neural arrays bằng learning.
Construction weights là initialization tường minh trước forward, không fit.

Runner đo wall/peakworking-set/peakprivatecommit trênWindows, perprocessJob
cap2GB và aggregateconservativebound=sum processpeaks gồm auditor. OSpeak
không là đo whole desktop; Hermes/live không là worker nghiên cứu này.

Save unique run, protocol/source SHA, inputs.json, raw.csv, traces.json,
checks.json, result.json và stdout/stderr. Auditor stdlib scalarALIF riêng,
không import model, không train, so input labels/predictions/features/
classconstant targets/counter/hash. Log exitcode và lỗi giữ nguyên, không
rerun để làm report đẹp. Budget gồm audit; không tính thời gian đọc literature
và viết code là trainingbudget, không báo chúng thành mô hình học120s.

Gate không đổi theo scores. PREFLIGHT_NO_GO khác pilotFAIL khác chưa kết
luận, không phải SNN impossibility. Không scientificCI/20stageretentionclaim.
File handoff/P0 artifacts chưa tìm thấy trong attached/project paths đã kiểm;
P0 percentages do user cung cấp, không được ghi là independently reproduced.
