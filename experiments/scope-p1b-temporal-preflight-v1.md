# SCOPE P1b: profiling và preflight temporal features, khóa trước forward

Ngày16/11/2026. Chỉ authorizes profiling/preflight, **0 learning updates**.
Audit P1data-only đã xác nhận NO-GO,0newforwards. Full directive private copy
`local/p1b-20261116/FULL_DIRECTIVE_20261116.md`, hash2e0fa1fd…849c6e5.
Không rerun P1 terminal-only. New branch research/scope-alif-p1b-20261116,
base6e08a361; no main/Hermes/live reset, oldTEST, push/merge/deploy.

## Một hypothesis mới, prior art và chống trùng

Sample-specific temporal filtered-spike P features can improve useful hidden
learning beyond identical final labels and class-conditioned matched traces,
including adverse initial readout signs. Feature KD/localeligibility có prior
art; không novelty claim. Khác P1: target không là terminalc_y; B tín hiệu
thay đổi qua thời gian, C tráo trong cùng label/read-clock. Không thêm oracle
intermediate target. Đây không là XOR/OR/MOVE/R13.2/EMA/HLOP/P0seed rerun.

Task directed temporal coincidence: hai events A/B trên32ticks, input64,
events có energychannel0 và signedkindchannel1. A=(a,+a), B=(a,−a).
Label1 iff A trướcB vàgap≤2; BA hoặcgap≥4 là0. Times/order đi vào mạng,
không labelchannel; endpointclock không đổi theo label.

Teacher hand-engineered, không claim P tự học: S64/P64,Htotal128,fan8FP32.
P56active:28detector weights(energy,kind)=(.358,−.158),14Arelays(.45,.45),
14Brelays(.45,−.45). R=0 vẫn giữALIF membrane/adaptation recurrence thật.
S8oldneurons inputchannel2=.9, oldhead frozen; remainingS56 là student rows
theo cố định identity mapping P64..119→S8..63. Đây là correspondence cho sẵn,
không learnedaligner hoặc bằng chứng general module composition.

Teacher first-event currentsA=.2a,B=.516a. ABgap2 u2=.6605a; BAgap1 u2=.6386a,
ABgap≥4 thấp hơnthreshold. Với a∈[.990,1.010], margins giữdetector theo label.
Không một intermediate oracle nào đi vào teacher; teacherforward chỉinput.
Studentwitness dùng cùng P rows trongS56, cùnghead/bounds; old8rows giữ nguyên.
Teacher/newhead weight3/28 trên28detectors, q=f/.1, biasclass1−.32; class0=0.

Studentinitial Win energy .118..122, kind±.002 theo seed, allsubthreshold.
Readout±3/28 trêndetectorrows; sourceoldrows/head bất biến. S hiddenR_SS
frozen0; trainmask Snew56×inputcolumns(0,1,3,4), cycle1 chỉ0/1 cóinput.
Fixed scale.1/clip±5, không fitnormalizer. No changes to neuron/config/dtype.

## Data/seeds chuẩn bị nhưng chưa P1registration

Seeds261161/261167/261173, paired±sign. QUAL48cases/seed theo source generator:
2orders×4gaps×2clocks×3rep=48; amplitudes.991..1.009, clocks18/22,
gaps1/2/4/6. QUALkhác TRAIN; **source exact** TRAIN:
2×4×2×12=192episodes, a.995..1.005, clocks16/20,gaps1/2/4/5.
DEV2×4×2×8=128, clocks18/22,gaps1/2/5/6,a.993..1.007.
UntouchedTEST2×4×3×8=192, clocks17/21/23,gaps1/2/6/7,
a∈[.990,.992] hoặc[1.008,1.010]. Không forward/selection trênTESTpreflight.
Oldvalid12cases: channel2 amplitudes.18/.22/.98/1.02, clocks16/20/24.
Construction parameters khóa bằng miền/đại số, không lựa theo QUALscores.

## Local mechanism dự kiến, chưa gọi learn

Own-neuron e_v/e_a/e_f lấy từ readout_eligibility.py, featuredecay.92, input
traces chỉ4columns được phép, không BPTT/RTRL/cross-neuron temporal Jacobian.
Teacherhint = q_P(t) từactualALIFprefix, không lookahead/targetlabel củaB.
e_f bank_hint = mean_t ((hint_t−q_S,t)/.1)×1[abs(rawS)<5]×e_f,t,
chỉS8..63. Endlabelsignal = Wout.T(onehot(y)−pS)×clipderivative/.1.
Direction = m_label×e_f,T + bank_hint choB/C; A chỉlabel, D hiddenfrozen.
Momentum.9/energy.99, .008 step, clip.01, weightbound3; headlocaldelta.04,
clip.025, bound3, unchanged sourcephysics. Evaluate **physical proposed delta**
after all transforms without writing it in preflight.

Matched C is permutation in each(label,readclock) bucket, sorted(order,gap,id)
rotated by half bucket. Same trace multiset/cost and labels, no identitypairing.
Teacheroutputs/cache computed once fromTRAINinputs; A/D may discard but matching
profile performs same tracebank arithmetic. Deployment costs without optional
teacher forA/D must be separately reported. Teachermanualprior not free skill.

## Preflight gates, không đổi sau khi xem kết quả

1. Teacher and S-only witness accuracy≥.95 trênQUAL vàoldvalid; P removal
   từteacherreadout phải giảm≥.05 absolute. Handconstruction không training.
2. Initial student all finalfeatures identical acrosslabel, compute ROceiling
   independently. Không fit head để provefeasibility.
3. Teacher P temporal traces có distinctvariation trong(label,clock) bucket,
   tracevariance>1e−6; shuffledpermutation bảo toànmultiset, donor khácsample.
4. On fixedstudent initialstate for bothsigns/all3seeds, B vsC proposed
   **physical update** median relative L2difference>1%; absolute normdiff>1e−6
   for at least onecase. Không dùng chỉvariance signal hoặc gradientnorm.
5. Inputcollision/leakage/split checks pass; actualnonzero weights carry kind,
   gap/amplitude throughdynamics; no TESTforward. Studentwitnesssamecapacity,
   maskonlyWin/head differ; không lén thayrecurrence/normalizer.
6. Source/numeric/deviceparity/resource checks. Nếu bất kỳ prerequisitefail,
   stop before training với ATTRIBUTION/REPRESENTATION/DATA/RESOURCE_NO_GO.
   Không đổisignal/task/weights/rate saufail để mở một pilot khác.

## Hardware profiling trước neuralstudy

Verify actualCPU/RAM/GPU/CC/runtime/CuPy/HermesPID, readonly. Existing assets
only, no installs/drivers/restarts. Probe matchedH64/D64/fan8,32tick ALIF+
local traces/proposedupdate, same inputs/init/opsFP32. Fixed3reps×2episodes,
one warmup/backend, reportcoldsetup/warmmedian/transfer/sync. No live weights.
CPUcore per-tick unusedprobability computation also executed onGPU for parity.
GPUonly if availablefree≥1GiB beforeload, minfree≥512MiB throughoutsamples,
hostRAM<2GB, numericpred parity/maxabsfeatureerror≤1e−5, median faster.
ElseCPU; GPUfail preserved, no switching neuron/precision to rescue.

## Budget và bước tiếp

1800s total active execution including originaldataaudit, profiling/preflight,
training/audit nếu sau này qualified; coding/readtime riêng không báo train.
Maximum24arm-runs total prospective A/B/C/D×3seeds×2signs, not executed here.
If prerequisitespass, **registerfull P1b protocol before first update**;
DEV gates beforeTEST/restart; G4continuation must be registered before execution.
Không scientificCI/20stagegatePASS từ preflight. Scalar auditors do not train.
