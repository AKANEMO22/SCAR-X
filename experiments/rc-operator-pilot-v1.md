# RC-OPERATOR-PILOT-v1 — preregistration draft 16/11/2026

**ROOT_REVIEWED_D0_ONLY. NOT_RUN.** Root cho phép implement D0 data-only;
chỉ execute một lần sau root commit source/protocol. Chưa GO neural/profile/
fit. [Design/prior art](../docs/research/REUSABLE_OPERATOR_DESIGN_20261116.md)
giải thích duplicate check và giới hạn identity-fold. User directive SHA256
`57f636019eba5e5301a5b6b5d45771c911cee6f707df43073a83909d3b006b25` mở pilot
mới sau valid protocol/preflight; không mở lại R13.2/P1b/TEST cũ.

Evidence base `1ea065be194d44000ba795138c5917f3d13adda0`. Implementation sẽ chỉ
ở `research/autonomous-reusable-20261116`, worktree `rc-1116`, source SHA/hash
phải khóa trước execution. Không main/live/Hermes/push/merge/xóa/auto-restart.

## 1. Estimands, hypotheses và scope

- E1: exactstate generalization của từng primitive MOVE/SWAP trên unseen
  fullstates, sau cùng 512 TRAIN examples/two epochs.
- E2: matched hidden contribution, SHARED_LOCAL_ALIF minus
  FROZEN_ALIF_READOUT, cùng init/codec/head/data/order/update count.
- E3: zero-additional-training exactstate/path trên given routes AB, BA,
  ABAB, BABA; intermediate error và counterfactual oracle-input gap.
- E4: quality/resource comparison với current-rule ALIF, GRU và kNN3.
- H1 memorization-only bị thách thức nếu unseen-state performance vượt memory
  controls; điều đó không tự chứng minh rule đã học trên toàn miền.
- H2/H3 chỉ xét primitive reachability/generalization và engineering handoff.
  Neutral deterministic identity interface làm đúng từng bước ⇒ đúng cả fold;
  không nhận diện một learned composition mechanism riêng.
- H4 chỉ bounded pilot cost; H5 faster new-skill learning/continual retention,
  autonomous routing và compilation **NOT_MEASURED**.

Không hypothesis novelty, language/video ability hoặc thắng oracle 100%.
Đây là exploratory pilot với locked triage, không confirmation.

## 2. Immutable provenance và records

DATA seed `261180`. Init/order seeds `261181,261187,261191`; cả ba dùng cùng
fixed data world. Within seed, mọi arm có cùng TRAIN ordering; epoch ordering
là RNG permutation dùng SeedSequence([model_seed, epoch, 6116]). Các prediction
không phải independent seeds. Không thay seed thất bại.

Unique run `YYYYMMDDTHHMMSSZ-rc-operator-pilot-v1-<random6>`; create-exclusive
run directory, không ghi lại path cũ. Manifest trước execution chứa source
SHA, tracked-source content hashes, protocol SHA256, config, Python/library
versions, machine/backend, data/split hashes và enabled ponytail status.
Mỗi arm/seed có output/checkpoint/log/resource riêng; atomic finalized files.
Controller không cho hai process ghi chung artifact/DB. Fail/abort giữ nguyên.

## 3. Task và codec khóa

`s∈{0,1,2,3}^5`; MOVE(i,l), i∈0..4/l∈0..3; SWAP(i,j), 0≤i<j≤4.
Oracle: thay đúng một location hoặc hoán vị đúng hai locations, giữ các slots
còn lại. Oracle là labeler/control duy nhất, tách module khỏi candidate.

Input64: state one-hot0:20, op20:22, arg1 22:27, arg2 27:32 (zero MOVE),
dest32:36 (zero SWAP), restzero. Eight repeated ticks; reset model h/v/a/z/f
mỗi primitive. Không hidden state qua operator boundary hoặc giữa examples.
Output một affine head20, five independent four-class groups; fullstate
argmax pergroup/tie-lowest. Handoff chỉ tuple→same state onehot; không nhận
op/args, không oracle/copy-mask/residual/diff/patch/symbolic update.

ALIF feature `q=10*core.features`, không fitted normalization hay input bypass.
Logits `W_out q+b`. Group loss mean của năm CE. Group error
`e[o,l]=(1[target_o=l]-softmax(logits[o,:])[l])/5`. Readout delta dùng cùng e;
full target20 nằm ở evaluator/learner TRAIN, không codec input.

## 4. Data specification và strict split

**D0 data-only phase**, sau root GO, trước neural profile/training:

1. Enumerate 1,024 states. Canonical location-shift orbit là lexicographic min
   của `(s+k) mod4`, k=0..3. Mỗi orbit bốn states; 256 orbits.
2. SHIFT là occupancy≥4: 64 states/16 orbits (60 exactly-four +4 all-five).
   ORDINARY là 960 states/240 orbits. Đây là arithmetic expectation cần kiểm.
3. Rank ordinary orbits bằng SHA256 UTF-8 string
   `rcop-v1|261180|partition|<canonical five digits>`. First128 orbits là
   TRAIN_STATESET512; remaining112 orbits là authorized IID_DEV_STATESET448.
   Không final TEST/confirmation split mở trong pilot này.
4. Rank remaining ordinary orbits bằng hash label `iid-eval`; first16 →64
   IID eval states. Changed-SWAP SHIFT evaluation chỉ lấy exactly-four occupancy:
   60 states/15 orbits; all-five cùng location không có changed SWAP về mặt toán.
   Rank15 orbits này bằng `shift-eval`; first8 →32 SHIFT eval states. All-five
   degenerate states không vào initial evaluation states hoặc initial composition
   states. Terminal/prefix có thể all-five nếu đáp ứng path constraints và phải
   được tag theo occupancy; đây không là riêng all-five initialstate evaluation.
   Đây là pool được phép chấm DEV, không dùng fit/calibration/tuning.
5. TRAIN pool gồm single transitions có **cả prestate và poststate trong
   TRAIN_STATESET**. Không composition targets. Select512 unique records:
   256 MOVE (224 changed/32 noop), 256 SWAP (224 changed/32 noop). Stable
   SHA256 record ranking label `train-record`; iterate MOVE arguments rồi SWAP
   pairs theo lexicographic order, scan ranked pool để cover mỗi MOVE argument
   ≥8 records và mỗi SWAP pair ≥16, rồi scan toàn ranked pool để fill quotas.
   Chỉ nhận record chưa chọn, không vượt changed/noop quota hoặc max2 records/
   prestate. Nếu greedy không đủ quota, STOP_DATA, không backtrack
   sang data seed/split khác hoặc nới min coverage sau outcome.
6. Require ≥256 distinct TRAIN prestates; mỗi location frequency từng object
   trong TRAIN input và target nằm trong [20%,30%]. Đủ 20 MOVE arguments và
   10 SWAP pairs. Không đủ →STOP_DATA. TRAIN generation không xem model.
7. Primitive DEV mỗi eval state: hai distinct changed MOVE và hai distinct
   changed SWAP, stable-hash ranked valid commands label `primitive-dev`.
   Tổng384 changed cases: IID128/operator, SHIFT64/operator. Require mọi
   operator argument xuất hiện, state labels balanced như orbit construction;
   report command counts. Thêm một noop MOVE và một noop SWAP/state, cùng
   ranking label `noop-dev`:192 noop cases, diagnostic tách khỏi gate.
8. Composition: initial32 IID +16 SHIFT states từ eval pools, rank label
   `sequence-initial`. Mỗi state một AB và BA twin cùng MOVE/SWAP arguments,
   oracle finalstates khác nhau. Mỗi state thêm ABAB và BABA, later arguments
   mới được sample. Tổng192 episodes:48/family, 96 length2 và96 length4.
   Mỗi oracle prefix phải changed, mọi true prefix/prestate ngoài
   TRAIN_STATESET, final khác initial và khác≥2 object slots. Trong mỗi
   length2 twin MOVE và SWAP share một affected object; destination khác hai
   initial locations. Candidate không nhận selection masks/targets.
9. Sequence sampling dùng RNG SeedSequence([261180,6116,seq_initial_index,
   family_index]); family_index0 tạo chung AB/BA twins (không sample BA độc lập),
   index1 cho ABAB và index2 cho BABA. Chọn command arguments uniform, reject chỉ theo constraints
   trên; tối đa2,000 attempts/initial-family. Thiếu episode hoặc command/order
   coverage →STOP_DATA. Không đổi luật sau neural outcome. Chỉ simulator/eval
   tính oracle path; TRAIN trainer không mở composition labels.
10. Hash arrays/record IDs và lưu intersections TRAIN inputs/targets vs tất cả
    true DEV inputs/prefixes. Require empty intersections; save novelty status
    của từng step. Predicted wrong state có thể rơi vào TRAIN; tag riêng.

Nếu strict split làm retrieval whole-tuple không thể đúng, phải ghi điều đó.
Không diễn giải 1NN=0 được bảo đảm bởi split là neural advantage. kNN3 per-slot
vote và GRU là controls mạnh hơn cho generalization.

## 5. No-learning preflight gates

D0 không import neural/GPU packages. Exhaustive codec roundtrip1,024states;
injectivity full input cho mọi valid state/command; shape/no-target channel;
oracle tests đối chiếu independent direct tuple implementation, SWAP involution,
MOVE idempotence, AB/BA noncommuting twins. Oracle control phải exact100%.

Score cheap refusals trên DEV labels: identity; last-destination constant
tuple; operator/argument-only lookup with TRAIN per-slot majority; state-only
identity; exact lookup; whole-tuple1NN và kNN3. kNN chỉ same exact command,
nearest prestates Hamming distance, sort(distance,TRAINrecordID), k3, unweighted
per-slot vote/tie-lowest. TRAIN có≥3 records/command; no fallback oracle.

Data gate: on changed IID primitives và length2 finalstates, best of identity/
constant/arg-only controls <50% exactstate; every order twin oracle differs;
roundtrip/label/split checks PASS. kNN3 được đo nhưng không chọn dataset để làm
nó thua. Nếu kNN≥80% trên both primitives và≥60% length2 path, route neural
comparison lacks cheap-baseline headroom: STOP_ADEQUACY trước train, report
memory suffices. Symbolic oracle không thuộc nhóm bị yêu cầu thua.

ALIF H128 analytic representability witness NOT_PROVEN. Không tạo handcoded
neural solver/witness trong candidate để lấy PASS. GRU training sau là
competence diagnostic, không preflight oracle hoặc chứng minh ALIF capacity.

D0 execution subcap được root khóa: CPU1, process-tree peak RSS≤300,000,000
bytes, active runtime≤20s. NumPy chỉ dùng SeedSequence/default_rng để khớp
registered sampling; không model/optimizer/fit import. Root phải commit source
và protocol trước launch; D0 PASS không tự mở profile/training. Không thay
quota/split/seed sau D0 FAIL.

Clarification trước source lock/D0, root đã chốt: DEV primitive argument
coverage tính global trên384 changed cases; composition command coverage tính
global trên192 episodes. Report command counts riêng theo stratum/family,
không thêm gate mỗi stratum/family phải có đủ30 commands. AB/BA twins và các
sequence constraints ở§4 giữ nguyên. Canonical record hash payload là
`<five-digit-prestate>|<op:first:second:destination>|<five-digit-target>`;
unused argument dùng -1. Source commit khóa serialization/sampler cụ thể trước
data execution; không đổi seed/partition/quota để cứu outcome.

Pre-run review clarification, trước code/data commit: primitive DEV true
poststate cũng là one-step prefix. Vì vậy primitive command pool phải lọc
target ngoài TRAIN_STATESET; split audit gồm cả primitive input và target,
cùng mọi sequence prefix. Không đủ hai changed commands/operator/state sau
lọc thì STOP_DATA ở fixed seed, không nới partition hoặc giảm counts. Đây là
làm rõ strict split ở§4.10, không thay threshold/data sau outcome; D0 chưa run.

## 6. Four neural arms và fixed learner settings

All ALIF H128/D64/fan8/K20, FP32, Config dynamics leak.85/threshold.65/
adapt_decay.95/adapt_strength.08/feature_decay.92/trace_decay.9; sparse topology
seeded, không thay sau init. W_in/W_rec/W_out initialization như SpikingCore,
biaszero. Init weights/topology của ba ALIF arms được copy bitwise withinseed.
Stable copies là init, không consolidate/replay/importance/capacity growth.

**CURRENT_ALIF:** adapt current core reward-STDP math trong research wrapper
cho grouped head. Readout lr.04, anchor.0001, clipdelta.025/weights±3. Hidden
rotating16 rows, hiddenlr.002; reward mean.95/.05, rawreward mean của năm
target probabilities, advantage clip±1; current pre/post eligibility và
anchors. W_in và sparse W_rec đều plastic theo math gốc. Không production edit.

**FROZEN_ALIF_READOUT:** W_in/W_rec/topology bitwise frozen. Head lr.04,
`ΔW=clip(.04*e⊗q,±.025)`, `Δb=clip(.04*e,±.025)`, weight/bias clip±3,
không anchor/normalization fit. Chính xác cùng head rule/init với candidate.

**SHARED_LOCAL_ALIF:** cùng head rule trên, W_rec/topology fixed. W_in-only
eligibility cho mỗi neuron/input, zero đầu example. Với theta=c.threshold+
beta*a_prev, pre-reset u, psi=logistic((u-theta)/.4)*(1-logistic)/.4,
raw sigmoid argument clip±20 và slopezero ngoài đó:

    Eu = leak*Ev_prev + x
    Ez = psi*(Eu - beta*Ea_prev)
    Ev = Eu - theta*Ez - spike*beta*Ea_prev
    Ea = adaptation_decay*Ea_prev + Ez
    Ef = feature_decay*Ef_prev + (1-feature_decay)*Ez

Không derivative đường recurrent qua neuron khác. End-of-eight-ticks:
`m=10*(pre-update W_out.T @ e)`, `direction=m[:,None]*Ef`.
Moment=.9*moment+.1*direction; energy=.99*energy+.01*direction², initialized
0 và1e-4. `delta=clip(.008*moment/(sqrt(energy)+1e-4),±.01)`;
W_in +=delta rồi clip±3. Head update sau khi đã tính m từ pre-update head.
Moment/energy persistent giữa examples; eligibility reset. Không teacher,
autograd, reverse-time tape, pseudolabel hoặc gradient through argmax/sequence.

**SHARED_GRU control:** torch.nn.GRU(input64,hidden36,num_layers1), one affine
20-logit head; native FP32, zero h0 mỗi primitive, tám identical input ticks.
Same mean grouped CE. Adam lr.003/betas(.9,.999)/eps1e-8/weight_decay0;
global grad_norm clip1; batchsize1, twoepochs, no scheduler/earlybest/teacher
forcing across operators. torch/BP code isolated under evaluation controls,
không import trong ALIF core/candidate. No Transformer substitution.

All neural arms512examples×2epochs =1,024 updates. No DEV checkpoint selection;
chỉ terminal checkpoint và init snapshot. Max12trajectories/12,288 updates.
Untrained init/revert evaluations là cùng trajectory, không được fit.

## 7. Resource preflight/profile và scheduler

300 seconds total active budget cho toàn operator campaign gồm D0, controlled
imports/profiler/worker/evaluator/controller/auditor/checkpoint I/O. Count sum
controlled-process active wall; parallel không nhân cap. Đồng thời ghi elapsed
wall và summed CPU seconds. Không tính human review idle; neural execution
không bắt đầu nếu không thể debit controller/resource monitor đúng.

Sub-budgets: data/contract audit≤20s; profile≤20s; training+DEV+serialization
≤220s; final data-only audit≤40s, tổng≤300s. Đây là ceilings, không chuyển
phần vượt thành budget mới. Trước launch estimate từ profile; projected
remainingwork >remainingbudget →STOP_BUDGET trước fit.

Experiment controlled-tree RSS và private commit mỗi metric<1,000,000,000
bytes; global owned research tree RSS<2,000,000,000 bytes gồm D1–D4 khi đồng
thời. Tối đa3 workers tổng pilot; monitor/controller là overhead được tính
RAM/time. Soft queue nếu forecast>900MB; hard stop owned research worker khi
chạmcap. Không kill live/Hermes/system process. Kiểm free physical≥4GB,
commit headroom≥2GB, SSDfree≥2GB, pagefile pressure và current jobs trước run.

Inventory CPU logical/physical, installed GPU/device/memory/processes, SSD.
OMP/MKL/OPENBLAS threads1/worker; torch intra/inter-op1. Không package install,
dataset/weight download hoặc dummyVRAM allocation.

Sau D0 PASS và root source review: matched profile cùng fixed TRAIN inputs,
eight-tick H128 ALIF with eligibility/proposal arithmetic, không applyupdates;
CPU vs available CUDA, synchronized warmup/timing, tối đa64examples/backend.
GRU profile cùng control forward/backward scratch, no optimizerstep, zero grad
giữa examples, no persistent fit. Separate import/context startup, peakmemory,
per-example forward/update arithmetic. Profile no DEV accuracy. Không CUDA
multiworker. Chỉ dùng installed dependencies; CUDA unavailable ghi rõ, CPU
chỉ chạy nếu measured projection/cap hợp lệ. Nếu CPU throughput tốt hơn thì
arm đó CPU, không dùng GPU. Profile phải phản ánh correct nativeFP32 path;
không suy từ P1b speed ratio.

Choose concurrency1/2/3 theo projected throughput **và** actual aggregate RAM;
không yêu cầu3 Torch processes. Có thể một GRUworker +hai NumPyworkers khi
fits. Root xác nhận global resource/D1D2 interaction trước launch. Run boundary
không bật Hermes/model user đã tắt. GPUreserve≥1GB nếu CUDA được chọn.

## 8. Evaluation DAG và immutable gates

`root review → D0 → profile/source audit → root launch check → all training →
primitive DEV → per-arm composition gate → data-only audit → report`.

1. Train all registered arms/seeds, terminal checkpoint. Log all updatecounts,
   loss, physical W_in/W_rec/head delta; bytes/hashes/counters. No earlybest.
2. Terminal primitive DEV changed+noop, no updates. Save prestate/command/
   target/logits/prediction, per-object margins and effect mask in evaluator.
3. **Primitive gate per arm/stratum:** every seed, both MOVE and SWAP changed
   exactstate≥.80. IID counts128/operator/seed; SHIFT64. If an arm fails any
   seed on IID, no composition for that arm at any seed. Candidate failure
   terminates candidate family; no more epochs/H/lr/seeds. Other registered
   arms may finish their own locked gate; no replacement.
4. SHIFT composition opens only if that arm's SHIFT primitive gate also PASS;
   otherwise report SHIFT_PRIMITIVE_FAIL, no SHIFT composition. Do not infer
   composition weakness from it. Gates have no cost-reallocation reward.
5. Open length2 AB/BA on eligible strata. Record predicted state every step,
   all-prefix path and final exact, first failure, state novelty and margins.
   Also evaluate each primitive once on the true oracle prestate, same reset.
   If prefix correct yet free-run differs from oracle-input, stop INVALID.
6. Length4 opens only if every seed in that eligible stratum has both AB and
   BA all-prefix/path exact≥.60. Otherwise LENGTH2_GATE_FAIL; no length4.
   Fixed routes/families/episode counts, no sequence rescue.
7. Candidate-vs-frozen primitive contrast/revert diagnostic: after candidate
   finish, restore init W_in while retaining candidate head/fixedR, evaluate
   same primitive cases only. No refit; report distribution shift of head when
   reverting, so not a standalone proof of pure representation content.

Toy passing thresholds mean pilot viability only. Hidden contribution needs
candidate−frozen positive on both operators each seed; practical signal flag
if≥5pp mean and no negative seed, with ranges/uncertainty. Lack of signal is
HIDDEN_GAIN_NOT_DEMONSTRATED, không đổi gate hoặc gọi weightmotion là success.

## 9. Metrics, uncertainty và decision partition

Primary: peroperator exactstate on changed IID; complete-path exact and final
exact by sequence family/length/initial stratum. Secondary: affected accuracy,
unaffected preservation, noop exact, oracle-input perstep accuracy, handoff
error, margin/NLL, kNN quality/cost, seed-matched deltas, updatecount/time/RAM/
VRAM/disk/parameter counts. Không average IID+SHIFT hoặc MOVE+SWAP để che fail.

Report correct/n, perseed, seedmean/minmax. State/initialstate cluster bootstrap
2,000 draws, fixed statsseed261199, paired arms, conditional on this fixed data
world; does not estimate population-of-training-seeds confidence. n3 exact
paired sign/permutation p-values cannot reach conventional .05 (one-sided
minimum.125); không claim population significance hoặc call12arms12seeds.

Outcomes priority: INVALID/LEAKAGE > RESOURCE/BUDGET_STOP > STOP_DATA/
STOP_ADEQUACY > PRIMITIVE_GATE_FAIL > LENGTH2_GATE_FAIL >
COMPOSITION_DIAGNOSTIC_COMPLETE. HIDDEN_GAIN_NOT_DEMONSTRATED và
NO_ADVANTAGE_OVER_GRU_KNN là orthogonal flags. Mỗi arm/stratum có NOT_RUN cho
descendants của failedgate, không zeroaccuracy giả. Không qualityPASS khi
receipt/audit thiếu. Không dùng final TEST hoặc choosewinner để advance.

Falsifiers: candidate primitive gatefail; no matched frozen/revert benefit;
trajectory primitive gap; broken identity invariants; strong GRU/kNN matches
or beats quality/resources. Nếu cả GRU và ALIF fail, capacity vs optimization
vs sample budget chưa tách. If all pass, it still demonstrates only this
finite toy under supplied symbolic state/route and supervised primitives.

## 10. Required output/audit và closure

Independent data-only implementation recomputes targets/metrics from raw,
checks leakages/counters/config/checkpoint/dataset hashes, confirms fixedR
candidate/frozen arrays and finiteFP32 values, recomputes logits for saved
feature rows, verifies feedback receives no args/oracle, verifies noop/order
counterfactuals. Audit level: same-team independent implementation, không
external peer review hoặc reproduced training.

Report perarm total/trainable parameters and native bytes; readout/hidden/
optimizer/state/eligibility/stable/rollback bytes; kNN index/examples; code/
runtime RSS/privatecommit; throughput total import/profile/eval/I/O; actual
MAC/call estimates, checkpoint/log/disk. Static ALIF parameters11,796;
candidate trainable10,772; frozen2,580; GRU11,756. Không equate parameter
matching với compute/memory matching hoặc offload với capacity miễn phí.

Preserve failed logs/checkpoints; source/protocol/data hashes, status, seeds,
run ID và resource receipt đi cùng report. Maximum new disk100MB; forecast
trước write; không xóa history. Stop family theo gate, không auto confirmation.
Future retention/new-primitive/sample-efficiency study cần protocol/data chưa
mở mới; không kéo H5/20stage vào pilot này. Hiện chỉ draft, **0 execution**.
