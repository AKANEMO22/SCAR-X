# Bản đồ dự án SCAR-X

## Điều hướng chính sau hợp nhất13/11

- [Master plan](RESEARCH_PLAN.md): định hướng/constraints, không coi lịch sử là approvalmới.
- [Research center](research/README.md), [experiment map](research/EXPERIMENT_MAP.md),
  [ledger](research/EXPERIMENT_LEDGER.md), [catalogJSON](research/experiment-catalog.json).
- `scripts/research/query_catalog.py`:list/search/show/verify, không load/trainmodel.
- `docs/research/archive/`:originalreports với SHA256 và sourceversions;
  `docs/maintenance/20261113/`:branch/worktreearchive + restore/protection.
- `runs/` giữ immutableevent/model/checkpoint evidence; không rename historicalrunID.
- `src/scarx/evaluation/`:isolatedresearch/controls; servingcode giữnguyên.
- `local/`:private/live/ignoredstate, dependencies vàmodelcaches; không publicmirror.
- ActiveR13-family ở `scar-x-gemini-r7`:EXCLUDED khỏi cleanup, source/cache/process giữ.

Các mục dưới là chỉ mục lịch sử, vẫn giữ đường dẫn để truy vấn từ báo cáo cũ.

**Capacity completed12/11:** [results](research/CAPACITY_H64_H128_RESULTS_20261112.md),
[audit](research/CAPACITY_RESULTS_INDEPENDENT_AUDIT_20261112.md). Finalrun160b6e
has252records/perseedCSV/data+reusehashes/modelcheckpoints/source-byte-bundle.
`audit_capacity_dev.py` scalar/eigenproof, independent`verify_capacity_independent.py`
manualALIF/SVD/thresholdoracle; `plot_capacity_results.py` actualscientificfigures.
CAPACITY_ONLY_SIGNAL: usefulH128directgain, chưa thêmhiddenadvantage≥2families.
Không main/Hermes/P2/confirmation. Passiveviewer8773 giữactualresults.

**Approvedcapacity12/11:** [protocol](../experiments/capacity-h64-h128-approved-v1.md),
[technical review](research/CAPACITY_INDEPENDENT_REVIEW_20261112.md),
`evaluation/capacity.py` dimension-onlyconstructoradapters, strictcapacitysnapshot
format trong `forensic/r8_snapshot.py`; oldH48formats không mởrộng. Runner
`run_capacity_dev.py`, passiveobserver `serve_capacity_observer.py`:8773.
H48oldqualityrecords reusehash; H64/H128 chạyfixed1024updates+equalCPU riêng.
Không thay scientificgates hoặc main/Hermes, không growbeyond128/P2/confirmation.

**ACR12/11:** [results](research/ACR_DEV_RESULTS_20261112.md),
[independent audit](research/ACR_STATISTICAL_HARNESS_REVIEW_20261112.md),
[router traces](research/ACR_ROUTER_ANALYSIS_20261112.md),
[literature comparison](research/ACR_VS_LITERATURE_20261112.md).
Masterprotocol `experiments/acr-dev-v1.md`; source/equations pin giữ nguyên.
`evaluation/acr.py` researchonly; runner `run_acr_dev.py`, explicitresumev1/v2/v3,
auditors `audit_acr_dev.py`/`audit_acr_mechanism.py`/`audit_acr_router_state.py`.
Raw final59ced4 có375models+CSVv2+source-byte-bundle, giữ parent55/203cells.
Passive viewer `serve_acr_results.py`:8771, không model/SQLite/live.
[Capacity diagnostic proposal](../experiments/capacity-diagnostic-proposal-20261112.md)
chưa phê duyệt training; chưa đổiH48 của lượt đã đo.

**P1 confirmation12seed đã duyệt:** [report](research/R8_P1_CONFIRMATION_20261112.md),
registration `experiments/r8-reset-credit-confirmation-v1.md`, runner
`scripts/run_r8_confirmation.py` / `r8_confirmation_worker.py`, statistics
`evaluation/r8_confirmation.py`, independent reference
`evaluation/forensic/confirmation_reference.py`. Receipt
`artifacts/r8-reset-credit-confirmation-v1.json`: numericalP1FAIL, integrityPASS,
giữ XOR gain và bằng chứng hidden learning; không deploy hoặc mở P2.

**R8 matched/reset DEV:** [kết quả](research/R8_REPAIR_AND_DEV_RESULTS_20261112.md),
protocols `experiments/r8-{matched-credit,reset-credit}-dev-v1.md`, prototypes
`evaluation/r8_matched_credit.py` / `r8_stopped_reset.py`, study oracle/probe
`r8_matched_study.py`. Runners `run_r8_*_dev.py`, raw verifier
`verify_r8_study_artifacts.py`, readonly observer `serve_matched_observer.py`.
Receipts `artifacts/r8-*-dev-v1.json`; fresh confirmation remains a proposal.

**R8 state repair P0:** [handoff](research/forensic-r6-20261111/R8_SNAPSHOT_REPAIR_HANDOFF_V1.md),
`evaluation/forensic/r8_snapshot.py`, `r8_snapshot_fixture.py`,
`scripts/verify_r8_snapshot_repair.py` / `r8_snapshot_cold.py` và15adaptertests.
Data-only research wrapper; candidate/main/app không đổi.

**Gemini R8 audit/repairs:** [review](research/forensic-r6-20261111/GEMINI_R8_REVIEW_AND_REPAIRS_VI.md),
[next proposal](research/forensic-r6-20261111/R8_REPAIR_AND_P1_PROPOSAL_V1.md).
`scripts/audit_gemini_r8.py` / `r8_engineering_worker.py`, independent gates
`evaluation/forensic/research_gates.py`, receipt `artifacts/gemini-r8-independent-audit-v1.json`.
Audit/fixtures only, no new quality training or live change.

**Gemini Cross-SNN review:** [report](research/forensic-r6-20261111/GEMINI_CROSS_SNN_REVIEW_VI.md),
protocol `experiments/cross-snn-transfer-review-v1.md`, runner
`scripts/review_cross_snn_transfer.py`, helpers `evaluation/cross_snn_review.py`.
Pinned Git inputs/raw arithmetic, primary-source corrections; không train mới.

**Data grounding43:** [báo cáo43](research/43_ADAPTIVE_DATA_GROUNDING.md),
`adaptive-curriculum-grounding-v1.md`, `evaluation/adaptive_grounding.py`,
`data/fixtures/adaptive_chat_references_v1.json`. Data-only, không pilot/modelchange.


**P0 usefulness42:** [báo cáo42](research/42_READOUT_USEFULNESS_P0.md), protocol
`readout-usefulness-p0-v1.md`; prototype `evaluation/readout_usefulness.py`,
`usefulness_validation.py`; runner `validate_readout_usefulness_p0.py`, restart
worker `usefulness_p0_cold.py`,15newtests. P0 đã kiểm; pilot chưa duyệt, no mainchange.


**Chuẩn bị41:** `experiments/readout-usefulness-proposal-v2.md` còn chờ duyệt.
Data/reference protocols `adaptive-curriculum-*.md`, script
`prepare_adaptive_curriculum.py` / `compare_adaptive_curriculum.py`; receipts
`artifacts/adaptive-curriculum-{data,reviewed,reference}.json`. Chỉ wording
development và reference, chưa train/import U hoặc thay phiên Hermes.


**Review cập nhật DeepSeek 40:** [báo cáo40](research/40_DEEPSEEK_REVIEW.md),
`experiments/deepseek-product-review-v1.md`, `evaluation/parameter_interventions.py`,
`evaluation/product_acceptance.py` (v1/v2 tách nhãn), `scripts/review_deepseek_product.py`.
Sửa before-clone failover/SQLitebackup/transplants/acquisition metrics; không
đổi ALIF/learning rule hoặc restore phiên thật. Git đã có trong workspace do
các cập nhật trước review; ghi nhận commit/hash, không init/commit/push ở đây.
`readout-protection-v1.md` vẫn là đề xuất CHƯA DUYỆT, không code/chạy.

**Credit alignment39:** [báo cáo39](research/39_LOCAL_CREDIT_ALIGNMENT.md), protocol
`experiments/local-credit-alignment-audit-v1.md`, `evaluation/credit_alignment.py`,
`scripts/audit_local_credit_alignment.py`; receipt`artifacts/local-credit-alignment.json`.
Read only15checkpoint36, áp hướng local unchanged35 lên **bản sao dùng một lần** và đo
ΔNLL thật so control; `0` neuralupdate, khôngTorch/mainchange. Bản chạy
`d8de33` có lỗi đo norm đã`do_not_cite`; dùng `db01fc`. UI`#research-lab` bảng AUDIT39
tách khỏi freeanswer.

**Wordobjectiveaudit38:** [báo cáo38](research/38_WORD_TRAINING_OBJECTIVE_AUDIT.md),
`experiments/word-training-objective-audit-v1.md`, `evaluation/word_training_audit.py`,
`scripts/audit_word_training_objective.py`. Readonly15models36/FREE-GOLDPREFIX/
wrong-zeroquestion/potentialdirections, khônglearner/headfit/encoder mới.
Receipt`word-training-objective-audit.json`; UI táchgoldprefix khỏi actualanswers.

**Directvisual37:** [báo cáo37](research/37_DIRECT_VISUAL_BRIDGE.md), protocol
`experiments/direct-visual-bridge-v1.md`, `evaluation/direct_visual.py`,
`visual_eligibility_validation.py`, `controls.py::DirectVisualLearnabilityControl`;
scripts`direct_visual_common.py`/`preflight_direct_visual.py`/`direct_visual_worker.py`/
`train_direct_visual_bridge.py`. Same32forward/cache, unchanged35localrule và
boundedpendingimageinputs/cachetrace choexactmidimagecontinuation. Receipts
`direct-visual-*.json`, archive`app.observatory.direct_visual_answer_archive`.

**Wordbridge36:** [báo cáo36](research/36_DIRECT_WORD_BRIDGE.md), protocol
`experiments/direct-word-bridge-v1.md` và premeasurementamendments36A/36B;
`evaluation/direct_words.py` + unchanged`readout_eligibility.py`,
`controls.py::DirectWordLearnabilityControl`. `direct_word_data.py` và scripts
`generate_direct_word_prompts.py`, `repair_direct_word_language.py`,
`review_direct_word_prompts.py`, `direct_words_common.py`, `preflight_direct_words.py`,
`direct_word_worker.py`, `train_direct_word_bridge.py`. 64/64actualAGYreviewed,
48ALIF/onecore,5methods; khôngmainimport/P1waiver/normalchatclaim.
Qwen64reference:`scripts/compare_qwen_direct_words.py`; receipts
`artifacts/direct-word-*.json`, `antigravity-direct-word-*.json`,
`qwen-direct-word-reference.json`. UI`direct_word_answer_archive` và
`runtime.experiment_status.direct_word_statistics` onlyreadpublicsource.

**Directreadouteligibility35 — proposal30 đãduyệt nghiên cứu:**
[báo cáo35](research/35_DIRECT_READOUT_ELIGIBILITY.md), protocol
`experiments/direct-readout-eligibility-v1.md`, `evaluation/readout_eligibility.py`,
`eligibility_validation.py`, `controls.py::EligibilityLearnabilityControl`.
Scripts`direct_eligibility_common.py`/`audit_readout_eligibility.py`; candidate
khôngTorch/BPTT, maincore/weights khôngđổi. Source32–34 pending là lịch sử.

**Visualinformationaudit34:** [báo cáo34](research/34_VISUAL_INFORMATION_PATH.md),
protocol`experiments/visual-information-audit-v1.md`,
`evaluation/visual_information.py`, `scripts/audit_visual_information.py`.
Chỉđọc32/33; exactcollisions/path và fixedridge ngoàicore vớiđúngtrainexposures.
Khôngneuraltraining/Torch/mainimport hoặcstateviewgrid; receipt`visual-information-latest.json`.

**Visuallearnabilitycontrol33:** [báo cáo33](research/33_VISUAL_WORD_LEARNABILITY_CONTROL.md),
protocol`experiments/visual-word-learnability-control-v1.md`, class
`evaluation/controls.py::VisualWordLearnabilityControl`, scripts
`preflight_visual_word_control.py`, `visual_word_control_common.py`,
`visual_word_control_worker.py`, `control_visual_word_learnability.py`.
BPTT chỉcontrol, NumPyALIFforward/head như32; khôngmain/proposal30change.
Receipts`visual-word-control-*.json`; archive tựhiệnactualoutputs cónhãnriêng.

**Jointvisualwords32:** [báo cáo32](research/32_JOINT_VISUAL_WORD_READOUT.md),
`evaluation/visual_words.py`, `visual_word_data.py`,
`scripts/build_visual_word_dataset.py`, `train_visual_word_readout.py`,
`audit_visual_word_readout.py`; protocols`visual-word-readout-v1.md`/
`visual-word-audit-v1.md`, receipts`visual-word-*.json`. Owneddataset192PNG,
hiddenfrozen/readoutonly, qualitySTOP, khôngmainimport hoặcproposal30code.

**Visionquestionpath:** [báo cáo31](research/31_VISION_QUESTION_PATH.md),
`experiments/vision-question-path-audit-v1.md`, `evaluation/vision_fixtures.py`,
`scripts/generate_vision_questions.py`, `review_vision_questions.py`,
`audit_vision_question_path.py`; receipts`vision-question-path-latest.json`/
`vision-questions-reviewed.json`/`antigravity-vision-questions-latest.json`.
OwnedPNGfixtures/caseannotations dướidata/candidates/<auditrun>. Khôngtraining.
`app/observatory.vision_answer_archive` chỉđọcpublicownedfixtures cóhash.

**Observedassociation:** [báo cáo29](research/29_OBSERVED_RESPONSE_ASSOCIATION.md),
`evaluation/observed_association.py`, `scripts/train_g1_observed_association.py`,
`audit_g1_observed_association.py`; protocols`g1-observed-association-v1.md`/
`g1-observed-association-audit-v1.md`, receipts`g1-observed-association-*.json`.
Familyv1STOP, khôngmainimport. [Đề xuất30](research/30_PROPOSED_LEARNING_SIGNAL_BOUNDARY.md)
chỉpendinguserdecision, chưa có code/rule mới hoặc đổi ranhgiới.

**Readonlycreditnoise:** [báo cáo28](research/28_CREDIT_SIGNAL_NOISE.md),
`experiments/g1-credit-noise-audit-v1.md`, `evaluation/credit_noise.py`,
`scripts/audit_g1_credit_noise.py`, receipt`artifacts/g1-credit-noise-latest.json`.
Không learningrule mới/scorev3 hoặc nguồnprivate; surrogate chỉcontrolcôlập.

**Đối chứng ALIF wordlearnability cô lập:** [báo cáo27](research/27_ISOLATED_WORD_LEARNABILITY.md),
`experiments/g1-word-learnability-control-v1.md`, `evaluation/controls.py`
(`WordLearnabilityControl`), `scripts/preflight_g1_word_control.py`,
`control_g1_word_learnability.py`, `word_control_worker.py`. Backward chỉ trong
controls, artifact cóformatcôlập không phải checkpointmain. IOfix:
`runtime/atomic_file.py`, `scripts/verify_observer_atomic_write.py`.
Qwenreference47case cònthiếu: `scripts/compare_qwen_word_controls.py` và
`artifacts/qwen-word-control-reference-latest.json`; không sourceproxy/main.
`app/observatory.control_answer_archive` đọc đủ95case đã đo/publicpairs,
`#answer-archive` tự theo tệp gần nhất; không có source/model/runselector.

**Vòng word-score input cuối:** [đề xuất/báo cáo26](research/26_FIXED_INPUT_NEURAL_TRANSFER.md),
`experiments/g1-word-score-input-transfer-v2.md`, `evaluation/fixed_word_codes.py`,
`scripts/train_g1_word_score_inputs.py`. Numericadapter riêng, không sửa core
serving. Parity oldsource: `scripts/check_score_word_legacy_parity.py`.
Readonlyconditionalaudit: `evaluation/word_conditioning.py` và
`scripts/audit_g1_word_conditioning.py`; không huấn luyện/câu trả lời đầy đủ.

**Nhật ký công việc tự động:** [OBSERVING](OBSERVING.md),
`runtime/work_trace.activity` → `app/observatory.feed` → `#agent-work`.
Runner `scripts/verify_passive_work_trace.py` và protocol
`experiments/passive-work-trace-acceptance-v1.md`; receipt
`artifacts/passive-work-trace-latest.json`. Không điều khiển chạy/chọn model.

**Representation/decoder và word-score transfer:**
[báo cáo24](research/24_REPRESENTATION_DECODER_AND_SEARCH.md),
[báo cáo25](research/25_WORD_SCORE_TRANSFER_PROPOSAL.md).
`evaluation/dialogue_probe.py`, `neural_beam.py`, `score_word.py`,
`input_identity.py`, `neuron_state_views.py`; runners tương ứng trong scripts.
Các lượt chẩn đoán/transfer chưa đạt, không servingimport hoặc scale.

**Hội thoại/state/curriculum:** [báo cáo23](research/23_REAL_DIALOGUE_STATE_AND_CURRICULUM.md),
`scripts/audit_g1_conversational_state.py`, `rehearse_g1_basics.py`,
`accept_g1_surface_normalization.py`, `train_g1_curated_curriculum.py`.
AGY generation/review/recovery và Qwenreference scripts có tên`g1_*` tương ứng.
Privateweights/rawoutputs ở`local/g1-*`, không đónggói. `app/project_supervision.py`
là explicitprojectlabelpath qua controller cũ, không response lookup.

**Dynamics views:** [báo cáo22](research/22_INFERENCE_AND_TIMESCALE_PROPOSAL.md),
`evaluation/inference_dynamics.py`, `scripts/audit_p2_inference_dynamics.py`,
`audit_p2_timescale_views.py`; screenSTOP, không learner mới/deployment.

**P2 paired-score diagnostic:** [báo cáo21](research/21_P2_RETENTION_AND_CUE_INFORMATION.md),
`evaluation/paired_retention.py`, `scripts/diagnose_p2_paired_retention.py`,
`scripts/audit_p2_cue_information.py`. Protocols`p2-paired-score-diagnostic-v1.md`
và`p2-cue-information-audit-v1.md`; receipts`p2-paired-score-diagnostic-latest.json`
/`p2-cue-information-latest.json`. Không candidate importance hay servingimport.

**P1 score integrity / paired-score:** [báo cáo20](research/20_SCORE_ELIGIBILITY_AND_VARIANCE.md),
`evaluation/score_integrity.py`, `neural/paired_score.py`,
`scripts/audit_score_integrity.py`, `scripts/run_p1_paired_score.py`.
Protocols`p1-score-integrity-audit-v1.md`/`p1-paired-score-v2.md`.
Đây là score-familyfinaladjustment, không servingimport/mở lại local-classifier.

**P1 control-factor audit:** [báo cáo19](research/19_CONTROL_OPTIMIZER_AND_COVERAGE.md),
`scripts/audit_p1_control_factors.py`, `experiments/p1-control-factor-audit-v1.md`.
IsolatedAdam/SGD×all/rotating trong`evaluation/controls.py`, khôngmainimport
hoặc localcandidate. Receipt`artifacts/p1-control-factors-latest.json`.

**P1 vòng cuối:** [báo cáo18](research/18_LOCAL_CLASSIFIER_FINAL_AND_LEARNABILITY.md),
`neural/local_classifier_conditioned.py`, `scripts/run_p1_local_classifier_v2.py`,
`scripts/audit_local_classifier_v2.py`; controls mới chỉở`evaluation/controls.py`.
Protocol`experiments/p1-local-classifier-v2.md`; candidate không servingimport.
FamilyđãNO-GO sau2vòng, không dùng receipt này để mở training lớn.

**Trace thụ động:** `app/observatory.py` đọc các transaction inference/learning
thật; `runtime/work_trace.py` nhận số đo từ registered runners. Website tự
theo dõi, không yêu cầu chọn/bấm chạy. `scripts/observe_live_work.py` là lượt
acceptance serving thật, protocol `experiments/live-observatory-acceptance-v1.md`.
`tests/test_passive_trace.py` kiểm số đo, inference không học và rollback.

**Quan sát localhost:** [OBSERVING](OBSERVING.md), [ADR0011](decisions/0011-local-observatory.md).
`app/capacity_lab.py` chạy chuỗi task quan sát; `runtime/experiment_status.py`
ghi/đọc journal có process identity; `app/generation_web/research.*` là UI.
`evaluation/runs.py` đăng ký lượt và kết quả. P1 mới ở
[báo cáo17](research/17_LOCAL_CLASSIFIER_ADAPTATION.md)/`scripts/run_p1_local_classifier.py`.

**RGB object diagnostic:** [báo cáo16](research/16_REAL_RGB_GROUNDING.md),
`scripts/fetch_cifar10_parquet.py`, `measure_rgb_object_grounding.py`,
`audit_rgb_grounding.py`. Ảnh/sourceparquet ở `local/datasets/cifar10`, không
đónggói; split/predictions/metriccorrection trongruns/receipts.

**Codec/decoder memory:** [báo cáo15](research/15_BYTE_CODEC_AND_WORKING_MEMORY.md),
`neural/byte_codec.py`, `evaluation/byte_dialogue.py`,
`evaluation/working_byte_dialogue.py`; stoppedprototype, không servingimport.
`scripts/train_byte_dialogue.py`, `train_working_byte_dialogue.py` có coldprocess
checks. BPTTcontrol chỉở `evaluation/controls.py`/`scripts/control_byte_learnability.py`.

**Prototype sinh câu thực:** [báo cáo14](research/14_EVENT_DIALOGUE_AND_QWEN_REFERENCE.md),
`evaluation/event_dialogue.py`, `scripts/train_event_dialogue.py`,
`scripts/verify_event_dialogue_cold.py`, `scripts/compare_qwen_dialogues.py`.
Corpus/version hiện hành ở `artifacts/antigravity-events-latest.json`; usage
phân biệt experiment training với live import. Tất cả chưa được serving nạp.

**P1/P2 hiện hành:** [báo cáo13](research/13_INPUT_AND_LOCAL_LEARNING_AUDIT.md),
`neural/input_events.py` và `neural/local_target.py` là prototype không được
serving import. `evaluation/readout_conditioning.py` là control đầu ra riêng.
Protocols `experiments/p1-*.md`, `p2-readout-conditioning*.md`; receipts dưới
`artifacts/p1-*.json`, `p2-*.json`, raw sources và predictions trong từng run.

**Tab Hỏi để học:** `src/scarx/app/learning_channel.py` giữ log hữu hạn;
`integrations/hermes/learning_plugin/` là plugin desktop/Python, cài bằng
`scripts/install_hermes_learning_tab.py`. API acceptance:
`scripts/verify_hermes_learning_channel.py`. [ADR0010](decisions/0010-learning-tab-and-cognitive-modes.md)
và [ba chế độ hoạt động](research/12_ADAPTIVE_COMPUTE_AND_IDLE_COGNITION.md).

**Đo tải kiến thức/retention:** [báo cáo11](research/11_KNOWLEDGE_LOAD_RETENTION.md),
protocol `experiments/knowledge-load-retention-v1*.md`, runner
`scripts/measure_knowledge_retention.py`, audit `scripts/audit_knowledge_retention.py`.
Qwen reference là process đối chứng đã tắt, không là backend sản phẩm.

**Hermes có trạng thái:** [adapter/hướng dẫn](../integrations/hermes/README.md),
installer `scripts/install_hermes_stateful.py`, public diff và test receipts ở
`integrations/hermes`/`artifacts/hermes-stateful-*.json`. Không thay neural core.
[Bảng đo memory/learning](research/09_MEMORY_LEARNING_MEASURES.md) là tham chiếu
cho master; không là master plan thứ hai hoặc số đo đã đạt.

**Điểm vào sản phẩm hiện hành:** [README](../README.md) → `run_app.py` →
[generation_server.py](../src/scarx/app/generation_server.py) →
[Generation](../src/scarx/app/generation.py) →
[UnifiedCore](../src/scarx/neural/unified.py) → một SpikingCoreALIF khởi đầu48,
capacity prototype48–128. [DialogueLearning](../src/scarx/learning/dialogue.py)
quản lý lesson, nguồn nhãn, review và câu hỏi; không là model ngôn ngữ thứ hai.
Frontend: [generation_web](../src/scarx/app/generation_web/). Session:
`local/generation1`. Release: `artifacts/generation1-release.json`.
Hermes chỉ còn `scarx-generation1`; old ZIPs offline ở `artifacts/retired`.
[ADR0006](decisions/0006-one-core-generation1.md) ghi thay đổi.
Những đường dẫn MVP/LAB bên dưới là bản đồ lịch sử để tái lập, không phải
hai hoặc ba core đang được sản phẩm mới nạp.

Điểm vào: [README](../README.md) → [RESEARCH_PLAN — master plan duy nhất](RESEARCH_PLAN.md) → [trạng thái](STATUS.md) → [kết quả Phase 2.1](research/08_PHASE21_RESULTS.md). [Phase 2](research/06_FEASIBILITY_VERDICT.md) là hồ sơ lịch sử.

**Hồ sơ sản phẩm V0.1:** [app/server.py](../src/scarx/app/server.py) → [app/state.py](../src/scarx/app/state.py) → PlasticCore/LifelongLearner. Frontend lịch sử ở [app/web](../src/scarx/app/web/), data cũ `local/mvp`. [ADR0005](decisions/0005-product-first-mvp.md) ghi phê duyệt Product-first trước khi thay bằng G1; không phải entrypoint hiện tại.

**Hồ sơ V0.2 LAB:** [neural/language.py](../src/scarx/neural/language.py), [app/pilots.py](../src/scarx/app/pilots.py) và [pilot-release.json](../artifacts/pilot-release.json) phục vụ regression/tái lập offline. [Run V2](../runs/20261108T190315Z-language-vision-pilot-v2-6a889e/report.md) giữ số đo language fail và digits diagnostic. Không được G1 nạp hai pilot này làm model đang chạy.

Roadmap/vision/gate nằm trong master. Architecture/metrics/design là tham chiếu; protocol nghiên cứu đã chạy không sửa thành roadmap cạnh tranh. Formal Phase2.2 còn là đề xuất, nhưng Product-firstMVP/code/UI và acceptance checks đã được phê duyệt; gate nghiên cứu không chặn tương tác/dạy thử.

| Thư mục | Nội dung | Quy tắc |
|---|---|---|
| `src/scarx/neural/` | ALIF, STDP, checkpoint | Không backprop; không truy xuất dữ liệu |
| `src/scarx/learning/` | Cập nhật có kiểm tra, replay, rollback | Phân biệt huấn luyện và đánh giá |
| `src/scarx/memory/` | SQLite, nguồn, truy xuất | Dữ liệu gốc trên đĩa, cache giới hạn |
| `src/scarx/runtime/` | C++ bridge, RAM monitor | Tách số đo khỏi ước tính |
| `src/scarx/evaluation/` | Bài thử, baseline, thống kê, báo cáo | Control backprop chỉ nằm ở đây |
| `src/scarx/interfaces/` | API văn bản, CLI | Không tự gọi cloud |
| `src/scarx/app/` | MVP localhost, task encoder, transaction và web UI | Prediction neural, rule verifier và memory phải phân biệt |
| `native/` | Mã C++ và cách build | Binary sinh ra nằm trong build/ |
| `configs/` | Cấu hình tái lập | Có version; lưu bản sao trong mỗi run |
| `experiments/` | Protocol đăng ký trước | Tên cố định, không sửa sau khi chạy |
| `data/fixtures/` | Dữ liệu nhỏ dùng kiểm thử | Có nguồn gốc; không chứa bí mật |
| `runs/<run-id>/` | Kết quả, log, metadata từng lần chạy | Không ghi đè; tra bằng runs/INDEX.md |
| `artifacts/` | Checkpoint và hồ sơ nguồn gốc | Tách model khỏi mã nguồn |
| `docs/RESEARCH_PLAN.md` | Kế hoạch nghiên cứu chính duy nhất | Tách đã triển khai/đề xuất; cập nhật theo phê duyệt |
| `docs/research/` | Kiến trúc, metrics và báo cáo nghiên cứu | Kết quả dẫn đến run; không sửa số liệu lịch sử |
| `docs/decisions/` | Nhật ký quyết định thiết kế | Có lý do và phương án bị loại |
| `tests/` | Kiểm thử hành vi, persistence, parity | Không thay cho benchmark năng lực |
| `scripts/` | Build, tái lập, thống kê | Chạy từ project root |
| `local/` | Dữ liệu cá nhân, cache thử nghiệm | Git-ignore; không tự công bố |

## Tìm nhanh

```powershell
rg --files src tests docs configs experiments
rg "elig_rec|modulation" src/scarx/neural
rg "gate|forgetting" runs --glob '*.json'
rg "NO-GO|PIVOT|GO" docs/research
```

Tên run: thời gian UTC + protocol + hậu tố ngẫu nhiên. `result.json` là dữ liệu máy đọc; `report.md` là tóm tắt cho người. `manifest.json` ghi seed, cấu hình, phiên bản môi trường, hash mã nguồn. Không dùng tên như `final2`, `new`, `test_latest` làm nguồn kết quả chính thức.

## Tìm cơ chế Phase 2.1

- `src/scarx/neural/plasticity.py`: stochastic ALIF, eligibility và importance, checkpoint/RNG.
- `src/scarx/learning/lifelong.py`: consolidation theo đồng hồ cập nhật, SI control tại readout.
- `src/scarx/learning/replay.py`: replay buffer có capacity cố định và RNG lưu được.
- `src/scarx/evaluation/phase21.py`: development selection, confirmation, thống kê.
- `src/scarx/evaluation/lifelong21.py`: stream không task ID, phản hồi nhiễu, selective learning.
- `artifacts/phase21-summary.json`: đường dẫn chính xác tới các run được trích trong báo cáo; các file hình nằm tại `artifacts/phase21-figures/`.

## Điểm mở và Phase 2.2 đề xuất

- [P0 kỹ thuật](RESEARCH_PLAN.md#p0--khép-an-toàn-kỹ-thuật-trước-consolidation-tự-động): `plasticity-v1`, whole-learner/replay rollback, evidence verification và peak RAM.
- [P1 representation/credit](RESEARCH_PLAN.md#p1--representation-và-credit-assignment) và [P2 readout/importance](RESEARCH_PLAN.md#p2--readout-stability-và-importance-theo-mức-hữu-ích) là ưu tiên nguyên nhân gốc.
- [P3–P5](RESEARCH_PLAN.md#p3--chuyển-nhiều-trải-nghiệm-thành-kỹ-năng): transfer, offline consolidation và tuổi stream; mở theo dependency/gate.
- [ADR0004](decisions/0004-master-plan-and-phase22-docs.md) ghi phạm vi docs-only. Không có Git local được khởi tạo, không di chuyển cấu trúc từ cập nhật này.
