# SCAR-X — BÀN GIAO NGHIÊN CỨU TOÀN DIỆN CHO CODEX

**Định dạng:** hồ sơ kỹ thuật, evidence ledger, forensic audit và nhiệm vụ thực thi.  
**Ngày chốt thông tin:** 11/11/2026 (giờ Việt Nam).  
**Repository:** https://github.com/AKANEMO22/SCAR-X  
**Người nhận:** Codex, với vai trò *independent engineering and scientific auditor*, không phải bên xác nhận kết luận của Grok, DeepSeek hay Gemini.  
**Độ tin cậy:** các số `R6A`, `R6B`, `R6D` được kiểm tra tĩnh qua GitHub và một số thống kê được tính lại từ `result.json`; **chưa có độc lập chạy lại Python/pytest/benchmark trong phiên lập báo cáo**. Các vòng cũ được ghi theo báo cáo trong repository, không tự coi là kiểm chứng độc lập.

---

## 0. LỆNH ĐIỀU HÀNH DÀNH CHO CODEX (đọc trước)

Bạn phải tiếp nhận SCAR-X như một dự án nghiên cứu neural SNN, không phải chiến dịch tối ưu điểm thi. Nhiệm vụ trước mắt là **xác minh tính đúng đắn của cơ sở bằng chứng**, bắt đầu từ R6D và các lỗi đo lường kế thừa, rồi đề xuất một vòng nghiên cứu mới chỉ khi protocol, evaluation và nguồn nhãn đáng tin cậy. Không tự tin lời của DeepSeek, Grok, Gemini hoặc báo cáo này. Mọi khẳng định phải truy ngược được đến mã thực thi, commit, dữ liệu đầu vào, metric implementation và raw measurements.

**Không được tự ý**: merge research branch vào `main`; sửa production learner/weight/checkpoint; xoá, ghi đè, sửa lại các `runs/` cũ; thay ALIF SNN bằng Transformer/LLM/RAG proxy; dùng BPTT/autograd/RTRL trong candidate; thay gate sau khi nhìn test; tăng neuron/model/dataset lớn để che lỗi; khai báo PASS khi gate thiếu/null; hoặc coi ridge/feature probe là bằng chứng hidden neurons đã học. Nếu cần thay đổi thuật toán ngoài đề xuất 30 đã được duyệt, hãy đưa ra protocol và xin duyệt trước. Có thể kiểm toán, tái lập, viết regression tests, sửa harness trong **nhánh nghiên cứu riêng**, và viết đề xuất protocol mới. Giữ dữ liệu `local/` của người dùng nguyên vẹn.

### Kết quả bàn giao bắt buộc

1. `BASELINE_INVENTORY.md`: HEAD/branch/commit, trạng thái worktree, run manifests, test environment, điểm không đồng nhất.
2. `FORENSIC_AUDIT.md`: claim-by-claim với link đường dẫn/line code, bằng chứng thô, counterexample, severity, verdict.
3. `tests/` và **evaluator độc lập** để tính lại raw prediction metrics; không gọi `summary` là oracle.
4. `PROTOCOL_LOCKED.md`: thiết kế và gates **commit trước bất cứ fresh confirmation nào**.
5. `runs/<unique-id>/`: immutable raw, per-seed, source hash, config, manifest, process-tree memory, commands.
6. `RESULTS.md` + `CLAIM_LEDGER.json`: `PASS / FAIL / INCONCLUSIVE / NOT_TESTED / INVALID_HARNESS`, không xóa các lỗi.
7. Git branch + final full SHA. **Không merge/deploy** khi chưa duyệt.

Nếu code không còn tái lập được đúng số lịch sử, ghi `REPRODUCTION_FAILED`, giữ nguyên lịch sử và tìm nguyên nhân. Không che bằng ngưỡng mới hoặc chọn seed khác.

---

## 1. SNAPSHOT GIT VÀ NGUỒN CHUẨN

### 1.1 Main/production

- Repo public: `AKANEMO22/SCAR-X`.
- `main`: `495dab2f4bac0180e4d1cd28674391f055ccf09d` tại thời điểm kiểm tra; commit merge release candidate PR #1.
- Trình chạy: `run_app.py` → `src/scarx/app/generation_server.py` → `src/scarx/app/generation.py` → `src/scarx/neural/unified.py` → `src/scarx/neural/core.py`.
- Model identity của app: `scarx-generation1`.
- Đọc theo thứ tự `AGENTS.md`, `README.md`, `docs/PROJECT_MAP.md`, `docs/STATUS.md`, `docs/RESEARCH_PLAN.md`; master plan v1.32 là nguồn quyết định/approval chính.
- App thật: một recurrent ALIF SNN, nền H48 lịch sử, prototype capacity có thể allocate 48–128; số neuron đã cấp trong một snapshot live có thể là 64. **Không được đánh tráo H48 nghiên cứu với allocated 64 ở live.**
- CPU-first, mục tiêu ngân sách RAM toàn process-tree **≤ 2,000,000,000 bytes**; một core chung text/digit vision; digit vision hiện là 8×8 (không nhận ảnh/screenshot tự nhiên); checkpoint SQLite/WAL, accepted/rejected teach guard, rollback, passive observatory.
- `src/scarx/neural/` không được import torch/autograd hoặc dùng backward toàn mạng. BPTT là **control độc lập** tại `src/scarx/evaluation/controls.py`.
- Các thí nghiệm từ nhiều branch có H48 frozen, feature map đọc ngoài serving: không được gọi chúng là model sản phẩm.
- Release RC được ghi nhận đã thử clean install và test suite (lịch sử báo cáo `278 passed, 14 skipped` sau các bản sửa; phải chạy lại tại SHA hiện hành trên môi trường thực tế). Những số `276/276`, `287 passed` ở các báo cáo nghiên cứu khác là môi trường/commit khác, không được gộp.

Nguồn: [AGENTS.md](https://github.com/AKANEMO22/SCAR-X/blob/main/AGENTS.md), [README](https://github.com/AKANEMO22/SCAR-X/blob/main/README.md), [RESEARCH_PLAN](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/RESEARCH_PLAN.md), [STATUS](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/STATUS.md).

### 1.2 Research branches đã hiện diện trên GitHub

| Nhóm | Branch | SHA chốt quan sát | Trạng thái kiến thức |
|---|---|---|---|
| Grok R1 | `research/grok/scarx-neural-round1-20261111` | `4cad9eb17f3660b241cefa21938ddee2359dd09b` | slow eligibility; no-go |
| Grok R2 | `research/grok/r2-dendritic-transfer` | `3bcf609a60ad66271c0429287eb902b899904d95` | DC-ALIF không transfer; STOP |
| Grok R3 | `research/grok/r3-temporal-stress` | `211380fecdcd05966ec78bb7686f54fa8179c11a` | temporal stress; STOP |
| Grok R4 | `research/grok/r4-online-readout-mechanism` | `8f200b988788dfb735a763fcbe085bf4546f731f` | readout mechanism; STOP |
| Grok R5 | `research/grok/r5-independent-audit` | `b0480c9cc02a3f35ba994480df2a7d984b6f89f1` | sửa cue bug, candidate vẫn STOP |
| Grok R6 | `research/grok/r6-causal-readout-repair` | `c39e9db7771582d9933af402a209917328b71119` | tái lập R5B, noclip STOP; runner audit cần sửa |
| DeepSeek R1 | `research/deepseek/scarx-neural-round1-20261111` | `61eeb77cff2b371ec314d3e5bcc9e6fd29d00eef` | head-conditioning không đủ |
| DeepSeek R2 | `research/deepseek/r2-readout-learnability` | `7d590f768b36eb9d60e5c9433f1966e4e620cff3` | spikesum tốt trên sim_xor; scoped GO |
| DeepSeek R3 | `research/deepseek/r3-temporal-substrate` | `42ed574264816e3a246a984a274f903f7eecdbfa` | 3-seed screen PASS, 8-seed confirmation FAIL |
| DeepSeek R4 | `research/deepseek/r4-relational-representation` | `10e1c7df9a585566db04e754af9ba7b419d8bab1` | bilinear query-conditioned diagnostic; corrected Question A |
| DeepSeek R5 | `research/deepseek/r5-independent-relational-binding` | `d6c609df6f90fc8df96fb3de424dd54388f8dc59` | two-value binding, both_A gate FAIL |
| DeepSeek R6A | `research/deepseek/r6-learning-scaling` | `ceb98e33b40e8b47c6788b7d483e8b52a97d3351` | data-scaling signal; defective harness; PR #11 draft |
| **DeepSeek R6D** | `research/deepseek/r6d-audit` | **`4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893`** | new corrected runs, non-integrated gates; PR #12 **draft** |
| Antigravity/MTB | `research/antigravity/scarx-neural-round1-20261111` | `5e39c8f8fb3f316c2713d1c2f6d5edea0598882f` | multiple timescales PIVOT |
| Gemini/DC-ALIF | `research/antigravity-gemini/scarx-neural-round1-20261111` | `81641f773145f5c441b5ad3b0f5771f645a5cbd6` | scoped success, generalized failure; **người dùng cho biết Gemini vẫn đang làm** |
| Product RC | `cursor/scarx-release-candidate` | `854250b43787d2abcecf4fa3ec4fc05118f22b4b` | RC branch HEAD khác commit đã merge vào main |

**Lưu ý:** GitHub HEAD là snapshot lúc kiểm tra. Codex phải `git fetch --all --prune`, kiểm tra lại trước khi thực hiện. Tồn tại branch không đồng nghĩa đã merge. Không đoán rằng Gemini đã kết thúc chỉ vì có report; người dùng nói Gemini vẫn đang làm. Không xung đột/cherry-pick công việc đang mở của Gemini.

### 1.3 Quyền nghiên cứu đã cho và chưa cho

ĐÃ duyệt: MVP/product engineering thường quy, guarded interactive teaching; nghiên cứu **đề xuất 30** (one-step `W_out.T @ error` + own-neuron forward eligibility **trong prototype tách biệt**); P0 toán/parity/codec/rollback cho readout usefulness.  
CHƯA duyệt: P2 usefulness pilot, production import learning rule mới, đại huấn luyện 10M+ tokens, tăng mô hình lớn, đổi encoder/tokenizer/vocab/main vì kết quả preview, tự động deploy DC-ALIF/MTB.  
Không được lấy sự cho phép xây MVP hoặc một P0 kỹ thuật để suy ra được phép chạy mọi bài P1/P2 hoặc đổi production.

---

## 2. KIẾN TRÚC, TÍN HIỆU HỌC VÀ ĐỊNH NGHĨA PHÉP ĐO

### 2.1 Production learning và khác biệt với research

`SpikingCore` / `UnifiedCore`: recurrent ALIF state gồm membrane voltage `v`, spikes `z`, adaptation `a`, filtered features `f`, sparse recurrent weights `w_rec` và input `w_in`; head/output synapses `w_out`. Historical `feature_decay=0.25` cho filtered spikes; `hidden_lr`/`readout_lr` khác nhau và pipeline nhiều bước chịu ảnh hưởng của frontend/decoder/guards. Historical production hidden learning dùng scalar reward-modulated local traces/heuristic, không phải BPTT/e-prop chính xác. Research R5/R6 dùng **hidden frozen H48** và readout riêng có thể dùng normalization, bilinear map, training budget hoàn toàn khác serving. **Tuyệt đối không gọi hai hệ là cùng một model đang học.**

Phân tích cần tách các ranh giới nguyên nhân:

- **Input information**: RGB→grayscale, token collisions, context serialization, query cue có thực sự tới neuron không?
- **Representation function class**: state có mang thông tin? linear probe/ridge có đọc được không? có cần cue × state hay dendritic coincidence?
- **Readout optimization**: cùng feature có đọc được với delta head theo single pass, normalization, clipping, order, projection?
- **Hidden credit assignment**: hidden synapses có thay đổi **đúng hướng** và làm feature mới hữu ích hơn frozen/shuffled/zero không?
- **Continual stability**: học cái mới có mất cái cũ không, do readout overwrite, state, hidden drift, insufficient features?
- **Deployment utility**: liệu có output đúng câu/ảnh mới, latency, guard, recovery, hoàn toàn không dựa vào teacher proxy?

### 2.2 Metrics tuyệt đối không nhầm

| Metric | Định nghĩa | Lỗi thường gặp |
|---|---|---|
| `direct` | `mean(pred == y)` trên các row held-out | Có thể bị baseline first-key ≈0.75 thổi lên khi `vA==vB` |
| `both_A` | trên counterfactual sibling pairs có `vA != vB`, **cả query A và B phải đúng** | Không thay bằng mean direct hoặc gộp sibling như hai seed độc lập |
| `both_B` | trên pairs `vA==vB`, cả hai đúng; nhãn bất biến | Đòi label flip cho case B là lỗi chấm |
| `needB` | query B khi `vA!=vB` (thách thức first-key shortcut) | Không đo thì dễ nhầm đọc key A thành hiểu quan hệ |
| `ridge` | linear diagnostic fitted offline trên frozen features, kiểm information accessibility | Không phải upper bound toán học, không phải learner online, không chứng minh hidden learning |
| `hidden-probe gain` | probe/refit cùng budget trên frozen vs learned hidden, holdout | Refit ridge có access label khác production, cannot claim skill |
| `correct - frozen` | paired improvement của arm hidden learning trên frozen hidden, cùng data/update/compute | Nếu frozen mạnh từ đầu, accuracy cao không chứng minh học |
| `correct - shuffled/zero` | tính specificity, causal effect hướng tín hiệu học | Phải cân bằng optimizer moments và update budgets |
| `mean / worst forgetting` | giảm từ mức kỹ năng đã đạt xuống cuối stream, theo context và seed | Ít quên khi chưa học được cũng không đạt gate |
| `train exact` vs `free utterance` | token training teacher-forcing vs tự sinh toàn câu | Không suy train 99% thành dev/chat 99% |

**First-key baseline không phải 50%** cho generator 2 independent stored bits: nó ≈75%. Shuffled labels và query-only ~50%. `both_A` loại bớt shortcut. Với R5B, q=A và q=B conditional accuracies lần lượt ~0.812/0.804 nên conjunction ≈0.668, không phải một phép đo tính trí tuệ tổng quát.

### 2.3 Quy chuẩn bằng chứng

- `REPORTED`: tệp report nói vậy, chưa đọc raw hoặc code.
- `SOURCE_CHECKED`: đối chiếu code path/logic tĩnh.
- `RAW_RECALCULATED`: tính lại trực tiếp từ số từng seed/raw `trials`, không chỉ summary (đã thực hiện một số metric R6B/R6D).
- `REPRODUCED`: **chỉ** khi Codex thực sự chạy lệnh trên checkout sạch và xác minh hash/output.
- `INDEPENDENT_CONFIRMED`: evaluator/implementation độc lập, fresh seeds prereg, kết quả thống nhất.
- `INVALID_HARNESS`: có bug khiến phép đo không đúng cái metric tuyên bố.
- `NOT_TESTED`: chưa chạy, không được điền PASS từ suy luận.

Không đưa mọi số trong tài liệu này lên mức `INDEPENDENT_CONFIRMED`.

---

## 3. LỊCH SỬ CÁC CHIẾN DỊCH CHÍNH TRÊN `main` (P0/P1/P2/G1)

*Bảng là sổ dẫn nguồn để Codex đi theo tất cả lịch sử. Các tệp `docs/research/01–06` thiên về nền tảng/thiết kế; những con số sau bắt đầu từ các protocol/run thực nghiệm được liên kết trong `docs/research/08–44`. Không phải mọi đường nghiên cứu đều cùng seed, sampler, dataset, encoder, learning rule hay compute; cấm lấy điểm từ hai hàng bất kỳ làm causal matched comparison.*

### 3.1 Từ synthetic temporal learning đến P1/P2

| Vòng/tệp | Thí nghiệm / dữ kiện quan trọng | Verdict đúng phạm vi |
|---|---|---|
| `08_PHASE21_RESULTS.md` | 303 trials; lifelong/paired score, 48 neuron, 2 GB. Stream 1,280 experiences, 8 seeds. Nhiều điều kiện acquisition khoảng 28–30%, final khoảng 27–29%; các binding task có shortcut bag-of-values 75%. | `PIVOT`, không scale; một số kỹ thuật ổn, learned skill/retention không đạt. |
| `09_MEMORY_LEARNING_MEASURES.md` | Định nghĩa `H(q)`, `C(q,I,B)`, retention, learning-to-criterion; proposal protocol, **không phải số benchmark thực**. | Không được tự gán memory capacity hoặc IQ từ spike count/context buffer. |
| `10_DIALOGUE_DATA_AND_ONLINE_LEARNING.md` | Data provenance, guarded human correction, anti-self-label, persistence/rollback; normal photo QA chưa có encoder đường học. | Engineering feedback được duyệt; neural acquisition chưa chứng minh. |
| `11_KNOWLEDGE_LOAD_RETENTION.md` | 16-item arithmetic load: full local 14.58%, readout-only 20.83%, replay 31.25%; paraphrase 0%, overload/retention không đạt; Qwen reference 160/160 trên số học không âm nhưng khác pretraining/RAM. | Không có transfer/retention hữu ích. |
| `12_ADAPTIVE_COMPUTE_AND_IDLE_COGNITION.md` | Ba chế độ cùng một core khác với ba tầng neuron vật lý. Autonomous reflection/ACT chưa được xác nhận; idle ticks ≠ suy luận. | Thiết kế, không claim năng lực. |
| `13_INPUT_AND_LOCAL_LEARNING_AUDIT.md` | Input collisions trước neuron; hai hidden learning families local-target-rate (~-1.43pp order, +0.91 XOR, +3.78 strict) và random-feedback (+1.43 order, -1.56 XOR, +1.95 strict) không đạt. Normalized output head: order ~95.53%, XOR ~86.99%, binding 61.74%. | Lợi ích conditioning thuộc readout, không phải hidden representation; P1 local candidates STOP. |
| `14_EVENT_DIALOGUE_AND_QWEN_REFERENCE.md` | Word decoder không tự sinh exact đúng trên các arm; sample outputs sai/cụt; Qwen 24 reference có 11 truncations. | Không có normal chat. |
| `15_BYTE_CODEC_AND_WORKING_MEMORY.md` | UTF-8 codec giải quyết khả năng biểu diễn ký tự nhưng exact free response vẫn 0%; isolated BPTT/Adam đối chứng chỉ ~23.18% dev trong một budget, snapshot không sửa. | Codec/working state ≠ học ngôn ngữ. |
| `16_REAL_RGB_GROUNDING.md` | RGB32×32, 640 train/160 dev, H48 frozen; readout raw/normalized khoảng 18.96%/20.42% (phân loại) dưới gate; không photo QA. | `PIVOT`, chưa vision grounded. |
| `17_LOCAL_CLASSIFIER_ADAPTATION.md` | Local auxiliary classifier: order 91.80% nhưng gain hidden +2.73pp; XOR 80.99% gain -0.52; binding 55.60% gain -0.65. | `PIVOT`, không đạt representation gain >=5pp. |
| `18_LOCAL_CLASSIFIER_FINAL_AND_LEARNABILITY.md` | Candidate probe gain order +2.60pp (CI crosses 0), XOR +3.78pp, strict +0.26pp. Isolated BPTT control đạt strict 91.67% vs local 65.49%, nhưng credit signal/optimizer/coverage khác. | `NO-GO` local classifier family, không suy local SNN bất khả. |
| `19_CONTROL_OPTIMIZER_AND_COVERAGE.md` | Diagnostic tách BPTT, optimizer, số synapse được update; strict binding một control lên 82.42%, các control khác ~61–63%. | Không được quy toàn bộ gap chỉ cho backprop; `DIAGNOSTIC_ONLY`. |
| `20_SCORE_ELIGIBILITY_AND_VARIANCE.md` | `paired-score v2`, 8-seed P1: hidden probe gain order +18.33pp CI [14.65,22.53], XOR +20.00pp CI [15.06,24.68], strict -1.29pp; direct order 91.99%, XOR74.56%, strict49.56%. | **`GO_P2_RESEARCH_ONLY`** trên order/XOR, binding FAIL, không deploy. |
| `21_P2_RETENTION_AND_CUE_INFORMATION.md` | 20 context, 3 seeds, 5,760 updates/arm; `full+reservoir`: acquisition 57.64%, final **34.74%**, mean forgetting **29.09pp**, worst context **71.88pp**; frozen+reservoir final28.67%, forgetting27.04pp. Offline-joint cũng <65%; cue engineered outer-product reference 100% nhưng không neural. | **P2 FAIL** (final ≥65%, mean forget≤3pp, worst≤5pp); readout overwrite lẫn thiếu representation. |
| `22_INFERENCE_AND_TIMESCALE_PROPOSAL.md` | 60 immutable views trên checkpoints: hetero time constants không vượt control mean-tau; best probe ~54.11% <65%; stochastic/MC và diagnostic inference views. | `STOP_TIMESCALE_VIEW_SCREEN_V1`, không đào tạo/deploy. |

**Học từ giai đoạn này:** có thể tăng điểm đọc tín hiệu mà hidden không học; hidden learning từng có tín hiệu trên 2 synthetic tasks, nhưng chưa giữ được kỹ năng sau interference. Đó là hai câu hỏi khoa học khác nhau.

### 3.2 Generation 1 language, vision và credit assignment

| Tệp | Kết quả quan trọng | Verdict / giới hạn |
|---|---|---|
| `23_REAL_DIALOGUE_STATE_AND_CURRICULUM.md` | Public/current cloned checkpoints khác nhau rất mạnh; curriculum/repair không đạt acquisition; capped guarded teaching; `CURATED_SCREEN_FAILED_KEEP_LIVE`. | Không import trial vào live; không lấy accepted updates làm hiểu câu. |
| `24_REPRESENTATION_DECODER_AND_SEARCH.md` | Raw/dev representation và decoder đều yếu; search width4 tăng dev exact 12.90→21.51% nhưng chưa qua gate; state view observer tối đa local 39.73%, ridge ~50.05. | `STOP_G1_WORD_DECODER_SEARCH`, `STOP_STATE_OBSERVABILITY_SCREEN_V1`. |
| `25_WORD_SCORE_TRANSFER_PROPOSAL.md` | Score-rule transfer to words: frozen dev exact 3.23%, full dev exact 3.23%; teacher token 52.24 vs frozen45.43, free utterance gain 0pp. | `STOP_WORD_SCORE_TRANSFER_V1`; token gain≠câu đúng. |
| `26_FIXED_INPUT_NEURAL_TRANSFER.md` | 27 trials, legacy/densebits/signed bucket encoders. Dense full dev7.53%, frozen8.60%; gain không ổn. | `STOP_WORD_SCORE_INPUT_TRANSFER_V2`; đổi input không đủ. |
| `27_ISOLATED_WORD_LEARNABILITY.md` | 6 isolated controls; densebits128: local score dev **7.53%** vs surrogate BPTT control **44.09%**, frozen8.60%; train local12.50%, BPTT91.67%. Không equal-compute, không production BPTT. | Chứng minh khoảng cách học trong budget, chưa quy một nguyên nhân. |
| `28_CREDIT_SIGNAL_NOISE.md` | 144 conditional inputs ở cùng checkpoints; phân tích nhiễu reward×eligibility, signal direction/variance. | Chẩn đoán, không learner mới. |
| `29_OBSERVED_RESPONSE_ASSOCIATION.md` | 12 trials; paired response association dev exact2.15% bằng frozen, prior thấp hơn frozen, không vượt gate. | `STOP_OBSERVED_ASSOCIATION_V1`. |
| `30_PROPOSED_LEARNING_SIGNAL_BOUNDARY.md` | Người dùng **đã duyệt** prototype readout error one-step × forward local eligibility; thừa nhận `W_out.T @ error` không thuần biological local. | Approval cho nghiên cứu hẹp, không deploy/BPTT candidate/scale. |
| `31_VISION_QUESTION_PATH.md` | 240 inference thật; câu hỏi bị bỏ trước neural vision path. | Input-path bug/omission, chưa có vision grounding. |
| `32_JOINT_VISUAL_WORD_READOUT.md` | 9 arm, frozen hidden; full RGB+question dev ~18.88%, no-image17.97%, no-question21.50%; không thể quy score cho hiểu RGB & question. | `STOP_VISUAL_WORD_READOUT_V1`. |
| `33_VISUAL_WORD_LEARNABILITY_CONTROL.md` | Isolated global BPTT control train48.43%, dev25.67%, novel27.44%, dưới gate; frozen dev18.88%. | `CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET`. |
| `34_VISUAL_INFORMATION_PATH.md` | Raw RGB+question semantic probe ~50.22% dev, learned 48-feature trace kém; token/feature/readout information bottlenecks, clip ít. | `VISUAL_INFORMATION_DIAGNOSTIC_ONLY`. |
| `35_DIRECT_READOUT_ELIGIBILITY.md` | 45 trials, 3 dev seeds. Correct accuracy order98.44%, XOR91.80%, strict63.41%; frozen94.92/81.25/62.24; XOR probe +10.94pp; strict probe -1.95pp. | `STOP_DIRECT_READOUT_ELIGIBILITY_V1`: chỉ XOR qua, không 8-seed confirmation. |
| `36_DIRECT_WORD_BRIDGE.md` | 15 trials, 3 seeds, 1,024 repeated training responses: correct **train99.48%**, **dev free utterance35.94%**; frozen dev16.67%, shuffled17.19%; BPTT control36.46%. Reset-head probe CI lower=0. | `STOP_DIRECT_WORD_BRIDGE_V1`, không normal chat/deploy. |
| `37_DIRECT_VISUAL_BRIDGE.md` | 21 trials; correct **macrodev25.10%**, novel25.76%; frozen15.05%, shuffled21.32%, BPTT26.30%; distractors/ablation không qua gate, reset-head probe âm. | `STOP_DIRECT_VISUAL_BRIDGE_V1`, chưa joint RGB QA. |
| `38_WORD_TRAINING_OBJECTIVE_AUDIT.md` | Train exact99.48% và dev normalized exact32.81% ở subset đo khác; later-word teacher prefix77.21% vs neural prefix47.87%, gap **29.34pp**. Incorrect question/no-question vẫn tạo nhiều đúng token. | Shortcuts teacher forcing/context, weak autonomous generation. **Không so trực tiếp dev32.81 vs 35.94 như mô hình suy giảm**, metric khác. |
| `39_LOCAL_CREDIT_ALIGNMENT.md` | One-step dev endpoint ΔNLL đúng hướng ~63.54%, nhưng lợi ích final NLL nhỏ/không chắc. Adam moment caveat: shuffle/update sign không chuẩn norm-equivalent; `0.995×` ≠ bit-identical. | Credit signal có tính hướng hẹp, không chứng minh skill/generalization. |
| `40_DEEPSEEK_REVIEW.md` | Audit guard, rollback, readout overwrite trên clones. 258 tests ở snapshot; accepted vs acquisition; 24 interventions chỉ cho thấy trong context hẹp w_out đủ gây quên một cặp, không chứng minh hidden luôn vô dụng. | GO correctness fixes, PIVOT learning/no scale. |
| `41_ADAPTIVE_CURRICULUM_AND_P2_PROPOSAL.md` | New source/prompt data chuẩn bị, không tự đưa vào train; usefulness learner mới **chưa được duyệt pilot**. | Proposal/data prep only. |
| `42_READOUT_USEFULNESS_P0.md` | 276/276 P0 kỹ thuật; finite difference/parity, matched norm, guard, rollback/restart; 95MB-ish RSS P0. | P0 engineering PASS, **P2 usefulness chưa duyệt/chưa chạy**. |
| `43_ADAPTIVE_DATA_GROUNDING.md` | Reference chat/vision provenance, DATA_GROUNDED_NOT_TRAINING_READY; majority blind vision QA 38.89% chỉ là một baseline, không chứng minh mọi score cao hơn đều nhìn pixel. | Data only, không train. |
| `44_P1_HANDOFF_EVIDENCE.md` | Handoff cho Claude/Gemini; audit input color loss ở `PilotLab.pixels(): convert('L')`; 18/18 cặp khác màu bị gộp; 670/671 answers thiếu ít nhất 1 output token; blind baseline 38.89%; read-only, không đổi live, chưa duyệt P1 candidate mới. | Input bottlenecks **có trước neural learning**; không được suy năng lực từ giới hạn trần. |

Nguồn gốc từng hàng: `https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/<SỐ>_<TÊN>.md` và `runs/` được link trong chính tệp đó. Đối chiếu theo đúng URL **tại SHA main cố định**, không tự thêm số khi nguồn không nói. Các mốc `01–06` còn có `01_RESEARCH_LANDSCAPE.md`, `02_SCAR_X_ARCHITECTURE.md`, `03_CRITICAL_REVIEW.md`, `04_ACI_CEI_METRICS.md`, `05_EXPERIMENT_DESIGN.md`, `06_FEASIBILITY_VERDICT.md` để xem các lý thuyết/metric và quyết định nền; **không** được nhầm thành R2/R6 experiments đã chạy.

---
## 4. CHIẾN DỊCH ĐA AGENT R1–R6D: THỰC NGHIỆM, GIẢ THUYẾT VÀ BÀI HỌC

Nhóm này nằm trên **research branches tách biệt**, không tính là cải tiến production. Khi diễn giải số liệu, cần phân biệt **bài synthetic khác nhau** (`sim_xor`, `xor_separated`, `order`, `pair_binding`, `strict_binding`, `independent key/value query`), representation/normalization khác nhau, 3 dev seeds hay 8 confirmation seeds. Không ghép các điểm chỉ vì cùng tên `ALIF`.

### 4.1 R1A — DeepSeek: normalization không tự sửa bài đọc tín hiệu

Nguồn: [DeepSeek R1 corrected report-v2](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/deepseek/report-v2.md).

- Bài `sim_xor`, frozen ALIF H48, `feature_decay=.25`; đầu ra raw/z-score/RMS. Ở seed 71001: ridge trên same production features ~0.6300, raw head0.5167, zscore0.5367, RMS0.5200; chênh còn ~9–11pp.
- Cùng frozen spike trajectory, ridge với `decay025`0.6300, `spikesum`0.9433, `bank5`0.9467. Đây là bằng chứng **mất thông tin khi tóm tắt thời gian**, không phải chỉ thiếu gradient/clip.
- Báo cáo có **đính chính**: từng nói precondition head≥0.65 không thể xảy ra vì ridge trần .63, nhưng ở seed 71007 ridge decay lên .6967; không được nói “mốc bất khả thi” theo một seed.
- H1 head-normalization không đạt gate; Stage B không mở. Một quan sát frozen vs hidden refit nhỏ ~3.8pp chưa đủ giải thích do head có train loss > chance.

### 4.2 R1B — Grok: slow eligibility

Nguồn: [Grok R1 report](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/grok/report.md).

- Frozen fast feature ridge0.4844, slow trace0.9375 trên delayed order. Slow head after training correct0.957 vs slow frozen0.896; shuffled0.914. Initial CI lower cho correct−frozen ~-0.00010, không qua gate +5pp.
- **Lỗi split:** một “confirmation” dùng lại dev seed 71004; bị phát hiện trong Codex review lịch sử. Amendment chạy seed 71005, mean hidden gain0.0273, CI lower **−0.00289**; vẫn no-go.
- Bài học: frozen slow memory có thể làm điểm cao trước khi học; kiểm tra split/seed rất quan trọng.

### 4.3 R1C — Antigravity/MTB: multi-timescale readout

Nguồn: [MTB report](https://github.com/AKANEMO22/SCAR-X/blob/research/antigravity/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/antigravity/report.md).

- Đổi **readout temporal receptive field**, giữ nguyên hidden ALIF và hidden rule. Một decay0.25×48 vs 5 decays(.10,.25,.50,.75,.90)×48 =240 features.
- 297 dev trials, 11 arms × 3 tasks × 3 seeds × 3 learning rates; P0 30/30; không mở confirmation vì screen fail.
- Matched dimension contrast MTB−prod_x240 **+3.60pp** CI [-1.30,+8.77]; MTB−single-timescale same-width +4.90pp CI [-0.22,+10.42]; MTB−shuffled-signal +0.65pp CI [-0.95,+2.30]. Không chứng minh cơ chế thắng.
- Hidden contribution `MTB − MTB_frozen = +0.43pp` CI [-0.91,+1.87]. `mtb_only_slow` order65.89% gần `mtb`68.88%; XOR/binding ~chance.
- Có **lỗi aggregator** lịch sử: gate `None` bị coi là vacuous True, grouping contrast trên learning rate (thay vì seed), thiếu key summary, spike-hash so sai arm. Đã sửa/report lại nhưng là dấu hiệu cần Codex audit framework gates chung.
- Verdict **PIVOT**, hidden not shown to learn.

### 4.4 R1D — Gemini/Antigravity: DC-ALIF dendritic coincidence

Nguồn: [DC-ALIF report](https://github.com/AKANEMO22/SCAR-X/blob/research/antigravity-gemini/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/antigravity/report.md). **Người dùng nói Gemini vẫn đang làm; không coi bản hiện có là final campaign**.

- Prototype riêng thêm slow dendritic trace không reset, quadratic cross-synaptic coincidence và bilinear dendritic–somatic gate, đọc bằng một head. One-step `W_out.T @ error` + neuron-local forward eligibility; không BPTT candidate, nhưng **one-step error không hoàn toàn biologically local**. `BPTT_control` cô lập.
- Formal `strict_binding` 8 seeds: correct **98.78%**, frozen75.22%, shuffled85.57%, no_gate62.43%, BPTT98.85%; correct−frozen **+23.56pp** CI [18.87,29.27]. Task `xor` correct98.14% vs frozen85.84%.
- **Regressions nghiêm trọng:** `order` correct64.04 vs frozen82.52% (−18.48pp, CI âm); `pair_binding` post-hoc correct50.51, no_gate78.39% (gate gây hại). Scope là order-3 structure, không general relational binding.
- **P2 20 contexts:** usefulness final56.13%, mean forgetting21.89pp, worst54.95pp (gates final≥65, forgetting≤3/≤5 FAIL). Không có continual learning đạt chuẩn.
- Protocol-v1 có **Amendment 1 post-hoc** và các arm registered chưa đo; cần đánh dấu deviation, không gọi preregistered full PASS. Grok R2 audit sau đó xác nhận sai khác docs-vs-executed parameter: prose leak.88 nhưng code .85, feature decay .92 và các hệ số dendritic không trùng; cần audit cả hai.
- Tại thời điểm chốt, SHA branch `81641f...`, không assume Gemini không đẩy commit tiếp.

### 4.5 R2A — DeepSeek: frozen substrate vs same online head

Nguồn: [DeepSeek R2A](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r2-readout-learnability/docs/research/collab/scarx-neural-round1-20261111/deepseek-r2/report.md).

- 8 fresh seeds, same `lr=.08`, clip=.025, hidden frozen, `sim_xor`: online head on `decay025` **0.5063** vs `spikesum` **0.9712**; ridge `decay025`0.7036, `spikesum`0.9719. `spikesum − decay025` +46.48pp CI [+43.31,+49.27].
- `bank5` online0.9243, `bank5_proj48` online0.5793 mặc dù ridge0.8831: số chiều/information trong frozen features **không** tự bảo đảm một online delta learner tìm được nghiệm.
- GO chỉ cho bài `sim_xor` và same-learning-head, **không phải** head không bao giờ là bottleneck. Grok R3B tìm thấy R2A screen có `S5_no_leakage: True` literal, không đủ làm leakage gate; phải audit.

### 4.6 R2B — Grok: DC-ALIF không transfer

Nguồn: [Grok R2B](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r2-dendritic-transfer/docs/research/collab/scarx-neural-round1-20261111/grok/report-r2b.md).

- Tính derivative được kiểm trên anchored smooth surrogate **không phải exact gradient của hard spike**. Discrepancy giữa documented DC-ALIF equations và actual coefficients/`coact`/`q` cần đối chiếu.
- Stage2 (3 seeds): `order` correct0.586 vs frozen0.747; `strict_binding` correct0.815 vs frozen0.685; `pair_binding` correct0.500 vs `no_gate`0.810, one-compartment0.966. No transfer, order regression; screen STOP.
- Lesion trên checkpoint strict_binding: xóa coincidence current/quadratic/plateau kéo về chance, trong khi forcing gate open vẫn 1.00 trên một seed; tức architecture may dominate, trained gate not necessarily necessary. Đây là lesion hẹp, không general theorem.

### 4.7 R3A — DeepSeek: temporal-bins screen thất bại confirmation

Nguồn: [DeepSeek R3A](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r3-temporal-substrate/docs/research/collab/scarx-neural-round1-20261111/deepseek-r3/report.md).

- `temporal_bins_3`, readout 144 dims, frozen H48. 3-dev screen pass với `xor` gain +7.4pp và order gain +16pp.
- 8-seed confirmation: `order`1.000, improvement +19.63pp CI [+16.21,+22.85]; `xor`0.6736, improvement **+0.34pp** CI [-3.78,+4.10]; `strict_binding`0.5007, regression **−4.71pp** CI [-7.23,-2.12]. STOP S2/S3.
- Counts alone discard some order relation, time bins help order but may damage query binding. Không tự chọn task thắng để claim general.

### 4.8 R3B — Grok: temporal stress và việc decoder có thể không đọc được code

Nguồn: [Grok R3B](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r3-temporal-stress/docs/research/collab/scarx-neural-round1-20261111/grok/report-r3b.md).

- Reproduced R2 screen: decay ridge~0.760, online~0.485; sum ridge~0.980, online~0.983.
- Order reversal: decay frozen **ridge** paired-flip 0.898–1.000 trong khi normalized **head** chỉ 0.102–0.180; bank5 head ~0.992–1.000. Phải tách linear decodability khỏi head learnability.
- Query permutation: ridge/head trên decay/spikesum/bank5 ~0.5–0.6; không đủ query binding. Random-label one screen seed .621 > registered .60, thêm 32 shuffle mean .485 max .590, nhưng **không xóa gate fail lịch sử**. Corrected distractor test dùng **paired noise**, vì trước đó thay cả background noise giữa arms.
- STOP, không train DC-ALIF.

### 4.9 R4A — DeepSeek: information path và corrected target definition

Nguồn: [DeepSeek R4A](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r4-relational-representation/docs/research/collab/scarx-neural-round1-20261111/deepseek-r4/report.md).

- Đầu tiên `R1 pre-state` ridge0.472 trên **query label** (không có query ở t=9); đã nhầm với khả năng lưu stored values. Later **correction** đo đúng value of key A/B pre-query: **0.9987/0.9987**, spike trajectory1.000. Không được trích bảng cũ để nói dữ liệu không được lưu.
- Query-conditioned bilinear map `φ(s,q)=[s;q;(Ps)·q_A;(Ps)·q_B]` dùng **P fixed random** và cue lấy từ input; ridge0.9928, online head0.9316 (3 dev seeds), trong khi các unconditioned map ~0.53–0.58 head.
- Later controls: matched-width no-interaction post-state ridge0.7637, head0.5508; cue shuffled ridge0.7708; zero query tick + supplied true cue ridge0.9961; zeroed query state alone ridge0.4668. Các controls **post-hoc** không được gọi prereg confirmation.
- Vì R4A generator làm vA/vB complementary, nó mang về bản chất 1 bit, không giải được hai independent facts. Cần R5B.

### 4.10 R4B — Grok: readout floor và small-variance coordinate

Nguồn: [Grok R4B](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r4-online-readout-mechanism/docs/research/collab/scarx-neural-round1-20261111/grok/report-r4b.md).

- Trên `order`, decay frozen ridge .949–1.000 nhưng online head paired-flip0.102–0.180, train accuracy~0.52–0.58. Geometry: median feature std **0.0042**; nhiều discriminative coords có sd<0.05 variance floor; effective rank~9.3; condition number~1.5e7.
- Noclip/nofloor/nonorm đơn yếu: order gain âm, một số division by zero; thay training order thay kết quả đáng kể. Không có candidate được promoted, STOP.

### 4.11 R5A — Grok: shift cue bug, order floor fix không chuyển nhiệm vụ

Nguồn: [Grok R5A](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r5-independent-audit/docs/research/collab/scarx-neural-round1-20261111/grok-r5/report.md).

- R4A temporal shift reported .6394 **do query cue đọc tick10 khi query đã chuyển tick11**, không phải evidence R4A không chịu được shift.
- Corrected replay F2 shift internal distribution0.982; F4 clean→pure-shift0.910. Giữ hồ sơ old fail và đánh nhãn corrected replay.
- Variance-safe head order paired flip~1.000 vs normal~0.2, nhưng query giảm~4.2pp vượt margin3pp; simultaneous XOR không đổi. STOP cross-task candidate. Chẩn đoán floor có ích cho order riêng, không nên thay serving head.

### 4.12 R5B — DeepSeek: independent key/value binding

Nguồn: [DeepSeek R5B](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r5-independent-relational-binding/docs/research/collab/scarx-neural-round1-20261111/deepseek-r5/report.md).

- Generator vA/vB **độc lập**; first-key null **0.7522**; 3-dev storage probes pre-query vA0.9967, vB0.9974, mixed-only binding0.9947. Cẩn thận: probe đọc frozen state, không proof synaptic learning.
- Dev screen 9/9 PASS, `R5_original` query-conditioned online direct0.9635, `needB`0.9254, ridge0.9948; case A pair both0.8419.
- 8 fresh seed formal confirmation: **clean direct0.8989**, **both_A0.6676 (<0.70 gate)**, `both_B0.9809`; stress distractor direct0.8208, both_A0.4536; noise direct0.8174, both_A0.4512; reversed writes direct0.9221. **STOP**.
- Historical initial shift had zero-pad no-op under zero-initialized recurrent network, discovered/fixed; commit invalid run as `INVALID_HARNESS` and correct run. Lặp lại lesson: test perturbation thực sự thay state và giữ nhãn.

### 4.13 R6B — Grok: reproduced R5B nhưng candidate noclip thất bại

Nguồn: [Grok R6B report](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/docs/research/collab/scarx-neural-round1-20261111/grok-r6/report.md), [raw JSON](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/runs/grok-r6-causal-v1/result.json), [runner](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/scripts/grok_r6_causal.py).

- Mình đã đối chiếu **content và Git blob SHA** của `r5b_probe.py`/`r5b_independent.py` trong nhánh Grok với `r5_probe.py`/`r5_independent.py` gốc ở DeepSeek R5B; hai cặp **trùng nội dung**.
- Đã tính lại mean từ 8 `reproduction[]` raw seed: `head_both_A=0.6676171928664432`, `ridge_both_A=0.9858055982735111`, `qA_given_differ=0.811903`, `qB_given_differ=0.803728`, product0.65255 gần joint0.6676. Đây là đối chiếu số học, **không** phải independent runtime reproduction.
- Candidate `NoClip` bỏ per-step clip, 3 screen seeds paired gains `[+0.06349, −0.07080, +0.03676]`, mean **+0.00982**; CI lower **−0.07080**, protected order paired flip giảm **0.1875**. STOP, không mở new confirmation.
- Projection sensitivity tại seed98001: projection draws1234/7/99 cho both_A .590/.746/.705 (một seed, không cherry-pick).
- **Audit còn mở:** `scripts/grok_r6_causal.py` định nghĩa lại `screen_seed`, `pair_stats`, `fit_predict` ở cuối file, ghi đè implementation phía trên. `tests/test_r6_toys.py` tự định nghĩa local `class NoClip(probe.Head)` thay vì import `scarx.evaluation.r6_candidate.NoClip`; do đó toy pass không trực tiếp chứng minh class candidate thật. Registered reversed writes, query swap, shifts/distractors/noise không có đầy đủ tương ứng trong screen runner; cần lập ma trận protocol–executed và kiểm tra từng metric, không chấp nhận đoạn prose “có kiểm” làm evidence.

### 4.14 R6A — DeepSeek: more unique episodes, but harness confounded

Nguồn: [DeepSeek R6A report](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6-learning-scaling/docs/research/collab/scarx-neural-round1-20261111/deepseek-r6/report.md), [R6A confirmation raw](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6-learning-scaling/runs/20261111T072852Z-r6-confirmation-v1-d3a510/result.json), [runner](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6-learning-scaling/scripts/r6_stage5_confirmation.py).

- Frozen ALIF H48 + bilinear cue map, single-pass local delta head. 8 fresh confirmation seeds121011…121043, nested-like data budgets nhưng actual calibration **không cố định**.
- Ladder: n=256: `direct0.8425`, `both_A0.5407`; 1,024: direct0.9138, both_A0.7176; 4,096: direct0.9532, both_A0.8445; 16,384: direct0.9736, both_A0.9237; ridge direct0.9965 at16k.
- Matched16,384 updates: repeated256×64 `both_A0.7920`, repeated1,024×16 `0.9014`, repeated4,096×4 `0.8948`, unique16,384×1 `0.9244`. Paired unique−repeated256 **+13.25pp** CI [+6.99,+19.40] trong raw và summary.
- Epoch scan 1,024 examples peaks~4 epochs (`both_A0.8318`) và 16 epochs giảm (`0.8036`); **không chứng minh delayed phase transition**.
- 6 defects được xác định trong R6A: (A1) R5B floor.05 clip±5 khác R6A floor1e-6 no clip; (A2) stress raw feature cho scaled-trained head; (A3) calibration lấy `trx[:256]` nhưng datasets khác budget không nested; (A4) G2 computed `direct` thay vì `both_A`; (A5) G4/G5 hardcoded True; (A6) một số registered gate G3/G7 thiếu.
- Một báo cáo nói G6 stress FAIL seed121017 temporal.5557/distractor.5713: **không nên diễn giải như model failure** vì A2 làm measurement invalid. Những số đó là lịch sử `INVALID_HARNESS` cho câu hỏi robustness; được sửa ở R6D.
- R6A đã nêu protocol chốt sau các stage exploratory trước 8 untouched confirmation; không gọi toàn bộ hypothesis prereg từ đầu. R6A `GO` cho hẹp data-effect nhưng **gated confirmation STOP**.

### 4.15 R6D — DeepSeek tự kiểm toán lại R6A, dữ liệu mới nhưng validation chưa khép kín

Nguồn: [R6D report](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/docs/research/collab/scarx-neural-round1-20261111/deepseek-r6d/report.md), [protocol](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/experiments/collab/scarx-neural-round1-20261111/deepseek-r6d/protocol-v1.md), [matrix code](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/scripts/r6d_matrix.py), [controls code](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/scripts/r6d_controls.py), [raw matrix](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/runs/20261111T080352Z-r6d-confirm-v1-ff539c/result.json), [raw controls](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/runs/20261111T080601Z-r6d-controls-confirm-v1-b53925/result.json), [PR #12](https://github.com/AKANEMO22/SCAR-X/pull/12).

**Thiết kế tự báo cáo:** commit protocol trước runs, fixed pinned 256-episode calibration offset+700000, compare data {1,024,16,384} × scaler {R5B-style floor.05 clip±5, R6A-style floor1e-6 unbounded}, matched update {256×64 vs16,384×1}, corrected stress. Fresh seeds130001/130007/130013/130019/130023/130029/130031/130037; dev replay120011/120017/120023 riêng. `FRESH_CONFIRMATION` là *nhãn trong artifact*, không là independent runtime của chúng ta.

**R6D RAW JSON — đã tính lại từ 80 `trials` (không chỉ copy summary):**

| Arm / metric | Giá trị |
|---|---:|
| n1,024, R5B-style `both_A` | **0.6872845675** |
| n16,384, R5B-style `both_A` | **0.9178224074** |
| data gain R5B-style (paired over 8 seeds) | **+0.2305378398**, bootstrap95% [+0.17161,+0.29482] từ summary |
| n1,024, R6A-style `both_A` | 0.7137801531 |
| n16,384, R6A-style `both_A` | 0.8942473631 |
| data gain R6A-style | +0.1804672100 |
| matched repeated256 (16,384 updates), R5B | 0.7806959343 |
| matched unique16k (16,384 updates), R5B | 0.9314543355 |
| matched unique−repeated, R5B | **+0.1507584011**, bootstrap95% [+0.09685,+0.22235] từ summary |
| temporal shift no retrain `both_A` | **0.9099176927**, worst seed **0.716667** |
| distractor no retrain `both_A` | **0.5502304532**, worst seed **0.340136** |
| random-label direct (separate controls run) | 0.4775390625 |
| first-key shortcut | 0.7501220703 |
| query-only | 0.4934082031 |

**Per-seed RAW matrix:**

| Seed | R5B-style both_A n1,024 | R5B-style both_A n16,384 | Gain | Repeated256×64 | Unique16k×1 | Temporal both_A | Distractor both_A |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 130001 | .851240 | .966942 | +.115702 | .801653 | .942149 | .876033 | .595041 |
| 130007 | .661017 | .966102 | +.305085 | .838983 | .949153 | .957627 | .559322 |
| 130013 | .792593 | .955556 | +.162963 | .874074 | .940741 | 1.000000 | .622222 |
| 130019 | .550000 | .816667 | +.266667 | .616667 | .741667 | **.716667** | .591667 |
| 130023 | .770492 | 1.000000 | +.229508 | .942623 | 1.000000 | .991803 | .483607 |
| 130029 | .563492 | .952381 | +.388889 | .761905 | .952381 | .952381 | .650794 |
| 130031 | .585034 | **.700680** | +.115646 | .598639 | .972789 | .816327 | **.340136** |
| 130037 | .724409 | .984252 | +.259843 | .811024 | .952756 | .968504 | .559055 |

**Nhận xét có cơ sở từ bảng thô:** Data gain >0 cho tất cả 8 seeds. **Seed 130031 clean n16k only 0.700680**; nếu protocol yêu cầu gate ≥0.75 trên **mọi seed** thì seed đó sẽ fail. R6D actual G1 chỉ chấm **mean** (0.9178), protocol G1 không ghi rõ every seed, vì vậy đây là **threshold interpretation ambiguity**, không được gán là lỗi cố ý. Temporal shift worst0.716667 cũng <0.75, dù mean0.9099. Distractor fail rõ ràng trên mean và mọi seed. Data diversity effect có tín hiệu, nhưng **không giải quyết robust temporal/distractor/hidden learning**.

**Điểm không thể bỏ qua:** `R6D` vừa có actual controlled evidence mới, vừa tự chứa **những khẳng định kiểm định chưa được code đảm bảo**. Xem forensic section tiếp theo. Không được bỏ qua lỗi validation vì số điểm đẹp.

---
## 5. FORENSIC AUDIT: VẤN ĐỀ CODEX PHẢI TỰ KIỂM TRA LẠI

Đây là **danh sách quan sát tĩnh**, không phải kết luận từ runtime. Định vị đường dẫn và dòng chính xác tại SHA được khóa. Sau khi chạy independent tests, thay nhãn `SOURCE_CHECKED` bằng `REPRODUCED/CONFIRMED` hoặc bác bỏ với evidence.

### 5.1 R6D — high priority P0/P1 forensic blockers

**F-01. Gate không được đóng trên cùng một kết quả formal (severity CRITICAL cho claim `confirmation_pass`).**  
`research/deepseek/r6d-audit:scripts/r6d_matrix.py` tạo `gates` với `G4_random_labels_chance=None`, `G5_shortcuts_ok=None`, `G8_forward_parity=None`. Raw `runs/20261111T080352Z-r6d-confirm-v1-ff539c/result.json` **vẫn có 3 null**. Matrix script ghi summary rồi kết thúc, **không** tính `confirmation_pass=all(...)`. `scripts/r6d_controls.py` báo riêng G4/G5 TRUE ở **một run khác** và G8 được cho là test, nhưng không có machine-check đã join cả 8 gate cùng identity/protocol hash/seed/digest. Report §5 nói “all computed, none hard-coded” và `confirmation_pass=False`; điều này **chưa phản ánh trong formal JSON**. Thực tế G6 đã là false nên verdict cuối vẫn FAIL, nhưng quá trình gate chính thức **chưa kín**.

**F-02. Gate-reference independence vs not the same trained head (MAJOR).**  
Matrix head `rng(seed + 31*n)` cho grid; stress head dùng `seed+9`; controls reference head dùng `seed+404`. Do đó `controls/reference_both_A=0.95035` khác `matrix/grid/r5b-16384=0.91782`. `G4/G5` về randomized-label và baseline vẫn là kiểm tra hợp lệ theo mục đích riêng, nhưng mô tả “the SAME cells ... as matrix” phải được ràng buộc bằng cùng train-data/scaler và có khai rõ khác update order. Không được gộp mù thành một experiment có cùng trained head. Cần một aggregator/manifest kiểm tra join keys, n_unique, train digests, scaler digest, seed, head initial state, update order hash.

**F-03. Tests mang tính cú pháp/tautology, không đủ test end-to-end (MAJOR).**  
`tests/test_r6d_audit.py`:
- `test_all_registered_gates_present()` kiểm tra chỉ `G1_`…`G7_`, **không kiểm `G8_`** hoặc tất cả gate thực sự được tính trong `result.json`.
- `test_no_gate_is_hardcoded()` chỉ grep không có một số chuỗi `": True"`; nó không assert `all(value is not None)`/fail-closed/threshold semantics.
- `test_calibration_is_actually_fixed()` lặp gọi `fixed_calibration(seed,"r5b")` trong vòng `n`, nhưng **không truyền/áp dụng `n`** vào bất kỳ lệnh nào; tự nhiên hàm pure deterministic sẽ trả digest giống nhau. Phải kiểm tra scaler digest trong **mỗi actual `build_cell`/trial**.
- `test_stress_uses_training_scaler()` kiểm tra clean `scaled` bằng clean `d['f_te']` từ cùng seed/data, **không thực thi cả temporal/distractor stress branch**. Có thể pass trong khi runner stress regression bị lỗi.
- `test_fresh_seeds_never_used_before()` so set trong module (`DEV_SEEDS`, `CONF_SEEDS`), **không so toàn repo/manifest các lần probe khác**.
- `test_head_update_matches_r5b()` dùng R5B `zscore=False`: chứng minh arithmetic local delta trên tensor đã chuẩn hóa, **không chứng minh cùng pipeline calibration/feature scale**.

**F-04. R5B-style không có nghĩa là R5B pipeline nguyên gốc (MAJOR cho attribution).**  
R5B `P.Head.observe()` tính mean/std từ **training features**, sau đó `basis()` sử dụng `max(sqrt(M2/(n−1)), .05)` rồi clip±5. R6D `Scaler.fit()` dùng mean/std trên **pinned 256-episode calibration pool** riêng, `.std(ddof=0)`, không phải training examples và không phải running `observe` trên toàn training. Dù cùng floor/clip convention, **không được gọi là bit-exact historical R5B experiment**. Test equality trong R6D dùng thủ công `Scaler(h.mean, sigma)` để chứng minh công thức, không chứng minh actual `fixed_calibration()` tái lập R5B. Để quy sự khác biệt R5B→R6D, cần factorial riêng phân biệt `scaler convention` × `calibration source` × `budget`.

**F-05. Per-seed acceptance còn mơ hồ (MAJOR).**  
R6D G1 dựa vào **mean** ở 16k=0.9178, nhưng `seed130031` chỉ **0.700680** (<0.75). Stress G6 dựa vào mean temporal0.9099, nhưng `seed130019` temporal0.716667 (<0.75). G6 distractor mean0.5502 fail dù diễn giải thế nào. Codex phải đọc literal protocol: gate trên mean hay mọi seed? **Không được sửa retroactively**. Ghi both (`mean_gate`, `all_seeds_gate`) và chỉ dùng rule prereg để kết luận lịch sử; nếu muốn strict gate thì đăng ký protocol *mới*.

**F-06. Stress validity checks có thể chưa kiểm đúng dữ liệu đang chấm (MAJOR).**  
`r6d_matrix.py::stress_valid(seed)` kiểm oracle cho **128 independent heldout rows** mới; scorer lại chấm cả transformed test `d['tex']` và `d['ptx']` paired siblings. Cần assert oracle/task-preservation trực tiếp trên **toàn bộ exact `tex` và `ptx` được đánh giá**, kiểm `pflips`, `vA/vB`, cue tick của **từng sample**, sister-pair common-prefix, category count, noise/distractor channels. `stress_valid=True` trên 128 rows không chứng minh 256 exact paired rows đã hợp lệ.

**F-07. Reproduction/G7/G8 và limits chưa hoàn chỉnh (MAJOR).**  
`G7` lặp train one cell ở **seed đầu tiên** và so head digest, không phải mọi seed/condition; G8 forward parity nằm trong unit test, không được join formal gate. Resource log ghi `hard_cap_enforced:false`, dù peak commit~248.6MB<2GB. `phase` xử lý `if == 'replay' else fresh confirm`, khiến typo cũng mở fresh run; nên fail-closed với choices `{replay, confirm}`. Đừng coi “không thấy vượt cap” thành đã hard-cap-enforced.

**F-08. Báo cáo audit tự mâu thuẫn về defect count (MINOR/MAJOR cho tài liệu).**  
`docs/.../deepseek-r6d/audit.md` đầu ghi “Four of the claimed defects are real. Two are not. One additional...” nhưng chính các mục A1–A6 đều `CONFIRMED`; `report.md` ghi “six defects all real” và N1–N3 không reproduce. Codex phải đối chiếu từng claim, sửa **tài liệu mới** bằng erratum có vết thay đổi; **không sửa lại lịch sử source `report.md` nếu mục tiêu preserve immutable**. Sai số đếm làm giảm tin cậy narrative, dù chưa đảo metric.

**F-09. Nested sample ladder và semantic diversity (P1 research confound).**  
`independent_query_set_v1(seed+offset,n)` tạo shuffle phụ thuộc n; train 1,024 không nhất thiết là prefix của train16,384. Cùng seed nhưng dữ liệu không nested. R6D đã sửa scaler drifting, không sửa paired nesting. Vì thế effect quantity còn trộn sampling variation (dù 8-seed positives hỗ trợ xu hướng). 16,384 `episodes` là 16,384 noisy synthetic samples, **không phải 16,384 khái niệm/quy tắc độc lập**; generator có 2 keys, 2 independent values và số trạng thái logic nhỏ. Dùng language chính xác: *number of distinct sampled episodes*. Muốn causal learning curve mạnh, prereg nested fixed parent pool + extra seeds + cross-distribution tests.

**F-10. Phân loại source truth (MAJOR).**  
`src/scarx/evaluation/r6d_audit.py` có `raw_features()` public bên cạnh `apply_features()`. Docstring “the ONLY way to get features for evaluation” là design convention, **không phải API-enforced impossibility**; một runner khác vẫn có thể gọi raw. Test phải bắt actual call-path và kiểm invariants bằng hash/dtype/feature range trên **mọi arm**. `Scaler.fit(..., convention="r5b")` có default mặc dù docstring nói convention bắt buộc; nếu giá trị mặc định được dùng vô ý, trace metadata phải phát hiện.

**Vì F-01/03/04/06 chưa xác minh runtime độc lập, trạng thái hiện tại là:** số R6D có signal `RAW_RECALCULATED` **không tương đương `INDEPENDENT_CONFIRMED`**. R6D report `PASS on data effect` là **kết luận của DeepSeek**, cần Codex công nhận hoặc bác bỏ sau independent run.

### 5.2 R6B — Grok forensic blockers

- **G-01:** Duplicate top-level functions cuối `scripts/grok_r6_causal.py` ghi đè các function đầu (`screen_seed`, `pair_stats`, `fit_predict`); phải phân tích **function objects thực được Python binding** tại `main()`; test chạy `inspect.getsource`/AST để chặn duplicated top-level definitions.
- **G-02:** `tests/test_r6_toys.py` định nghĩa `class NoClip` trong test, không test chính `scarx.evaluation.r6_candidate.NoClip`. Phải so exact update rule, module import, perturbing training features, finite values, clip boundaries.
- **G-03:** Protocol ghi nhiều stress/reversed/query-swap/noise, runner screen chủ yếu train original/permuted và `protected()` XOR/order; lập **registered vs executed matrix**, đánh dấu missing. Nếu thiếu metric, verdict STOP có thể giữ vì candidate fail, nhưng không gọi đã đo đủ mọi registered control.
- **G-04:** Noclip candidate regress order−18.75pp; CI gain caseA chứa 0, nên **không có cơ sở tăng budget** hoặc deploy. Không chuyển R6B STOP thành GO theo R6D n16k; khác data budget/protocol.

### 5.3 Cross-campaign hazards đã gặp thật

| Hazard | Vòng xuất hiện | Regression cần có |
|---|---|---|
| Train/test cue tick mismatched khi temporal shift | DeepSeek R4A, Grok R5A corrected replay | query locator runtime được kiểm cho **từng** shifted row |
| Zero-padding tạo shift no-op ở zero-initialized RNN | DeepSeek R5B initial invalid run | `max|state_clean-state_shift| > epsilon` và oracle unchanged, paired noise |
| Wrong metric “decode stored value” vs “decode query answer” | DeepSeek R4A correction | explicit target tensors `vA`,`vB`,`y` và tests phát hiện swap |
| Hardcoded `True`, `None` vacuous pass | R2A/R6A, AGY MTB, R6D integration thiếu | closed typed gates require keys==registered and all booleans non-null |
| Non-independent seed reuse | Grok R1 amendment; risk for any confirmation | manifest inventory + heldout split check over **all runs** |
| Data-size-dependent scaler drift | R6A | same scaler digest across budget cells, raw and scaled tensors tagged |
| Feature mismatch between train and stress | R6A | compare source/normalizer contract and exact stress caller end-to-end |
| Control differs in optimizer coverage/moments | Main report19/39 | match or disclose actual update vector and state of Adam moments |
| Reader high from frozen code | R1/R2/R3/R4/R5/R6 | frozen/shuffled/zero + learned-hidden refit and feature-delta interventions |
| Screen3-seed attractive but confirmation8-seed fails | DeepSeek R3A, R5B, Gemini order | gates fixed before 8+ fresh seed confirmation |
| Different modality information bottleneck pre-neuron | Reports31/34/44 | exact collisions/color/token coverage + matched-input information control |

**Checklist bắt buộc:** Thí nghiệm mới phải có negative tests “nếu tái xuất từng lỗi trên, test sẽ FAIL”. Tập trung test anti-confound, không chỉ test smoke code chạy.

---

## 6. NHỮNG KẾT LUẬN THỰC SỰ CÓ VÀ NHỮNG KẾT LUẬN PHẢI BÁC BỎ

| ID | Mệnh đề | Trạng thái hiện tại | Chứng cứ và giới hạn |
|---|---|---|---|
| C01 | Spike trace chứa thông tin thời gian mà fast decay mất | **SUPPORTED_SCOPED** | R1/R2 frozen feature probes, synthetic xor/order, không transferable chung |
| C02 | Readout delta luôn là bottleneck chính | **REFUTED_AS_UNIVERSAL** | R2 spikesum0.9712, R4 order delta fail dù ridge high; tùy geometry |
| C03 | Frozen ALIF lưu 2 independent key/value bits | **REPORTED_STRONG_PROBE** | R5 pre-query probes >.996; chỉ synthetic generator/label probe |
| C04 | Query-conditioned bilinear map giải mã tốt | **SUPPORTED_DIAGNOSTIC** | R4/R5 ridge/head; fixed P + input-supplied cue, không neuron tự khám phá |
| C05 | Online head R5B học cả differing-value siblings đủ ≥.70 | **FAILED_CONFIRMATION** | both_A .6676, gate .70; Grok reproduced raw means |
| C06 | Bỏ clip đơn lẻ sửa R5B | **FAILED_SCREEN** | Grok R6B mean gain .00982 CI lower âm; order regression. |
| C07 | Tăng sampled episodes cải thiện readout trên R5 generator | **PROMISING/RAW_SUPPORT** | R6D correct scaler ~+.23054, per-seed all positive; requires independent replication |
| C08 | Cứ thêm update/replay giúp bằng thêm unique episodes | **NOT_SUPPORTED** | matched updates unique > repeated, nhưng data distribution/normalization phải kiểm |
| C09 | Delayed phase transition/emergent jump | **NOT_SUPPORTED** | R6A second-difference 18/36 chance, no phase transition established |
| C10 | R6A temporal stress .8256 chứng minh seed fragile | **WITHDRAWN_INVALID_HARNESS** | raw stress passed unscaled features; R6D temporal remeasure .9099 |
| C11 | R6D chịu được distractors | **FAILED** | raw both_A .55023 worst .34014, verify perturbation semantics độc lập |
| C12 | R6D all gates computed in one formal result | **REFUTED_BY_STATIC_SOURCE** | G4/G5/G8 null in matrix raw; no integration confirmation_pass |
| C13 | DC-ALIF giải `strict_binding` synthetic | **REPORTED_SCOPED** | Gemini 98.78, protocol deviation; independent transfer still fails |
| C14 | DC-ALIF cải thiện mọi relational tasks | **FAILED** | order regress18.48pp, pair_binding .505 chance vs no_gate .7839 |
| C15 | Hidden synapses học biểu diễn bền vững rộng | **NOT_PROVEN** | main P1 narrow order/XOR gains; many tasks fail, R6 hidden frozen |
| C16 | SCAR-X P2 continual memory đạt gate | **FAILED** | main report21: final .3474, forgetting29.09pp; Gemini P2 final .5613/forget21.89pp |
| C17 | Readout usefulness pilot P2 thành công | **NOT_TESTED/NOT_APPROVED** | Only P0 math/parity/rollback checks approved |
| C18 | G1 normal chat tốt | **FAILED** | word free utterance .3594 vs train .9948, token shortcut teacher prefix |
| C19 | G1 normal RGB photo QA đạt | **FAILED** | visual word macrodev .2510, RGB path/codec bottlenecks, digit8×8 only serving |
| C20 | Thêm neurons 48→128 tự tăng thông minh | **NO_EVIDENCE** | capacity allocator is engineering; no learning/retention gain demonstrated |
| C21 | Qwen/GPT can be used as serving answer instead of SNN | **PROHIBITED** | user constraint; only isolated control/reference |
| C22 | More data alone solves structural input/encoder loss | **REFUTED_AS_GENERAL CLAIM** | grayscale color collapse, output-vocab cap, missing question path, H48 task confounds |
| C23 | Main product release is stable enough to benchmark | **ENGINEERING_BASELINE** | RC smoke/pytest historical, must rerun & check 14 skips on current environment |

Bất kỳ claim nào trong Codex report phải ghi rõ *task, dataset, seed, baseline, probe vs actual inference, normalization, confidence, status*. “Mô hình đã học X” chỉ khi synapses đã cập nhật, learned-hidden gain > frozen/shuffled/zero, feature map dùng để infer, causal test/transfer và retention có bằng chứng.

---

## 7. ROOT-CAUSE MAP: PHẢI TÁCH NGUYÊN NHÂN TRƯỚC KHI XÂY NEURON MỚI

```text
CÙNG MỘT VẤN ĐỀ QUAN SÁT: output sai / quên / không tổng quát
|
+-- (A) Input không mang thông tin -> tokenizer collision, RGB->L, missing question
|       -> kiểm raw->encoded collision, oracle, information-preserving controls
|
+-- (B) Hidden dynamics/feature không giữ đúng tín hiệu
|       -> same spike train, different filtered readout; state pre/post query
|       -> readout time constants and sparse temporal signal, without adding neuron
|
+-- (C) Có thông tin, nhưng linear head không biểu đạt phép kết hợp
|       -> R4 bilinear cue map; Gemini dendritic coincidence gate
|       -> compare capacity-matched no-interaction, cue shuffle, query-zeroed
|
+-- (D) Head có thể biểu đạt, nhưng online local updates không học được
|       -> variance floor, clipped updates, training order, data budget, projection
|       -> frozen features, same exact tensor, matched updates, multiple scalers
|
+-- (E) Hidden synapses update nhưng không học useful representation
|       -> correct vs frozen/shuffled/zero, loss/effective rank, refit-head transfer
|       -> signal tracing / credit assignment and geometry/optimizer controls
|
+-- (F) Có kỹ năng ban đầu nhưng học mới ghi đè kỹ năng cũ
|       -> readout overwrite, hidden drift, capacity, consolidation/replay
|       -> context-stream P2 with acquisition+forgetting+plasticity joint gates
|
+-- (G) Harness sai khiến một trong các giả thuyết trên trông như đúng
        -> wrong tick, no-op shift, wrong label target, raw/scaled mismatch,
           pre-reg gates absent, train/test seed leak, cherry-picked dev
        -> FIX INSTRUMENTATION FIRST
```

**Quy tắc ra quyết định:** từ điểm C/D/E mà không có ablation sẽ không biết có nên nghiên cứu dendrite, multicompartment, sparse recurrence hay optimizer nữa. Không thay neuron trước khi biết lỗi thuộc representation function, optimizable feature geometry, hay hidden credit. Phương pháp biologically inspired là ràng buộc thiết kế, không phải bằng chứng “càng giống hệ thần kinh thì tự thông minh”.

---
## 8. KẾ HOẠCH THỰC THI ƯU TIÊN CHO CODEX — KHÔNG ĐƯỢC BỎ QUA GIAI ĐOẠN KIỂM TOÁN

**Định nghĩa phạm vi:** nhiệm vụ hiện tại của Codex là tạo một **audit độc lập có thể chạy lại**, phát hiện các sai lệch giữa code/protocol/report, và chuẩn bị thiết kế R6E/R7. Đây **không phải** lệnh tự ý mở fresh confirmation, thay learning rule, triển khai model mới hoặc sửa sản phẩm. R6D đang `FAIL` trên G6 theo chính dữ liệu của nó; lỗi gate aggregation làm cho nhãn toàn vòng không hoàn chỉnh, không thể hợp thức hóa bằng báo cáo.

### P0 — Inventory và chain-of-custody (bắt buộc, làm đầu tiên)

**Tác vụ**

1. Kiểm tra repo root, branch, full HEAD, dirty files và remote; không overwrite thay đổi chưa commit của người dùng. Không chạy lệnh reset/clean. Nếu có concurrent edits, chỉ lập nhánh audit mới từ commit rõ ràng hoặc dùng worktree biệt lập.
2. Đọc `AGENTS.md`, `README.md`, `docs/PROJECT_MAP.md`, `docs/STATUS.md`, `docs/RESEARCH_PLAN.md` trước khi viết code.
3. Đối chiếu `main` `495dab2...` với các nhánh R2–R6 và R6D. Phân biệt branch tip và commit thực tế sinh run; không áp dụng source mới cho artifact cũ mà không xác minh.
4. Với mỗi run được audit: lấy `manifest.json`, protocol nguyên bản, SHA-256 protocol, code hash, seed set, split, calibration, generator, forward sampler, update order, NumPy/Python version, thread/resource configuration và raw output. Kiểm source.zip nếu có, đối chiếu với HEAD của branch. Có mismatch → `SOURCE_MISMATCH`, không tự bỏ qua.
5. Phân loại từng run `EXPLORATORY`, `REGISTERED_SCREEN`, `HISTORICAL_CONFIRMATION`, `CORRECTED_REPLAY`, `FRESH_CONFIRMATION`, `INVALID_HARNESS`, `SUPERSEDED`, `NOT_RUN`. Chỉ nhãn cuối hợp lệ theo đăng ký mới có thể hỗ trợ một verdict mới.
6. Lập bảng kiểm tra không trùng seed và, quan trọng hơn, **không trùng raw episode / counterfactual group / calibration item** giữa calibration/train/held/test nếu protocol yêu cầu độc lập. Seed không trùng **không tự chứng minh dữ liệu không overlap**.

**Artifact:** `BASELINE_INVENTORY.md` với các bảng: `branch | SHA | base | report path | protocol SHA | run ID | code path | status | reproducible?`; bảng `data lineage`; `environment`; `known deviations`.

**Gate P0:** chưa khẳng định bất kỳ kết quả khoa học nào khi chưa biết source nào và dữ liệu nào sinh ra kết quả ấy.

### P1 — Xây evaluator độc lập, sửa phép chấm trước khi sửa learner

Codex phải tạo một **reference evaluator** nhỏ, độc lập tối đa với `src/scarx/evaluation/r6d_audit.py`. Chấm từ `predictions`/`labels`/`vA`/`vB`/`flips` gốc; nếu run hiện không lưu predictions, ghi rõ thiếu provenance và tạo một **new reproduction run** ở nhánh audit, **không sửa run cũ**.

**Các bất biến phải được test tự động:**

- Hai phần tử kề nhau của `paired_set` là anh em của cùng `pair_id`, có cùng facts/noise-prefix theo đúng contract; chỉ query được phép thay đổi nếu protocol yêu cầu.
- `both_A` chỉ xét `vA != vB`, yêu cầu **hai dự đoán cùng đúng**; `both_B` chỉ xét `vA == vB`, yêu cầu invariance và đúng nhãn; `both_all` là trên toàn bộ pairs. Không đánh đồng `direct` hoặc `needB` với `both_A`.
- Prediction/label/order phải khớp sau mọi permutation; test cố tình tráo một sibling hoặc query và xác nhận evaluator phát hiện.
- `query_tick` phải lấy từ biến đổi thật, không gắn cứng 10 khi marker chuyển sang tick 11; replay cổ điển Grok R5 xác nhận lỗi đã xảy ra.
- Shift phải làm thay đổi spike/features thực trên same-noise counterfactual, không biến thành no-op do zero padding; oracle phải không đổi khi bài test định nghĩa temporal invariance.
- Distractor phải giữ target, query và cả two-sibling relation; kiểm trên **chính tất cả examples được chấm**, không chỉ 128 ví dụ phụ.
- Calibration train-only hoặc separate-pinned phải trùng digest trên các arms mà protocol nói dùng chung; không lén fit trên held-out; test cố tình cho truyền raw features vào scaled head phải thất bại.
- `frozen` thực sự không cập nhật hidden; hash cả `w_in`, `w_rec`, state và local traces đúng phạm vi; `zero`/`shuffled` controls tách learning attribution.
- Random-label control không rò label thật; baseline first-key/query-only được tính trên đúng held-out distribution và cùng paired/independent split phù hợp.
- Tính CI trên **đơn vị seed độc lập**, ghép cặp đúng seed/task/condition và báo từng seed; không bootstrap mọi sibling như mẫu độc lập; không tính trung bình condition rồi thay cho per-seed gate.
- `null`/`NaN`/missing gate phải là **UNKNOWN → không PASS**, không có `all({gates thuộc nhóm con})` hoặc literal `True`.

**Mutation tests tối thiểu:** cố tình tạo sai query tick, bỏ scaler, thay labels ngẫu nhiên, tráo sibling pair, zero-pad shift no-op, dùng metrics `direct` thay `both_A`, hard-code `True`, bỏ G8, thay training order giữa matched controls. Ít nhất một test phải fail trong từng mutation. Test chỉ grep một chuỗi trong source không đủ.

**Artifact:** `reference_evaluator.py`, `tests/test_forensic_evaluator.py`, `tests/test_forensic_mutations.py`, raw-digest comparison và bảng failed/passed cases có trace.

### P2 — Kiểm toán DeepSeek R6D trước bất kỳ experiment mới

**A. Kiểm tra cấu trúc R6D:**

- Đọc `scripts/r6d_matrix.py` và `scripts/r6d_controls.py`, đối chiếu protocol §5 G1–G8; giải thích vì sao matrix lưu G4, G5, G8 là `null`. Gộp full gate trong một **artifact kết quả mới** bằng protocol/seed/condition identity. Không ghi đè JSON gốc. Khi một nhánh nguồn không có thể nối trực tiếp với nhánh kia do khác training order, ghi `NON_EQUIVALENT_CONTROL`.
- Ghi lại `reference_both_A=0.9503508` ở controls nhưng grid mean `0.9178224`; tìm nguyên nhân đúng bằng replay **cùng order và cùng features**. Kiểm khác biệt `seed+404` so với `seed+31*n`, và `seed+9` ở stress. Không lắp reference .95035 như một arm thống nhất với .91782 khi không match.
- Kiểm **ngữ nghĩa của G3:** mean unique−repeated >0 khác với mọi-seed >0; nếu protocol chỉ khóa mean, không được sửa ngưỡng lịch sử, nhưng công khai cả hai và đặt gate mạnh hơn cho protocol tương lai nếu được duyệt.
- G1/G6 đánh giá theo mean hay từng seed? R6D min clean tại seed `130031` là `0.70068` và min temporal seed `130019` là `0.71667`. Báo verdict theo **đúng ngưỡng lịch sử**; tuyệt đối không tự áp điều kiện per-seed sau đó để sửa verdict, nhưng đừng công bố mean như mọi seed đã qua.
- G7 chỉ re-run một `seed` hay đủ toàn bộ arms? G8 đã được test parity trong tests nhưng không được import vào gates; xác minh đầy đủ.
- `tests/test_r6d_audit.py::test_no_gate_is_hardcoded` chỉ tìm các mẫu literal `True`, không bắt `None`/missing integration; `test_calibration_is_actually_fixed` gọi cùng hàm bất biến mỗi budget, không chứng minh code `build_cell` dùng đúng scaler; kiểm bằng negative/mutation tests thực sự.
- Kiểm `phase` bất hợp lệ có thể bị gán sang confirmation; bỏ implicit default nguy hiểm và yêu cầu explicit CLI enum. Đây là lỗi vận hành, sửa harness branch mới nếu cần.
- Tái lập exact R5B `Head.basis` không chỉ phép tính `(f−mu)/sigma` mà cả nguồn population statistics, population vs sample ddof, thứ tự `observe()` và clip/saturation. Đặt tên `R5B_STYLE_FIXED_CALIB` cho pipeline R6D để khỏi bị nhầm là reproduction R5B gốc nếu khác semantics.
- Kiểm memory: sampled `peak_commit_or_vms` <2GB nhưng `hard_cap_enforced=false`; ghi `MEASURED_UNDER_CAP_NOT_HARD_ENFORCED`, không nói 'enforced'.

**B. Đối chứng nhân quả đã có:**

- Đối chiếu same-seed 2×2 normalization (R5B style / R6A) × (1024 / 16384). Tính `both_A`, gradient clip saturation, preprocessing × data interaction trên **raw trials** chứ không sao chép summary.
- Đối chiếu matched 16,384 updates: repeated 256×64 vs unique 16,384×1, giữ dimension, train source, seed, hidden, projection, scaler digest, evaluation, trained head init và lịch update ghi rõ. Lưu ý repeated-vs-unique vẫn khác phân bố thứ tự/gradient path: không suy ra chỉ có entropy là nguyên nhân cho mọi môi trường.
- Phân loại 16,384 'unique episodes' là **distinct generated realizations**, không phải 16,384 khái niệm/task độc lập; generator hai keys/hai values và vài pattern/noise. Cần phân tách `logical_state_coverage`, `noise_coverage`, `temporal-structure diversity`.
- Stress: head phải nhận feature đã chuẩn hóa đúng training. Cần stress trên **cùng trained head** (chứ không retrain bản khác cho mỗi evaluation), đồng thời phân tích sự khác nhau giữa stress baseline `.9178` (grid), `.95035` (controls) và stress script riêng `seed+9`.

**C. Output bắt buộc:** `R6D_SOURCE_AUDIT.md`, bảng sai lệch code/protocol/báo cáo, `R6D_CLAIM_RECOMPUTE.json`, `R6D_GATE_INTEGRATION.json`, CI độc lập và verdict `HISTORICAL_FAIL_G6` + `INCOMPLETE_AGGREGATION`, trừ khi có chứng cứ rõ ràng thay đổi cách hiểu (không rewrite history).

### P3 — Kiểm toán Grok R6B / DeepSeek R5B với evaluator độc lập

- Kiểm sibling storage, independent values, query-conditioned bilinear map, permutation invariant, ridge alpha, calibration, projection seed, per-seed predictions; sinh `both_A` bằng code độc lập từ raw arrays; historical head `.6676172`, ridge `.9858056` là **mục tiêu tái lập**, không là constants buộc evaluator phải trả ra.
- Test `NoClip` **thực từ** `src/scarx/evaluation/r6_candidate.py`, không test lớp `NoClip` shadow trong test file.
- Xóa mã bị định nghĩa trùng chỉ trong bản audit/harness mới; giữ commit gốc và ghi chênh lệch. Bảo đảm tests được viết để fail trên lỗi duplicate-shadow nếu nhận nhầm implementation.
- So sánh danh sách conditions Grok protocol (`reversed writes`, `query swap`, `shift`, `distractors`, `noise`, `XOR`, `order`) với những nhánh thực sự chạy. Nếu chỉ chạy clean/perm/XOR/order, bỏ claim cover stress khác hoặc đánh `NOT_TESTED`.
- Tái lập probe on same feature tensor: cấm đổi feature dimension/calibration/data/seed giữa head và ridge. Phân biệt ridge diagnostic (optimizer khác) với local update learner, không gọi ridge là achievable online ceiling.
- Điều tra R5A/SafeHead variance floor giải được order nhưng không giải simultaneous XOR/query: chống kết luận 'chỉ cần một normalizer tốt hơn' dựa trên một bài.

**Artifact:** `R5B_R6B_REPRODUCTION.md`, bảng `pred pair | expected | actual`, tests, source code digests, per-seed summary và `DEVIATIONS.md`.

### P4 — Gemini / DC-ALIF cross-review sau khi nhánh được đóng băng

Gemini vẫn được người dùng cho làm tiếp. **Không thay đổi nhiệm vụ của Gemini**; Codex chỉ audit branch đã commit. Kiểm toán hai bộ dự án `research/antigravity-gemini/...` và `research/grok/r2-dendritic-transfer`, đặc biệt:

- Công thức dendritic plateau / coincidence / gating trong prose so với code đang chạy, thông số leak/decay/gain thực, và `W_out.T @ error` (không gọi hoàn toàn biologically local).
- Bản số strict_binding98.78% vs no_gate62.43% dưới deviation, nhưng order64.04% vs frozen82.52%, pair_binding50.51% vs no_gate78.39%, P2 final56.13% / forgetting21.89pp. Không gom task lợi và hại vào một 'overall improvement'.
- Thiết kế train/held/test, correct vs frozen vs zero vs shuffled vs no_gate vs BPTT-control; check resuming/rollback/calibration and intervention semantics.
- So sánh với single-compartment H48 cùng data, target, updates, time, memory, noise, readout dimension, learning rule; nếu không matched, chỉ báo observation không causal attribution.
- Chạy thêm chỉ trong scope audit đã được cho phép; mọi mới-train/candidate/protocol cần duyệt. Báo `PIVOT_TASK_SCOPED`, không thay production.

### P5 — Đề xuất khoa học sau forensic audit: trả lời câu hỏi NEURON HỌC GÌ?

Không tự động chạy. **Chỉ soạn protocol R6E/R7 chờ duyệt**, tách ba giả thuyết:

1. `FEATURE_READOUT_ONLY`: hidden ALIF đóng băng + head local học tốt hơn trên nhiều episode; cần tách data diversity khỏi calibration/order/projection.
2. `LEARNED_REPRESENTATION`: correct hidden plasticity thực sự tăng khả năng đọc biểu diễn ở heldout so với initial/frozen/zero/shuffled, với common **fresh refit** trên cùng head budget và cùng test. Không dùng endpoint coadaptation hay trained-head lesion đơn lẻ để tuyên bố representation.
3. `CONTINUAL_REUSE`: sau khi học task B, vẫn giữ task A và học task C, đồng thời kiểm acquisition, final, mean/worst forgetting, plasticity và head-only/hidden-only overwrite trong stream không task boundary.

Nếu P2 acquisition dưới threshold ngay từ đầu, số forgetting thấp là **không có kỹ năng để quên**, không PASS. Không hạ gate để được mở rộng. Không dùng phản hồi model tự sinh làm nhãn đúng nếu không có external verifier/human-confirmed source.

---

## 9. GỢI Ý PROTOCOL R6E — BẢN THẢO ĐỂ CODEX ĐỀ XUẤT, CHƯA ĐƯỢC PHÊ DUYỆT

**Tựa đề:** *Where do the 16K gains come from? Data support, preprocessing, optimization path, shortcut or hidden representation?*

**Mục tiêu chính:** chứng minh (hoặc bác bỏ) rằng dữ liệu đa dạng cải thiện local readout một cách nhân quả trên frozen ALIF, xác định giới hạn nhiễu; đồng thời đặt thí nghiệm hidden learning riêng, không đánh đồng hai mục tiêu.

**Đăng ký trước:** source SHA, protocol SHA, hypothesis, H0/H1, generator version, transformations, task ontology, exact metrics, thresholds, screen seeds, untouched confirm seeds, exclusions, time/memory budget và significance correction. Không đọc kết quả confirm trước khi khóa; seed mới phải kiểm không trùng các bộ của Grok/DeepSeek/Gemini. Không kéo metric từ R6D lỗi vào làm fresh confirmation mới.

**Cấu trúc thiết kế tối thiểu (những yếu tố phải tách):**

| Trục | Các arm gợi ý | Câu hỏi |
|---|---|---|
| Biểu diễn | frozen H48 original; optional local hidden-learning arm **nếu được duyệt** | readout vs representation |
| Normalization | R5B original online train stats; R5B-style pinned; R6A pinned | scaler semantics và calibration |
| Số episode | 256, 1024, 4096, 16384 | hiệu ứng data; ghi rõ nested vs independent |
| Update budget | equal 16384; fixed one-pass ladder | uniqueness so với repetitions |
| Data variety | same logical patterns/noise variants; new logical structure; new temporal arrangement | học noise distribution hay quy tắc |
| Stress | same-noise temporal shift, distractor within facts, OOD perturbation; every paired sibling | robust causal binding |
| Controls | first-key, query-only, label-shuffle, cue-shuffle, frozen, zero, shuffled hidden, ridge diagnostic | shortcuts và credit attribution |
| Readout | exact historical R5B learner, fixed init and order; other readouts only as separately registered arms | isolate optimizer |

**Đầu ra chính:** `both_A` on independent paired counterfactuals (not `direct`). Additional metrics: direct, both_B, needB, OOD `both_A`, seed-worst, label random direct/paired, cross-task regression and time/memory, per-sample prediction hashes. `both_A` is a *pair* outcome; bootstrap unit primary is model initialization seed, not individual query pair.

**Stop/promotion rules (Codex phải ghi số threshold trước khi xem kết quả mới):**

- Fresh confirmation phải có **zero missing gate**, cùng source SHA và source zip, no train/test leakage, no broken counterfactual semantics, no mock hidden changes; nếu thiếu, toàn vòng = `INVALID_HARNESS`/`INCONCLUSIVE`.
- Với exploratory comparison, 95% paired seed bootstrap CI phải tính trước khi gọi 'improvement'; kiểm độ ổn định theo thứ tự train, projector và calibration.
- Nếu một task bảo vệ như order/XOR/strict binding bị regression đã đăng ký vượt margin, không chọn candidate dù điểm `both_A` tăng.
- Không gọi architecture 'general relational learner' khi chỉ một binary two-key generator. Chứng minh học một bài mới và giữ cũ đòi **tác vụ khác**, mới được khóa trước.
- Nếu distractor cùng facts làm `both_A` giảm mạnh như R6D `.5502`, nghiên cứu nguyên nhân perturbation/noise/feature geometry trước khi tăng neuron hay train data lớn.

**Khoảng phân tích nguyên nhân cần ưu tiên (không phải cách chọn winner):** đo feature-distance, effective rank/covariance, gradient clip activation fraction, class-conditional small-variance directions, query-cue ablations, per-seed projection sensitivity, fraction of corrupted vs unchanged paired states, and head-order stability. Với distractors, so sánh train-time augmentation (nếu đăng ký) với blind test-time robustness, tuyệt đối không dùng held-out stress làm tập huấn luyện hậu kiểm.

**Yêu cầu tách protocol:** (A) Harness/reference evaluator; (B) Frozen readout scaling; (C) Hidden synapse learning; (D) P2 continual transfer. R7 không được gọi một kết quả ở B là PASS ở C/D.

---
## 10. REPRODUCTION RUNBOOK — CODEx CẦN THỰC HIỆN VÀ GHI LẠI

**Lưu ý:** các command này là điểm khởi đầu từ scripts đã có, không phải log chúng tôi đã chạy. Vì không có quyền/khả năng tự chạy repo trong phiên soạn báo cáo, Codex phải ghi kết quả thực của mình, không lấy câu '15 tests pass' trong report làm kết quả mới. Dùng Windows PowerShell nếu đang ở máy người dùng; bên dưới là portable commands tương đương cho shell Bash. Luôn kiểm tra current branch và dirty changes trước khi chạy.

```bash
# 0. Inspect only; không git reset, clean hoặc chạm model/data người dùng.
git rev-parse --show-toplevel
git status --short
git rev-parse HEAD
git branch -vv
git remote -v

# 1. Fetch branch tips. Chỉ khi remote/network được phép và đang hoạt động.
git fetch --all --prune

git rev-parse origin/main
git rev-parse origin/research/deepseek/r6d-audit
git rev-parse origin/research/grok/r6-causal-readout-repair

# 2. Tạo worktree THÍ NGHIỆM RIÊNG, chọn path mới chưa tồn tại.
# Không đổi branch của serving process đang chạy; không sửa local/generation1.
git worktree add -b research/codex/forensic-audit ../scarx-codex-audit origin/main
cd ../scarx-codex-audit

# 3. Record environment before measurements.
python --version
python -m pip --version
python -m pip install -e '.[vision,test]'
python -m pytest -q -rs
```

**Nếu muốn chạy trực tiếp R6D**, tạo thêm worktree riêng từ `origin/research/deepseek/r6d-audit` (không sửa worktree main), dùng source đúng HEAD của R6D. Những lệnh sau sẽ **tạo run mới**, có thể mất CPU/RAM/disk; gắn nhãn `HISTORICAL_REPRODUCTION` hoặc `CORRECTED_REPLAY`, *không* gọi là seed mới:

```bash
# Trong worktree của branch R6D; one numerical thread, CPU-first.
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m pytest tests/test_r6d_audit.py -q
python scripts/r6d_matrix.py replay
python scripts/r6d_controls.py replay

# Chỉ chạy nếu người dùng cho phép dùng lại bộ confirmation cũ như REPRODUCTION:
# python scripts/r6d_matrix.py confirm
# python scripts/r6d_controls.py confirm
# Những seed 130001...130037 đã bị 'seen' từ R6D. Đây KHÔNG phải new confirmation.
```

**Grok R6B:** trong worktree/checkout Grok riêng:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -m pytest -q tests/test_r6_toys.py
# Runner gốc có OUT.exists() guard; nếu output tồn tại, không xóa để chạy lại.
# Tạo bản runner audit riêng có output path duy nhất và lưu source hash.
# python scripts/grok_r6_causal.py
```

**R5B và product:** chỉ khi cần smoke serving, chạy trên data-dir riêng và cổng khác `8765`. Không đụng `local/generation1`, không tự dạy dữ liệu mới vào phiên cá nhân. BPTT controls không được import sang `src/scarx/neural`.

**Bất biến khi chạy:** (i) mỗi run ID mới, (ii) 1 numerical thread, (iii) process-tree RSS/commit <2GB và ghi `hard_cap_enforced` chính xác, (iv) file source zip và hash/manifest, (v) một run lỗi giữ nguyên, (vi) trước/sau run so Git worktree và production checkpoints hash, (vii) không đặt output trong data cá nhân.

### 10.1 Thứ tự mở/khóa benchmark

1. `pytest` và mutation tests trên data nhỏ; chưa chấm fresh seeds.
2. Recompute từ raw arrays hiện có (đối chiếu số published).
3. Replay seed lịch sử để xác định pipeline đúng; không dùng replay để sửa hypothesis.
4. Soạn protocol mới và gửi user duyệt scope; `git commit` protocol + hash **trước** confirm.
5. Chạy screen trên development seeds công bố trước; **khóa candidate** trước confirm.
6. Chỉ nếu screen gate qua: chạy confirmation untouched; không sử dụng confirm để chọn hyperparameters, feature projection hay post-hoc thresholds.
7. Chỉ sau đó đọc Gemini revision mới/đóng băng và cross-review; không cherry-pick phần tốt.

### 10.2 Định nghĩa dữ liệu mỗi trial tối thiểu

Một record nên có `experiment_id`, `task`, `seed`, `sample_group_ids`, `train_split_sha256`, `held_split_sha256`, `calibration_sha256`, `raw_feature_sha256`, `scaled_feature_sha256`, `feature_map_id`, `query_tick`, `scaler_id`, `train_order_sha256`, `weights_initial_sha256`, `weights_final_sha256`, `predict_ids_sha256`, `prediction_file`, `label_file`, `n_unique`, `n_updates`, `n_epochs`, `loss`, `direct`, `both_A`, `both_B`, `both_all`, `needB`, `stress_kind`, `hidden_frozen`, `runtime_seconds`, `peak_rss_bytes`, `peak_commit_bytes`, `state_restore_digest`, `code_sha`, `protocol_sha`, `status`.

Nếu nguồn cũ không có từng prediction, **ghi thiếu**; đừng tự tạo `prediction_file` tưởng tượng. Một số raw result lịch sử chỉ có aggregate trên từng seed, đủ tái tính mean nhưng **không** đủ kiểm oracle/sibling prediction; cần chạy lại để tạo evidence mới dưới đúng nhãn.

### 10.3 JSON schema đề xuất cho audit verdict

```json
{
  "artifact_type": "SCARX_INDEPENDENT_AUDIT",
  "auditor": "Codex",
  "date_utc": null,
  "repository": "AKANEMO22/SCAR-X",
  "source_sha": null,
  "protocol_sha256": null,
  "run_class": "HISTORICAL_REPRODUCTION",
  "run_id": null,
  "comparison_id": "r6d_source_vs_reference_evaluator",
  "source_files_verified": [],
  "seed_ids": [],
  "metrics": {
    "both_A_mean": null,
    "both_A_per_seed": [],
    "distractor_both_A_mean": null,
    "paired_gain_ci95": null
  },
  "gates": {
    "G1": {"status": "NOT_TESTED", "value": null},
    "G2": {"status": "NOT_TESTED", "value": null},
    "G3": {"status": "NOT_TESTED", "value": null},
    "G4": {"status": "NOT_TESTED", "value": null},
    "G5": {"status": "NOT_TESTED", "value": null},
    "G6": {"status": "NOT_TESTED", "value": null},
    "G7": {"status": "NOT_TESTED", "value": null},
    "G8": {"status": "NOT_TESTED", "value": null}
  },
  "all_gates_complete": false,
  "scientific_verdict": "INCONCLUSIVE",
  "harness_verdict": "NOT_TESTED",
  "limits": [],
  "reproduction_command": null,
  "new_data_added": false
}
```

**Contract:** `PASS` chỉ khi **mọi required gate true** và không có invalid-harness condition. `FAIL` khi gate hợp lệ nhưng không đạt. `INCONCLUSIVE` khi bằng chứng thiếu/mâu thuẫn. Không tự ghi 'pass' chỉ vì code đã chạy xong. Mọi `null` phải ngăn aggregate PASS; có thể giữ một *partial per-gate pass*, nhưng global verdict phải `INCOMPLETE`/`INCONCLUSIVE` hoặc `FAIL` nếu một gate khác đã fail hợp lệ.

---

## 11. SOURCE INDEX — BẮT BUỘC CODEx ĐỌC ĐÚNG NHÁNH VÀ ĐÚNG COMMIT

### Main/master

- [Repo](https://github.com/AKANEMO22/SCAR-X) · [main pinned `495dab2`](https://github.com/AKANEMO22/SCAR-X/tree/495dab2f4bac0180e4d1cd28674391f055ccf09d).
- [`AGENTS.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/AGENTS.md) · [`docs/RESEARCH_PLAN.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/RESEARCH_PLAN.md) · [`docs/STATUS.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/STATUS.md).
- [`docs/research/20_SCORE_ELIGIBILITY_AND_VARIANCE.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/20_SCORE_ELIGIBILITY_AND_VARIANCE.md): narrow P1.
- [`docs/research/21_P2_RETENTION_AND_CUE_INFORMATION.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/21_P2_RETENTION_AND_CUE_INFORMATION.md): retention baseline.
- [`docs/research/35_DIRECT_READOUT_ELIGIBILITY.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/35_DIRECT_READOUT_ELIGIBILITY.md): approved local credit prototype but failed P1.
- [`docs/research/36_DIRECT_WORD_BRIDGE.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/36_DIRECT_WORD_BRIDGE.md): actual neural free utterance failure.
- [`docs/research/37_DIRECT_VISUAL_BRIDGE.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/37_DIRECT_VISUAL_BRIDGE.md): visual joint grounding failure.
- [`docs/research/38_WORD_TRAINING_OBJECTIVE_AUDIT.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/38_WORD_TRAINING_OBJECTIVE_AUDIT.md): teacher prefix vs neural free generation.
- [`docs/research/39_LOCAL_CREDIT_ALIGNMENT.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/39_LOCAL_CREDIT_ALIGNMENT.md): synthetic directionality, Adam confound caveats.
- [`docs/research/40_DEEPSEEK_REVIEW.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/40_DEEPSEEK_REVIEW.md): release/guard/rollback and causal readout overwrite.
- [`docs/research/42_READOUT_USEFULNESS_P0.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/42_READOUT_USEFULNESS_P0.md): P0 only, P2 pilot not approved.
- [`docs/research/44_P1_HANDOFF_EVIDENCE.md`](https://github.com/AKANEMO22/SCAR-X/blob/main/docs/research/44_P1_HANDOFF_EVIDENCE.md): actual image/vocab bottlenecks.

### DeepSeek research branches and artifacts

- [R1B report v2](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/deepseek/report-v2.md).
- [R2A](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r2-readout-learnability/docs/research/collab/scarx-neural-round1-20261111/deepseek-r2/report.md).
- [R3A](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r3-temporal-substrate/docs/research/collab/scarx-neural-round1-20261111/deepseek-r3/report.md).
- [R4A](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r4-relational-representation/docs/research/collab/scarx-neural-round1-20261111/deepseek-r4/report.md).
- [R5B](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r5-independent-relational-binding/docs/research/collab/scarx-neural-round1-20261111/deepseek-r5/report.md).
- [R6A](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6-learning-scaling/docs/research/collab/scarx-neural-round1-20261111/deepseek-r6/report.md) · [R6A source](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6-learning-scaling/scripts/r6_stage5_confirmation.py) · [R6A PR #11](https://github.com/AKANEMO22/SCAR-X/pull/11).
- **[R6D report](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/docs/research/collab/scarx-neural-round1-20261111/deepseek-r6d/report.md)** · [R6D audit](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/docs/research/collab/scarx-neural-round1-20261111/deepseek-r6d/audit.md) · [argument ledger](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/docs/research/collab/scarx-neural-round1-20261111/deepseek-r6d/argument-audit.md).
- **[R6D protocol](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/experiments/collab/scarx-neural-round1-20261111/deepseek-r6d/protocol-v1.md)** · [R6D matrix](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/scripts/r6d_matrix.py) · [controls](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/scripts/r6d_controls.py) · [eval](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/src/scarx/evaluation/r6d_audit.py) · [tests](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/tests/test_r6d_audit.py).
- **[R6D raw confirmation result](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/runs/20261111T080352Z-r6d-confirm-v1-ff539c/result.json)** · [manifest](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/runs/20261111T080352Z-r6d-confirm-v1-ff539c/manifest.json) · [controls raw](https://github.com/AKANEMO22/SCAR-X/blob/research/deepseek/r6d-audit/runs/20261111T080601Z-r6d-controls-confirm-v1-b53925/result.json) · [R6D PR #12](https://github.com/AKANEMO22/SCAR-X/pull/12).

### Grok research

- [Round 1 slow eligibility](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/grok/report.md).
- [R2B DC-ALIF transfer](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r2-dendritic-transfer/docs/research/collab/scarx-neural-round1-20261111/grok/report-r2b.md).
- [R3B temporal stress](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r3-temporal-stress/docs/research/collab/scarx-neural-round1-20261111/grok/report-r3b.md).
- [R4B online mechanism](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r4-online-readout-mechanism/docs/research/collab/scarx-neural-round1-20261111/grok/report-r4b.md).
- [R5A independent audit](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r5-independent-audit/docs/research/collab/scarx-neural-round1-20261111/grok-r5/report.md).
- **[R6B report](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/docs/research/collab/scarx-neural-round1-20261111/grok-r6/report.md)** · [R6B protocol](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/experiments/collab/scarx-neural-round1-20261111/grok-r6/protocol-v1.md) · [runner](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/scripts/grok_r6_causal.py) · [toy tests](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/tests/test_r6_toys.py) · [raw](https://github.com/AKANEMO22/SCAR-X/blob/research/grok/r6-causal-readout-repair/runs/grok-r6-causal-v1/result.json).

### Gemini and Antigravity

- [Gemini DC-ALIF report](https://github.com/AKANEMO22/SCAR-X/blob/research/antigravity-gemini/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/antigravity/report.md).
- [Antigravity MTB report](https://github.com/AKANEMO22/SCAR-X/blob/research/antigravity/scarx-neural-round1-20261111/docs/research/collab/scarx-neural-round1-20261111/antigravity/report.md).

**Độ tin cậy của index:** những đường dẫn trên được suy ra từ cây Git đã duyệt và những file đã được `fetch_file` đọc đúng nhánh; Codex phải xác minh lại `ref/sha` trước khi dùng để đo. Nếu branch được force-push, dùng SHA cố định trong §1, không so số mới với source thay đổi mà không ghi chép.

---

## 12. THANG ĐÁNH GIÁ DỰ ÁN VÀ TIÊU CHÍ KẾT THÚC NHIỆM VỤ CODEX

### 12.1 Tình trạng khoa học hiện tại

| Trụ cột | Bằng chứng tốt nhất hiện có | Mức độ |
|---|---|---|
| Serving local có state/rollback | Generation 1 + release smoke, tests trong `main` | Prototype, không là năng lực AI |
| Nhận dữ liệu kiểu event và chữ số | Shared ALIF + text; digit 8×8 | Prototype hẹp |
| Frozen H48 giữ thông tin quan hệ | R5B stored vA/vB ≈0.997, ridge both_A ≈0.986 | Diagnostic synthetic |
| Local readout học ánh xạ có cue | R6D n16k both_A ≈0.918, +23.05pp vs n1024; chưa independent re-run | Kết quả sơ bộ có raw, chưa xác minh toàn bộ gate |
| Khả năng chịu distractor | R6D `.55023` both_A | FAIL |
| Hidden plasticity học representation mới | P1-score narrow pass; DC-ALIF task-scoped | Chưa tổng quát/chưa stable multi-task |
| Giữ nhiều kỹ năng qua stream | Main P2 final34.74%, forgetting29.09pp; Gemini P2 final56.13%, forgetting21.89pp | FAIL |
| Sinh câu mới / thị giác có câu hỏi | Word dev35.94%, visual macrodev25.10% | FAIL |
| Năng lực hội thoại/vision/suy luận đa năng | Chưa có benchmark đạt gate | NOT DEMONSTRATED |

**Không gán số % tiến độ AGI.** Phần infrastructure đã có prototype thực; core scientific goal representation–learning–retention chưa được xác nhận. `STOP` trong một protocol không có nghĩa mọi kiến trúc neuron bất khả thi; `PASS` của một bài synthetic cũng không có nghĩa sản phẩm thông minh.

### 12.2 Khi nào nhiệm vụ kiểm toán Codex được tính là HOÀN TẤT?

Mọi điều kiện sau phải đáp ứng:

1. Có **bản đồ source** xác nhận commits + manifest + protocol + raw results đủ dùng; mọi thiếu sót nêu rõ.
2. Có **independent metric implementation**, mutation tests, và per-seed recomputation không lệ thuộc summary của agent.
3. Có **bảng từng claim** từ R1→R6D: confirmed-by-source / independently-reproduced / contradicted / invalid-measurement / untested; không biến 'reported' thành 'verified' trong văn bản.
4. R6D gate G1–G8 được audit với historical semantics; tất cả `null`, mismatched-run identities và thiếu integration ghi rõ. G6 `.55023` không tự động biến mất sau khi ghép kết quả.
5. Grok R6B shadow-definition/tests và partial protocol coverage được giải quyết trong audit harness, **không sửa bằng chứng lịch sử**.
6. Có protocol R6E/R7 **chờ duyệt** với giả thuyết loại trừ nhau, controls, seeds độc lập, prespecified gates, compute cost, stop conditions.
7. Đề xuất kiến trúc mới phải gắn với một lỗi nhân quả đã xác nhận, **không** từ trực giác rằng SNN càng giống não càng tốt.
8. Không thay production `main`, không thay live weights/model data, không tự merge, không mở P2 chưa duyệt, không dùng cloud LLM/teacher làm thành phần core.
9. Mọi commit/PR có full SHA; báo cáo chỉ ra failures và những thứ chưa làm. Nếu không thể chạy thí nghiệm vì thiếu môi trường/permission, nói thẳng và bàn giao phần audit tĩnh có thể xác minh.

**Verdict bàn giao cho Codex:** `PROCEED_WITH_FORENSIC_AUDIT; DO_NOT_MERGE; DO_NOT_SCALE; REQUIRE_USER_APPROVAL_FOR_NEW_RESEARCH_CANDIDATE`. Sau audit, nếu chứng cứ đủ thì chuẩn bị protocol tiếp; nếu phát hiện thêm sai lệch, kết luận `INCONCLUSIVE` và giữ lại artifacts lỗi.

---

## 13. ĐIỀU BẢN BÀN GIAO NÀY KHÔNG ĐƯỢC TỰ TUYÊN BỐ

- Không nói 'đã chạy lại toàn bộ các thí nghiệm': phiên soạn báo cáo đã đọc mã, báo cáo và một số raw `result.json` qua GitHub connector, tính lại một số aggregates, **không có môi trường chạy repo độc lập**.
- Không nói 'đã đọc từng file trong repository': đã duyệt source tree và đọc nhiều source/protocol/report đại diện; Codex chịu trách nhiệm đọc toàn bộ file có liên quan đến một claim cụ thể.
- Không nói một agent nói thật chỉ vì có PR, nhiều tests, CI hoặc hash; những thứ ấy là **bằng chứng về khả năng truy vết**, không bảo đảm design đúng.
- Không tự hồi tố biến R6A 'fragile temporal' thành xác thực hay R6D 'full PASS'; những kết quả này đã bị lỗi hoặc fail gate.
- Không đọc từ `r6d` rằng hidden synapses học được: hidden được frozen trong 2x2 và matched arms.
- Không lấy Gemini báo cáo `strict_binding` 98.78% thành cải thiện rộng; regressions đã được ghi.
- Không dùng saved Qwen/Gemini/Claude agent outputs làm labels đúng cho SCAR-X; chúng chỉ là công cụ nghiên cứu/đối chứng và phải có verifier.

**Thông điệp ngắn gửi Codex:** Nhiệm vụ của bạn không phải khiến SCAR-X thắng benchmark bằng mọi giá. Nhiệm vụ là tạo một chuỗi chứng cứ khiến những kết luận sai không thể tiếp tục được chấp nhận, rồi chỉ phát triển neuron/learning rule từ các nguyên nhân đã được xác minh.
