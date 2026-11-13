# Workspace, tool loops, compression và execution mode — 14/11/2026

**Kết luận: GO kỹ thuật cho unifiedGPU/CPU và vòng vận hành; PIVOT về chất lượng
phân tích/context-use, chưa lifelong hoặc LoRA PASS.** Không đổi kiến trúc
ALIF, học cục bộ, precision hoặc scientific gates. LoRA tự động vẫn chưa đạt chuẩn.
Các lỗi/runs trước giữ nguyên. Phê duyệt mới nhất yêu cầu GPUmode áp dụng cho
cả VLM/vision, ASR và forward/local learning ALIF; CPUmode không dùng GPU rời
cho tính toán neural. Tokenization, filesystem/network, SQLite, RNG/serialization
và điều phối vẫn là công việc host. Đây không phải yêu cầu CPU utilization=0.

## Nguyên nhân được xác minh

1. `Greeting #6` đã có đúng cwd của project `app z fold` trong Hermes DB.
   Lượt đó thực hiện31toolcalls; phần cuối có22web_search cùng args/result rỗng.
   Tool guard của desktop mặc định warn-only nên lời cảnh báo không chặn thực thi.
2. Bản tích hợp trước cho web tools eager nhưng defer filesystem/terminal sau
   discovery. Snapshot systemprompt của session cũ không được lưu trongDB, nên
   chưa đủ chứng cứ khẳng định API cũ có/thiếu cwd hint; không suy ra từ trườngNULL.
   Model nhỏ hiểu “app z fold” thành Samsung và đi sai công cụ. Có project/folder trong UI
   không có nghĩa nội dung file đã được model đọc.
3. Chỉ main-agent64K floor đã có ngoại lệ local16K; feasibility của auxiliary
   compression vẫn từ chối16K. Sau sửa floor còn phát hiện gateway mặc định
   output512token trong khi Hermes yêu cầu summary khoảng2000token.
4. Chuyển GPU cũ chỉ chuyển VLM/vision. ALIF NumPy và Whisper`-ng` vẫn CPU.
   Đây là giới hạn của bản cũ, không phù hợp yêu cầu mode mới.
5. Auto-rest cũ nhả VLM sau180s, mất prefix cache. Lượt sau phải tải native
   model/CUDA lại; gateway boot còn checksum toàn model, numeric parity, init
   CUDA/ALIF. Full Hermes startup, tool I/O và vòng tool lặp là các độ trễ khác.

## Thay đổi triển khai

- Giữ nguyên toolsets được phép; file/terminal/video và web có schema trực tiếp,
  các công cụ khác vẫn qua discovery. SCAR-X loopback8775 có hard stop sau3
  idempotent no-progress results, tối đa6websearch/turn và24APIiterations/turn.
  Model/provider khác giữ policy cũ. Installer có backup/hash và từ chối patch
  nếu source Hermes đổi, không ghi đè mù hoặc sửa branch main của SCAR-X.
- Nghiệm thu DEV chứng minh0.8B có thể bỏ qua named`tool_choice`, chép marker
  prompt dù cwd đúng. Vì vậy yêu cầu đọc workspace rõ ràng được một runtime
  prelude đọc giới hạn directory/docs. Hermes thực thi terminal/approval bình
  thường; không có terminal service mới. JSONtoolcall ghi rõ producer/actor là
  runtime, không tuyên bố VLM/SNN tự nghĩ ra. Câu trả lời VLM giữ nguyên.
  Chỉ chạy một lần trước tool result, không sửa history/schema/user text.
- Native16K aux route được chấp nhận, model khác vẫn có64K floor. Summary nhận
  output budget theo tokenizer thật, tối đa4096 và chỉ phần trống còn lại trong
  context16K; normal output tối đa1024. Không quảng cáo context giả. Khi summary
  nhỏ vẫn thất bại, dùng ngay deterministic/redacted recovery đã có trong Hermes
  cùng transaction thay vì tiếp tục retry đến overflow. Đây là extractive memory,
  không phải semantic compression thành công hay synapse học kiến thức.
- Inference repeat_penalty1.05 hạn chế output lặp; native numeric parity endpoints
  không đổi sampling. Đây là tham số suy luận, không nén/đổi trọng số.
- GPUmode: original VLM+vision CUDA0, Whisperb5454CUDA12.4 dùng checkpointFP32,
  ALIF cùng phương trình bằng CuPyFP32. RNG/checkpoint định dạng cũ giữ nguyên;
  rollback/baseline/probe cũng dùng device của learner. Không torch/autograd/BPTT.
- CPUmode: native CPU VLM/Whisper`-ng`, NumPyALIF; không cần import CuPy. iGPU
  chưa có backend được đo; không gọi CUDA là iGPU. GUI Windows vẫn có thể dùng
  GPU để vẽ cửa sổ, tách khỏi device của model.
- GPU auto-rest giữ VLM warm mặc định (`gpu_keep_warm=true`); manual Ngủ vẫn
  nhả model, CPU auto-release giữ nguyên. Debug hiển thị placement từng phần.
  Observer reset cursor khi đổi process để không mất trace lúc chuyển lab/live.

## Bằng chứng và giới hạn hiện biết

Protocol: [hybrid-workspace-gpu-reliability-v1](../../experiments/hybrid-workspace-gpu-reliability-v1.md).
Runner: `scripts/hybrid_reliability_soak.py`; actual Hermes process:
`scripts/hybrid_agent_workload.py`. Read-only plugin chỉ ở home test, không cài
vào profile người dùng. Private project transcripts giữ local, public receipt
chỉ có checksum/path. Nhánh main, activeR13 và checkpoint người dùng được bảo toàn.

Hermes regression:2tests đỏ trên bản cũ (filesystem disclosure/loop boundary,
aux64K), sau sửa64tests qua bằng canonical`run_tests.sh` trong source copy và PM
test env cô lập; các route khác vẫn từ chối nhỏ hơn64K. Tests không được phép
đọc/ghi real Hermes home; lượt đầu phát hiện điều này và đã chuyển sang source
copy, không nới file-I/O guard. NumPy không được cài thêm vào Hermes để chạy helper.

CUDA kiểm toán:3seed×20updates, tất cả state/trace/weights CPU↔GPU trong
atol/rtol3e-5; cùng device save/reload/next update khớp byte. Có test cấm gọi CPU
`SpikingCore.step` trong GPUlearner, DBfault/rollback giữ device. Đây là parity
kỹ thuật, không phải20chặng retention chất lượng. Chưa có hidden advantage mới.

Workspace DEV: nhiều lượt đầu không đọc file và chép marker prompt. Prelude sửa
được truy cập nguồn: terminal/read_file thật, không web. Chất lượng chưa ổn định:
một lượt Việt lặp văn bản bị Hermes chặn; lượt Anh giải thích đúng fixture;
short prompt có thể tiếp tục đọc thừa đến ngân sách. Không gọi tất cả là PASS.

Compression DEV thật:54messages/71.056ký tự →14messages/46.231ký tự, không64K
error; tokenizer đo13.481input và2.871token outputroom. VLM vẫn có thể sinh đến
cap, fallback bảo toàn anchors. Kiểm tra hỏi lại trả mãJADE-472 đúng nhưng cổng
8123cũ thay vì9237mới. Vì vậy memory/update-understanding quality FAIL trên mẫu
này, dù compression transaction chạy được. Không dùng đáp án sai này làm label.
Kiểm tra rawcompact: giá trị9237 vẫn hiện ở hai tin nhắn cuối được giữ nguyên.
Vì vậy lỗi trên mẫu này bao gồm VLM không ưu tiên thông tin mới, không chỉ mất
dữ liệu do cửa sổ/context; đây cũng không phải phép đo forgetting do train synapse.

Trọng số/phiên người dùng không nhận các cập nhật lab. Các số3updates trong
nghiệm thu là action-policy từ arithmetic verifier + replay; không phải VLM đã
học toán hay ngôn ngữ. Pilot automaticLoRA trước vẫn NOTQUALIFIED vì unsafe
create/reuse proposals; không bật lên để che lỗi pipeline.

## Runs và thất bại được giữ

- `20261113T192235Z-workspace-dev-v1-65ee6b`: harness chưa bind sessioncwd;
  lấy launchcwd. CácDEV kế tiếp sửa cwd nhưng model vẫn không chọn tool.
- `20261113T192929Z-workspace-dev-v1-9bd8b4`: prelude + file reads thật, đọc được
  fixture; output còn diễn giải thiếu chính xác, không suy rộng năng lực.
- `20261113T193050Z-hybrid-reliability-v1-755916`: helper import packageSCAR-X
  làm đòi NumPy trong Hermes env;5jobprecondition lỗi, đã dừng process lab đúng
  PID/creationtime/command, giữ abortedreceipt và partialprogress. Không30minPASS.
- `20261113T193253Z-hybrid-reliability-v1-670196`: workspace/jobs thật; compression
  output512truncated rồi95stimeout. Dừng trước mixed-endurance theo preflight.
- `20261113T193900Z-compression-dev-v1-2946dc`: có outputroom nhưng summary vẫn
  đến cap; caller retry ladder. Dừng ownDEV, giữ aborted/log, không đổi thànhPASS.
- `20261113T194210Z-compression-dev-v1-9db304`: fallback chạy, hỏi lại sai cổng.
- `20261113T194426Z-hybrid-reliability-v1-6da252`: run đang tiếp tục; kết luận cuối
  phải dựa trên receipts/elapsed thực, không số phút dự kiến.

## Nghiệm thu đã hoàn tất

[Run6da252](../../runs/20261113T194426Z-hybrid-reliability-v1-6da252/manifest.json),
[audit từ raw](../../runs/20261113T194426Z-hybrid-reliability-v1-6da252/independent-audit.json)
và runner/auditor được giữ để kiểm tra lại. Source-byte-bundle khớp manifest;
supplement dependency được ghi rõ là capture sau khi bắt đầu, không sửa protocol.

| Kiểm tra | Số đo/kết quả |
|---|---|
| Mixed-endurance | Đủ1.800s,45real requests, không exception trong mixed loop |
| Sleep/wake | Lượt đầu +4lượt theo chu kỳ đều hoạt động; actual worker unload/reload |
| ALIF |1machine-verified update+2replays, tổng3updates/3hidden; không tự mở thêm budget |
| Checkpoint | Full rollback exact; cold GPU→CPU→GPU arrays/RNG/replay/review/budget exact |
| Native gate | Original-precision storage và3probability cases qua ở cả3process |
| CPU | VLM/vision, ASR, ALIF đềuCPU; peakconservativeRAM3,971GB<4,5GB; không resource exceed |
| GPU | Tất cả neuralCUDA; peakhostRAM5,208GB, đúng profileGPUuncapped; không OOM |
| Nvidia sampling |10.125samples, peakdeviceutil96%; globaldevicepeak3700MiB, không đồng nhất với owned-PID allocation |
| Warm stream TTFT | Median0,141s; p950,203s, trên45lượt nhỏ; không phải full-agent latency |
| Explicit cold VLM wake | TTFT1,515s; native resume riêng~1,297s |
| Gateway ready từ coldprocess |GPU15,141s; CPU11,265s; restartGPU13,390s, gồm checksum/CUDA/P0 |
| ASR cùng waveform |GPU1,844s; CPU3,610s;1clip, không suy ra quality hoặc realtime cho mọivideo |
| H128 microbenchmark |CPU0,197ms vsGPU3,407ms median/forward,50repeats sau10warmup; tinyGPU work chậm hơn ở đây |

Soak tổng2076,485s gồm preflight,30phút mixed và đóng/lưu; CPUcheck17,984s,
restartGPU16,5s. CPU/GPUspeedHello không được lấy warmGPU chia cho coldCPU.
MicrobenchmarkH128 giữ FP32/seed/input, outputdiff5,96e-8; suy đoán overhead
kernel/transfer, chưa kernel-profile. Yêu cầuGPU vẫn được thực hiện, không âm
thầm fallbackCPU để đẹp tốc độ. Đây chưa là8giờ release soak hoặc dài ngày.

Finalregressions:128SCAR-Xchecks gồm4nativeCPUparity và4actualCUDAcases;
64Hermeschecks bằng canonicalrunner;4Nodecomposerchecks. KhôngskipCUDA rồi gọi
làPASS. CuPy có warning về toolkitroot trong cách cài wheel tối thiểu; CUDA
execution/parity thực đã qua, không coi warning là lỗiCPUfallback.

### Luồng đọc folder sau soak

Sau runfrozen trên, [repairv2](../../experiments/hybrid-workspace-purpose-repair-v2.md)
thêm overview budget4results, giữ budget qua Hermes empty-response nudge.
Các yêu cầu code/edit/debug không bị overviewbudget. Khi model chưa trả lời
được, runtime báo dừng rõ ràng với nguồn đọc được; **không thay bằng bản phân
tích giả rồi chấm như output VLM**. Source v2 không được gọi là đã qua30minrunv1.

DEV phát hiện first-tooloutput~18.960chars vượt inlinebudget16K, bị spill ra
cache; readonlyharness lại chặn cache của chính nó. Đã sửa denial củaharness
và giảm payload overview còn≤3200serializedchars với paths bình thường, có cờ
truncation/source snippets. Không nén hoặc hạ precision trọng số.

[Finalv2summary](../../runs/20261113T204311Z-workspace-purpose-v2-1ef03e/summary.json):
actualVi3terminalrepeats bị hardstop; actualEn1terminalread→bản tóm tắt trong8,85s,
nhận đúng một manifestcom.zing.zalo/Android nhưng còn kết luận thiếu nguồn
(`custom SDK`, xemapktool như kiến trúc ứng dụng). Không thể gọi toàn bộ folder
nhiều artifact là một app chỉ từ một manifest. Vì vậy chưa PASS phân tíchproject.
Rawactualproject outputs giữprivate local, summary lưuSHA, không publiccode.

**Đính chính metric:** trường`semantic_quality_pass` trong một số DEVsummary
chỉ là kiểm từ khóa, tên gọi sai. OriginalJSON được giữ; runner mới đổi tên
`keyword_probe_pass`; review không lấy từAndroid xuất hiện trong controlled
notice hoặc trong outputVLM làm semanticPASS.

Fixture lồng trong SCAR-XGit tree còn kế thừa researchAGENTS, gây confound.
Fixture trung lập ngoài repo
[ee37c6](../../runs/20261113T203810Z-workspace-clean-fixture-v1-ee37c6/result.json)
đọcREADME/main thật, trả đúng9237 trong6,78s, không cóSCARXresearchinstructions.
Đây chỉ là fixture đọc đơn giản, không phải generalization hoặc hidden learning.
Compression trung lập
[81a191](../../runs/20261113T204807Z-compression-neutral-v1-81a191/result.json)
không còn confoundAGENTS nhưng vẫn chọn8123cũ thay9237; transaction thu gọn được,
context-use còn lỗi. Không nhập những outputs này làm teacher labels.

Sửa readonlyharness/overviewpayload xong, run`1ef03e` cho actualEn đọc1tool,
actualVi còn lặp3terminalcalls và bị hardstop. Fixture trong repo vẫnconfounded;
không lấy keywordprobe làm semanticPASS. Không có tuyên bố đã giải quyết mọi
phân tíchcode tiếngViệt hoặc đã học lâu dài sau những sửa đổi runtime này.

GPU keepwarm mặc định đã bật; explicitSleep vẫn nhảVLM. MainSCAR-X/SNN-only và
R13 không bị sửa/dừng; khôngmerge/push hoặc bật adapter. CPU4,5GB giữ nguyên.
ComputerUse đã xác nhận trực tiếp Hermes hiển thị **VLM/visionGPU · ASRGPU ·
ALIFGPU**, backendready, learner5updates production (khác3updates lab). CLI đã
restart gatewayHermes để loadcompatibilitypatch; các phiên cũ không bị xóa.
NativeP0 của model vẫn mất khoảng13–15s khi khởi động process mới; keepwarm
tránh lặp chi phí đó sau mỗi lần rảnh, không biến coldboot thành0s.

## Nguồn dependency và xem GPU

Whisper officialb5454CUDA12.4 archiveSHA256:
`afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a`.
[Upstream release](https://github.com/ggml-org/whisper.cpp/releases/tag/b5454).
CuPy14.2, NVRTC/cudart12.9 pinned trong private SCAR-X runtime-env; tái dùng
cuBLASDLL pinned từ native VLM, tránh tải thêm553MB bản trùng. Không Docker hoặc
system-wide CUDA Toolkit. [CuPy installation](https://docs.cupy.dev/en/stable/install.html).

Task Manager có nhiều engine GPU; xem đúng RTX4050 và CUDA/Compute, không chỉ
graph3D. GPU0/1 là thứ tự Windows, CUDA0 là thứ tự thiết bịCUDA trong máy này.
[Microsoft giải thích engine/counter](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).
Nvidia utilization là toàn GPU, có thể gồm app khác; PDHmodel allocations và log
backend được báo riêng. Không cần utilization100% khi không có phép tính hữu ích.
