# Tool failures, explicit feedback and remaining GPU VRAM — 14/11/2026

## Quyết định vận hành

Giữ **Qwen3.5-0.8B nguyên precision + ALIF H128 + Whisper base FP32**, tất cả
tính toán neural trên CUDA trong GPU mode. Host I/O, tokenizer, SQLite và điều
phối vẫn dùng CPU. Không chạy hai VLM đồng thời, không đổi công thức ALIF,
không nhập weights lab vào live, không bật LoRA. Main gốc và active R13 giữ nguyên.

Khoảng 3 GB người dùng nói là **VRAM GPU rời còn trống**, không phải RAM máy.
Đã thử dùng dung lượng ấy cho VLM lớn hơn và buffer thực. Không cấp phát rỗng
để làm GPU đầy: occupancy, tốc độ và khả năng giải bài là ba phép đo khác nhau.
GPU mode không giới hạn host RAM/commit; CPU mode giữ 4.500.000.000 byte.

## Các lượt thử dùng thêm VRAM

Giữ context16K, native BF16/F32, KV BF16 và local ALIF; không quantization,
pruning, CPU weight offload, giảm context hay hạ precision để ép qua gate.

| Candidate/profile | Global VRAM free nhỏ nhất | Quyết định |
|---|---:|---|
| Qwen2B lượt đầu | 388 MiB | Dưới reserve512; không promote |
| Qwen2B batch128/micro64, FAauto, graphs | 423 MiB | Dừng sau load, không thử ảnh lớn |
| Qwen2B batch64/micro32, FAon, graphs | 418 MiB | Dừng sau load |
| Qwen2B batch64/micro32, FAon, no graphs | 424 MiB | Dừng sau load |
| LFM2.5-VL-1.6B, batch128/micro64 | 964 MiB | Resource probe qua, quality gate không qua |
| Qwen0.8B đối chứng cùng lượt LFM | 2.198 MiB | Giữ bản đang phục vụ |

Các số là sampling Nvidia toàn GPU, gồm ảnh hưởng ứng dụng khác và transient,
khác allocation riêng của worker. Không đóng app của người dùng để tăng headroom.
Ba buffer profiles đăng ký trước không giải quyết được thiếu VRAM của Qwen2B.
Mỗi arm dừng đúng preflight; không diễn giải các ca chưa chạy thành quality FAIL.

Đã thêm config batch_size, microbatch_size, flash_attention và cuda_graphs.
Mặc định vẫn128/64/auto/on. CUDA graph env chỉ áp dụng owned child. Trước ASR,
runtime kiểm tra freeVRAM; dưới 538 MiB workspace + 512 MiB reserve thì nhả
nguyên VLM, chạy ASR CUDA rồi nạp lại. Đây là scheduling component, không paging
layer qua SSD hoặc fallback neural CPU. Có chi phí reload/prefix cache.

- [2B buffer protocol](../../experiments/hybrid-native-2b-buffer-v2.md)
- [Raw ba profiles](../../runs/20261113T214816Z-native-2b-buffer-v2-028963/manifest.json)
- [Lượt 2B đầu và matched RAG/SNN](RAG_SNN_AND_VRAM_20261114.md)

## LFM1.6B: vừa VRAM nhưng chưa đủ lý do thay model

Nguồn chính thức [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B),
revision919fde3d022e3f90a4716006f993938ee8c2eb97, source3.193.334.216 byte,
SHA256 7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d.
Giữ LICENSE local; không commit/redistribute weights. Model card không quảng
cáo tiếng Việt trong tám ngôn ngữ hỗ trợ; không suy benchmark Việt từ tên model.

Original reference lần đầu FAIL vì loader Transformers đổi prefix vision key,
không phải mismatch giá trị. Giữ reference.json ban đầu. V2 chỉ cho ánh xạ đúng
prefix đã xác định, kiểm dtype/shape/toàn bộ tensor byte, không bỏ tensor; kết
quả không mismatch/unused. Native probability parity3case qua cho cả hai model.
Config/manifest xác minh model identity thực, không gắn nhãn LFM thành Qwen.

| Phép đo cùng 12 ca engineering đã khóa | Qwen0.8B | LFM1.6B |
|---|---:|---:|
| Đúng strict output contract | 8/12 | 6/12 |
| Đúng strict tiếng Việt | 1/4 | 2/4 |
| TTFT lớn nhất trong12ca | 0,859s | 1,453s |
| Load + native qualification | 10,844s | 15,203s |
| Mixed ASR+hello cycles | 3/3 | 3/3 |
| Full-state rollback exact | Có | Có |

LFM trả “Blue.” thay “blue”, hoặc thêm câu giải thích khi yêu cầu chỉ một số:
đây là lỗi contract, không bằng chứng không nhận ra màu/số. Giữ strict scoring
đã khóa; không xóa dấu chấm/Markdown sau khi thấy kết quả để đổi FAIL thành PASS.
Cả hai vẫn chọn port8123 cũ thay9237 trong strictJSON, dù JSON hợp schema.
Grammar sửa cú pháp không bảo đảm chọn đúng dữ kiện mới.

LFM loaded global used4307 MiB; transient minfree964 MiB, sharedGPU khoảng180,2
đến182,3 MB trong nhóm tiến trình đo. Ba chu kỳ ASR+hello hoàn tất không cần nhả
VLM; số thời gian cycle bao gồm cả câu hello, không là ASR-only benchmark.
Không OOM/crash trong probe ngắn. Không suy thành 8h production PASS.
Gate yêu cầu không kém đối chứng ở tổng và Việt: **6<8 nên NOT_QUALIFIED**.
Không promote, không thử lại cấu hình khác trên chính12ca này.

- [Protocol khóa trước inference](../../experiments/hybrid-native-lfm16-vram-v1.md)
- [Raw baseline](../../runs/20261113T222819Z-native-lfm16-v1-ae86cd/paired-baseline.json)
- [Raw candidate](../../runs/20261113T222819Z-native-lfm16-v1-ae86cd/paired-candidate.json)
- [Audit độc lập từ chunks/case bytes/source hashes](../../runs/20261113T222819Z-native-lfm16-v1-ae86cd/audit-rescore-v1.json)
- [Auditor không train/sửa live](../../scripts/audit_hybrid_lfm16.py)

Audit tính lại score/Vi từ expected đã khóa và raw SSE chunks, kiểm source bundle,
CSV minimumVRAM, identity và parity. Không có integrity problem.12ca DEV nhỏ không
đủ kết luận LFM hay Qwen tốt hơn tổng quát hoặc ngang model thương mại.

## Tool lỗi → điều tra giới hạn → phản hồi học

Đã nối published Hermes pre_api_request/post_tool_call hooks vào cùng gateway:

> Tool result thật → detector nonprogress → incident có nguồn → local learning
> goal → existing ALIF đề xuất hành động → runtime kiểm quyền/tiến độ → chủ máy
> đánh giá rõ ràng → guarded local update + checkpoint.

Hook chỉ theo đúng scarx-hybrid trên loopback8775 và có session/turn identity.
Gửi hashes của arguments/results cùng status, error, empty/spillover, duration;
không gửi nội dung file, command, câu hỏi riêng hoặc URL vào observation API.
Không có server/daemon mới. Plugin dùng public ctx, không import Hermes core.

Sau3kết quả giống nhau liên tiếp tạo một incident trong turn. Polling process
hợp lệ được miễn; callID trùng không được đếm lại. Bound64turn buckets,
16recent observations/bucket và32incidents; controller goal/decision cũng hữu hạn.
Hook lỗi không làm hỏng tool bình thường. Missing scope/route khác không góp dữ liệu.

Nếu read_file đã được quảng cáo trong lượt và ctx có project scope, có thể chạy
một readonly probe qua chính ctx.dispatch_tool và guards sẵn có: cùng basename
.md/.txt ở projectroot nếu path khác, hoặc README cho localfolder request bị
websearch rỗng. Không shell, recursive search, đọc secret, đổi quyền hoặc tự gửi
webquery. Read success chỉ hỗ trợ giả thuyết path/source binding; không chứng
minh file đó giải quyết câu hỏi. API không cho gắn verified=true để tự tạo nhãn.

Incident xuất hiện ở Hỏi để học/trace. Log ghi proposed action, executed action,
selector và số updates. Lúc nghiệm thu, ALIF đề xuất defer, runtime_progress_guard
chuyển sang ask: **không gọi đây là neural tự nhận biết nguyên nhân**.
Goal của các incident khác nhau có identity riêng; không sửa neural input features.

Trả lời thường không có reward; Bỏ qua không là nhãn âm. Chỉ nút đánh giá
“Trả lời · hữu ích” hoặc “Chưa hữu ích” tạo +1/-1 qua learn đã có. Text người dùng
vẫn human_statement, không tự trở thành sự thật phổ quát. Negative feedback
để goal deferred. Guard có thể từ chối update, không bảo đảm mỗi penalty đều tốt.

Synapse update, RNG/replay, question closure và goal state commit cùng checkpoint.
Lỗi persistence khôi phục trạng thái, giữ câu hỏi chờ để retry không học hai lần.
Audit incident/evidence có chủ ý vẫn giữ khi model rollback; không xóa lịch sử lỗi.

## Nghiệm thu và giới hạn

[Protocol](../../experiments/hybrid-tool-experience-loop-engineering-v1.md) và
[run mới nhất5e0ead](../../runs/20261113T230549Z-tool-experience-v1-5e0ead/result.json):
23,953s, real Hermes filetool trả missing3lần, readonly alternative đọc được,
1incident/1goal, observation không đổi weights. Injected negative fixture feedback
đi qua API, reward−1 tạo1local/hidden update, fullrollback trả0exact; audit còn.

Đây là **fault injection thủ công trong sandbox** với projectscope cung cấp
cho fixture, không phải model tự chọn ba lời gọi hoặc liveGUI chứng minh tự sửa.
Owner rating trong run là tín hiệu fixture, không là đánh giá thật của người dùng.
Các run cũ feedc8/90dede/d99a08 giữ nguyên;18c0ad chỉ prepare rồi sourceguard
chặn trước run.5e0ead gồm bổ sung cleanup dedupstate khi đổi modelroute và
giới hạn một probe/turn dù có nhiều signature hoặc pre_api_requests.

169 SCAR-X Python checks gồm CUDA3seed×20updates và4nativeCPU parity,64Hermes
compat/guard/compression checks qua canonical runner,5Node checks qua. Node kiểm
trực tiếp widget: trả lời/bỏqua không có helpful; chỉ explicit rating có flag.
Fault tests bao phủ positive/negative/unrated/skip save failure và retry.
CuPy cảnh báo không tìm thấy toolkit path đầy đủ; CUDA kernel/parity thực vẫn qua.
Plugin validation qua publicSDK, installer có backup; không claim UI mới đã
được kiểm trực quan nếu chỉ có component/API checks.

Production giữ0.8B/GPU và5local updates thật trước lab, không nhập update fixture.
Trace tại [localhost8775](http://127.0.0.1:8775); actor orchestrator ghi đo/thử nghiệm,
khác neural_policy/runtime. Debug hiển thị riêng hostRAM và measured dedicatedVRAM.

## Điều còn thiếu trước production

1. Matched DEV hiện **chưa chứng minh hidden SNN hơn mạnh nhất RAG/rule/readout**;
   encoder64D còn mất metadata/quan hệ, retention chưa qua3/5pp. Không scale để né.
2. Chưa chứng minh model tự chẩn đoán lỗi semantic, chọn probe mới hoặc dùng bài
   học sang task mới. Loop vừa nối là plumbing/explicit policy feedback.
3. Chưa chọn được native VLM lớn hơn vượt đồng thời quality/headroom/latency.
   0.8B vẫn yếu về Việt, tool use, code và chọn dữ kiện sau compression.
4. Video dài/realtime, ASRVi, 100+100 videoheldout, formalP1/P2 và8hP3 còn mở.
   30phút soak trước không được tự nâng thành8h hoặc lifelong PASS.
5. AutomaticLoRA vẫn NOT_QUALIFIED/tắt. Không tự luyện VLM từ câu trả lời của nó.

Quyết định: **GO kỹ thuật cho bounded incident/feedback; PIVOT về representation,
credit và semantic reliability. Không có bằng chứng scientific/production PASS.**
