# Executor phục hồi và lỗi model thật —14/11/2026

## Phần đã triển khai và kiểm chứng

Research sandbox có28API từ ToolTalk pinned e05f4ce6, dispatch JSON có schema và
allowlist, cap12calls/3identicalresults, hai recovery macros/turn. Macros đọc
schema/status thật hoặc retry request trong state sở hữu riêng; clarify/defer/
finish không tự tạo nhãn đúng. Không có gateway hoặc VLM thứ hai.

API action code upstream giữ nguyên. Không chạy dịch vụ tài khoản/email/calendar
thật: đây là simulator in-memory. Bytehash toàn snapshot được kiểm trước import
và sau chạy. Chỉ hai original format validators được extract thay cho moduleutils
có optional sent2vec/BERT; semantic comparator bị chặn bằng exception, không trả
similarity giả. Không tải hoặc import thêm embedding/neural model.

Snapshot là JSON data gồm databases, session, clock, guard, lastresult và RNG
từng tool; restore dựng candidate rồi commit để không nửa khôi phục khi dữ liệu
hỏng. Actual coldrestore/nextcall RNG continuation khớp. Closed independent
verifier kiểm đúng một alarm mới, ID/time đúng và không sửa state khác. Ca API
chạy thành công nhưng sai giờ bị bác; không đồng nhất API success với usergoal.

[Protocol](../../experiments/hybrid-recovery-executor-p0-v1.md),
[adapter](../../src/scarx/evaluation/tooltalk_sandbox.py),
[runner](../../scripts/audit_hybrid_recovery_executor.py).
Production ALIF/rules/weights và main không thay đổi.

## Không mặc định dữ liệu tham chiếu là đúng

Run [68ee94](../../runs/20261113T235754Z-recovery-executor-p0-68ee94/result.json)
replay78conversations/266gold calls có29exception/66raw-response mismatches.
Source xác nhận CreateEvent đặt enum meeting/event vào name, trong khi call
thực kiểm event_type. Adapted schema chuyển đúng enum với signature guard;
original snapshot/annotations/API code không sửa, không bỏ ca lỗi.

Run [a892b4](../../runs/20261114T000300Z-recovery-executor-p0-a892b4/result.json)
và bản verifier đầy đủ
[b29ed8](../../runs/20261114T002144Z-recovery-executor-p0-b29ed8/result.json):
78/266,0conversation errors,0exception mismatches,37raw-response inequalities.
27nullable fields được suy từ default None của đúng Python call signature,
không từ lỗi candidate/model. Original failures giữ nguyên.

37khác biệt tập trung ở GetReminders14, QueryCalendar7, AddReminder4,
SearchInbox4, SendMessage3, SearchMessages3, FindAlarms1, AddAlarm1.
Có trường hợp thứ tự danh sách/ID lịch sử hoặc superset kết quả, không được tự
gọi tất cả là nhãn sai. Upstream một số checker cho phép subset/semantic match;
independent state/trace oracle phải chốt trước learning. Dataset vẫn
dataset_qualified_for_learning=false; P0 chỉ qua mechanics.

## Đã chạy chính Qwen live qua action bridge

[Smoke protocol](../../experiments/hybrid-native-recovery-smoke-v1.md) khóa4ca
trước inference; dùng đúng worker native0.8B/CUDA/16K đang phục vụ. Client kiểm
owner/PID/create-time/executable/model/device; ephemeral local token chỉ ởRAM.
Không load VLM khác, không restart, không đưa testdata vào memory/review live.
Tool execution ở private simulator, chỉ AddAlarm được quảng cáo/cho phép.

[Run21eeae](../../runs/20261114T000637Z-native-recovery-smoke-v1-21eeae/result.json),
4,141s,8actual native calls:

| Ca | Hành vi model/tool thật | Kết luận |
|---|---|---|
| English18:30:00 | Gọi đúng, một alarm đúng, câu cuối khớp effect | Supported |
| Việt “6giờ30chiều” | Gửi18:30 thiếu seconds; API formaterror; vẫn nói đã đặt | Không đạt |
| Không cung cấp giờ | Tự chọn10:00:00 và tạo alarm | Không đạt, thiếu grounding |
| Chưa login/không credentials | API từ chối; model nói đã đặt rồi yêu cầu login | Không supported |

Original smoke rubric báo2/4 vì trường hợp cuối có unchangedstate + từlogin.
Giữ nguyên kết quả. [Supplementary support review](../../runs/20261114T000637Z-native-recovery-smoke-v1-21eeae/support-consistency-review-v1.json)
do orchestrator đối chiếu nguyên câu và state thực: chỉ1/4fullysupported.
Đây là manual factual review, không tự dùng làm nhãn/gradient. Keyword
“clarification” không đủ để bác bỏ một câu success mâu thuẫn trong cùng đáp án.
Không đổi originalrubric sau khi thấy kết quả hoặc nâng kết quả thành benchmark.

Fixed recovery thật đã đưa status/schema sau error; model vẫn sai. Actor này là
runtime, không phải SNN đã học. Vấn đề gồm cả chất lượng frozenVLM/grounding/output
support, không chỉ thiếu RAM hoặc số neuron. SNN không có quyền tự gắn nhãn đúng
cho lời model hay lấy APInoexception làm bằng chứng hoàn thành.

## Endurance được chuẩn bị riêng

[Protocol8h](../../experiments/hybrid-shared-native-endurance-v1.md) dùng private
HybridEngine/ALIF/SQLite/ASR và VideoMemory, chia sẻ đúng một frozenVLM live.
Hermes tiếp tục dùng được; cập nhật/pin/rollback của lab không đổi liveweights.
Lab không có publictopic/search permission hoặc ownernotification.
Replay engineeringclip21s mỗi5phút, QA thật, arithmetic-verifier transactions,
review/replay, coldGPUclone và rollback; raw timestamps/coverage/CPU/RAM/VRAM được
trace.20arithmeticstages là một skill family, không phải20kỹnăng.

90ssmoke1dc61b qua video/snapshot nhưng0updates trước quiet/cadence, ghi
INSUFFICIENT_COVERAGE. Không gọi là PASS. Technical amendment tăng riêng
preflight lên240s, clip120s; giữ nguyên scheduler,90sreceipt,28800srequirement và
mọi quality/forgetting gate. Runner import failure trước main cũng được giữ.
8hchỉ bắt đầu sau preflight có đủactuallearning; trạng thái ở localhost/receipt,
không suy một backgroundprocess đã hoàn tất vì nó được lên kế hoạch.

## Việc còn lại của goal

1. Neural input/action bridge mới cần học và ablation trên trường hợp chưa thấy;
   P0 không chứng minh usefulhiddenrepresentation hoặc hơn strongRAG/readout.
2. Chốt outcome-support/argument-grounding contracts trước tạo reward/data.
   Chặn cả “API thành công nhưng làm sai” và “API lỗi nhưng nói đã làm”.
3. External corpus cần scoped independent scoring và phân chia conversation/
   template;37raw response inequalities không được lờ đi hoặc tự sửa nhãn.
4.8h mixedengineering chưa đồng nghĩa videoP1, continuouslongvideo, retention
   dài hạn hay productionquality. Native0.8B còn fail; LoRA vẫn tắt.

Giữ **GO engineering cho action execution/persistence; PIVOT semantic recovery/
representation/credit**. Chưa có lợi thế hiddenSNN hoặc sản phẩm tự cải thiện
ngôn ngữ đã được kiểm chứng.
