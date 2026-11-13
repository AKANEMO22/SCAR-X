# Vì sao SNN chưa học được nguyên nhân lỗi? — P0 14/11/2026

## Kết luận đã kiểm bằng source và tool thật

**Đường thông tin hiện bị thiếu trước khi tới synapse.** Detector lưu status và
probe, nhưng HybridEngine.goal_sequence chỉ mã hóa câu hỏi chung cùng attempts,
publictopic/evidence-present, reviewed/searched/verified-attempted và presence.
Nó không đọc reported error/probe outcome hoặc goal.reason. Vì vậy neuron
không được cung cấp phần phân biệt nguyên nhân mà giao diện đã hiển thị.

Thêm một hạn chế khác: action verify hiện chỉ có verifier số học. Action review
tìm evidence local; SNN chưa có executor tổng quát để chọn và thực thi một cách
sửa file/HTTP, rồi nhận bằng chứng độc lập về kết quả sửa. Thay đầu vào riêng
lẻ chưa làm cho các hành động đó tự xuất hiện.

Không đổi ALIF, công thức học, serving input/checkpoint, precision, live weights
hoặc LoRA trong lượt này. Qwen0.8B/Whisper/ALIF live tiếp tục GPU. Phần code mới
ở evaluation; không import nó vào engine đang dùng.
189Python checks qua, gồm14encoder/time/label guards,6schema regressions và
CPU/native/CUDA parity hiện có. Không dùng số tests này làm quality PASS.

## Phép thử thông tin từ ba lỗi file thật

[Protocol](../../experiments/hybrid-incident-information-p0-v1.md),
[runner](../../scripts/audit_hybrid_incident_information.py),
[actual Hermes helper](../../scripts/hermes_incident_information_fixture.py).
Dùng riêng temp HERMES_HOME/project; actual read_file dispatcher và public
readonly_probe chạy thật. Projectscope được cấp bằng fixture, không phải bằng
chứng liveGUI tự tìm ra scope. Không đọc project/chat/video riêng của người dùng.

| Trạng thái nguồn thay thế sau ba missing calls | Tool/probe quan sát được | Annotation bước tiếp theo |
|---|---|---|
| Có file README.md | read_ok=true, not_found=false | Kiểm tra nguồn đọc được |
| Không có file | read_ok=false, not_found=true | Hỏi nguồn còn thiếu |
| README.md là thư mục | read_ok=false, not_found=false; actual tool báo not a regular file | Làm rõ kiểu file/đường dẫn |

Cả ba đều tạo **cùng một ma trận input8×64, cùng SHA1657f01744ab…** từ serving
goal_sequence, dù probe và reason khác nhau. SNN/learning rule không thể khôi
phục phần thông tin không được đưa vào input. Đây là giới hạn ở đường dữ liệu,
không chứng minh kiến trúc SNN nói chung không thể học.

Research encoder giữ flags thực theo thứ tự, dùng nguyên row features64D hiện
có; mỗi observation8tick, ba observations24×64. Nó tạo ba input khác nhau,
không dùng label/causeID/scopehash/pathidentity/future success hoặc câu trả lời
người dùng. V2 bắt buộc actual tool-return epochs và decision cutoff, từ chối
future/backwards/nonfinite times; không đưa absolute timestamp vào vector.

Cận phân loại deterministic trên **ba annotation cụ thể** là1/3 với input cũ,
3/3 với input phân biệt được. Số3/3 là khả năng phân biệt input, **không phải
accuracy model**, không có training hay prediction trong P0. Các annotation chi
tiết cũng không đồng nhất với sáu action coarse của production: ask có thể là
một bước an toàn ở nhiều trường hợp. Không suy cận1/3 thành điểm trần mọi task,
không gọi đây là representation learning hoặc thắng RAG.

Probe failed/nonmissing còn có thể do permission/binary/nonregular source khác.
Không được suy riêng từ flags rằng đã hiểu chính xác nguyên nhân; cần thêm ca
đối chứng phân biệt và verifier. Trường hợp thư mục ở đây có filesystem type
do fixture đo riêng, không bí mật chuyển type/gold đó vào neural input.

Hai run immutable:

- [fd128e](../../runs/20261113T232035Z-incident-information-p0-fd128e/result.json):
  10,609s; encoder đầu dựa chronology do caller cấp.
- [2d5a24](../../runs/20261113T232337Z-incident-information-p0-2d5a24/result.json):
  10,609s; thêm actual return times và cutoff guards.
- [Auditor data-only](../../scripts/verify_hybrid_incident_information.py) tính
  lại groups/hashes/bounds từ ma trận và raw tool JSON, kiểm frozen source bytes;
  [audit2d5a24](../../runs/20261113T232337Z-incident-information-p0-2d5a24/independent-audit-v1.json)
  không có integrity problem.

Cả hai0synapse updates, arrays không đổi, data-only coldrestore exact. Bounded
read thật3/20lines có truncated=true; fullread20lines false. Không dùng số ca
hoặc số token của fixture làm tuyên bố đã học kỹ năng dài hạn.

## Chuẩn bị đối chứng độc lập: ToolTalk

Để tránh tự thiết kế vài ca dễ rồi gọi thành công, đã kiểm nguồn
[Microsoft ToolTalk](https://github.com/microsoft/ToolTalk) ở commit cố định
e05f4ce6132c80ed33392b81535b077d56ab28fd.
[Intake manifest](../../runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json)
xác nhận28easy+50hard conversations,503turns,266gold calls,28APIs/7suites;
không có exact full-JSON duplicates. Đây chưa là kiểm hết near/template leakage.
Archive261.057byte, source682.193byte, SHA/file inventory và MIT/LICENSE/NOTICE giữ.
Không install dependency hoặc inference/training trên corpus này.

Hai phát hiện P0 trước khi dùng scorer:

1. Upstream to_openai_doc đặt required ngoài parameters. JSONSchema validator
   chấp nhận emptyargs dù khai báo một field bắt buộc. Research-only normalizer
   đặt required vào đúng parameter schema; emptyargs bị từ chối, validargs qua;
   conflicting/unknown/duplicate declarations bị chặn. Không sửa snapshot gốc.
2. Upstream success kiểm đủ gold calls và không có successful unintended action.
   Fixture một call đúng cộng24failed reads giống nhau vẫn success=true;
   tương tự24failed mutation attempts. Successful unintended mutation thìfalse.
   Đây là giới hạn phạm vi metric đối với mục tiêu SCAR-X tránh loop, không
   phải tuyên bố ToolTalk sai với định nghĩa riêng của họ.

[Protocol harness](../../experiments/hybrid-tool-benchmark-harness-p0-v1.md),
[raw method/schema/scorer fixtures](../../runs/20261113T233655Z-tooltalk-harness-p0-0c9bc7/result.json).
Chỉ compile hai pure methods đã đọc từ AST, bỏ descriptor decorators để gọi
trực tiếp; whitelist calls/globals và dùng local fixture comparator. Không import
toàn upstream package, không chạy API gửi email/account/calendar hay tạo tài
khoản thật. LICENSE/NOTICE đi kèm extracted methods.

Lượt đầu1a783b dừng trước khi gọi method vì namespace hạn chế thiếu annotation
type bool. Original runner và failure receipt được giữ; lượt sửa0c9bc7 mới chạy
fixtures. Không ghi đè lỗi cũ thành PASS.

Upstream semantic checker có lazy sent2vec Vectorizer; không được âm thầm tải
thêm embedding/BERT vào một arm rồi báo cùng model/RAM. Independent executor/
state+trace scoring và budget cần được chốt trước inference. Normalizer và
technical fixtures không biến SCAR-X thành hệ thắng benchmark.

## Điều cần làm tiếp, theo thứ tự

1. Nối observation đủ thông tin tới một shadow policy version rõ ràng; duy trì
   encoder/checkpoint compatibility, không thay input của weights live im lặng.
2. Thiết kế action bridge có quyền giới hạn và independent recovery verifier.
   Tách lựa chọn của ALIF khỏi runtime override; reward action do guard chọn là
   teaching signal, không chứng minh on-policy neural credit assignment.
3. Khóa external DEV/TEST theo conversation/template, chống future gold và trải
   nghiệm xuyên split. Kiểm upstream schemas/executor/scorer bằng regression
   trước; thêm unresolved-loop rate, số failed calls/cost và đúng finalstate.
4. Cùng VLM/prompt/tools/memory/source/time budgets: strongRAG/rule experience,
   frozen SNN, readout-only, hidden, shuffled và hidden-reset. Giữ3seeds,
   >=5pp/lower95%CI>0 và acquisition+20stage forgetting3/5pp. Không gọi việc
   encoder tách được ba state là đã qua những gate này.
5.8h production/mixedvideo soak và dataset video100DEV+100TEST vẫn thiếu.
   Resource probes trước không đủ thay thế. AutomaticLoRA vẫn tắt.

[Reflexion](https://arxiv.org/abs/2303.11366) và
[ExpeL](https://arxiv.org/abs/2308.10144) là lý do cần đối chứng experience/text
memory: cải thiện hành vi có thể xảy ra mà không cập nhật parametric weights.
Chưa có bằng chứng lợi thế riêng của hidden SNN trong SCAR-X.

Quyết định hiện tại: **PIVOT information path + action/verifier bridge; chưa
train trên ba fixture, chưa promote encoder, chưa đạt autonomous/lifelong PASS.**
