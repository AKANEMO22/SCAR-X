# P0 đề xuất: bằng chứng hoàn thành tác vụ cho learning controller

**Đã duyệt P0 bằng “làm đi” ngày15/11/2026; chưa mở P1/train.** Đây là bước sửa
định nghĩa outcome trước nghiên cứu neural. Không đổi hoặc khởi động lại history
P0 FAIL; không mở DEV, train, model mới, corpus mới, LoRA hoặc live integration.

## Literature-first và khoảng trống thực tế

- ToolTalk tại pin đã intake `e05f4ce6`: audit SCAR-X trước chỉ ra gold calls
  cộng24 failed attempts vẫn có thể success theo metric của họ. Tái dùng source/
  audit đó, không chạy benchmark/API/model thêm để lặp kết quả.
- [ToolSandbox paper](https://aclanthology.org/2025.findings-naacl.65/) và
  [code tác giả](https://github.com/apple-aiml-research/ToolSandbox/blob/main/tool_sandbox/common/evaluation.py):
  đã đọc abstract, README/license, SnapshotConstraint, CachedSimilarityCalculator,
  EvaluationResult, milestone mapping/evaluate và defaults theo cột. Blob
  `af9faacc65c0e700a52093fb0c5d77ac6c36c449`. Không claim đã kiểm toàn repo/paper.
  REUSE nguyên tắc snapshot trước/sau, dependency order và forbidden events;
  không copy/install framework, Polars/NetworkX/ROUGE hoặc chạy RapidAPI.
- Code upstream có partial similarity, ROUGE-L ở một số cột text; guardrail và
  minefield có logic riêng, và không có milestone mặc định cho similarity1.
  Không nhập aggregate đó thành nhãn học đúng/sai tuyệt đối cho SCAR-X.
- RC13 đã chặn phép tính tình cờ cấp nhãn cho sai yêu cầu. Khoảng trống còn lại:
  chưa có contract đủ rõ để phân biệt API thành công với task thực sự hoàn thành,
  rồi nối outcome đó về đúng quyết định/action được chọn. SNN input hiện còn bỏ
  status/probe; P0 này **không sửa encoder hoặc giả đã giải quyết credit**.

## Phạm vi được đề nghị

Viết evaluation-only contract trên owned temp SQLite/files, không import vào
engine/gateway. Mỗi trace có task ID, scope/source ID, decision cutoff,
proposed/executed action và selector origin; event có thứ tự, tool outcome,
hash của snapshot trước/sau. Checker giữ target/required effects riêng, không
đưa gold/future state/familyID vào observation mà một policy được phép thấy.

Outcome gồm riêng task completion, forbidden attempts, unresolved repeats,
cost và insufficient-info/clarification. Một trạng thái thiếu bằng chứng là
unverified, không thành failure fact hoặc nhãn negative synapse tự động.
Không đồng nhất quyền cho một action với neural đã lựa chọn; không ghi credit
cho SNN khi guard/fixture chọn action. Không forward, không synapse update.

## Chỉ12 fixture kỹ thuật, không là dataset chất lượng model

1. Đọc đúng file/scope và giá trị yêu cầu: chứng minh bound source/effect.
2. Đọc được file khác, cùng kiểu nội dung: không đủ hoàn thành task gốc.
3. Tool trả success nhưng không có effect được yêu cầu: không task proof.
4. Tool không lỗi, thiếu required effect: không task proof.
5. SQLite thay đúng record/field được yêu cầu: effect proof hợp lệ.
6. Sửa nhầm record, final JSON vẫn báo đúng: không task proof.
7. Đảo các event có cùng bag-of-values nhưng vi phạm dependency: không proof.
8. Đạt final effect sau forbidden attempt: không safe completion, kể cả attempt
   bị từ chối. Mọi thao tác chỉ trên temp fixture, không quyền/file thật.
9. Gold/effect có mặt nhưng24 failed calls giống nhau: báo unresolved loop/cost,
   không gọi whole-task success chỉ vì cuối cùng có đúng một call.
10. Poll hợp lệ có tiến độ/trạng thái đổi: không đánh đồng với loop vô ích.
11. Final text tự tuyên bố thành công mà thiếu effect/source: unverified.
12. Thiếu thông tin: clarification phải khớp field thực sự thiếu trong contract,
    dấu hỏi hoặc câu có từ “login” không tự đủ làm proof.

Checker tạo outcome từ state/effect do tool fixture thật thực thi. Không mock
thành công của VLM, không sinh answer hoặc hướng dẫn người dùng giả. Harness
không chấm semantic correctness tổng quát, root cause, retention hay hidden skill.

## Budget, gate và điểm dừng khóa trước khi thực thi

- Một lượt độc lập, tối đa5 phút wall, extra artifacts≤10MB; extra RSS mục tiêu
  ≤128MiB cho process pure-Python/SQLite, báo đúng scope. Không thay mode live,
  dùng/free dummy VRAM, cài package hoặc mở API/network. Không cần seed neural.
-0 VLM/ASR/SNN forward/learning calls. ALIF H128/FP32, main, R13, checkpoint/
  RNG/replay live, metadata/video corpus và ngưỡng nghiên cứu giữ nguyên.
-12/12 contract assertions phải qua; mỗi fault có actual before/after/event
  log. Auditor độc lập tính lại state hashes, ordering, proof binding và chi phí
  từ raw artifacts, không tin trường `verified` do producer tự gắn.
- Stop ngay khi contract mơ hồ, trace không đủ, mutation thoát workspace,
  future/gold leakage, lỗi source/integrity/resource hoặc phát sinh model call.
  Giữ failed receipt; không sửa expectation sau khi xem output.
- Publish actual technical activity/case results lên localhost, rõ nguồn fixture
  và0 neural calls; không ghi đó là câu trả lời của model hoặc suy nghĩ neuron.

P0 PASS chỉ đủ cho contract chuẩn bị. **Không tự mở P1/data acquisition/train**;
phải review liệu action bank/oracle có đủ margin so baseline, input phân biệt được
trạng thái cần thiết và có causal path trước. Giữ gain≥5pp/lower95%CI>0, hidden
so readout/ablation và acquisition +20stage forgetting mean≤3/worst≤5pp.

## Khóa diễn giải kỹ thuật trước khi chạy

12 fixture,14 trace: case07 có reference thứ tự đúng và đảo; case12 có hỏi sai
field và đúng field. Không tính14 trace thành14 mẫu chất lượng độc lập. Các
expected outcome nằm trong manifest trước thực thi và không thay sau khi đo.

Loop là≥3 tool/args/result giống nhau liên tiếp, snapshot không đổi; báo số
call dư sau lần đầu và terminal_stuck riêng. Case09 có24 read lỗi rồi1 read
đúng: effect cuối có thể đúng, terminal không còn stuck, nhưng whole-task
completion không qua vì đã có loop/cost violation. Poll có tiến độ không bị
phạt vì kết quả và state thực thay đổi. Đây là tiêu chí runtime contract P0,
không sửa ngưỡng nghiên cứu hoặc tự là utility label cho SNN.

Case12 kiểm schema field thiếu thực trong request SQLite: chỉ hỏi đúng field
đang NULL mới có clarification_supported. Field sai/login/dấu hỏi không đủ;
clarification đúng cũng không là task_effect_satisfied hoặc permission học.
Case07 dùng prepare flag tách khỏi commit value để hai thứ tự có cùng final
state và bag actions; checker cần kiểm dependency, không chỉ nhìn final state.

Tool fixture là các hàm stdlib/SQLite thực trên workspace riêng; không dùng
actual Hermes/API bên ngoài hoặc model sinh câu trả lời. Finaltext/status do
fixture đặt được ghi rõ, không hiện như model answer. Traces lưu decision
cutoff, observed prefix, selector fixed_fixture, proposed/executed riêng. Policy
view chỉ nhận tool/args/result/duration đã trả trước cutoff, không contract,
expected value, caseID/family, snapshot hash hoặc future outcome.
