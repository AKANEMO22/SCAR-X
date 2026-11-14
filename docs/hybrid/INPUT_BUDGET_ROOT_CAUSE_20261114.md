# Dữ liệu đã mất trước synapse: kiểm toán ngân sách đầu vào

**Phát hiện mới:** trong các câu hỏi dài hợp lệ, encoder có thể bỏ toàn bộ
phần phân biệt sự kiện trước khi SNN nhận input. Tăng neuron hoặc chạy thêm
tick trên cùng input không bổ sung lại thông tin đã bỏ. Đây là một nút thắt
đường dữ liệu, không phải kết luận tất cả thất bại của SCAR-X đều do encoder.

[Protocol trước đo](../../experiments/hybrid-input-budget-audit-p0-v1.md),
[source/cases40c5f3](../../runs/20261114T100120Z-input-budget-audit-p0-v1-40c5f3/manifest.json),
[kết quả](../../runs/20261114T100120Z-input-budget-audit-p0-v1-40c5f3/result.json),
[đối chiếu độc lập đường tính](../../runs/20261114T100120Z-input-budget-audit-p0-v1-40c5f3/independent-data-audit-v1.json).

## Đã đo gì, không được suy rộng thành gì?

| Đường dữ liệu thật | Phép đối chiếu cố định | Kết quả |
|---|---|---|
| LocalALIF.relevance → features | Cùng câu hỏi, `lamp color blue`/`lamp color pink`, cùng độ dài văn bản để lengthchannel không tiết lộ khác biệt | 5/10cặp có input giống byte: câu hỏi125/126/127/128/160từ. Các ca16/64/120/123/124 vẫn khác input |
| Research incidentV2 | Cùng flags/probe; chỉ đổi missing_file/permission_denied | 128/160từ: lexicalerror bị bỏ, input giống nhau.16/64/120 còn khác |
| Store.search/SQLiteFTS5 | Keywordblue tại1/15/16/17/32/129, các từ trước không có trong evidence | Có nguồn đúng ở1/15/16; mất nguồn ở17/32/129 do query chỉ dùng16từ đầu |
| Dữ liệu predicate032f30 đã thất bại | Kiểm1.840trường text thực vào pair_sequence | Tối đa11từ/trường, không trường nào>128; truncation dài không giải thích failure cũ này |

125 là boundary của **cặp fixture cụ thể**, không phải ngưỡng chung cho mọi
sự kiện. Encoder giữ128lexicalwords của chuỗi `query + eventmarker + text`;
trong fixture trên, tokenmarker và phần đầu evidence còn chiếm chỗ. Đây cũng
không phải128token của tokenizer Qwen. ContextVLM16K và sốtickALIF là các đại
lượng khác. Các words đều được đếm bằng regex thực của encoder.

Đầu vào relevance được lấy bằng chính method hiện có với **input-only tap**
thay forward: nó chỉ chụp argument, trả placeholdertransport không được xuất
thành score. Không tạo/chạy một neuralmodel, không CUDAforward, không synapse
update hoặc VLMcall. Auditor dùng accumulator/hash arithmetic riêng để so các
mảng đã lưu, đọc lại SQLiteFTS thật, kiểm rawsource/cases/hashes. 0,5/1 trong
artifact là cận phân biệt input trên cặp nhãn đối nghịch tại cùng state, **không
là accuracy model**, không phải100%recall hoặc kết quả assistant đầy đủ.

Khi input và startingstate giống nhau, tăng hidden không giúp head phân biệt
hai cặp đó. Nhưng VLM có thể còn đọc rawtext, SQLite có thứ tự/tie-break và
controller có trạng thái khác; không áp cận0,5 cho toàn trợ lý. Research
incidentV2 chưa được nhập vào live; thậm chí phân biệt error_type cũng không
đồng nghĩa xác minh đúng nguyên nhân thực tế.

## Hướng sửa cần giữ phạm vi rõ ràng

1. **Ngân sách theo trường:** query, evidence và observation phải có slot riêng,
   không ghép thành một prefix rồi cắt. Lỗi/trạng thái/nguồn/thời gian cần đường
   biểu diễn rõ ràng, không phụ thuộc câu hỏi có chiếm hết lexicalbudget hay không.
2. **Lịch đọc có giới hạn:** nếu nhiều chunk, khai báo phần nào đã đọc/chưa đọc,
   giữ chronology/source binding và công bố chi phí. Chunking không tự chứng
   minh SNN giữ được quan hệ qua nhiều bước; cần delayed/binding heldoutcontrols.
3. **Truy xuất toàn yêu cầu:** chọn terms/chunks trên toàn câu hỏi trong ngân
   sách cố định, đối chiếu vớiBM25prefix hiện tại. Không bỏ nguồn chỉ vì keyword
   ở cuối hoặc lén đưa đáp án/gold vào queryrewrite. Đây là sửa memory/retrieval,
   chưa là hiddenlearning.
4. **Version và đối chứng:** encoder mới cần envelope riêng, không dùng weights
   cũ rồi gọi là đã học. P0 phải kiểm phân biệt input/timestamp/no-labelleak,
   coldresume/nextupdate/rollback và resourcebudget. Chỉ sau đó mới dùng fresh
   DEV với cùngALIF/rule, frozen/readout/hidden/strongRAG và hiddenreset.
5. **Credit/retention vẫn độc lập:** shortpredicate fail vẫn cần giải quyết
   binding, readout và credit. Không nới acquisition/3–5ppforgetting/5ppgain;
   current-only/history rule mới vẫn chờ duyệt, không suy phép cắt input này
   thành bằng chứng variant sẽ tốt hơn hoặc tự train nó.

Chưa code/nạp encoder sửa, chưa thayStore.search hay weights đang phục vụ.
Không dùng audit này để mở lại các score-family cũ đãSTOP. Sau mixed8h kết thúc,
triển khai cần versioncompatibility và kiểm attribution để tách input/SQLite/
readout/replay khỏi học hidden, không chỉ làm điểm tổng tăng rồi gọi neuralgain.

## Trạng thái vận hành cùng lúc

Smallmixed8h32cd13 vẫn có actualPID46940/source36files nguyên vẹn khi đo; không
đổi các pathPython đã khóa. Chỉ localhost renderer bổ sung panel trong mục
“Các phép đo học và trí nhớ đã hoàn tất”. 16trace-rendering checks qua. Giao
diện ghi0neuralforward/update/VLMcall và không cộng số liệu audit vào live.
Main và R13 giữ nguyên. Goal toàn phần chưa đạt, tiếp tục PIVOT input/credit/
representation/retention; đây là thêm bằng chứng về nguyên nhân cần sửa.

Theo dõi UI sau đó: đã thấy panel hiện đúng dữ liệu40c5f3 trên localhost thật.
Banner `Failed to fetch` cũ còn treo sau khi poll đã hồi phục; nay tách lỗi kết
nối, lỗi runtime và lỗi thao tác. Poll thành công chỉ xóa cảnh báo kết nối,
không che lỗi thao tác hoặc lỗi backend đang có.17tracechecks cuối qua, có
controlledtransport failure/recovery riêng (không modelinference), nguồn/log ở
`ui-followup-v1`. Không restart gateway hoặc sửa sourcePython của32cd13.
