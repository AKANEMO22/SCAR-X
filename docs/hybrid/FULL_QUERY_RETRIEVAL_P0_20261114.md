# Bản vá truy xuất toàn câu hỏi — đã qua P0 cô lập, chưa nạp live

Đã sửa được lỗi bỏ từ khóa sau16từ trong **bản sao Store/SQLite**. Lượt8hsmall
32cd13 còn dùng source cũ đang khóa; không áp bản vá, restart gateway hoặc đổi
trọng số đang phục vụ trong khi đo. Đây là sửa memory/retrieval, không phải
SNN học thêm, tăng contextVLM hoặc giải quyết hiddenrepresentation/retention.

[Protocol trước đo](../../experiments/hybrid-full-query-retrieval-p0-v1.md),
[run7a9ee0](../../runs/20261114T101526Z-full-query-retrieval-p0-v1-7a9ee0/manifest.json),
[rawSQLiteoutputs](../../runs/20261114T101526Z-full-query-retrieval-p0-v1-7a9ee0/result.json),
[dataaudit](../../runs/20261114T101526Z-full-query-retrieval-p0-v1-7a9ee0/independent-data-audit-v1.json).

## Thay đổi cụ thể

[Recipe đúng SHA](../../integrations/hermes/full_query_retrieval_patch.py) chỉ
thay đoạn tạoMATCHquery. Câu hỏi≤16lexicalwords giữ nguyênchuỗiSQL cũ, gồm
cả từ lặp và thứ tự. Với câu dài hơn, đọc toàn bộ trong6000Unicodecharacters,
loại từ trùng theo lowercase và tạo câyOR cân bằng từ literalwords được quote.
Không chọn terms dựa trên gold/nhãn/nội dung đáp án; không prefix/tail-only cap.
SQL vẫn dùng tham số. Stream/timefilter, resultlimit≤32 và BM25 giữ nguyên.

6000characters là giới hạn request đã có, không phải6000VLMtokens. Tất cả
distinctwords của câu hỏi dài được giữ; ví dụ lớn nhất đã đo có MATCH11236char,
độ sâu ngoặc10. Không thêm table/schema, embeddingmodel, service hoặc daemon.
Unknown/nonstring/query6001char bị từ chối. Bản vá chưa đổi fileStore active.

## Kết quả thật

Fixture có1028bản ghi riêng; baseline/candidate lấy từ hai coldSQLitebackup
cùng row/indexdigest.21case đã khóa trước đo, gồm từ khóa cuối, tiếngViệt,
punctuation/literaloperators, khônghit, queryrỗng, repeatedwords, stream/time,
resultlimit và nhiều từ duy nhất. Không dùng dữ liệu hội thoại riêng.

| Vị trí keywordblue | Base16wordprefix | Candidatefullquery |
|---|---|---|
|1/15/16|Có nguồnblue|Có nguồnblue|
|17/32/129|Thiếu nguồnblue|Có nguồnblue|
|Cuối câu≈6000characters|Thiếu nguồnblue|Có nguồnblue|

10casequeryngắn giữ chính xác orderedIDs. Sourcehit/filter/boundary cases đều
qua, kể cả cuối câu tiếngViệt. Cận tối đa32results giữ. Independentauditor
đọc rawSQL/MATCH và thực thi lạiSELECT trên readonlySQLite, kiểm toàn bộterms
có mặt, thứ tự shortqueries và coldrow/indexhash. Không sửa source/query/gold
theo kết quả.2SQLiteunitchecks riêng qua;18Node tracechecks cuối qua.

Thời gian đo lớn nhất từngquery: cảbase/candidate16ms; peakPythontracemalloc
củacandidate358157byte. Đây là cóSQLtrace vàtracemalloc, độ phân giải timer
hữu hạn, **không phải benchmarktốc độ hoặc peakRAM toàn SCAR-X**. Cả lượt gồm
setup/SQL~0,218s trên1028records; chưa đo chi phí trên kho lớn/realcorpus.
Fullquery có thêm terms/cost, không tuyên bố cùngcompute với16prefix hoặc
tốt hơn mọiquery. Longqueryranking có thể thay đổi do termset đầy đủ/dedup.

Auditor lần đầu nhầm rằng tracecallback chỉ có1statement; SQLiteFTS còn phát
commentnội bộ `-- SELECT/PRAGMA`. Đã giữ source/failure ởaudit-harness-attempt-v1,
sau đó chỉ tái thực thi đúng primarySELECT, giữ toàn rawtrace gốc. Không đổi
runner/cases hoặc ghi đè resultP0; audit không gọi model/train.

## Những gì chưa giải quyết

- Candidate chỉ tìm đúng nguồn trong hợp đồng kỹ thuật trên. Chưa có heldout
  semanticQA/P1/fullproductiongain hoặc so sánhneuralskill mới.
- SNNrelevance/incident vẫn có128wordprefixloss đã thấy ở40c5f3. VLMprompt còn
  giới hạntext mỗi evidence1100char. Tìm đượcrecord không bảo đảm phần chứa
  đáp án đã tới neuralinput hoặc generationprompt; cần kiểm source/chunkbinding.
- Chưa khắc phục credit/history, readoutforgetting hoặc importanceinert bằng
  việc đổi truy xuất. Không tăng model/neuron, bậtLoRA hoặc nhập labcheckpoint.
- Chưa nạp live vì source32cd13 phải giữ nguyên đến actualterminal+dataaudit.
  Sau đó mới làm source/runtime regression, backup và giao dịch restart có
  maintenancelease; không đánh tráo kết quả8hsource cũ sang bản vá mới.

Localhost tự hiện kết quả dưới “Các phép đo học và trí nhớ đã hoàn tất”; ghi
shadow/chưalive/0neuralcalls rõ. Main và protectedR13 giữ nguyên. Quyết định:
**GO cho bản vá truy xuất ở bước tích hợp tiếp theo; PIVOT nghiên cứu neural
vẫn giữ nguyên, chưa production hoặc học liên tụcPASS.**
