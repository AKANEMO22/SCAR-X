# Nguồn web và task trong vòng tự xem lại — RC11 → RC12

## Lỗi quan sát được

RC10 đã sửa routing cho câu JoJo mới nhất và follow-up “tìm kiếm đi”: gọi
`web_search` thật, không tìm file. Nhưng lần kiểm canonical Hermes ở RC11 cho
thấy nguồn không đi đến idle reviewer: cả hai lượt nhận5 kết quả thật, trong
khi answer record có `evidence=[]`, `web_coverage.partial=true` và0 provenance.

Nguyên nhân là Hermes bọc JSON trong `<untrusted_tool_result source="web_search">`
cùng cảnh báo dữ liệu không đáng tin. Parser chỉ nhận JSON trần. Test JSON
trần trước đó qua nhưng không phủ frame của host thật; RC11 không được gọi là
end-to-end source-persistence PASS. Receipt cũ giữ nguyên trong vùng private.

VLM còn trả “tập11” từ snippet nói **tổng11 tập dự kiến**. Hai lần review tự
nhiên ở RC11 lại chấp nhận câu đó và nhầm với chương manga. Đây là lỗi semantic
còn tồn tại, không phải bằng chứng model tự nhận ra lỗi. Guard không biến review
thành nhãn đúng, và không cập nhật synapse từ những câu tự nhận xét này.

## Bản sửa RC12

- Parser chỉ giải mã bản sao của frame `web_search` đúng định dạng host đã đọc;
  frame và cảnh báo được giữ nguyên trong messages gửi model. Wrapper sai nguồn
  hoặc prefix tùy ý vẫn không được nhận như JSON hợp lệ.
- Task gốc, user message thực tế, interpretation riêng và nguồn của đúng tool
  call đi đến answer/review/feedback. Không lấy personal FTS không liên quan làm
  bằng chứng cho câu hỏi public. Mỗi snippet giữ trạng thái `observed`, không
  mặc định tin `verified:true`, không coi observation time là publication time.
- Reviewer nhận giờ tạo câu trả lời/giờ review UTC để đánh giá “mới nhất” tại
  thời điểm trả lời. Metadata nhắc phân biệt tổng dự kiến với tập đã phát hành;
  metadata này không là đáp án hoặc verifier sự thật.
- Không đổi phương trình ALIF, input dimensions, checkpoint schema, LoRA hoặc
  trọng số VLM; không chạy lại history P0 hoặc mở nghiên cứu mới.

## Kiểm chứng nguồn trước đóng gói

- **35 targeted wrapper/review/feedback checks qua**, bao gồm thực thi riêng hai
  helper framing của Hermes đã cài bằng AST và kiểm messages không bị đổi.
- **28 package checks qua**. Không cộng các suite chồng nhau thành mẫu thống kê.
- Full332 checks RC11 là baseline trước sửa; không tuyên bố chạy lại toàn bộ cho
  RC12. Qualification có hash source/recipe/JUnit tại
  [WEB_WRAPPER_TIME_SOURCE](evidence/WEB_WRAPPER_TIME_SOURCE_20261115.json).
- Replay **dữ liệu đã có** của canonical session RC11 nay giải mã đúng5 URL,
  khớp raw tool result, `partial=false` và không sửa messages. Không tìm web,
  gọi VLM hoặc train thêm trong replay này.

## Activation và lượt Hermes thật

[Activation RC12](evidence/RC12_ACTIVATION_20261115.json) đã cập nhật thật hai
file từ predecessor RC11, qua maintenance lease/CAS và backup đầy đủ. Arrays,
learned metadata, RNG, replay giữ nguyên; live updates vẫn5. [Read-only installed
verification](evidence/RC12_INSTALLED_ARTIFACT_20261115.json) xác nhận57 payload
files, native assets, Hermes/dependency pins và GPU worker đúng cấu hình.

[Một follow-up canonical thật](evidence/RC12_WEB_FRAME_LIVE_20261115.json) trên
session nghiệm thu riêng trước đó gọi đúng `web_search` với chủ đề JoJo gốc,
không tìm file/paid extractor.5 URL/snippet khớp raw tool result và persisted
provenance; task gốc được giữ riêng với user message “tìm kiếm đi”, `partial=false`.
Không bỏ frame cảnh báo khỏi messages. Runtime sau đó tự review khi rảnh:
events15656→15661 nhận cùng5 source IDs, không gọi review thủ công.

Native answer vẫn khẳng định “tập11”; snippet nói tổng dự kiến11 tập, chưa đủ
để xác định tập mới nhất. Reviewer nêu câu hỏi cần xác minh nhưng reason vẫn
coi con số đó là bằng chứng. Kết luận **pipeline source/review đã chạy thật;
answer/critic quality chưa đạt kiểm chứng**, không claim model đã tự hiểu lỗi.
`weights_updated=false`, reason rõ không dùng generated review làm nhãn học.

Mã audit đọc lại receipt, không tìm web/infer/train:
`scripts/audit_hermes_web_frame_followup.py --run <owned-private-run> --output <new-json>`.
[Release checks](evidence/RC12_RELEASE_CHECKS_20261115.json) giữ hash/JUnit đúng
phạm vi. Các giới hạn review/answer quality vẫn giữ cho đến khi có bằng chứng
độc lập; test plumbing qua không chứng minh học kỹ năng hoặc hidden advantage.
