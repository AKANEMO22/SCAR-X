# Chat dài, workspace và compact — nghiệm thu có phạm vi

## Lỗi đã sửa

Một request thật qua gateway canonical Hermes với 7.028 ký tự bị HTTP 400:
`Search query exceeds bounded 6000-character working request`. Giới hạn query
SQLite vô tình thành giới hạn của toàn tin nhắn. Nhánh non-streaming còn có
guard debug 6.000 ký tự đặt trước đường chat Hermes.

Đã thêm `Gateway.search_turn_evidence`: tin nhắn ngắn giữ nguyên đường cũ;
query dài chia thành các nhóm từ, mỗi query FTS tối đa 6.000 ký tự. Pool kết
quả interleave/deduplicate tối đa 32 trước bộ relevance có sẵn. **Không cắt
nội dung gửi tới VLM**, không đổi encoder hoặc quy tắc học. Guard 6.000 ký tự
của debug được giữ sau nhánh Hermes. Store/P0 query protocol lịch sử không đổi.

Trần xử lý text tại helper là 65.536 ký tự, không phải cửa sổ 65K token. Model
vẫn có context 16.384 token; system/history/tools cũng chiếm chỗ. Token hóa và
context guard thực tế quyết định request có vừa không. Từ đơn quá 6.000 ký tự
không đưa vào FTS và có counter audit; tin nhắn gốc vẫn nguyên. Bounded pool
không đảm bảo tìm mọi bằng chứng; đây chưa là cải thiện chất lượng retrieval.

## Đã chạy thật

Project thử trung lập có `.git` riêng, README mô tả Jade Inventory, receipt
JADE-472 và port 9237. Bài kiểm không đọc workspace riêng của người dùng.

- Canonical Hermes session có CWD đúng project; một lệnh terminal đọc nguồn
  do runtime prelude chọn, sau đó VLM trả đúng mục đích/code/port. Không ghi
  thành SNN đã tự biết nên chọn tool nào.
- Tin nhắn 7.028 ký tự sau sửa được trả lời đúng, thay lỗi format trước đó.
- Năm lượt thực tiếp theo, mỗi lượt 7.115 ký tự, đều trả đúng code/port và
  kích hoạt **3 compact tự động**. Context max vẫn 16.384, peak context_used
  quan sát trong 5 lượt là 13.048 token; không lấy usage tích lũy làm context.
- Compact thủ công qua `session.mutate(operation=compress)` đúng wire Desktop
  hoàn tất 8 → 6 message, không abort/no-op. Owner snapshot giữ cả 7 đoạn user
  nguyên văn; raw history có 4 hàng handoff nhưng display projection có 0.
- Sau compact thủ công, một lượt hỏi nữa trả `JADE-472 | 9237`, không gọi tool
  và không lộ marker compact. Đây là ngữ cảnh/truy xuất runtime, không phải
  chứng minh kiến thức đã nằm trong synapse.
- Source/activation check xác nhận mảng/RNG/replay được giữ qua restart, 5 cập
  nhật live vẫn nguyên. Không train hoặc chạy lại test 8 giờ. **171 Python
  checks qua**, gồm long-input streaming/non-streaming và CPU/native parity.

[Audit receipt](evidence/LONG_INPUT_AND_COMPACTION_20261114.json) đối chiếu
actual RPC/events/snapshot. Raw dữ liệu thử và các lỗi harness được giữ private.
Follow-up sau manual có receipt riêng, không ghi đè receipt chuỗi 5 lượt.

## Các lỗi của bài kiểm được giữ lại

1. `surface=desktop` không hợp lệ trên canonical submit; cửa sổ app thông
   thường bỏ trường này. Refusal xảy ra trước admission, không phải lỗi model.
2. Create cùng `title` có nghĩa create-if-missing/resume. Các lần thử dùng
   title cố định đã quay về project thử trước. Bản đúng dùng title và request ID
   riêng, rồi kiểm CWD trước khi gửi prompt; project có root trung lập để không
   thừa hưởng AGENTS nghiên cứu. Không dùng câu trả lời ở harness sai để chấm.
3. `session.compress` là tên phía client cũ, Desktop ánh xạ sang mutation có
   revision/generation fence. Lượt gọi sai wire trả unknown_method được giữ;
   không sửa server để thêm một API sai.

## Giới hạn kết luận

Đây là nghiệm thu ngắn với dữ liệu engineering đã biết, không phải benchmark
ngôn ngữ, reasoning, video chưa thấy hay retention 20 task. Không có hidden
ablation/readout control trong bài này. Không suy ra SNN vượt cùng LLM + RAG,
tự học nguyên nhân tool loop hoặc đạt production. Đã kiểm owner display
projection; không chụp lại UI Desktop ở lượt này và chưa nghiệm thu mọi luồng
capture màn hình. Các ngưỡng nghiên cứu trước đó giữ nguyên.
