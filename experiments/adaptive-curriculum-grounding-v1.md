# Ground development references — không học, không pilot

2026-11-10. Theo master data/verifier/Experience-to-Skill và chỉ đạo tạo data.
P0 U đã hoàn tất, **pilot P2 vẫn chưa duyệt**. Bước này chỉ hoàn thiện corpus
development127wording đã review; không thêm rule/encoder hoặc mở lại familySTOP.

## Khóa trước dựng dữ liệu

- Nguồn: `artifacts/adaptive-curriculum-reviewed.json`, sha đúng;95chat và32vision.
- Chat:95reference dự án tự viết, đáp ứng từng câu cụ thể, kèm rubric và nguồn
  code/docs cho các nhận định về model. Greeting/thanks/bye là style, không là
  sự thật thực nghiệm. Không lấy Qwen/AGY output làm đáp án factual.
- Giữ câu trả lời tự nhiên, không tự cắt từ, đồng nhất mọi đáp án theo intent,
  hoặc thay đáp án bằng output model để tăng compatibility.
- Vision:36ownedRGB128×128;12single (3shape×2color×2geometry),12pair
  (6orderedshape pair×2color orders RG/GR),12triple(6shape permutations×2color
  patterns RRG/GGR). Seed43301, jitter geometry nhỏ, không occlusion.
- Ghép32question: singlecolor/singleshape chỉ single; count mọi36scene;
  leftcolor chỉ12pair. Tổng576QA; tất cả development, không finaltest/train split.
- Nhãn primitive từ manifest phải khớp oracle độc lập đọc connectedcomponents,
  màu/boundingbox/row-width shape trên pixel. Thêm reference sentence Việt/Anh;
  không gắn nhãn lớp hoặc caption vào pixel input. Tên PNG opaque SHA.
- Kiểm answer balance và empirical question-only ceiling, left bag-of-objects
  ceiling. Không phải performance của SCAR-X; không tìm corpus dễ cho model.
- Counterfactual color variants giữ nguyên geometry; thay màu không được đồng
  thời đổi vị trí/shape/size. Decode bằng `PilotLab.pixels` hiện hành (chỉ tiền
  xử lý, không gọi core/model) để đo collision/ceiling theo cùng question+64pixel.
  Không thay palette/decoder để giấu informationloss. Đây không là qualityscore
  của API chữ số hoặc lý do giải thích mọi lỗi của RGB prototype37.
- Audit token/vocabulary trên publicrelease24token và metadata phiên thật đọc
  trong RAM (chỉ xuất aggregate, không xuất vocabulary/weights riêng). Audit
  cả maxanswer24token/vocabcap256 trong code, không tự thay giới hạn hoặc schema.
- Không train/generate SCAR-X, không gọi Qwen/AGY mới, không dùngU prototype.
  1thread,2GBmonitor, wall<=120s. Lưu rawsources/results; không ghi đè data41.
- Đầu ra có `automatic_learning_allowed=false`. Không tuyên bố corpus nhỏ này
  là normalchat/generalvision, knowledge4B hoặc năng lực đã vào synapse.

## Đầu ra

DatasetJSON/chatreferences/imageQA, sourcehashes và36PNG; preview chỉ để review,
không input train. Báo cáo coverage/compatibility và các giới hạn, không mở
pilot hoặc mainteaching dù data validate thành công.
