# Chuẩn bị dữ liệu mới và thiết kế P2 sau review DeepSeek

2026-11-10. **Chưa đạt normalchat/vision. Chưa triển khai learner mới.**
[Master1.31](../RESEARCH_PLAN.md), [review40](40_DEEPSEEK_REVIEW.md).

## Vì sao chưa tiếp tục huấn luyện bằng cách tăng dữ liệu

35–37 đã STOP theo gate. Bộ chat8style cũ có shortcut firstword/prefix;
train cao nhưng generalization thấp. Review40 xác nhận readout có thể đủ gây
mất một cặp, không xác nhận hidden vô ích. Tạo thêm wording không tự khắc phục
credit assignment, representation hoặc retention. Không dùng data mới như cách
vòng qua STOP, không đưa Qwen hoặc Transformer vào SCAR-X để làm điểm đẹp hơn.

P2 usefulness có thể giúp ổn định đầu ra; nó **không** là cơ chế tạo hidden
representation mới. P1/binding, language/vision generalization và tích lũy kỹ
năng vẫn là ưu tiên gốc trong master, không bị thay bằng protection-only.

## Antigravity CLI đã chạy thật

Protocol: [adaptive-curriculum-data-v1](../../experiments/adaptive-curriculum-data-v1.md).
Contract công khai: [wording chat/vision](../../data/contracts/ANTIGRAVITY_ADAPTIVE_CURRICULUM_V1.md).
CLI `agy.exe` đã cài, model `gemini-3.8-flash-low`, một structured finish;
không gửi history/ảnh/DB/weights riêng. CWD tạm ngoài repository.

Lượt đầu terminal sau1,06giây vì process-tree commit1,207GB vượt cap1GB.
Chưa candidate. Giữ toàn bộ run/log; amendment trước dữ liệu cho đúng một retry
sau sửa env numericalthreads trước NumPyimports. Không tăng cap hoặc đổi contract.
Lượt hai hoàn tất81,43giây:128rows schema hợp lệ; peakRSS751.927.296 byte,
peakcommit743.645.184 byte, dưới ngân sách1GB cho pipeline tạo wording.

- 96câu chat: greet/identity/thanks/bye/learning/capability, Việt/Anh.
- 32câu hỏi vision: màu/hình dạng/số vật/màu vật bên trái.
- Đọc toàn bộ128; giữ127, tách1câu identity hỏi thêm nguồn gốc phát triển.
- Không trùng exact-normalized trong candidate hoặc bốn nguồn wording công khai
  đã kiểm. Điều này không loại được mọi semanticduplicate hoặc trainingleakage.
- Có10ghi chú editorial; câu hỏi về cách sửa lỗi cần đáp án quy trình, không
  được ép về một câu trả lời yes/no chung. Identity mỗi model phải đúng model đó.

[Receipt sinh data](../../artifacts/adaptive-curriculum-data.json),
[receipt review](../../artifacts/adaptive-curriculum-reviewed.json).
`reviewed.json` là **developmentwording**, chưa là dataset training có nhãn
đúng. AGY không tạo đáp án.32visionwording chưa có bộ ảnh mới/target annotation;
không gọi đây là generalvision dataset hoàn chỉnh. Finaltest cho nghiên cứu
tương lai phải khóa độc lập, không tái gọi các câu đã xem là unseen finaltest.

## Đối chiếu thật 24 câu chat mới

[Protocol reference](../../experiments/adaptive-curriculum-reference-v1.md),
[đầu ra và tài nguyên](../../artifacts/adaptive-curriculum-reference.json).
Lấy cố định id01/id02 của6intent×2language trước khi xem output.

SCAR-X là bản sao snapshot SQLite nhất quán của phiên hiện tại, giữ64ALIF,
vocabulary/encoder/topology, canonicalstate reset riêng từng câu; không học hoặc
retrieval. Canonicalclone không là bằng chứng hành vi continuousstate lúc chat.
Modelcopy giữ nguyên toàn bộ state/weights sau đo; không restore/dạy phiên thật.

Qwen3-4B-Instruct-2507 Q4_K_M kiểm size/SHA256 trước chạy; CPU2threads, ctx1024,
temp0, seed919, max128tokens. Qwen chỉ nhận đúng một publicprompt, không
family/đáp án/câu SCAR-X. Reference hoàn tất24/24 sau209,79giây,10output bị cắt
do length; giữ nguyên, không chạy lại để chọn câu đẹp. PeakRSS3.167.940.608 byte,
commit3.209.920.512, ngân sáchreference4.5GB riêng. Ownedserver đã tắt, không9B.

| Prompt mới | SCAR-X sinh thật | Nhận xét giới hạn |
|---|---|---|
| What name do you usually go by? | Hello. | Không trả lời câu hỏi danh tính. |
| Mọi người thường gọi bạn bằng danh xưng nào? | Tôi đang tôi | Không thành câu có ích. |
| Mình xin phép dừng cuộc trò chuyện ở đây. | Tôi. | Chưa đáp ứng ý định kết thúc. |
| Câu trả lời vừa nãy giúp ích cho mình rất nhiều. | Tạm | Không phải lời đáp cảm ơn hoàn chỉnh. |

Qwen phần lớn tạo văn bản trôi chảy, nhưng không mặc định đúng: có câu nói quá
khả năng tự học từ feedback; câu “Làm sao để tôi hướng dẫn bạn học…” bị đổi thành
“hướng dẫn người khác học…”. Không dùng các câu đó như nhãn factual hoặc bằng
chứng runtime Qwen có synapse học online. Các output dài bị cắt không là câu
hoàn chỉnh. Không chấm một điểm intelligence/equivalence từ24câu; identity
Qwen đúng theo Qwen, không phải cần tự nhận là SCAR-X.

Đầu ra SCAR-X vẫn yếu ngay trên6intent. Đây là bằng chứng khoảng cách còn lại,
không là một cải thiện chất lượng sau training. Trên localhost `#test-answers`
có cả24cặp rawoutputs, timestamp, nguồn COPY và truncation thật.

## Đề xuất cơ chế tiếp theo — vẫn chờ duyệt

[readout-usefulness-proposal-v2](../../experiments/readout-usefulness-proposal-v2.md)
giữ draft A cũ nguyên vẹn, bổ sung:

1. Đối chiếu importance magnitude/anchor đã có ở Phase2.1; không quảng bá EMA
   như phát minh mới. Phân biệt localupdate strength với usefulness.
2. Đề xuất loss-counterfactual ở readout: mức loss tăng khi xóa một đóng góp
   synapse trên mẫu train đã có nhãn rõ ràng và đang dự đoán đúng. Chỉ xấp xỉ
   usefulness tại đầu ra, không tự thành causalimportance dài hạn.
3. Control shuffledimportance và scalarLR, bank khớp appliedΔWnorm để tách chọn
   đúng synapse với giảm tốc; bank rule tự nhiên/equal-time báo riêng.
4. P0 math/parity/codec/RNG/importance rollback và timing trước pilot. Eight fresh
   networkseeds, không chọn3checkpoint tốt. Tối đa168job sau P0 và amendment
   tài nguyên, không tự sweep β/LR hoặc chạy nếu prerequisite chưa qua.
5. Giữ joint gate forgetting mean3đ/worst5đ, chất lượng65%, acquisition và
   relativeCI theo master. Giữ cũ nhưng học mới thấp thì PIVOT, không GO.

**Chưa code learner U, chưa chạy P0/pilot P2.** Phê duyệt30 trước đây cho local
eligibility/readout-error; không tự coi đó là phê duyệt mọi cơ chế importance.
Yêu cầu duyệt gửi kèm đường dẫn cụ thể; dữ liệu/reference độc lập vẫn được làm.

## Tình trạng bàn giao

Các script mới chỉ tạo/kiểm dữ liệu và đo reference; không đổi source neural
hoặc cấu hình Hermes. Bộ dữ liệu, protocol, runmanifest, rawoutputs có nguồn
và các lần thất bại đều giữ lại. Không đưa DB/privateweights vào gói nguồn.
Đích hội thoại bình thường, vision và lifelong learning vẫn **chưa đạt**.


261/261tests qua (pytest84,40giây), gồm3test mới cho bounds/metadata/Unicode và
không đọc localprivate khi indexpublicdata. [Receipt kiểm thử](../../artifacts/adaptive-curriculum-software-validation.json).
[Receipt bàn giao](../../artifacts/adaptive-curriculum-delivery.json) xác nhận
liveweights/updatecounter/64ALIF và source neural giữ nguyên. Qwen đã tắt.
ZIP chỉ kèm releaseweights công khai có sẵn; không kèm snapshot64neuron riêng
của người dùng. Những commit phát sinh đồng thời được phiên khác thực hiện;
review này chỉ ghi nhận head/hash, không commit/push.
