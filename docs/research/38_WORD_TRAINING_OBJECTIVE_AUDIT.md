# Cách dạy hiện tại có phù hợp mạng phát xung không?

2026-11-10. **Đã hoàn tất audit. Không huấn luyện hoặc cập nhật trọng số.**
[Protocol38](../../experiments/word-training-objective-audit-v1.md) kiểm tính hợp
lý của bài học sau câu hỏi của người dùng, không mở thêmtraining hoặc coi việc
SNN khácTransformer là đủđểkếtluận mọi supervisedlearning sai.

Phải tách dữliệu/mục tiêu khỏi kiếntrúc và quy tắc học. DirectWords36 vẫn
ALIF48 + directreadouterror/own-neuroneligibility35; nhưng corpus8style và
teacherpreviousword có thể cho phép hoàn tất mẫu mà không hiểu câu hỏi.
Audit chỉđọc15checkpoint36, so FREEprev-neuralword với GOLDPREFIX, wrong/zero
question, tách first/laterword và đo potentialcredit ởcheckpointcuối. Không
neuralupdates, khôngTorch/backward, khôngrewrite thuậttoán hoặc lịch sử.

Probe/tokenmetrics không thay wholeutteranceaccuracy. Prefixambiguitytable
là mô tả corpus, không response lookup được đưa vào model. SourceQwen64public
textrefs cósẵn chỉ đểđối chiếu, không sinhlabel hoặc giảvision.
MainALIF/Hermes giữnguyên; mọi đềxuất selectivelearning mới cầnprotocol riêng.

## Kết quả thực tế

[Lượt đo](../../runs/20261110T072222Z-word-training-objective-audit-v1-6a768d/report.md)
đọc 15 checkpoint của vòng 36: 3 seed × 5 phương pháp, mỗi model có 64 câu
train và 64 câu dev. Tổng cộng 7.680 đoạn forward, 193.236 tick neural;
**0 cập nhật synapse/readout/optimizer, không import Torch hoặc backward**.
Toàn bộ câu tự sinh và số đo teacher forcing khớp lại với nguồn vòng 36.

Corpus có 8 mẫu đáp án, 8 từ đầu khác nhau. Trong 35 vị trí được chấm đều theo
mẫu, 27 vị trí nằm sau từ đầu, 16 vị trí là dấu câu hoặc EOS. Mọi tiền tố đúng
không rỗng đều xác định duy nhất từ tiếp theo trong bộ mẫu này. Đây là một
giới hạn của bài học: hoàn tất mẫu không đòi hỏi tạo nội dung mới. Bảng mô tả
này chỉ dùng để audit, không được đưa vào model làm bảng tra đáp án.

### Model thực sự dựa vào tiền tố đến đâu?

Trung bình 3 model `correct` của vòng 36:

| Phép đo | Train | Dev |
|---|---:|---:|
| Cả câu tự sinh đúng token và EOS |99,48%|32,81%|
| Từ đầu đúng, chỉ có câu hỏi và BOS |99,48%|54,69%|
| Từ sau đúng khi được cho tiền tố chuẩn |100,00%|82,20%|
| Từ sau đúng với tiền tố chuẩn nhưng câu hỏi sai |71,88%|71,88%|
| Từ sau đúng khi xóa vector câu hỏi |82,64%|82,64%|

**32,81% không phải model bị giảm từ 35,94%.** Vòng 36 dùng điểm normalized
style, bỏ dấu cuối câu. Audit này kiểm đúng mọi token và EOS. Cả 30 phép so
seed/method/split đều khớp với chỉ số strict đã lưu của vòng 36; text sinh ra
không thay đổi. [Đối chiếu phạm vi metric](../../artifacts/word-objective-metric-scope.json)
được lưu bổ sung, không sửa kết quả hoặc gate lịch sử.

Cờ `POTENTIAL_PREFIX_RELIANCE` đã đăng ký không đạt ở cả 3 model correct,
cả train và dev. Đổi câu hỏi vẫn làm accuracy từ sau giảm 22,92–35,42 điểm
trên train, vượt mức tối đa 10 điểm của cờ. Vì vậy **không đủ bằng chứng nói
model chỉ nhìn tiền tố, hoàn toàn bỏ qua câu hỏi**. Cấu trúc corpus có shortcut
khác với việc model đã tận dụng shortcut ấy một cách ổn định.

### Khoảng cách giữa lúc được gợi ý và lúc tự nói

So đúng các vị trí dev mà model tự sinh đã đi tới:

- Với tiền tố chuẩn: accuracy từ sau **77,21%**.
- Với tiền tố neural do model tự sinh: **47,87%**.
- Chênh lệch ghép cặp: **29,34 điểm**, CI95% **26,69..31,90** trên 3 seed.

Các model đã đi tới 559/648 vị trí từ sau; 89 vị trí còn lại chưa được xử lý
do model dừng sớm. Chúng được ghi là censored, không điền dự đoán giả hoặc
lặng lẽ coi là đúng. Đây là chênh lệch token trên cùng vị trí, không phải
chênh lệch accuracy cả câu. Cả 3 seed đạt cờ `EXPOSURE_SENSITIVITY` trên dev,
nhưng không đạt trên train đã fit gần hết.

Kết quả cho thấy độ nhạy với tiền tố sai và lỗi nối tiếp là vấn đề thực trong
pilot này. Nó **không tự chứng minh** teacher forcing là nguyên nhân duy nhất,
hoặc một cách thay teacher forcing sẽ khắc phục được tổng quát hóa.

### Tín hiệu học cục bộ có xung đột không?

Audit chỉ tính hướng cập nhật tiềm năng tại checkpoint cuối, không áp dụng nó.
Cosine trung vị giữa hướng từ đầu và tổng hướng các từ sau trên train lần lượt
−0,097 / −0,084 / −0,040; trên dev là +0,103 / +0,114 / +0,158.

Tín hiệu trên train gần hội tụ có norm nhỏ hơn nhiều so với dev. Hướng tiềm
năng ở checkpoint cuối không tái dựng được các cập nhật trong quá trình học
trước đó. Vì vậy chưa thể kết luận các từ sau đã phá hỏng representation,
hoặc tự chọn chỉ cập nhật hidden ở từ đầu. Norm lớn cũng không có nghĩa là
tín hiệu hữu ích. Chưa triển khai selective credit hay một learner mới.

## Kết luận về sự phù hợp với kiến trúc SNN

**Dữ liệu/mục tiêu có thể dùng chung giữa các kiến trúc; cơ chế cập nhật phải
phù hợp với động lực neural cụ thể.** Prototype hiện vẫn ALIF và eligibility
cục bộ với readout feedback một bước đã được duyệt. Audit không phát hiện
việc chuyển nó thành Transformer; cũng không xác nhận cách dạy hiện tại đủ
cho mục tiêu lifelong neural của SCAR-X.

Bài 8 kiểu đáp án chủ yếu kiểm nhận diện cách hỏi và hoàn tất câu ngắn. Nó
không đủ để chứng minh học quy luật, tổng hợp trải nghiệm, hiểu ảnh hoặc duy
trì kỹ năng qua thời gian. Điểm token/teacher forcing cao không thay được
free generation, heldout transfer và retention. Các gate P1/P2, các quyết định
STOP 35–37 và mục tiêu cuối giữ nguyên.

Bước tiếp theo cần một giả thuyết về representation hoặc cách gán tín dụng
và bài đánh giá không dựa vào tám mẫu cố định. Phải có protocol, đối chứng và
ngưỡng trước thử nghiệm; không mặc định đổ thêm dữ liệu/epoch hoặc chỉ bỏ tín
hiệu các từ sau. Audit này chưa phê duyệt hay triển khai thay đổi ấy.

## Tái lập và quan sát

Nguồn AGY 128 prompt đã review và Qwen 64 câu trả lời của vòng 36 được giữ
nguyên, không gọi teacher/API mới hoặc dùng output model làm nhãn. Checksum
của cả 15 checkpoint và các trường trọng số, stable state, topology, head
statistics, Adam, bộ đếm học, RNG không đổi. Trạng thái làm việc của bản sao
thay đổi do forward; không mô tả sai rằng mọi byte runtime đều bất biến.

Audit chạy 18,84 giây, sampled peak RSS 52,74 MB, commit 37,49 MB; có cap
Windows 2 GB. 201 kiểm thử đầy đủ qua; 4 kiểm thử chẩn đoán chạy lại sau khi
bổ sung tick/spike capture cũng qua. Không nới guard, không iGPU/offload hoặc
suy ra điện năng từ FLOPs. Localhost tách FREE/GOLDPREFIX, token/cả câu,
censoring và số liệu vòng 36; không có learning/ΔW giả. Main SCAR-X vẫn một
lõi ALIF, không thay trọng số hoặc cấu hình Hermes.
