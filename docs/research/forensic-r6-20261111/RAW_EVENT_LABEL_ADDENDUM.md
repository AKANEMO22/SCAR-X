# Bổ sung kiểm toán event/nhãn Gemini — 11/11/2026

Bổ sung cho bàn giao `24453bdcfbd61f9fa36804b98e6f4d23af180182`.
**Không thay điểm benchmark, không huấn luyện/forward neural, không đổi learner
hoặc đánh giá seed mới.** ZIP và các run đã bàn giao giữ nguyên.

## Khoảng trống đã khép lại

Đã kiểm độc lập135.168**lượt row** của8seed lịch sử121011…121043, qua train/
held/paired và9loại condition. Đây không phải135.168kiến thức hay mẫu độc lập:
nhiều condition là biến đổi cùng dữ liệu gốc. **0nhãn không khớp event được sinh.**

Oracle độc lập đọc marker key/value theo thời gian và lấy lần ghi cuối cùng;
không dùng y/vA/vB/metadata của agent làm đáp án. Đối với noise.16, bộ kiểm toán
tái tạo đúng background từ seed rồi trừ nó chỉ trong evaluator để kiểm event
latent. Model không nhận tensor đã trừnoise. Đối với move_query, kiểm riêng phép
dời tick tuyến tính và pulse tensor; không âm thầm thay ngưỡngmarker.5.

| Condition | Row được kiểm | Nhãn sai | Cặp khác nhiễu ngoài query / tổng |
|---|---:|---:|---:|
| Clean |16.384|0|0/2.048|
| Shift+1 |16.384|0|2.048/2.048|
| Reversed writes |16.384|0|0/2.048|
| Query tick variation |16.384|0|0/2.048|
| Distractors |16.384|0|2.030/2.048|
| Noise change |16.384|0|2.048/2.048|
| Same-sample paired distractors |4.096|0|0/2.048|
| Structured third write/windows |16.384|0|0/2.048|
| Key overwrite |16.384|0|0/2.048|

Hai bài auxiliary nay có bằng chứng nhãn/event độc lập ở các tensor lịch sử này.
Điều đó tăng độ tin cậy của data, không biến gain thành hidden representation
learning hay sửa các lỗi preregistration/ablation đã báo.

## Các phát hiện bổ sung

1. **Nhãn đúng không bảo đảm bộ đọc cue đúng.** Ở noise condition, hàm cue hiện
   dùng `channel5 > .5` khác query latent trên4/16.384rows. Những lỗi này thuộc
   input/query decoding của condition, không phải nhãn bị sửa. Không suy bốn
   trường hợp này giải thích toàn bộ lỗi CKCER hoặc model đã tự biết sửa chúng.
2. **Raw marker oracle có thể mơ hồ với Gaussian noise.** Cả24dataset
   train/held/paired noise được kiểm có ít nhất một marker ambiguity với bộ
   đọc.5; query-move có thêm1dataset như vậy vì cộng hai background ticks.
   Latent label vẫn khớp sau kiểm event/background độc lập. Do Gaussian không
   bị chặn, không được tuyên bố mọi observed tensor đều có oracle marker tuyệt
   đối rõ ràng. Raw run dừng đầu tiên được giữ, không chuyển thành PASS hồi tố.
3. **First-key không phải first-written.** Trên16.384clean rows, baseline source
   luôn trả keyA. Nó khác giá trị key được ghi sớm nhất ở4.126rows. Vì vậy nhãn
   “first-key bias” phải phân biệt key identity với chronology. Với bài nhị phân,
   `wrong_first_key_qB_diff` theo keyA còn chỉ là phần bù của accuracyqueryB khi
   hai giá trị khác nhau; không phải một chẩn đoán độc lập về ưu tiên write sớm.

## Bằng chứng và giới hạn

- Run thành công: `runs/20261111T105004Z-forensic-gemini-events-v1-a41c3e`.
- Run raw ambiguous được giữ: `runs/20261111T104720Z-forensic-gemini-events-v1-dd3666`.
- Source Gemini được pin `8f24a5abde0598201e558eee559564ccde889145`.
- Source oracle: `src/scarx/evaluation/forensic/event_oracle.py`; script
  `scripts/forensic_gemini_event_audit.py`;6regression tests độc lập.
- Sampled peak treecommit41.320.448B, hard per-process2GB bật;12,32s;
 0neuralforward/0learningupdate. Không đụng main/live/Hermes.

**Quyết định không đổi:** giữ CKCER làm ứng viên nghiên cứu readout, chưa đưa
learner vào main; R6E/R7 chờ duyệt. Không cần sinh thêm dataset hoặc sửa
neuron để xử lý khoảng trống audit dữ liệu vừa được khép lại này.
