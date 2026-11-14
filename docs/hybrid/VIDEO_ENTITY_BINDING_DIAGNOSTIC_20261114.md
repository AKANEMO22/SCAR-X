# Lỗi gắn người với sự kiện video — diagnostic14/11/2026

Đã dùng cùng Qwen0.8B/nativeGPU trên đầu ra ASR thật của lượt a98122, khóa
protocol trước16unique native calls. Không train, thay prompt sau kết quả, nạp
checkpoint live hoặc sửa profile8h.28armrecords dùng chung output khi inputSHA
giống nhau, không coi là28mẫu hay3seed độc lập của VLM.

## Phát hiện

| Câu hỏi Adam Hall hạng nào? | Rank đọc từ đáp án | Exactname trong evidence? |
|---|---:|---|
| BaseASR, chronological/reverse |15/15|Không|
| SmallASR, chronological/reverse |15/9|Không|
| SmallASR, untrainedSNNseed260113/137/171 |15/9/15|Không|
| Humanreference, chronological/reverse |9/9|Có|
| Humanreference, chỉ câu đúng target |9 (đúngstrictinteger)|Có|
| Không evidence |10 (bịa)|Không|

ASR ghi “Aramhorn” thay AdamHall. Giữ cảhai đoạn nhưng đổi thứ tự là đủ làm
đầu ra đổi15→9 với smallASR. SNN chưa học ởseed137 tình cờ xếp order thuận lợi,
không phải lợi ích representation học được. Transcript người xác nhận cho
đúng9 ở cảhai thứ tự; đó là oraclepositivecontrol, không đối chứng fair mà
SCAR-X được quyền truy cập khi dùng video bình thường.

ArlyVelasquez: cảASR/humancontext trả15 theo parserdiagnostic; nonevidence
bịa hạng1 trong WWE. Không nói thiếu evidence thì model luônbiết từ chối.
Exact lexicalname check không khẳng định alias lỗiâm là người khác; nó chỉ báo
chưa có anchor phù hợp. Không tự nhập “Aramhorn=AdamHall” thành nhãn học.

## Giới hạn và hướng sửa

- Lỗi hiện có cả mất entity quaASR và phụ thuộc order tronglanguagebinding.
  Không quy toàn bộ cho SNN, retrieval hoặc VLM bằng mộtknowncase.
- System instruction hiện bắt citation/time trong khi user yêu cầu chỉsố.
  Strictformat failure có conflict; diagnosticrank được đăngký riêng trướcrun,
  không rewrite strictscore thànhPASS. Số9vs15 là lỗi dữ kiện riêng.
- Có thể giữ thứ tựtime/metadata, bảo toàn tên riêng hoặc yêu cầu người dùng
  xác nhận entity chưa rõ; không tự tái xếp order theo outputgold hay nới gate.
- Muốn sửa sinhđápán có cấu trúc/provenance cần bộ mới source-disjoint và cùng
  budget/strongRAG/readout/frozenhiddencontrols.16calls không phải videoP1.

[Protocol](../../experiments/hybrid-video-binding-diagnostic-v1.md),
[raw outputs/source/cache](../../runs/20261114T022448Z-video-binding-diagnostic-v1-2a3230/result.json),
[runner](../../scripts/diagnose_hybrid_video_binding.py).
ALIF/topology/weights và model live khôngđổi; bản này chỉ cung cấp chứng cứ để
chọn bước nghiên cứu tiếp theo, chưa deploy heuristicbinding.
