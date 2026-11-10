# Câu hỏi ảnh có thật sự tới neuron không?

10/11/2026. **Audit chỉđọc,240inferences thật; không huấn luyện hoặc đổi main.**
[Protocol](../../experiments/vision-question-path-audit-v1.md),
[run](../../runs/20261109T215943Z-vision-question-path-audit-v1-41c917/report.md).
Phần này độc lập với đềxuất30 đangchờngườidùng; không suy ra approval từ goal
continuation và không triển khai errortransport mới.

## Dữ liệu và đường đo

AntigravityCLI đã sinh32câu hỏi thực (8/kind,4vi/4en) trong16,03s, toolsteps0.
[Generation](../../runs/20261109T215349Z-antigravity-vision-questions-v1-bfbd17/report.md)
và [editorialreview](../../runs/20261109T215556Z-vision-question-semantic-review-v1-d2f2c5/report.md)
giữ nguyên câu chữ, có3caveat (cách hỏi hình học hơi cứng, “mépngoàicùngbêntrái”
được hiểu leftmost trong controlledfixture). Khôngteacheranswers/privateinputs.
Review đủchoaudit đường vào, không là điểm naturalconversation/visionsemantic.

12ảnhhình học128×128 doPillowrenderer của dựán tạo: circle/square/triangle,
red/green,1–3objects. Đápán màu/hình/count/leftcolor lấy từ manifest có kiểm
palette, nonoverlap và connectedcomponents trênpixel thật. Không lấy output
teacher/model làm nhãn. 240eligibleimage-questionpairs, không balancedgeneral
accuracybenchmark; automatic_learning_allowed=False, chưa nhập vào học.

Mỗi case gọi đúng `Generation.vision` trên một sessionpublicreleaseclone,
backgroundoff, restore cùngstate,48neurons/capacity cốđịnh. Recorder passthrough
đọc actualcore.step vectors mà không đổi phương trình hoặc output. Không dùng
privateweights/feedback và không sửa app trongHermes. Source/headweights/
counter/dialoguestate không đổi; fullsessionclone khôi phục đúng sauaudit.

## Kết quả: câu hỏi bị bỏ trước neuralcore

| Nhóm ảnh | Sốảnh | Câu hỏi/ảnh | Input neural khi đổi câu hỏi | Xác suất neural | PrototypeRGB+question |
|---|---:|---:|---|---|---|
| Một vật thể |6|24|Giốnghệt|Giốnghệt|24sequencekhác/ảnh|
| Nhiều vật thể |6|16|Giốnghệt|Giốnghệt|16sequencekhác/ảnh|

240/240calls có0giátrịkhác0 trong128questioncontextchannels. Cùngảnh/state/
budget tạo cùng16tickinputtape và output10digitprobabilities, dù renderer
targets giữa color/shape/count khác nhau. Source `plan_workload` chỉ đo độdài/
capacity khi cópixels, không encodequestion; `UnifiedCore.image` chỉ nhận64
pixel và visioncontrol rồi đọc10digitrows. API nhận prompt không có nghĩa là
prompt đã được neuralcore xử lý.

Capacity mặc định có thể thay khi inputvolume khác; audit cốđịnh48 để phân
lập semanticinput, không nói mọi câu hỏi trong mọi hìnhthức đều giống nhau.
Không gọi đây là regression của model, chỉ kiểm chứng hạn chế pipeline hiện tại.
Tăngdata, tick hoặc replay không khôi phục được thôngtin chưađưa vào neuron.

6/6cặpRGB khácmàu trở thành cùng8×8graypixels. Đây là **tái kiểm giới hạn đã
công bố ởbáo cáo13**, không một phát hiện mới. RGBInputEvents prototype giữ
khácbiệt màu và câu hỏi trong cùngchannelbudget; chưa bật runtime và khác
vector không chứng minh neuron đã họcvision hoặc languagegrounding.

## Phạm vi kết luận và bước tiếp theo được phép

Đã xác định thiếu đường jointquestion+image→wordreadout cóhọc, không chỉ thiếu
kiếnthức. RawdigitJSON thật được giữnguyên; renderer đápán chỉbảngđối chiếu,
không dùng host trả lời thay neuralmodel. Không benchmarkQwenVL: Qwen3-4B-
Instruct hiện có làtext-only. Không gửi image-as-text/sceneanswer cho nó rồi
gọi đó làvision, không tải/bậtQwenVL hoặc9B, không ngụy tạoreference.

Bước độc lập tiếp theo có thể nối **prototype** question/RGBevents tới cùng
ALIF và wordreadout, sau protocol/noleak/parity/ablation, không đổi tínhiệuhọc
đangchờduyệt. Chưa triển khai trongaudit, chưa quyền bậtmain hoặc nhận normal
vision. Phải giữ causalimage/questionablation, renderercombinationheldout,
freewordoutput khác class-nameformatter và nguồn/datareview rõ. P1/P2gates,
closedfamilies và CPU2GB vẫn giữ, không scale model.

## Tài nguyên và quan sát

Audit≈6,94s, peakRSS59,06MB/commit42,59MB; mộtpubliccheckpoint, không240seeds.
Khôngweight/headupdate, iGPU/offload/energyclaim. Mainprivateweights không
được dùng hoặc thay.128×128ảnh gốc và8×8graythumbnail (phóngnearestneighbor)
cùng câu hỏi/rawneuralJSON lênlocalhost tự động, không userstart/selector.
Reader chỉ đọc PNG thuộc ownedfixture cóhash, không trả arbitrarylocalphotos.
Protocol/source/generation/review/audit được giữ, không rewrite lịch sử.
165softwarechecks qua; sourceweight/protocol/input/mathSHA và240outputrecords
đãkiểm. Mainprivateweights/topology/vocab/encoding/counter241363 giữ nguyên;
reader/UI được quan sát thật. Softwarepass không là visionqualitygate.
