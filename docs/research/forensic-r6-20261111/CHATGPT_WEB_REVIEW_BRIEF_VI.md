# SCAR-X — bản chuyển ChatGPT web đối chứng, 11/11/2026

## Kết luận hiện tại của Codex

**Có tiến bộ về đọc thông tin từ frozen ALIF; chưa chứng minh một learner hidden
synapse học representation rộng và giữ được kỹ năng lâu dài. Chưa duyệt learner
mới vào main.** Đây là kết quả audit, không phải tuyên bố SNN không thể học.

Codex đã đọc đầy đủ hai tệp handoff, đối chiếu19nhánh local/remote tại snapshot P0,
rồi bổ sung Gemini R6C khi commit mới được push. Các JSON handoff chỉ là đối chiếu.
Các phép replay dùng seed lịch sử, không phải fresh confirmation.

## Source và kết quả đã chạy thật

| Nhánh/source | Bằng chứng Codex |
|---|---|
| Main `495dab2f4bac0180e4d1cd28674391f055ccf09d` | Không sửa/merge main hoặc data live |
| DeepSeek R6D `4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893` | Replay8seed:80trial metrics và summary khớp chính xác; chấm lại prediction độc lập |
| Grok R6B `c39e9db7771582d9933af402a209917328b71119` | Replay8seed R5B + screen3seed: mọi trường khoa học đã công bố khớp; test actual NoClip |
| Gemini R6C `8f24a5abde0598201e558eee559564ccde889145` |5test qua; replay8seed,344nhóm prediction chấm độc lập. CKCER A/B accuracy/both_A khớp; toàn seed_results không bit-exact do margin rounding và1prediction baseline R5 reversed-write khác |

R6D: `both_A`1K→16K **.6872846→.9178224**, paired gain **+.2305378**,
CI95% **[.1716115,.2948206]**. Equal16Kupdates: unique−repeated **+.1507584**,
CI95% **[.0968468,.2223458]**. **Hidden không được học trong các arm này.**

Grok R6B: head `both_A` **.6676172**, ridge **.9858056** trên cùng feature tensor;
NoClip gain **+.0098201**, CI chạm giá trị âm; protected order giảm **.1875**. STOP đúng.

Gemini R6C: CKCER clean `both_A` khoảng **.995968**, paired distractor **.965844**.
Đây là kết quả head/query-gate trên frozen H48, chưa phải hidden recurrent learning.

## Những vấn đề còn mở — đề nghị đối chứng

1. **Attribution hidden/readout:** R6D/R6B/R6C chủ yếu frozen hidden. Oja ở
   `CKCERHead.w_q` là query-readout plasticity; không phải `w_in/w_rec` học mới.
   P1-score/main20 và DC-ALIF có tín hiệu trên một số bài hẹp, nhưng chưa chứng minh
   tổng quát hóa/giữ nhiều kỹ năng. Không được kết luận mọi hidden learning bằng0.

2. **R6D thiếu khép kín gates:** matrix raw giữ G4/G5/G8=null; control run dùng
   khác training order. Codex đã tính lại controls nhưng giữ nhãn khác trained
   head, không ghép mù thành một thí nghiệm. G6 ghi distractor **.5502305** vẫn fail.

3. **Counterfactual stress không còn cùng nhiễu:** trên actual512paired rows mỗi
   seed R6D,256/256cặp temporal bị thay phần ngoài query; distractor lệch252–255cặp.
   Oracle từng row vẫn đúng. Điểm đo là hợp lệ cho cặp có hai realization khác
   nhau, nhưng chưa cô lập tác động query như same-noise counterfactual tuyên bố.

4. **Gemini đã nhìn confirmation FAIL rồi sửa/rescore:** run
   `20261111T095646Z-r6c-confirmation-v1-d215bc` STOP vìG8. Sau khi đổi protected
   generators/baseline features, cùng8seed chạy lại ở
   `20261111T100240Z-r6c-confirmation-v1-f8d321` thànhGO. Cần amendment và nhãn
   corrected replay. Bank121011…121043 cũng đã xuất hiện ở DeepSeek R6A.

5. **Protocol Gemini khác code:** protocol noise.01→.03 và distractor lúc test;
   code dùng.08→.16, train lại ở noise cao và train có distractor. Windows/ticks
   cũng khác. Không so96,58% Gemini như cùng phép thử zero-shot với55,02% R6D.

6. **Ablation chưa đơn yếu tố:** Gemini matched/Step2 dùng hard perceptron,
   CKCER dùng softmax delta. Codex kiểm class thật trên fixture nhỏ: cùng nhãn0,
   ablation không đổiW, CKCER đổi max|W|=.075. Gain không được quy hết cho bỏ trunk.

7. **Hai suy luận toán cần sửa:** inactive-branch weights đúng là không đổi,
   nhưng shared bias làm scores query khác đổi **.025** sau1update. R5 đọc
   post-query state, nên shared trunk không luôn query-invariant; counterexample
   thật có chênh state/margin **1.80248**. Không thể áp định lý cần cùngstate
   trướcquery cho R5 gốc mà bỏ điều kiện đó.

8. **Data size không đồng nghĩa số kiến thức:**8seed không có raw episode overlap
   giữa train/calib/held/paired. Nhưng train1K không nằm trọn train16K; đều chỉ
   có8logical states `(vA,vB,q)`. Data/order/noise-support khác nhau, không có
   bằng chứng một phase transition hay16Kkhái niệm mới.

9. **Retention và product còn fail:** main P2 lịch sử final34,74%, mean forgetting
   29,09đ; DC-ALIF P2 final56,13%, forgetting21,89đ. Các vòng sinh câu/ảnh chưa qua
   quality gates. CKCER không đo continual dialogue, chat, normal vision hoặc
   learn-new-without-forgetting trong shared production session.

10. **Engineering:** R6C chưa có persistence/rollback product đầy đủ cho head,
    normalization, Oja/gating/latched traces/controller. Full pytest cùng process
    vượt2GB khi cha và child control cùng nạp Torch; bounded partition chạy được.
    Đây là lỗi cách chạy kiểm thử, không bằng chứng serving G1 vượt2GB.

## Những gì nên giữ để nghiên cứu, chưa đưa vào main learner

- Pre-gating normalization và nhánh tắt=0.
- Pre-query count/adaptation temporal summary.
- Neural-transient query gate Oja, đúng phạm vi readout.
- Independent evaluator, null-safe gate validation, source/data/order/scaler
  digests và regression chống lỗi harness — có thể đề xuất nhập hạ tầng research
  sau review riêng. Không tự đổi production forward/learning rule.

## Câu hỏi đề nghị ChatGPT web phản biện

1. Kiểm lại các hash/source/run trên GitHub: kết luận nào ở trên sai hoặc quá rộng?
2. Trong CKCER, lợi ích nào do representation, normalization, optimizer hay
   supervised routing? Cần matched control nào để phân biệt?
3. Theorem của Gemini có thỏa điều kiện cùngstate ở actual R5 post-query không?
4. Có cách giữ branch-isolation mà không bỏ qua shared bias và gate drift không?
5. Protocol R6E/R7 nên kiểm hidden-learning bằng common refit/frozen/zero/shuffled
   thế nào trước khi nghĩ đến neuron mới hoặc tăng quy mô?
6. Có công trình cùng hướng nào đã có matched benchmark tốt hơn? Xin dẫn paper,
   code, revision và learning-signal assumptions; không lấy pretrained/BPTT làm
   bằng chứng SCAR-X local learning đã giải xong.

Không duyệt mở thí nghiệm mới, merge, scale hoặc thay architecture chỉ từ trao đổi
này. Đề xuất cần protocol được người dùng xem và duyệt.

## Còn bao nhiêu task đến production?

Với **production trong miền kiến thức cơ bản hữu hạn**: **10 nhóm việc còn mở**,
không phải10commit và không phải ước lượng thời gian. Các nhóm nghiên cứu có thể
cần nhiều vòng hoặc không đạt. Chat/vision đa năng là mục tiêu xa hơn, không được
đánh đồng với bản miền kiến thức nhỏ này.

| # | Nhóm việc | Điều kiện xong |
|---:|---|---|
|1|Khép kín audit/harness/protocol|Gates đầy đủ, source/seed/calibration/order đúng; sửa nhãn post-hoc và counterfactual|
|2|Chốt cơ chế readout công bằng R6E|Same learning rule, bias/gating/timescale ablation và stress đã khóa; untouched confirmation nếu được duyệt|
|3|Chứng minh hidden representation R7|Learned hidden thắng initial/frozen/zero/shuffled bằng common refit và causal tests; giữ P1 gates|
|4|Học liên tục/retention|Acquisition/final≥65%; mean forgetting≤3đ, worst≤5đ, mọi seed; P2 duyệt riêng|
|5|Data và input/output codec|Nguồn/nhãn/paraphrase/unknown rõ; khôngOOV/collision/EOS lỗi; mọi codec change có migration approval|
|6|Neural QA và unknown có ích|A≥80% factual heldout, B≥75% paraphrase + contrasts, C≥95%unknown; khônglookup/LLM/template thay neural|
|7|Checkpoint/migration/rollback|Lưu/khôi phục cả hidden/head, traces, normalizer, gate/Oja, RNG, replay và controller; cold exact resume|
|8|Runtime dài hạn|CPU-first, toàn system≤2GB khi infer/learn/consolidate/rollback; bounded idle, foreground priority,8hkhông leak|
|9|Tích hợp Hermes/observer thật|Mộtcore; output/uncertainty/updates/guard/version/RAM/neuron thật, không fake thinking hoặc Production Ready|
|10|Release acceptance và user approval|Independent final test, clean install/API/E2E/regression/resource/provenance và duyệt deploy|

**Đường găng:** nhóm3và4chưa có lời giải đạt chuẩn; nhóm1/2phải làm đúng trước
khi dùng kết quả để chọn thuật toán. Không thể cam kết production chỉ bằng
số lượng task hoặc thêm dữ liệu/neuron.
