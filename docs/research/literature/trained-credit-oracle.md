# Credit tại checkpoint đã học — diagnostic đóng băng, không train

**Đã chạy đủ180/180rows; trạng thái `COMPLETE_CONTROL_ONLY`.** Không có cập
nhật học, không chọn candidate hoặc thay thuật toán. Bằng chứng này giới hạn
ở hướng credit của một loss có normalizer/readout đóng băng, không phải
accuracy, retention hoặc delta vật lý sau optimizer.

## Đăng ký và nguồn

Protocol `experiments/trained-credit-oracle-v1.md` được viết trước khi tính
row đầu. Script `scripts/verify_trained_credit_oracle.py` ghi protocol/source
hash vào manifest trước compute, kiểm lại sau cùng; reservation riêng một lần.
Ponytail không có trong tool inventory. Source model/adapters không bị sửa.

Nguồn checkpoint duy nhất là reference run đã hoàn tất132cells:
`20261112T080533Z-literature-reference-dev-v1-839e07`, source
`a3b79ea66cb6676f8b03a22b0cb907d158e5b2a2`.
Năm methods own,stop,eprop_matched,eprop_aligned,braintrace × ba primary tasks
strict_binding,xor,distractor_binding × ba DEVseeds140011,140017,140023.
Chỉ final equal_updates/unshuffled checkpoint;45checkpoint.
Mỗi checkpoint lấy TRAINindices0,63,511,1023, tổng180rows.

Không đọc ID/OOD input/labels, không mở confirmation bank, không thêm seed,
không gọi `learn_episode`. Nhãn đã thấy TRAIN chỉ tính loss tại endpoint;
không biến diagnostic thành dữ liệu huấn luyện hoặc oracle cho candidate.

Output unique:
`runs/20261112T083456Z-trained-credit-oracle-v1-462171/`.
`manifest.json` giữ source/protocol/parentmanifest hashes; `result.json` giữ
180rows và aggregate;45`raw/*.npz` giữ signal, local/fullvector, eligibility,
features,target,loss,indices cùng checksums. Không ghi lại raw reference.

## Phép đo chính xác

Clone từ checkpoint, reset dynamic state và chạy cùng sequence bằng local
step. Giữ nguyên W_in,W_rec,Wout,bias, f_mean/f_var và toàn optimizer state:

```text
scale=max(sqrt(f_var),.05)
raw=(f-f_mean)/scale; fhat=clip(raw,-5,5)
p=softmax(Wout@fhat+bias)
signal=Wout.T@(onehot(target)-p) * (abs(raw)<5) / scale
local_vector[j,k]=signal[j]*local_eligibility[j,k]
full_vector=signal @ full_surrogate_sensitivity(frozen_core,sequence)
```

Dấu là **ascent** theo convention1_y-p của source. Fulloracle giữ toàn bộ
recurrent sensitivity, kể cả self-edges, với cùng surrogate và hard forward.
Nó là control phi cục bộ chạy trong workspace research; candidate vẫn dùng
eligibility cục bộ. Numerical correctness của fulloracle đã được kiểm bằng
anchored-smooth whole-sequence finite-difference/JVP/VJP trong
`tests/test_literature_oracle.py`, không phải finite-difference Heaviside.

Đã kiểm hash datasetZIP và modelZIP trước đọc. Snapshot original model trước/
sau mỗi diagnostic khớp byte; clone giữ nguyên weights/readout, syn_trace,
syn_energy,f_mean/f_var, initweights, gamma, RNG và các learning counters.
Chỉ neuron/eligibility state và total_ticks của clone được phép tiến do forward.
Original model không thay cả trạng thái này. Tất cả180row kiểm state đều qua.
AST script không có call `learn_episode`; `new_learning_updates=0`.

## Kết quả descriptive

Mỗi ô cosine trung bình12rows:3seed×4episode. Đây không phải12independentseeds.
Không p-value/confirmation hoặc candidate ranking từ bảng này.

| Estimator tại checkpoint của chính nó | Binding cosine | XOR cosine | Distractor cosine | Mean toàn36rows | Mean norm local/full |
|---|---:|---:|---:|---:|---:|
| own |.56668|.47807|.67455|.57310|1.38717|
| stop |.64378|.57152|.79716|.67082|2.34422|
| eprop_matched |.68804|.56994|.78216|.68005|2.16723|
| eprop_aligned |.89978|.78673|.90410|.86354|1.49229|
| braintrace |.92928|.71448|.95269|.86548|.78568|

BrainTrace trained-checkpoint breakdown:

| Task | Cosine min–max | Mean sign agreement | Mean norm local/full |
|---|---:|---:|---:|
| Binding |.78647–.97056|.88835|.96396|
| XOR |.54804–.85397|.83561|.37372|
| Distractor |.93726–.97267|.92611|1.01936|

Không có row localnorm hoặc fullnorm bằng0, không có undefinedcosine bị bỏ.
Đã chấm lại cả180cosine từ rawvectors bằng scalar sum/dot/norm độc lập;
khớp receipt tới1e-12. Toàn bộ45rawNPZ hash khớp và original_model_unchanged
đều true.

## Diễn giải được và không được

1. Initial frozen fixture từng cho BrainTrace cosine gần.998. Ở các checkpoint
   đã học, estimator không còn gần1 trên mọi sequence: XOR mean.7145 và
   min.5480; có credit đáng kể bị local approximation bỏ đi.
2. **Không quy mức giảm đó riêng cho quá trình học recurrentweights**: fixture
   ban đầu khác input, signal và checkpoint với phép đo này. Chưa chạy paired
   initial/final cùng input/signal; không tự thêm phép thử ngoài registration.
3. Mỗi method được chấm trên finalweights/readout/stats của **chính method đó**.
   Vì vậy các hàng không tạo matched-weights causalranking giữa năm algorithm.
   Bảng chỉ đo độ lệch estimator-vsfull tại từng checkpoint.
4. Alignment cao ở binding/distractor của BrainTrace chưa đủ tạo directskill
   tốt trên reference DEV. Đây là phản ví dụ hữu ích đối với suy luận rằng
   “gradient gần oracle” tự bảo đảm chất lượng. Chưa xác định duy nhất nguyên
   nhân: representation/readout conditioning, surrogateobjective, clipping và
   optimization trajectory đều còn khả năng.
5. Signal ở đây freezef_mean/f_var. Sourcelearner thực sự cập nhật statistics
   rồi dùng momentum/energy normalization và clips; fullvector này không là
   gradient xuyên mọi operation của learner đó, cũng không là physicalupdate.
6. Normratio khác1 nhắc cần kiểm actualnorm sau optimizer/cast/clip trước khi
   quy lợi ích cho hướng credit. Không dùng cosine để miễn matchedcontrols.
7. Không dùng kết quả này tune/“cứu” ACR; protocol ACR và scientificgates giữ.

## Tài nguyên và tái lập

Lượt diagnostic wall **10,913253giây**, giới hạn đã khóa180giây.
Observed treeRSSpeak50.929.664B, committedpeak70.438.912B; nativeprocess
high-waterRSS51.916.800B và committed71.225.344B. Không measurementerrors,
không vượt2GB. Windows per-process cap receipt `enforced=true`; monitorflag
`hard_cap_enforced=false` riêng chưa được nối tới receipt, không đánh đồng.
Không GPU/iGPU hoặc phép đo năng lượng.

Lệnh thực thi gốc:

```powershell
python -X utf8 scripts/verify_trained_credit_oracle.py
```

Runner từ chối chạy lại cùng reservation để tránh lặp budget vô ý. Tái lập
độc lập cần checkout source/hash và reservation/output nghiên cứu mới được
ghi rõ; không xóa reservation hoặc ghi đè evidence hiện tại. Audit số liệu chỉ
đọc45rawNPZ/result, không cần mở training. Main/Hermes/SQLite/port8767 giữ nguyên.

**Kết luận cơ chế:** phương trình localJacobian và feature alignment có thể
kiểm chứng, nhưng giới hạn credit vẫn hiện rõ ở trainedXOR. Đây là
`PARTIALLY_SOLVED`, chưa phải chất lượng đa kỹ năng hoặc `P1_PASS`.
