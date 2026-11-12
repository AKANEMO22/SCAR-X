# Đề xuất phân biệt capacity và learning — CHƯA PHÊ DUYỆT CHẠY

Người dùng hỏi liệu H48 quá nhỏ, RAM còn dư và tăng neuron có giúp không.
Đây là phân tích/đề xuất, không quyền tăng model hoặc mở training mới. Giữ lượt
ACR H48 đang chạy nguyên vẹn. Không suy ra giới hạn SNN từ48neuron, cũng không
suy ra lợi ích chất lượng từ RAM còn trống. Primary scientific gates không đổi.

## Số hiện có và dự toán, không phải benchmark H64/H128

H48 D8 fan8 O2 có18H+2=866unique trainable scalar weights, không phải48tham số.
Core còn stablecopies, traces, optimizer, metadata. Với đúng layout hiện tại và
plastic_rows12, mảng ACR tính theo944H+864byte (không gồm Python/runtime/tempZIP).

| H | Unique trainable weights | ACR array bytes theo công thức |
|---|---:|---:|
|48|866|46.176|
|64|1.154|61.280|
|128|2.306|121.696|

Đây là phép tính. Adapter ACR/reference/checkpoint hiện pinH48, không hỗ trợ
đổi ConfigH rồi chạy ngay. Cần prototype secondary riêng và parity H48 trước;
không sửa candidate/kernel/source đang dùng trong lượt đã đăng ký. Processpeak
thực tế khoảng168MBRSS/212MBcommit ở snapshot279jobs; array bytes không là
tổng RAM và số đo này không phải bản AI đa năng đã đạt2GB.

## Giả thuyết phân biệt

- CAPACITY: H64/H128 giúp biểu diễn sự kiện/quan hệ tốt hơn trong cùng local
  learner, heldout/OOD và causal gains tăng, không chỉ frozenrandomfeatures tốt.
- LEARNING: learner lớn hơn vẫn khai thác representation kém hoặc readout đè
  kỹ năng; eligibility/signal/optimizer hoặc data budget còn là nút thắt.
- COMPUTE: tăngH chỉ thắng vì dùng thêm CPU. Phải báo equalupdates và equalCPU
  riêng; một kết quả không thay kết quả kia.

Không cải thiện khi tăngH chưa đủ chứng minh architecturelimit; model lớn có
thể cần dữ liệu/optimizationbudget khác. Chưa có scalinglaw chuyển thẳng từ
LLM/SpikeGPT sang ALIF local learner này.

## Thiết kế đề xuất nếu được duyệt

H48/64/128, D8/fan8/O2, hardALIF; cùng cơ chế học và profile theo intrinsic
quy tắc, không chọn theo task. Arms own-reset, globalACRgroups_up vàreadout-only.
ThayH đồng thời thay mật độ sparsefan8/H: phải báo, không coi là chỉ thay một
số vô hại. Không âm thầm đổi fan-in để giữ mật độ; đó là một factor khác.
H48 endpoint/equations phải khớp raw đã hoàn tất trước khi mở H64/H128.

Ba DEVseed đã đăng ký140011/17/23,4tasks và frozen splits cũ, cùng calibration/
labelbudget/optimizer/sourceorder. Init **trong cùngH** exact giữa arms; giữaH
dùng paired seed và cùng phân phối/normalization, không giả vờ weights khác
shape là bitwise-identical. H48 data/modelresults reuse có hash nếu portparity
đủ; không train lại H48 chỉ để có score đẹp.

Equal1024updates primary; equalCPU secondary tách riêng, locktime/overshoot;
readout-only, causal hiddenrevert và refitprobe giữ nguyên. Chỉ gọi lợi thế
local-learning khi learned-vsfrozen contrast tốt hơn, không chỉ tăng random
reservoir/readout. Task direct/probe/OOD/order đo riêng, không chỉ accuracyTB.
Tập language/vision/Hermes không thuộc capacityprobe này.

P0 phải xử lý snapshot layout mới/allstates/RNG/coldresume/rollback và memory
preallocation trước training; cùng versionkhông âm thầmload H48→H128 bằngzero
padding. Giữ endpoint normalization/init policy, mộtforward/core. Trần2GB đo
cảlearn/checkpoint/rollback; max60minworker, resource/seed/condition budget cuối
phải đăng ký trướcscores. Sốjob tối đa cần khóa sau đo tốc độ kỹ thuật, không
dùng tắt bớt seed sau khi thấy chất lượng.

Chưa P2: không gán retention tốt hơn nếu chưa kiểm20context theo gates
acquisition/final65%, meanforget<=3đ/worst<=5đ theo seed. Không freshconfirmation,
main/Hermes/liveSQLite hoặc C++/iGPU từ đề xuất này.

## Diễn giải trước khi xem kết quả

- Lớn hơn thắng cả learned-vsfrozen, direct/OOD và equalCPU: bằng chứng capacity
  có ích ở bài đã công bố; mới cân nhắc duyệt quy mô tiếp.
- Chỉ frozen/probe tốt lên: thêm capacity, chưa chứng minh synapse học tốt hơn.
- Chỉ equalupdates thắng cònequalCPU thua: tradeoffcompute, không efficiencygain.
- H128 vẫn không cải thiện: giữ kết luận chưa phân biệt learning/data/capacity,
  đề xuất control optimization/oracle cô lập; không tuyên bố SNN bất khả thi.

Không đổigates để cứu H48 hoặc H128. Tăngquymô là giả thuyết kiểm chứng được,
không là cách che lỗi học hoặc kết luận rằng mọi scaling đều là sai lầm.
