# Đối chiếu continual-SNN sau P1B action-bank STOP

Không có learner/dataset mới được chạy trong intake này. Mục đích: kiểm xem
các dự án gần hướng có giải được nút thắt SCAR-X hiện tại hay chỉ khác task/
representation/hardware. Giữ nguyên gate và raw failures; không suy paper claim
thành bằng chứng SCAR-X đã học được kỹ năng.

## CLP-SNN: prototype và representation cần phân biệt

Đã đọc [README tác giả](https://github.com/elvinhajizada/CLP-SNN),
[Replay.py](https://github.com/elvinhajizada/CLP-SNN/blob/main/models/Replay.py),
và metadata/abstract [paper2511.01553](https://arxiv.org/abs/2511.01553) được repo
dẫn. Chưa audit toàn repo hoặc tái lập scores. Decision **REFERENCE/DEFER**,
không cài Torch framework/Loihi emulator, không download1,2GB feature corpus.

Tác giả dùng frozen/pre-extracted EfficientNet-B0 features1280D trên OpenLORIS
object classes; simulator có adaptive prototypes/metaplasticity, còn deployment
Loihi2 là allocation-only/imprint/freeze. README phân biệt rõ hai path. Số đo
continual classification trên semantic backbone features không chứng minh task
understanding, recurrent credit hoặc SNN/RAG advantage trong SCAR-X.

Replay.py là **softmax baseline dùng Torch**, không chính local SNN rule. Có
optionalFP16 và feature replay; không copy vào ALIF hoặc dùng half/quantization
để đạt ngân sách trái yêu cầu. Chi tiết reusable là baseline/feature-state/
sample count/compute/retention phải cùng điều kiện và khai báo rõ. Prototype
allocation freeze cũng không tự là hidden representation learning.

## Local STDP và forgetting

Đã đọc abstract [Continuous learning of spiking networks trained with local rules](https://arxiv.org/abs/2111.09553),
chưa fullpaper/source audit. Tác giả nghiên cứu importance qua Langevin dynamics
không cần global loss gradients và forgetting ở local STDP; **REFERENCE**, chưa
đủ để nhập một rule hoặc công bố hiệu năng. Không rebrand magnitude importance
trước đây của SCAR-X đã FAIL thành tái lập bài này. Cần inspect equations/code/
license nếu thật sự chuyển cơ chế; không chạy sweep ngay từ abstract.

Đã có e-prop/BrainTrace/SI/DECOLLE intake và raw reference cũ. Không chạy lại
132cells để chứng minh trace tồn tại. Unknown feature/action utility không
được sửa bằng thêm replay khi input thiếu hoặc oracle không hơn baseline.

## Khoảng trống của SCAR-X còn cần giải quyết

1. **Information:** signed-hash64 và giới hạn128words khác semantic backbone
   pretrained1280D. Causal blocks P1A giữ task/error history nhưng chưa chứng
   minh mã hóa quan hệ hữu ích; không coi input khác nhau là skill đã học.
2. **Action utility:** bank P1B cũ có6actions, TESToracle10/16 bằngRAG10/16.
   Không learner nào chọn trong chính6actions tạo được>5pp gain; đó là giới
   hạn môi trường/action/LLM, chưa là verdict toàn bộ SNN. Reuse bound audit,
   không huấn luyện nhiều seed chỉ để xác nhận cùng cận.
3. **Credit:** outcome verified gắn đúng decision đã được kiểm frozen. Còn
   thiếu study học có useful actions và heldout transfer; replay positive
   verify labels hoặc LoRA success không đáp ứng claim hidden-learning.
4. **Retention:** cần acquisition rồi old-skill probes sau20learning stages.
   Snapshot save/load hoặc không-update nên không-quên không là bằng chứng.
   LoRA control12câu0observedregressions không đủ noninferiority3/5pp.

**Định hướng:** ưu tiên phiên bản task/action contract mới chỉ khi có giả thuyết
phân biệt được và benefit headroom từ actual tools; cùng nguồn/hardware/action
budget cho RAG/readout/hidden. Giữ model/capacity và local rule. Nếu representation
phải đổi, ghi riêng adapter/encoder version và sự đóng góp của pretrained parts.
Không tự thay ALIF bằng prototype classifier hoặc kéo paper scores vào product.
