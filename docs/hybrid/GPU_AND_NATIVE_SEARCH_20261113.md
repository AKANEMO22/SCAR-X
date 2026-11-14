# GPU, CPU và SearXNG native — 13/11/2026

**Cập nhật14/11:** [profile hiện tại và cách chuyểnmode](ASR_RUNTIME_AND_8H_20261114.md)
đã có ASRsmallFP32/allneuralGPU. Khi chuyểnCPU cần chọn base/name/revision đúng
cùngmode; không giữ smallCPU chưaqualified. Các cấu hình/ASRCPU và số đo bên
dưới ghi trạng thái13/11, không thay cho telemetry/runtime hiện tại.

**Đang dùng GPU RTX4050; người dùng đã bỏ trần RAM/committed của SCAR-X cho
GPUmode.** CPUmode vẫn giữ4.500.000.000byte. Cùng model0.8B, precision gốc,
context16K và ALIFH128. Không quantization/pruning/LoRA trong lượt này.

## Cấu hình và vận hành

File duy nhất chọn chế độ: `local/hybrid-v1/config.json` trong worktree này:

```json
{
  "execution_mode": "gpu",
  "gpu_device": 0,
  "searxng_enabled": true
}
```

Đổi execution_mode thành cpu hoặc gpu, sau đó từ rootworktree chạy:

```powershell
python scripts/hybrid_service.py stop
python scripts/hybrid_service.py start
```

Đây là chuyển backend thực, không chỉ thay nhãn giao diện. CPU dùng runtime
llama.cpp-b12360 gốc; GPU dùng official matching CUDA12.4, CUDA0, offload các
layerVLM và visionprojector. ASRFP32 và SNNlocal hiện vẫn chạyCPU. ModeGPU không
tự biến toàn bộ hệ thành kernelGPU, không tăng neuron/kiến thức.

`HybridConfig.effective_ram_limit` và softlimit bằngnull ởGPU; CPU vẫn4,5/4,25GB.
WindowsJob giữ lifecycle containment cho children ở cả2mode, chỉCPU có memory
limit. RSS/privatecommit/sharedGPU vẫn đo50ms; dedicatedVRAM đoPDH riêng2s.
Trần vật lýVRAM6GB và OOM của hệ điều hành/driver vẫn có thật. Không trừVRAM
khỏi RSS/commit để làm thấp số. UI/Debug báoGPU không đặt trần vàVRAM riêng.

CUDAassets tải từ releaseb12360 của [llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b12360),
kiểm SHA256 do release công bố. CPUbinary giữ nguyên; không cài CUDAtoolkit hoặc
thay driver toàn máy. Receipts/binaries chỉ ởlocal, không commit hàngGBassets.
`setup_hybrid_cuda.py` có hash pin và từ chối member/pathzip không an toàn.

## Số đo tốc độ thực

Run `20261113T163628Z-cpu-gpu-speed-v1-1ab607`:
[protocol trước đo](../../experiments/hybrid-cpu-gpu-speed-v1.md),
[inputs](../../runs/20261113T163628Z-cpu-gpu-speed-v1-1ab607/inputs.json),
[receipt gốc](../../runs/20261113T163628Z-cpu-gpu-speed-v1-1ab607/receipt.json),
[chấm lại từSSEthô](../../runs/20261113T163628Z-cpu-gpu-speed-v1-1ab607/speed-audit.json).

Median3lượt/bài, cùnginput/tokenizer/max128tokens, greedyseed260113,
cache_prompt=false, context16K, batch128/ubatch64. CPU4threads; GPUCUDA0.
Ba ca numericalparity đều qua ở mỗi mode với ngưỡng cũ top1same/Δprob≤0,03.

| Bài | CPU tổng | GPU tổng | Nhanh hơn | CPU→GPU token/s khi sinh |
|---|---:|---:|---:|---:|
| Sinh tiếng Việt |4,859s |1,375s |3,53× |27,48→99,72 |
| Đọc log80records |9,938s |1,578s |6,30× |26,51→98,83 |
| Mô tả ảnh fixture |1,766s |0,344s |5,13× |27,50→99,74 |

TTFTlog:5,203s→0,266s; TTFTảnh:0,797s→0,094s; TTFTVi:0,234s→0,063s.
Đây là **VLMruntime**, không phải latency toàn lượt Hermes có prompt/tool/browser.
Timing dùng Windowsmonotonic nên có độ phân giảiwallclock hữu hạn;3repeat chưa
đủ chứng minh p95/CI dài hạn. So profiledeployment: sourceb12360 giống nhau,
CPU và CUDAbinary khác build. Firstparity đã chạy trước các case; không gọi
mọi số này là fullycoldstart. Imagekernel/cachewarming có thể góp phần chênh lệch
firstrepeat. Không mở training hoặc chọn prompt sau thấy kết quả.

Hai bài text đều dừng ở128tokens, có phần câu bị cắt. Log vẫn bỏ sót một trong
ba record được hỏi; ảnh vẫn gọi rectangle thành square. GPU không sửa những lỗi
đó. OutputVi có vài khác biệt do floating-point/backend, log/ảnh trùng nhau.
Không coi speedup hoặc numericalparity3case là quality/lifelongPASS.

RAMđỉnh trong lượt đo:CPU3.575.451.648byte; GPU3.888.988.160byte. GPUcommitđỉnh
4.846.395.392byte; dedicatedVRAMđỉnh2.240.278.528byte cho các processAI/Hermes.
VRAMtoàn card có thể cao hơn vì appkhác. DedicatedVRAM không cộng vào hostRAM;
sharedGPU có cộng bảo thủ. Không tuyên bốGPU luôn dùng ít RAM hơnCPU.

Chương trình chấm lại kiểm18SSEanswers, input/sourcehash, cache_n=0 và công thức
decodeTPS=(predicted_n−1)/predicted_ms×1000; không phát hiện sai lệch. Dataset là
fixture kỹ thuật nhỏ, không đối đầu modelonline hoặc đánh giá trí nhớvideo lâu dài.

## Lỗi trước đó được giữ nguyên

Ba lượtGPU với trần4,5GB bị guardchặn do committedmemory lúc nạp, peak khoảng
4,66/4,68/4,51GB. Các lần thử hostbuffer/microbatch không được gọi thành công.
[Receipts lịch sử](../../runs/20261113T163628Z-cpu-gpu-speed-v1-1ab607/startup-failures-4_5gb-historical.json)
giữ nguyên FAIL; chỉ sau phê duyệtGPUuncapped mới có profilehoạt động ở trên.
Không sửaCPUcap hoặc nới qualitygate để nhậnPASS.

RunnerV1 gán lifecyclejob lên chính parent. Gateway được restart trongfinally
thành con, rồi bị jobđó kill khi runner thoát. Số đo18request hoàn tất vẫn giữ,
nhưng response"started" không chứng minh restorebền. Đã khôi phục từ parent
ngoài job và xác nhậnready. [Recoveryreceipt](../../runs/20261113T163628Z-cpu-gpu-speed-v1-1ab607/persistent-recovery.json),
[harnessv2](../../experiments/hybrid-cpu-gpu-speed-harness-v2.md). Không chạy lại
speedbenchmark chỉ để xóa lỗi lifecycle. Serving thật vẫn dùng job của nó.

## SearXNG không Docker/WSL

Người dùng chọnSearXNG và từ chốiDocker. Docker đã không chạy; không bậtWSL,
không cài distro, không thay tính năngWindows. Dùng official source pin
`219b63b207c98b85471de5abab773e76432f1961`, venv riêng và Waitress3.0.2,
chỉ bind127.0.0.1:18778. Nó là child do gateway quản lý, RAM được tính trongtree.
Kiểm tra gần thời điểm này: launcher8,43MB + worker77,86MB, khoảng86,3MBRSS;
không phải fixedmemoryguarantee khi mọi workload.

NativeWindows cần một patch hẹp cho `searx/valkeydb.py`: không importUnixpwd
trênWindows và log lỗiValkey không dùngUnixUID. Valkey không được bật. Không
mocksearchresults hoặc thay engineHTTP. Versionmetadata được đóng băng theo
đúngsourcecommit, không vô tình lấy gitversion củaSCAR-Xparent. Một symlink
templateApacheLinux được bỏ khi giải nén; không ảnh hưởng appWSGI đang chạy.
OriginalZIP/SHA và filetrước patch lưu tronglocal. Đây là nativeWindowsdeployment
cần giữcompatibilitypatch khi update; không tuyên bốWindows là upstreamsupported
deployment. SearXNG và licenseAGPL-3.0-or-later giữ nguyên trongsourceđã tải.

Entrypoints: `setup_native_searxng.py`, `run_native_searxng.py`, supervisor
`src/scarx/hybrid/search_service.py`. Supervisor từ chối portbị processkhác
chiếm; khi dừng, reap cả Windowsvenvlauncher lẫn basePythonchild có PIDidentity
đúng. Không có orphanserver hay systemservice mới. Configengines/secret ở
`local/searxng/settings.yml`; update helper giữ config đã có.

Hermes đã pin `web.search_backend=searxng`, `SEARXNG_URL=http://127.0.0.1:18778`.
ActualHermeswebtool trả3nguồn Chính phủ thành công1,968s;
[receipt](../../runs/20261113T160831Z-hermes-free-web-v1/searxng-native.public.json).
SearXNGdirect trước đó trả10hits0,781s. Đây là phép tìm thật; không phải chỉ
HTTP200/importPASS. **SearXNG chỉ search**: web_extract hiện vẫn là Exafree/keyless
riêng, chưa chuyển thành fetcherlocal. Không mô tả toàn bộ webpipeline là không
dùng dịch vụ bên thứ ba. Upstreamengines củaSearXNG cũng vẫn cần Internet.

## Giới hạn và công việc đang chờ

Model0.8B vẫn có thể lặp tool_search hoặc bịa dữ kiện. Cho thấy trực tiếp
web_search/web_extract trong profileSCAR-X giảm một tầngdiscovery nhưng chưa
chứng minh agent sử dụng nguồn tin cậy. Lượt thử đầu bịa100thành phố, phải giữ
FAIL dù công cụsearch hoạt động. Một lượt test reuse cùngsession nên bị loại;
lượt gọi fresh sau đó vẫn có oldhistory và chưa là clean tool-usebenchmark.
Không lấy các lượt đó làm bằng chứng sửa năng lực model.
Fullwebpagebodies/nativeownersnapshots giữlocal+SHA thay vì commit toàn bộ
HTMLhoặc oldhistory; publicrun có metadata/kết quả/lỗi và private-raw-index.

DataAntigravity đã xong96wordings bằng FlashHigh/effortHigh. CLI báo1.5.8 lúc
tạo (lầnhelp đầu là1.5.5), cả2init xác nhận đúngmodel; rawlogs/receipts giữ riêng.
Sau khi search/GPU/speed hoàn tất, đã chạy [ALIFpolicy pilot](ADAPTER_POLICY_PILOT_20261113.md)
3seed: có học nhưngNOTQUALIFIED do wrongcreate/reuseproposals; khôngđưaweights
nghiêncứu vào serving. Chưatrain/activateVLM-LoRA. Khôngcoidatađãsinhthành
kiến thứcmodel hoặcnhãnđúng. Cácmoduleevaluation khôngđượcimport vào serving;
khôngcó adapterLoRA mới trong phiênGPU này.
