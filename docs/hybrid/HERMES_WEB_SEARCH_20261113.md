# Tìm kiếm web trong Hermes — 13/11/2026

**Lựa chọn sau đó của người dùng:** đã thay searchExa bằngSearXNGnativeWindows,
loopback18778, khôngDocker/WSL. ActualHermeswebtool qua với3nguồn Chính phủ.
Exa chỉ còn là extractbackendfree vìSearXNGsearch-only. [Trạng thái mới nhất](GPU_AND_NATIVE_SEARCH_20261113.md).
Các phép Exa/DDGS bên dưới giữ kết quả thời điểm trước khi đổi lựa chọn.

`Searching tools` trong ảnh là **tool discovery**: tìm tên/schema công cụ
trong catalog Hermes. `tools/tool_search_catalog.py` thực hiện BM25 trên
catalog; nó không tìm bài viết Internet. Sau đó model phải gọi `web_search`
hoặc công cụ đã tìm được. `waiting on provider` là trạng thái lượt model;
không đủ để kết luận websearch đang lỗi. Model0.8B vẫn có thể gọi sai hoặc lặp.

## Đã cấu hình tuyến hoạt động

Hermes có sẵn provider plugins; không thêm proxy/server hoặc core tool mới.
Trước sửa resolve search=perplexity/extract=keenable. Nay dùng cấu hình chuẩn:

```yaml
web:
  search_backend: exa
  extract_backend: exa
  provider_tier:
    exa: free
```

Đây là **dịch vụ bên thứ ba ở free/keyless tier**, không phải search engine
offline hoặc tự host. Không tạo tài khoản/APIkey mới, không bật paid tier.
Cấu hình ép free kể cả máy có credential khác. Free service có giới hạn nguồn
cung cấp, không cam kết unthrottled. Query và URL được gửi tới dịch vụ như công
cụ tìm web; phép thử chỉ dùng câu hỏi công khai, không gửi video/hội thoại riêng.

`scripts/configure_hermes_free_web.py` dùng chính `hermes config set`, sao lưu
nguyên config trong `local/hybrid-v1/hermes-free-web-backups`, so cấu trúc YAML
để chỉ3path trên được đổi. Không sửa model/providerLLM, secrets hay code Hermes.
Configloader của Hermes tự invalidation theo file signature; không cần dừng
gateway đang có hội thoại chỉ để đổi các web backend này. Chat mới được dùng
cho bài kiểm tra agent, giữ snapshot công cụ của chat cũ.

## Thử nghiệm thực và giới hạn

[Receipt search/extract](../../runs/20261113T160831Z-hermes-free-web-v1/receipt.public.json)
gọi actual `tools.web_tools.web_search_tool` và `web_extract_tool` bằng đúng PM
environment của Hermes, không mock. Search3kết quả Chính phủ thành công1,6335s;
đọc trang đầu thành công0,3686s. Wrapper extract trả `{results:[...]}`, không
có fieldsuccess; kiểm `error:null` và nội dung thực thay vì chấm theo missingflag.

Trang [Chi tiết34đơn vị hành chính cấp tỉnh](https://xaydungchinhsach.chinhphu.vn/chi-tiet-34-don-vi-hanh-chinh-cap-tinh-tu-12-6-2025-119250612141845533.htm)
hiển thị cập nhật13/11/2026 và ghi28tỉnh/6thành phố. Không dùng lời model cũ
“3thành phố” làm nhãn. Đây là bằng chứng công khai có ngày, không kết luận model
đã học vĩnh viễn kiến thức đó. Tool success không bảo đảm model0.8B sử dụng đúng.

## Các repo mã nguồn mở

- [DDGS](https://github.com/deedy5/ddgs): thư viện metasearch, Hermes đã có provider
  search-only. Đã cài đúng pin9.16.0 qua `pm.sync_venv(['ddgs'], explicit=True)`.
  Thử auto gặp kết nối nguồnYahoo lỗi10051; thử DuckDuckGo/Bing không có kết quả.
  Không chọn làm mặc định chỉ vì import được. Không tắt TLS hoặc vượt CAPTCHA.
- [SearXNG](https://github.com/searxng/searxng): search engine tổng hợp có thể tự
  host, Hermes hỗ trợ `searxng` sẵn. Vẫn cần Internet/upstreamindex; tự host không
  có nghĩa sở hữu bản sao toàn web. Chưa cài thêm server/Docker trong trần4,5GB.
- [Providerweb của Hermes](https://github.com/NousResearch/hermes-agent/tree/main/plugins/web)
  là lớp tích hợp hiện dùng; [tài liệu chính thức](https://hermes-agent.nousresearch.com/docs/user-guide/features/web-search)
  phân biệt web_search, web_extract và free/paid/backend riêng.

Hướng gọn hiện tại: dùng provider đã thử thành công, giữ DDGS là tùy chọn đã
cài, để SearXNG thành phương án tự host khi thật sự cần tự vận hành. Không có
mã nguồn mở nào bảo đảm thông tin web mới khi không có kết nối hoặc index local.

Task Antigravity/adapter vẫn giữ riêng:2lượt FlashHigh/effortHigh đã tạo96wordings
song ngữ ở `data/candidates/20261113T155741Z-adapter-data-v1-614f34`. Đây là
candidate cần review, chưa phải nhãn LoRA hoặc adapter đã được train/kích hoạt.
