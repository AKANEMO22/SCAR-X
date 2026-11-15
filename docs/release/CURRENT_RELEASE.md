# SCAR-X Hybrid — RC14 đang cài trong Hermes

**Dùng trong chat Hermes bình thường**, chọn **Scarx Hybrid**, model ID
`scarx-hybrid`. Một gateway `127.0.0.1:8775`, một Qwen3.5-0.8B nguyên BF16/F32,
Whisper-small FP32 và ALIF H128 học cục bộ trên GPU. Không LoRA, quantization,
pruning hoặc thay quy tắc học.

Gói hiện tại: `artifacts/releases/SCAR-X-Hybrid-1.0.0-rc.14+44c928df0eea.zip`,
865.770 byte, 58 payload files. SHA256:
`c79b4df28d38b621ca3d6314a1b2841fda6e22ccfbe5a3a3feda0c3f555c2238`.
Git SHA nguồn: `44c928df0eead1d3829043b8334af4f07e1b1ded`.
Sidecar `.installed-verification.json` xác nhận payload khớp code đang cài,
asset/Hermes/dependency pins khớp. Không main/R13 change, merge hoặc push.

## Cách dùng và phạm vi gói

- Chat và ảnh/video dùng các thao tác sẵn có của Hermes. Tab Debug chỉ quan sát.
- Video đính kèm được phân tích local trước khi gửi câu hỏi, chờ tối đa 35 giây
  cho bằng chứng đầu tiên; lỗi/timeout giữ bản nháp và cùng retry identity.
- Video ở draft mới gắn theo **session + mã lượt gửi thực**, không theo câu hỏi
  giống nhau. Metadata không chèn vào lời người dùng; restart/retry không đổi chủ.
- Xem màn hình cần yêu cầu rõ và chọn cửa sổ. Sau stop/restart có thể hỏi rõ
  về “cửa sổ vừa xem”, dùng nguồn còn giữ mà không capture lại. Không mặc định
  theo dõi toàn màn hình hoặc microphone.
- Trace: `http://127.0.0.1:8775/?tab=trace`; các bản sửa/check thật có event riêng.
- Gói là overlay cho **deployment Windows/Hermes đã cài bản metadata admission
  tương ứng trên máy này**, không chứa model/venv/dữ liệu riêng. Installer kiểm
  fingerprints, từ chối app khác; chưa phải bộ cài cho máy mới. Source patch
  `HERMES_MEDIA_ADMISSION.patch` đi kèm phục vụ bảo trì, không tự sửa app lạ.

## Đã sửa và kiểm

| Phần | Bằng chứng, kết luận đúng phạm vi |
|---|---|
| Video draft lấy nhầm giữa chat | Bỏ unscoped text matching; callback sau resolve session, canonical executor truyền request ID qua ContextVar/header; pending/binding/focus commit atomic. Có test hai chat cùng text, queued inputs, retry/restart và lỗi transaction. |
| Snapshot ghi đè | JPEG immutable theo hash; hai nguồn khác có path khác, publication không overwrite trên Windows. Có 32 request concurrency, integrity và archive quota tests. |
| UI video mới | Upload qua Files ở chat mới, frame local có trước bind; DB request ID khớp attachment, user text giữ nguyên, 0 tool calls sau readiness fix. RC6 nhận đúng màu nhưng trả tiếng Anh và thiếu timestamp; receipt lỗi đó được giữ. |
| Trả lời video sau sửa presentation | Canonical Hermes thật trên clip đã biết trả “Hình đầu tiên là một hình vuông màu đỏ.”, 0 tool events. Runtime hiển thị nguồn `[E1 · 00:05]`, log riêng với native answer; chưa phải model học dẫn nguồn hoặc xác minh đáp án. |
| Marker ở lời chào | Repro đúng ảnh người dùng, sửa ví dụ steering không dùng ở lượt xã giao tại host; không hardcode/lọc output. Canonical Hermes trả “chào bạn” bình thường; chất lượng xã giao khác còn hạn chế. |
| Screen history | Capture thật cửa sổ FFplay thử: 13 frame/2 VLM descriptions, dừng sạch; hỏi sau stop và restart vẫn chọn đúng nguồn/màu đỏ. Câu không liên quan không nạp ảnh cũ, hỏi lịch sử không cấp quyền capture mới. Audio/mic tắt trong thử này. |
| Lỗi âm thanh màn hình | FFmpeg thật từ chối WAV hỏng: code cũ dừng toàn ingestion, code sửa vẫn phân tích hai frame bằng VLM GPU thật (`red`, `blue`). Failed/skipped intervals rõ ràng; ba lỗi audio liên tiếp chỉ pause ASR, không giả transcript; resource breach vẫn fatal. Đây là owned journal, chưa live loopback acceptance. |
| Nâng cấp/bảo trì | Pin payload RC7 thay vì hash candidate đang sửa; terminal source gate nhận đúng predecessor/next bytes và giữ custody không-PASS. Đã chạy thật installer cập nhật ba file, cùng clone CAS/rollback giữ dữ liệu mới; không no-op install claim. |
| Feedback/retry RC9 | Checkpoint, nhãn, journal và receipt cùng transaction; actual Handler + GPU ALIF tái hiện lỗi trước sửa và kiểm socket mất phản hồi sau commit, concurrent retry, restart/rollback, SQLite ABORT và batch rollback. Cùng mã khác nội dung bị từ chối. Không áp nhãn thử vào model live. |
| Web routing RC10 | Câu JoJo mới nhất và “tìm kiếm đi” đã gọi web_search thật đúng cùng chủ đề, mỗi lượt5 kết quả; không tìm file hay gọi extractor. Native answer vẫn lấy ngày ra mắt thay số tập hiện tại, không chấm answer-quality PASS. |
| Source/review RC11→RC12 | RC11 nhận5 nguồn nhưng parser bỏ sót frame cảnh báo của Hermes, nên review không có nguồn; lỗi cũ giữ nguyên. RC12 giải mã bản sao để lưu, không bỏ cảnh báo khỏi model input. Một follow-up thật lưu đúng5 snippet `observed`, gắn task gốc, `partial=false`; tự review sau khi rảnh với đủ5 nguồn. Native vẫn nhầm tổng dự kiến11 tập với tập mới nhất; critique không xác minh được, không tự học từ nhãn đó. |
| Verifier scope RC13 | Câu hỏi màu/code/trích dẫn chứa12+3 không còn tự được cấp nhãn số học. Cùng checker toàn-câu cho mask, action, review và feedback;0 neural updates trên3 controller fixtures trước đây cấp nhãn sai. Đây là sửa quyền cấp nhãn, không neural intent learning. |
| Verifier scope RC14 | Sau review Antigravity high được kiểm độc lập, bare3-1/03-01 không tự thành phép trừ vì có thể là tỷ số/ngày.5 actual isolated controller reviews dùng declared reflection fixture giữ unverified,0 forward/learn; cold restore exact. Proof IDv3, chưa general intent understanding. |
| Tool/workspace/compact/search | Observer đọc đúng envelope; readonly probe thật 0→1 sau 3 lỗi. Guard replay chặn lần thứ tư giống hệt. Project trung lập/7K-char turns và 3 auto compact +1 manual qua; owner display không lộ handoff. Dispatcher SearXNG native trả nguồn thật. |
| GPU/state | Cả VLM/vision, ASR, ALIF trên GPU; startup/native parity đã kiểm. Full saved arrays/RNG/replay qua activation cuối giữ nguyên, live updates vẫn 5; không nhập lab weights. |

**332 full release Python checks qua ở RC11** được giữ làm baseline. RC14 chạy
**114 targeted arithmetic/review/cognition/feedback/web/package checks**, đã gồm28 package cases, không
tuyên bố rerun toàn332. Không cộng các receipt chồng nhau thành mẫu độc lập.
Có 9 actual HTTP/CUDA feedback cases,
CPU/native/CUDA parity, audio fault, upgrade CAS/rollback và persistence.
**29 Node checks +3 UI HTTP feedback
scenarios qua ở RC7**; 12 Hermes Desktop tests/typecheck qua ở lần sửa admission
RC6, không chạy lại vì RC14 không đổi renderer/plugin/composer. Các receipt này
được ghi đúng phiên bản;
không dùng test transport hoặc test xanh để chấm năng lực đa năng.
Renderer build/typecheck thành công, Electron main/preload giữ nguyên.

Khi trả lời dựa vào media local, native call không được cấp tool schemas để
tránh biến observations riêng thành automatic web/remote query. Đây là policy
runtime, chưa là SNN tự học lúc nào nên tìm kiếm. Tác vụ thông thường vẫn có tool.
Context vẫn **16.384 token**; trần text 65.536 ký tự và FTS chunk 6.000 ký tự
không tạo context 65K token. Cache/quota và SSD memory không bằng trí nhớ synapse.

## Bằng chứng và bảo trì

- [Sửa ownership/snapshot và UI](MEDIA_ADMISSION_REPAIRS_20261114.md),
  [audit dữ liệu](evidence/MEDIA_ADMISSION_REPAIRS_20261114.json),
  [Desktop contracts](evidence/HERMES_MEDIA_ADMISSION_DESKTOP_TESTS_20261114.json).
- [Antigravity review/triage](ANTIGRAVITY_FULL_QA_20261114.md): review đầu 6 finding
  đã đối chiếu, hai lỗi kỹ thuật đã sửa. CLI không chạy xong tests vì sandbox;
  Codex thực thi suite. Follow-up review source timeout, không tính approval.
- [Trả lời media và ranh giới footer](MEDIA_REPLY_PRESENTATION_20261114.md),
  [audit actual Hermes](evidence/MEDIA_REPLY_PRESENTATION_20261114.json),
  [RC10 code/runtime pins](evidence/RC10_INSTALLED_ARTIFACT_20261115.json),
  [RC10 activation giữ state](evidence/RC10_ACTIVATION_20261115.json),
  [RC10 regression](evidence/RC10_RELEASE_CHECKS_20261115.json),
  [RC7 functional receipts](evidence/RC7_RELEASE_CHECKS_20261114.json).
- [Feedback atomicity](FEEDBACK_TRANSACTION_20261115.md),
  [exact source qualification](evidence/FEEDBACK_ATOMIC_FINAL_SOURCE_20261115.json).
  RC7 UI fixtures không phủ backend commit gap; RC9 thêm actual Handler/socket
  coverage. Review CLI cho bản sửa này timeout, không approval từ Antigravity.
- [Public web routing](WEB_SEARCH_ROUTING_20261115.md),
  [actual Hermes calls/results](evidence/ACTUAL_HERMES_WEB_SEARCH_20261115.json).
  Runtime chọn tool đầu tiên và giữ query của người dùng; không nói SNN đã học
  lựa chọn web. SearXNG vẫn search-only; không ngầm dùng Exa extraction của cấu hình.
- [Web source/task/review và lỗi wrapper thật](WEB_EVIDENCE_REVIEW_20261115.md),
  [qualification RC12](evidence/WEB_WRAPPER_TIME_SOURCE_20261115.json),
  [actual RC12 calls/review](evidence/RC12_WEB_FRAME_LIVE_20261115.json),
  [RC12 installed pins](evidence/RC12_INSTALLED_ARTIFACT_20261115.json).
  Snippet và self-critique vẫn chưa được xác minh, không tự trở thành nhãn học.
- [Sửa phạm vi verifier](ARITHMETIC_VERIFIER_SCOPE_20261115.md),
  [actual controller fixtures](evidence/RC13_ARITHMETIC_CONTROLLER_20261115.json),
  [RC13 installed pins](evidence/RC13_INSTALLED_ARTIFACT_20261115.json),
  [RC13 activation](evidence/RC13_ACTIVATION_20261115.json),
  [RC13 checks](evidence/RC13_RELEASE_CHECKS_20261115.json).
  Grammar hẹp có thể từ chối câu toán hợp lệ ngoài phạm vi; vẫn chat/review
  nhưng không tự xác minh hoặc coi lời sửa người dùng sai bằng checker này.
- [RC14 installed pins](evidence/RC14_INSTALLED_ARTIFACT_20261115.json),
  [RC14 activation](evidence/RC14_ACTIVATION_20261115.json),
  [RC14 five controller cases](evidence/RC14_ARITHMETIC_CONTROLLER_20261115.json),
  [RC14 checks](evidence/RC14_RELEASE_CHECKS_20261115.json),
  [Antigravity review/disposition](evidence/ARITHMETIC_ANTIGRAVITY_REVIEW_20261115.json).
  RC13 receipts ở trên giữ lịch sử; pointer đầu trang là gói hiện đang cài.
- [Audio resilience](SCREEN_AUDIO_RESILIENCE_20261115.md),
  [upgrade source custody](RC_UPGRADE_PATH_20261115.md),
  [goal completion audit](GOAL_COMPLETION_AUDIT_20261115.md) tách kỹ thuật đã
  kiểm với SNN advantage/retention/causal recovery chưa đạt.
- [Screen/greeting](SCREEN_HISTORY_AND_GREETING_20261114.md),
  [chat dài/compact](LONG_INPUT_AND_COMPACTION_20261114.md),
  [feedback HTTP RC6](evidence/RC6_FEEDBACK_HTTP_20261114.json) giữ lịch sử.
- `scripts/verify_installed_hybrid_rc.py` kiểm ZIP với deployment read-only.
  `scripts/audit_media_admission_repairs.py` đọc actual UI/admissions/source pins,
  không gọi model thêm. Backup source/renderer/checkpoint và log ở vùng private.
- Các artifact/receipt trước được giữ nguyên. Code rollback không ghi đè DB,
  xóa journal hoặc hoàn tác trải nghiệm mới để làm phép kiểm trông thành công.
  RC8 `+fa8aada0fac1` là candidate trước sửa upgrade path, không phải gói đang
  cài; giữ nguyên thay vì ghi đè hoặc gọi là nghiệm thu thành công.

## Giới hạn còn lại

- [P1A research f39bc6](../hybrid/NEURAL_OUTCOME_BRIDGE_P1A_RESULT_20261115.md)
  có frozen neural proposals/tools thật nhưng STOP do harness guard; không
  nhập wrapper/adapter/snapshots vào RC14.6 pure guard-fix tests không chứng
  nhận model skill. Confirmation v2 chưa được duyệt/chạy; không reset budget.

- Model 0.8B còn yếu, có thể hiểu sai, sai ngôn ngữ hoặc không dẫn nguồn như yêu
  cầu. Clip/project đã biết không chứng minh chất lượng video chưa thấy, năng
  lực đa năng hoặc ngang model online. Hai lỗi QA được sửa không giải quyết
  toàn bộ chất lượng model.
- SNN chưa chứng minh vượt cùng LLM + RAG, retention chưa đạt ngưỡng đã đăng ký,
  chưa tự học chắc chắn nguyên nhân tool loop hay kỹ năng sửa lỗi lâu dài.
  Không gọi guard/retrieval/host routing là hidden-neuron improvement.
- [History P0](../hybrid/HISTORY_P0_RESULT_20261115.md) đã dừng ở parity gate:
  177/600 calls cả diagnostic, không DEV/live promotion. Feedback RC9 không
  đổi rule P0, tăng budget hoặc nhập checkpoint nghiên cứu vào serving.
- Request feedback sau RC9 có ID mới nhận bảo đảm atomic retry. Không gửi ID
  giữ legacy semantics; không tự suy đoán receipt cho lời phản hồi mất trước sửa.
- Chưa nghiệm thu live speaker-loopback, mọi cách hỏi màn hình, mọi workflow
  Desktop hoặc mọi tổ hợp chạy đồng thời. Không tuyên bố full UI/production PASS.
- Lượt small8h `32cd13` đã hủy theo yêu cầu: giữ 38 chu kỳ/7 checkpoint,
  `USER_CANCELLED_NOT_EIGHT_HOUR_PASS`, không worker sót và không mở lại.
- LoRA tự động, learner mới, dataset/benchmark mới ở backlog theo ưu tiên sản phẩm.

Đây là **RC dùng thử có giới hạn**, không phải production hoặc lifelong PASS.
