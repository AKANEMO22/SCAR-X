# Hermes — compact nội bộ lộ trong chat13/11

Lỗi thực tế: hội thoại“Review SCAR-X interface”hiện nguyêncarrier gồm
`PRIOR CONTEXT`,delimiter và`CONTEXT COMPACTION`như assistantreply.
Đây là lỗi chọn projection trongDesktop, không phảiSCAR-X tự gửi lời tóm tắt
cho người dùng. Không dùng regex/CSS để che nội dung có chữ“compact”bất kỳ.

## Nguồn gốc và sửa

OwnerHermes giữ hai view: raw`messages`cho continuation/rewind và
`display_messages`cho transcript. NativeRPCđúngprojection đã trả17rawrows,
12displayrows:1rawcarrier,0displaycarrier, không mất bằng chứng công cụ.
Desktop có adapter`CanonicalDesktopProtocol`riêng, chưa đồng bộ sharednodeclient:
không gửi`display_projection:true`lúcresume/activate, đồng thời bỏ qua
`display_messages`khi dựngsnapshot. Vì vậy backend đúng nhưngUIvẫn lấyraw.

Sourcefix ởadapter:
- Resume dùng để vẽchat và warmactivate yêu cầu displayprojection. Lazywatch
  attach cũng dùngprojection; metadata-only attach`omit_messages:true`không
  tải thêm transcript. Profile và revision/generationcontrols giữ nguyên.
- Snapshot`messages`choUIlấyprojectiondoownerphát; raw giữ riêng trong
  `stored_messages`. Projectionrỗng vẫn có thẩm quyền, không fallbackraw.
- Olderowner không có trườngprojection giữđườngtươngthích cũ. Không sửa
  prompt/modelhistory/checkpoint hoặc xóa/cải biên databasechat đã lưu.

## Kiểm chứng và triển khai

3regressiontests mới đỏ trênbase, xanh saufix: requestcontract, actualWebSocket
client nhận cảhaihistoryviews+toChatMessages, emptyprojection/olderowner.
139checks ở9files qua, gồmcanonicalwire/watchwindow/busycontrol/clarify/
hydration/toolgroup. ScopedESLint vàPrettiertest qua; rendererproductionbuild
qua với toolchainHermesNode26.12.1/pnpm12.15.0. Sourceadaptercũ giữformat sẵn,
không formatterrewrite toànfile hoặc thay dependency.
Fullrenderer`tsc -p . --noEmit`qua sau khi bổ sungtypeschoRPCresult trong test;
3regressionchecks được chạy lại sau chỉnhtypes và tiếp tụcqua. Giữfailed
typecheckbanđầu trongfinalvalidationreceipt, không gọi lượt đó làPASS.

Patchreversible ở`integrations/hermes/patches/compaction-display-projection-v1.patch`.
Dùng`git apply --unidiff-zero --ignore-space-change`và kiểmsourcehash trướcáp;
reverse-check trênsourceđãsửa qua. Existingtoolcollapsefix vẫn trongbuild.
Installerchung chỉ đổi rendererassets/index/manifest, giữhashElectronmain,
preload/buildmetadata. Không cài modelmới hoặc restartbackend.

Backupsource:`local/hermes-compaction-display-current.json`; backupdeployment:
`local/hermes-compaction-display-renderer-current.json`. Rollback restore
index/manifest trongbackup, assetcũ còn nguyên. SourceHermesbase3b42157858;
các sửaPython/tíchhợpSCAR-X vàpeerresearch không bị động tới.

NativeGUIverification sauCtrl+R và mở lại chính hội thoại bị lỗi: không còn
3marker/preamble nội bộ, tinuser vànhóm“Used7tools”vẫn còn, toolgroupđóngkín.
Không tạo lượt hỏi/modelcall mới hoặc thay nội dungdatabasechat để đạt kết quả.
Receipt cuối được ghi trongrunriêng;
không dùng screenshot có dữliệu riêng làm testfixture hoặc đẩylênGitHub.
Read-only helper`verify_hermes_compaction_display.py`chỉ incounts/digest,
khôngincredential hoặc nguyênhistory. Không có modelbenchmark/training mới.

Kiểmtra trạngthái trong đợt sửa phát hiện workerHybrid đã bịguard4GBdừng
trong một hội thoại đangretry. Đã khôi phục riêng Hybridruntime bằnghelper
graceful/identitychecked, không thay sốRAM/precision hoặc restartHermesowner;
trạngthái cuối ghi trong`runtime_recovery.json`. Đây là lỗi tài nguyên riêng,
không dùng việcUIđãẩncompact để gọi modelvận hành ổn định/khôngbịngắt.
