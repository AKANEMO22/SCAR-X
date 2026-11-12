# Dọn và hợp nhất workspace —13/11/2026

**Đã hoàn tất.** 30remoteheads và33localbranches cũ đã dọn;36oldworktrees
được gỡ sau archive, cộng một worktree tạm hợp nhất. Còn8registeredworktrees:
main, activeR13 và6source dependencies bảo vệ. GitHub chỉ cònmain và cácR13heads
được bảo vệ; sốhead có thể tăng khi thí nghiệm đangchạy tạo/push thêm nhánh.
75archive tags đã verify,11historicalPRs đềuclosed, discussion giữ nguyên.
[Biên bản máy đọc được](cleanup-result.json), [PR readback](pr-final-state.json),
[nhật ký worktrees](worktree-removal.jsonl), [nhật ký refs](ref-cleanup.jsonl).
LocalbranchR13.2 mới và các R13aliases không bị xóa; branch mới ngoài inventory
không bị sweep theo danh sách cũ.

Người dùng cho phép kiểm các branch, đưa kết quả hoàn tất vềmain, merge khi
phù hợp và dọn GitHub/local. Sau đó người dùng yêu cầu **không đụng thí nghiệm
đang chạy trong scar-x-gemini-r7**. Quy tắc bảo vệ áp dụng toàn bộ R13-family,
kể cả nhánhR13.1/R13.2 mới tạo sau inventory, không chỉ tipf4272ba ban đầu.

## Những phần được hợp nhất

- Các nhánh Codexresearch đã kiểm: capacity/ACR/R8matched, literature, forensic,
  codecP0/neuralhandoff và correctedGrok integration. Có mergecommits thật,
  giải conflict tài liệu và strictsnapshotformat bằng union schema.
- Không đổi production source `neural/`, `app/`, `learning/`, `integrations/`,
  run_app/capabilities/project hoặc existingservingconfigs. Chỉ thêm research
  registrations/configs và isolatedevaluation/tools/tests. Không thay weights
  hoặc SQLiteprivate, không train mới, không deploylearner chưađạtgates.
- 227historicalreports/protocol/metricreceipts từ27peerbranches copy nguyênbyte,
  sourceSHA/SHA256/versionURL ở [source-index](../../research/archive/source-index.json).
  Peer algorithms không tự merge vào serving. Raw/source cũ giữ bằng archive tags.
- [Catalog120entries](../../research/experiment-catalog.json),
  [sơ đồ](../../research/EXPERIMENT_MAP.md), [sổ kết quả](../../research/EXPERIMENT_LEDGER.md).
  Đây là navigationcounts, không120independentexperiments hoặc qualityPASS.

## Bảo toàn trước khi dọn

FullGitbundle được tạo và verify trước mutation:
`local/maintenance/20261113-cleanup/scar-x-before-cleanup.bundle`.
Archive tags `archive/2026-11-13/...` được push và kiểm SHA trênorigin trước khi
gỡ worktree. [branch-archive.json](branch-archive.json) ánh xạ từngref/worktree
đến SHA/tag/phục hồi; không đổi tag đã xuất bản để “chỉnh lại” lịch sử.

36completedworktrees đã được gỡ; các bản chưacommit có snapshotcommit/tag và
ZIP local gồm staged/unstagedpatch, file gốc vàhash. Ignorednoncache research
state được giữ local, không tự publicprivate sessions. Cache/build có thể tái
tạo không được coi là evidence. ArchiveZIP được kiểm lại nội dung/hash trước
khi xóa bản sao worktree. Gitbundle/tags không thay thế ZIP đối với ignoredstate.

Các bản sao còn giữ vì có thể là dependency của activeexperiment:
`review-gemini-r8-607ffca`, `review-gemini-r8-fe472b3`, `scarx-acr-20261112`,
`scarx-capacity-h64-h128-20261112`, `scarx-literature-first-20261112`,
`scarx-r8-matched-controls-20261112`. Chưa biết dependency của các file R13.x
đang viết thì giữ bảo thủ, không tự dọn. Folder `scar-x-claude-research` không
phải registeredworktree và được giữ nguyên.

## Bảo vệ thí nghiệm đang chạy

- Không xóa/di chuyển/checkout/reset/sửa file của
  `C:\Users\hachimi\Downloads\model agent\scar-x-gemini-r7`.
- Không dừng process, đổi model/provider, xóa HuggingFace/download/cache hoặc
  scan/copy uncommittedbenchmarkfiles vào completedcatalog.
- Không xóa remote/local refs có prefix `research/gemini/r13` hoặc currentref
  đang gắn với protectedworktree. Những ref mới sau inventory không nằm trong
  deletionplan. LegacyR13f4272ba report/source được đọc bằng Gitobject thôi.
- Không chạm `local/generation1`, Hermes, modelweights hoặc SQLiteproduction.

## Phục hồi

```powershell
git fetch origin --tags
git worktree add --detach local/reproduce-grok-r6 archive/2026-11-13/branches/research/grok/r6-causal-readout-repair
git worktree add --detach local/recover-old-worktree <tag trong branch-archive.json>
```

Tạo checkout mới, không thay activeR13.x. Cần đúng file chưacommit/rawbytes thì
dùng ZIP trong `local/maintenance/20261113-cleanup/worktree-backups/` và so
FILE_SHA256.json. Không giải nén chồng lên livefolder. Fullbundle có thể dùng
`git clone <bundle> <folder-mới>` khi không có mạng; privateignoredstate vẫn cầnZIP.

Historicalruns/checkpoints/protocolhash giữ nguyên. Mainresearchcode có adapter
tìm pinnedinputs từ main hoặc worktree; `scripts/research/prepare_pinned_peers.py`
chỉ tạo đúng2detachedsourceinputs nếu chưa có, từchối overwritekhácSHA/dirty.
Các runner cũ có literalpath/sourcehash nên muốn rerun cần archivedSHA/source-byte
bundle và output mới; không nới checksum, không ghi đè runID/reservation.

## Đóng nhánh và PR

Completedremoteheads chỉ được xóa sau khi kết quả/docs đã có trênmain và
immutabletag đã verify. Deletion dùng expectedSHA lease; nếu tác giả push tiếp
thì không xóa theo snapshot cũ. Branchđang chạy được loại theo cảprefix/path.
PR lịch sử có thể đóng với trạng thái archived/superseded, không đổi thành
“merged learner/quality approved”; discussion và evidence vẫn trênGitHub.
Không gửi comment hoặc tin nhắn thay người dùng.

Kết nốiGitHub cho phép đọc nhưng PATCHPR trả403`Resource not accessible by
integration`. Không tăng permission hoặc lấy credential khác. Xóaheadbranches
đã archive bằng Git lease khiến GitHub tựđóng cácPR cũ; đã readback cả11PR
đềuclosed/merged=false. Không đánh dấu chúng là merge/prodapproval hoặc sửa
body/title/comments.

## Kiểm tra hợp nhất

614tests đã chạy đủ:6APImodule trong một process riêng và608tests còn lại ở
process khác,0fail/0skip. Test/lô mới gồm catalog/hash/protectedpath và union
checkpointkinds. Buildnative/shared-coreparity đã chạy. [Log/XML/receipt](validation/README.md)
được giữ, không dùng test kỹ thuật này làm kết quả ML mới.

Một lượt monolithic trước đó có613PASS/1HTTP507 trongAPIteaching. Endpoint
map507 cho MemoryError hoặcTimeoutError; body cụ thể của failure cũ không
được capture, nên không khẳng định độc nhất nguyên nhânRAM. Cùng test qua khi
chạy riêng trênoriginalmain và toànAPImodule qua trên cây hợp nhất. Isolation
tránh phụ thuộc tài nguyên tích lũy của numerical/researchtests; không nới
guard, không sửa serving source hoặc bỏ testcase để đạt xanh.

Kết quả thực thi cuối, branch/worktreecounts, testresults và consolidationSHA được
ghi trong `cleanup-result.json`. Finalmainref có thể thêm commit chỉ ghi metadata
cleanup sau consolidationSHA đã kiểm. Không coi xóa branch là xóa
evidence: report/tag/SHA/localrecovery mới là đường truy vấn lâu dài.
