# Bản đồ thí nghiệm SCAR-X

Sơ đồ thể hiện **dòng câu hỏi và bằng chứng**, không phải mọi node là một
thí nghiệm độc lập hay một lần PASS. Các trial/seed chi tiết ở
[sổ thí nghiệm](EXPERIMENT_LEDGER.md) và [catalog](experiment-catalog.json).

```mermaid
flowchart TD
  P2["Phase2: local STDP / replay<br/>forgetting22,29đ, hidden contribution chưa rõ"]
  P21["Phase2.1: trace / metaplasticity / strict binding<br/>PIVOT, không continual advantage"]
  G1["G1 trials09–44: dialogue / vision / readout / credit<br/>có prototype, nhiều STOP/P0-only"]
  R1["Round1: Grok slow trace / DeepSeek binding / Gemini DC-ALIF<br/>đọc được thông tin khác với hidden học được"]
  GH["GrokR2–R6: transfer / temporal / head / norm controls<br/>chưa GO learner"]
  DH["DeepSeekR2–R6D: readout / independent binding / scaling<br/>confirmationfail + self-audit/errata"]
  D7["DeepSeekR7 rounds1–43<br/>readout/replay lợi ích từng phần; nhiều retraction; gates chưa đạt"]
  F["Codex forensicR6/R7: audit source / labels / metrics<br/>không tin self-report mặc định"]
  R8["R8 repaired own/stopped local eligibility<br/>DEV rồi12freshseedconfirmation"]
  P1F["P1_FAIL_RESEARCH_ONLY<br/>chỉ XOR đủ toàn bộ gates"]
  ACR["ACR: static / neuron-specific / state router<br/>345jobs375records; SPECIALIZED_GAIN"]
  LIT["Literature-first e-prop / BrainTrace / SI math<br/>132referencejobs, REFERENCE_MIXED"]
  CAP["Capacity H48→H64→H128<br/>252records; directgain có ích, CAPACITY_ONLY_SIGNAL"]
  G9["GeminiR8/R9/R9b/R9c/R9d<br/>harness/gates/cache/seed issues, không formalGO"]
  G10["GeminiR10 replication<br/>authorclaim + source/summary checks; chưa full independentPASS"]
  G11["GeminiR11 shared joint checkpoint<br/>IID multi-task signal; không continual retention"]
  G12["GeminiR12 reservoir replay<br/>retentiongain hai orders; forgetting vẫn vượt gate"]
  G13["GeminiR13 cũ f4272ba<br/>invalidcontrols/symbolic bridge; archive-only"]
  LIVE["R13-family đang chạy (ảnh ban đầu R13.1)<br/>GIỮ NGUYÊN worktree / branch / cache / process"]
  NEXT["Còn mở: credit quality, readout stability,<br/>task-agnostic retention, shared skills, language/vision"]
  P2-->P21-->G1-->R1
  R1-->GH-->F
  R1-->DH-->F
  DH-->D7
  F-->R8-->P1F-->ACR
  P1F-->LIT
  ACR-->CAP
  LIT-->CAP
  R8-->G9-->G10-->G11-->G12-->G13-->LIVE
  LIT-->G10
  ACR-->G10
  D7-->NEXT
  CAP-->NEXT
  G12-->NEXT
  LIVE-."chưa có kết quả cuối trong cleanup".->NEXT
  classDef measured fill:#dcefe6,stroke:#25734c,color:#10291b
  classDef blocked fill:#f9e1d9,stroke:#a34226,color:#3d150b
  classDef active fill:#fff0bb,stroke:#9c7a00,color:#352800
  class ACR,LIT,CAP,F,R8 measured
  class P1F,G13 blocked
  class LIVE active
```

## Ba phân biệt xuyên suốt

```mermaid
flowchart LR
  RAW["Dữ liệu/trace chứa thông tin"] --> PROBE["Probe/refit đọc ra được"]
  PROBE --> DIRECT["Model direct sử dụng được"]
  DIRECT --> RETAIN["Học skill mới vẫn giữ skill cũ"]
  RETAIN --> GENERAL["Áp dụng ở tình huống chưa thấy"]
  PROBE -."không tự suy ra".-> RETAIN
  RAW -."truy xuất không phải synapse đã học".-> GENERAL
```

Các mũi tên là câu hỏi phải kiểm tiếp, không bảo đảm kéo theo nhau. Dữ liệu
replay/SSD, frozen random features, readout fitting và hidden plasticity phải
có đối chứng riêng. Số neuron, số config hoặc token không thay quality gate.

## Xem chi tiết theo cụm

- Foundation/G1: [reports01–44](EXPERIMENT_LEDGER.md#foundation-và-g1).
- Neural mechanisms: [Grok/DeepSeek/Gemini](EXPERIMENT_LEDGER.md#các-nhánh-tác-giả).
- Các phép đo đã kiểm gần nhất: [Codex R8/ACR/literature/capacity](EXPERIMENT_LEDGER.md#các-lượt-đã-kiểm-gần-nhất).
- Bản gốc và retractions: [archive index](archive/README.md).
- Active R13.1: protected; source và kết quả đang thay đổi không được copy vào
  một báo cáo “đã hoàn tất”.
