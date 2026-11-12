# ACR đối chiếu literature — review adoption, 12/11/2026

**Chỉ giữ ở nhánh nghiên cứu. Chưa quality GO hoặc promotion main/Hermes.**
Reviewer `/root/reference_audit_resume` chỉ chấm lại raw và kiểm source/hash/init; không tạo learning update, không đổi learner/protocol hoặc chọn profile theo task. Ponytail không có trong inventory. Norm/stress final audit của toàn ACR chưa hoàn tất tại lúc viết; nhận xét cuối về ACR phải chờ các control đã đăng ký.

## Nguồn và phép so sánh hợp lệ

- ACR: `20261112T121634Z-acr-dev-v1-resume-59ced4`, source `c54ee689126a79e695df98eb81eb60053acc2621`, một global nominee **groups_up** đã khóa từ đủ225A/B/C cells.
- Literature: `20261112T080533Z-literature-reference-dev-v1-839e07`, source `a3b79ea66cb6676f8b03a22b0cb907d158e5b2a2`, đủ132cells; global reference nominee **eprop_matched**, integrityPASS/REFERENCE_MIXED.
- Cùng seeds140011/140017/140023, bốn isolatedtasks,1024updates, TRAIN1024/refit256/ID512/OOD512, calibration32TRAIN, H48D8K8 hard-ALIF, pre-update readout feedback và optimizer source.
- Hash five shared source bodies(core,matched,stopped,surrogate,generator/study) exact giữa hai worktree; mọi hash manifest của cả hai còn đúng. So actualsource, không chỉ tin branch name.
- Kiểm96datasetarray pairs R/L exact; kiểm60cross-method initial pairs, gồm toàn bộcore arrays, normalizer, momentum vàenergy. Không coi thay đổi filter eligibility đã đăng ký là khác init weights.
-72selectedcells(6methods×4tasks×3seed) được chấm lại147.456ID/OOD direct/probe entries. Checkpoint/initial hashes khớp receipts. Không trộn equalCPU vào so sánh chất lượng này.
- Endpointalpha0/1 đã được đối chiếu exact với own/stopped thực; static10012cells còn khớp12.288per-update losses, core,optimizer vàRNG sau1024updates. Các observer không dùng LLM trả lời thay.

Raw result hash index kiểm chéo72files (sorted relativepath→SHA256 JSON, separatorscomma/colon) có digest:
`aa9d77fd2fdac05d5b6acb932e2891e78a16d3d96e75301098aa6abd1878888b`.
FullSHA mỗi result file được ghi cùng bảng per-seed cuối tài liệu; raw source path xác định bởi run/method/task/seed ở trên.

## Direct output: không chọn endpoint/alpha riêng cho từng task

Đơn vị mọi score là%; mỗi ô là mean3seed.

| Method | Binding direct | XOR direct | Distractor direct | Order direct |
|---|---:|---:|---:|---:|
| groups_up | 85.03 | 98.11 | 59.24 | 100.00 |
| own | 85.42 | 91.86 | 56.97 | 100.00 |
| stop | 82.62 | 99.35 | 55.66 | 99.61 |
| eprop_matched | 78.58 | 98.24 | 60.03 | 100.00 |
| eprop_aligned | 69.92 | 99.54 | 55.01 | 99.93 |
| braintrace | 68.62 | 99.22 | 53.32 | 99.93 |

groups_up giữ Binding gần own(−0,39đ), cải thiện XOR+6,25đ và Distractor+2,28đ so own. Nhưng so endpoint tốt hơn theo từng metric, XOR giảm1,24đ và Distractor vẫn59,24%<65%. Đây là một profile cố định, không có bằng chứng router học được policy chuyên biệt. Không lấy own choBinding, stop choXOR và epropchoDistractor rồi gọi là một learner thống nhất.

| Method | Binding probe | Binding OOD direct/probe | XOR probe | XOR OOD direct/probe | Distractor probe | Distractor OOD direct/probe |
|---|---:|---:|---:|---:|---:|---:|
| groups_up | 88.54 | 85.81/89.00 | 98.44 | 85.22/90.89 | 60.48 | 56.05/57.75 |
| own | 86.98 | 83.59/85.03 | 96.68 | 85.55/89.19 | 56.97 | 54.10/56.18 |
| stop | 85.29 | 82.88/85.03 | 99.54 | 86.65/90.04 | 63.41 | 56.32/62.76 |
| eprop_matched | 82.75 | 79.30/82.62 | 99.61 | 87.17/92.19 | 61.98 | 59.51/58.66 |
| eprop_aligned | 70.77 | 68.42/66.02 | 99.80 | 91.21/90.62 | 58.07 | 53.19/55.86 |
| braintrace | 74.22 | 67.97/71.61 | 98.96 | 89.13/88.48 | 55.79 | 52.99/54.69 |

groups_up cao hơn eprop_matched ởBindingdirect6,45đ, nhưng thấp hơn ởDistractor0,78đ và OODDistractor3,45đ. Eprop_aligned/BrainTrace rất mạnh trênXOR nhưng regressionBinding lớn; không thể suy từXOR≈99% rằng đã tìm được general learner. Chuyển phương trình đã công bố không tự bảo đảm thắng khi forward/signal/optimizer vàbenchmark thay đổi. Chưa có bằng chứng đủ để thay toàn bộ nền SCAR-X bằng một reference này hoặc bác bỏ literature nói chung.

Hidden probe gain và causalrevert trước đó chứng minh có thay đổi representation ở phạm vi eventtasks; probe không thay directskill. Directreadout và robustrepresentation còn là hai mục tiêu khác nhau. Tất cả distractor direct của sáu learner dưới65%; không có learner nào giải xong nút thắt này.

## State router và control âm tính

| Profile | Binding direct/probe | Binding OOD direct/probe | XOR direct/probe | Distractor direct/probe |
|---|---:|---:|---:|---:|
| state | 80.47/84.18 | 80.86/83.46 | 99.09/99.54 | 59.24/59.77 |
| state_shuffle | 88.28/89.39 | 88.67/90.49 | 96.61/98.31 | 59.83/61.52 |
| random_alpha | 84.70/83.85 | 84.83/83.98 | 97.66/98.57 | 58.20/58.27 |
| state_signal_shuffle | 52.47/53.97 | 53.12/55.73 | 76.82/87.89 | 51.37/54.43 |

**state_shuffle là control, không phải winner candidate.** TrênBinding, nó cao hơn state-correct7,81đdirect và5,21đprobe; OODdirect cũng cao hơn7,81đ. Việc làm đúng tương quan alpha–spike chưa tạo bằng chứng causal rằng routing hiện tại hữu ích. Không chọnstate_shuffle hậu nghiệm làm production hoặc thiết kế routermới theo tên task. Globalnomination loại shuffled/random-alpha controls theo protocol.

Alpha của statev1 là heuristic coefficients cố định, không phải router được học. Two-streammix afterfilter không feedback vềstream; với updatecuối episode, alphaởticktrước không trực tiếp tạo cập nhậtweight. Routerdiagnostic thực xác minh biến thiên trạng thái/correlation, chưa xác minh kỹ năng suy luận hoặc lifelong consolidation.

## Tài nguyên và fair controls

ACRdualstream có46.176Bpersistentarray, reference/own/stopped29.472B; thêm16.704B gồm trace và diagnostics. Đây không phải processRAM; buffer/runtime/checkpoint được đo riêng dưới2GB. Cảhai giữ48neuron/8inputs/8fanin vàkhông BPTT/pretrainedmodel trong prediction.

CPU của ACR endpoints vẫn tính hai stream, còn originalown/stopped ởreference tính một stream. Chỉ so equal1024updates ở tài liệu này. Không nhập equalCPU của hai run rồi khẳng định efficiencyadvantage; actualnormpaired ACR vẫn đang được kiểm riêng. D-RTRLlocalblock cũng là adapter phương trình, chưa phải upstreamcompiler runtime hoặc fullgradient trong candidate.

Ba seedDEV chỉ là exploratory; chọnglobalnominee sau nhiềuprofile tạo selectionbias. Paired95% descriptiveCI và minexacttwo-sidedp=.25 không chứng minh significance hoặcfreshconfirmation. Cácconfig không phải independentseeds.

## Nên giữ gì, chưa đưa gì vào sản phẩm

| Thành phần | Đề xuất | Căn cứ/giới hạn |
|---|---|---|
| Strict data-only checkpoint/RNG/coldresume/rollback, endpoint parity, raw scorer | Giữ trong research/tooling | Có tests và đối chiếu thô; production migration/liveplasticity vẫn cần review riêng |
| Pinned e-prop/BrainTrace equations, upstream license receipts, Jacobian/oracle tests | Giữ làm reference nghiên cứu | Tận dụng phương trình và numericalverification có sẵn; không gán novelty cho port |
| SI math helper và phân biệt task-boundary/task-agnostic | Giữ reference +roadmap | Chưa có retention/P2; approximatecredit không kế thừa guarantee của exactlossgradient |
| groups_up hoặcstateACR thành learner mặc định | Chưa nhận | Distractordirectfail, jointregressions, chưa đủnorm/stressfinalaudit |
| eprop_matched/aligned hoặcBrainTrace thay productioncore | Chưa nhận | Cótemporalhiddenlearning, nhưngBinding/OOD tradeoffs vàDistractor<65%; referenceadapter khôngpaperoriginal |
| DECOLLE sourcecopy/DGN dynamics vào primary | Chưa nhận | CurrentDECOLLElicensephi-commercial; DGN thayneurondynamics ngoàiforwardđangkhóa |

Jointgroups_up aggregate65,17% không che được per-familyguardfailBinding/Distractor/Order. Pooledjoint không chứng minh task-agnostic accumulation hoặc chống catastrophicforgetting; isolatedtaskchecks cũng không phải một checkpoint nhớ cảbốnskill. P2 chưa mở. Không lấy Order100% làm exemption causal/retention.

## Khuyến nghị có điều kiện

Giữ nghiên cứucredit theo equations/test đã công bố; hoàn tất đúng controls norm/shuffle/stress đã đăng ký, rồi dừng vòngrouterv1 nếu không có lợi thếmatched. Hiện chưa có lý do mở search lớn hoặctuneCcoeff thêm. Nếu finalcontrols xác nhận không đạtgate, **PIVOT hẹp** vào phân biệt creditapproximation/filter/readoutusability vàdistractorinterference; mỗi cải tiến phải có baseline/equation/parity sẵn trước khi phát minh. Không tăngH48, không ghép LLM, không đổi goal2GB/CPU-first.

Không chọn dựa vào costđã đầu tưACR. Mặt khác cũng không nói literature đã giải distractor/retention củaSCAR-X khi reference hiện chưa đạt. Phầnđãgiải trongliterature: factorization vàlocaltraceconstruction; phầnđãtáilập: adaptermath/checkpoint/parity. Generalhiddenrepresentation/readoutstability/continualretention cònUNRESOLVED. ACR chưa đủevidence đểgọi POTENTIAL_NEW_CONTRIBUTION.

Kết luận tại thời điểm viết: **chưa production; chưaP1/P2; chưafreshconfirmation**. Nhận định cuối về ACRđợi finalnorm/stress vàintegrityaudit. Sốlearningupdates tạo bởi reviewer:0.

## Per-seed raw scores và result hashes

R:`jobs/<task>-<seed>-groups_up-equal_updates/result-0.json`; L:`jobs/<method>-<task>-<seed>-equal_updates/result.json`. Chỉequal1024updates, khôngshuffle. Điểmđơnvị%; hashSHA256 full củaJSONresult (checkpoint/initialhashes nằm trongJSON vàđãkiểm).

### groups_up

| Task | Seed | Direct | Probe | OODdirect | OODprobe | ResultSHA256 |
|---|---:|---:|---:|---:|---:|---|
| strict_binding | 140011 | 84.77 | 86.72 | 83.98 | 83.20 | `8d82aa49ce630e973051804caa2658d3bd666b2af460610007dbe54a7d56fccc` |
| strict_binding | 140017 | 89.65 | 87.30 | 92.77 | 90.04 | `7edc939125a98f4c1bd488332cfd74173d60a6cae96ef8962881b493afe926d8` |
| strict_binding | 140023 | 80.66 | 91.60 | 80.66 | 93.75 | `da906843f05f9a4e2fc58b6d084fc96e913bf16a27f9d4c7c051a18f279da5e7` |
| xor | 140011 | 99.80 | 99.80 | 92.19 | 95.70 | `9a1dbf39b2671108d486216595d77785d3c37fb8fa7754bee693feb7e8c11239` |
| xor | 140017 | 99.02 | 99.61 | 82.81 | 90.04 | `4feea43f03e68c52b84fee2c2c3e14934eef8d13258c6a90e01032cae369ea96` |
| xor | 140023 | 95.51 | 95.90 | 80.66 | 86.91 | `565d8cc2d54262cfcfb2b4bdc8ed2637caaeea60af64e710a10111b3121d3e56` |
| distractor_binding | 140011 | 56.45 | 56.45 | 53.12 | 51.56 | `cec39a5961c38355a02d4304ecbfc9be9b10c96beda9be190d4dab053c426563` |
| distractor_binding | 140017 | 58.98 | 58.79 | 56.45 | 58.79 | `e677b29e9253e322bbb2545790b68b51b4acbd5177c394c3559ecbb420eb033d` |
| distractor_binding | 140023 | 62.30 | 66.21 | 58.59 | 62.89 | `0c53c5f26d446ef038051eda244e0fecaf316b51f841f8b66c1ea9780072f7e4` |
| order | 140011 | 100.00 | 100.00 | 100.00 | 100.00 | `99aec056a7a96583316840ff42d8e2b53f8bcc4341b4a149cac38e7eb08c2d18` |
| order | 140017 | 100.00 | 100.00 | 99.80 | 93.95 | `61a07a914355b67d3efc7fd932df8929ef644579aed17a53c72df45c2b1dbc1f` |
| order | 140023 | 100.00 | 100.00 | 100.00 | 100.00 | `756875247b30902f7c48b1eeffb076c1686a3c8a55307b0aced8016aec759c3e` |

### own

| Task | Seed | Direct | Probe | OODdirect | OODprobe | ResultSHA256 |
|---|---:|---:|---:|---:|---:|---|
| strict_binding | 140011 | 86.91 | 87.50 | 86.13 | 86.13 | `bb0b96e3661447dac6d4e30a73ed37999367fb2d226c0ee532e392b23685250e` |
| strict_binding | 140017 | 81.45 | 81.25 | 81.64 | 78.12 | `76c6f5a1bc1d2599b43d01bafa69a514b029d3be301ffef37e18ccb25813fef6` |
| strict_binding | 140023 | 87.89 | 92.19 | 83.01 | 90.82 | `258966cf59f7239119cf5aa571b62d33f1a0707068241c78e363efe9d91dda2b` |
| xor | 140011 | 98.05 | 98.63 | 87.11 | 87.50 | `48ba981c925522e724df7becaf23283c2448086d5be34b1c2d2df39b21946ca3` |
| xor | 140017 | 95.31 | 98.24 | 89.84 | 95.31 | `399a61222219067ce2a677f9a2ca661d2a20badc3bc2958566c6537cf768f3a6` |
| xor | 140023 | 82.23 | 93.16 | 79.69 | 84.77 | `90b57c41c0fc6ea59242a69e931ce0f1b74deac08977f2e00a0ad3b70f46cced` |
| distractor_binding | 140011 | 55.86 | 52.15 | 51.56 | 54.69 | `3d9e1b87e0c4e2c6d330cf5b9bda7c61d4a8557ad5ca14ecb774f6db9d7d0f09` |
| distractor_binding | 140017 | 51.17 | 58.40 | 56.05 | 56.84 | `eac17fae1a5c0bbbbde85743c1660cb0fc684517b63c9f27e9fa6b83d135f193` |
| distractor_binding | 140023 | 63.87 | 60.35 | 54.69 | 57.03 | `e20367fceadbd4add20dbe96c1cc7590e7d340103de3f05b3e0979157d0eafc6` |
| order | 140011 | 100.00 | 100.00 | 100.00 | 100.00 | `1b1c0a68efc96cc42b5100ca6a3469d16970f29f7ce5d75e884cff5d7e2d6da1` |
| order | 140017 | 100.00 | 100.00 | 100.00 | 100.00 | `be9a72ad51719179f6654872e896b2a5370cde1e0b60e59c6f56ab354755c71f` |
| order | 140023 | 100.00 | 100.00 | 100.00 | 100.00 | `09d10e59e67dd53ca08a52dfe0c78436b10069be48380c429e0448c2bc7ed4e9` |

### stop

| Task | Seed | Direct | Probe | OODdirect | OODprobe | ResultSHA256 |
|---|---:|---:|---:|---:|---:|---|
| strict_binding | 140011 | 85.74 | 92.19 | 86.72 | 94.34 | `82527a2ff0448fa0510e46fae3972dde272efe293fe617218bd9425016969d49` |
| strict_binding | 140017 | 73.63 | 75.39 | 72.66 | 75.78 | `8e4ea26e98b994d32a1b361ae86f325fccc4d3f337311ae6e7652b2ec4913bec` |
| strict_binding | 140023 | 88.48 | 88.28 | 89.26 | 84.96 | `fa5078817c3dafc466562cf5cfa07dbdec68daad20b3fbfd84567667938e67e0` |
| xor | 140011 | 99.80 | 100.00 | 84.57 | 88.28 | `70803684481f1600e7882feae39a0c8ad7fa5d268ba0d88cc5466f8b6d73ffd1` |
| xor | 140017 | 99.02 | 99.41 | 91.60 | 93.95 | `4450f3b6b47a8a156e349576567e2c90581c394c39f236fea305d70b0cdb6641` |
| xor | 140023 | 99.22 | 99.22 | 83.79 | 87.89 | `4ce62ff49e0ab5e338006a645c5f725ea5f861bfdf0afae39c4bb85fd62be8fe` |
| distractor_binding | 140011 | 55.66 | 60.35 | 56.84 | 63.09 | `c51a5f12ce054c0f6029401aae0cd04b9cb9b1ee86d8ae6719f36bed7a3a4d86` |
| distractor_binding | 140017 | 50.00 | 58.59 | 52.73 | 59.57 | `d69aa9d23bacf29dcd635f0c364f01fad798cdf538717a394232769718c64ae5` |
| distractor_binding | 140023 | 61.33 | 71.29 | 59.38 | 65.62 | `67efd175581cf1b8c04756dc5f2a5d3ce19ffacbdc1d412c30ead50397c9a1d7` |
| order | 140011 | 98.83 | 100.00 | 100.00 | 100.00 | `2c287f5837f9d9bad22aefc8d2f9fe69ab5a4779e6b753b6a66495d75de9dfbe` |
| order | 140017 | 100.00 | 100.00 | 100.00 | 100.00 | `17df7301471760c917e7e0fd6571014c055c977843c7b1f2f485f9baa111965d` |
| order | 140023 | 100.00 | 100.00 | 100.00 | 100.00 | `0bef338381cc93c1345f7d0ceeba5e8d40d17566f5f7e21630402e58da4dc674` |

### eprop_matched

| Task | Seed | Direct | Probe | OODdirect | OODprobe | ResultSHA256 |
|---|---:|---:|---:|---:|---:|---|
| strict_binding | 140011 | 78.71 | 87.50 | 79.49 | 87.70 | `ceeb0e011ac2deb91a88102039aa3810ec51eb6c23ffd1cad34fda63d6937255` |
| strict_binding | 140017 | 66.99 | 68.55 | 65.82 | 67.97 | `19cba112bab5d009032a24e4692e2448e7faf338f3bd420cc7322d93383c9b0c` |
| strict_binding | 140023 | 90.04 | 92.19 | 92.58 | 92.19 | `f410c849eeb07b0c3961dd36d1e8380db1919fc6003193417fad03997ebba8b5` |
| xor | 140011 | 100.00 | 100.00 | 90.62 | 92.58 | `c457568e26fdc354e598104e6a9ffa7fbfc68b5512d978f3ddf3d23a464a89a3` |
| xor | 140017 | 99.02 | 99.61 | 88.87 | 95.31 | `be97ee026d5788b7b98fdc483bd45e1a3a59e0193faf554358e1152d6111da04` |
| xor | 140023 | 95.70 | 99.22 | 82.03 | 88.67 | `7f50781f5ae5d0feb220fe45cd4d1fde02658d815e10216a72c055e789f7f23a` |
| distractor_binding | 140011 | 55.08 | 57.81 | 53.52 | 55.66 | `48e948fff26bfdf65ee4e162cfa4f3464f541f5ffeab62e27b878042c5bf0418` |
| distractor_binding | 140017 | 57.62 | 61.72 | 57.23 | 57.23 | `7e0d0639b0cec7414cdbaddb528dc64c21c05f21a770b016679c71da41db5527` |
| distractor_binding | 140023 | 67.38 | 66.41 | 67.77 | 63.09 | `38cbf274cd45f0273ad09ac0cee062c4dfad131292d82fcac8f1c9e1ac1b7b51` |
| order | 140011 | 100.00 | 100.00 | 100.00 | 100.00 | `cb1f453e8b0688b58f8aec264eece52be6cb8996bd47dcad55868d02de22c713` |
| order | 140017 | 100.00 | 100.00 | 100.00 | 100.00 | `e6d032ddf0e59e4974f581aae594364498a5a2819172fe08bbef1891781b0415` |
| order | 140023 | 100.00 | 100.00 | 100.00 | 100.00 | `53461fe415461bc842cd264d5bc5b57ffa871a80a770b7bd5a117fe779e3bf9a` |

### eprop_aligned

| Task | Seed | Direct | Probe | OODdirect | OODprobe | ResultSHA256 |
|---|---:|---:|---:|---:|---:|---|
| strict_binding | 140011 | 66.80 | 73.44 | 64.45 | 70.31 | `919c0b0de924bdfdcf9681d39dd83fd993ae94607c2de911c4a531c54423b54a` |
| strict_binding | 140017 | 68.16 | 63.48 | 71.09 | 61.33 | `ddc2d5073941b4f6537acf5198052725913a63e2c789951e8532e37f24b8f8e7` |
| strict_binding | 140023 | 74.80 | 75.39 | 69.73 | 66.41 | `9eda4279b2c6887fae4284b6071cfe83735bcd813b17db08546374054be97b9e` |
| xor | 140011 | 100.00 | 100.00 | 93.75 | 91.60 | `acfd13d94c94ac349d3db03e6968b3ae7f17b05a4fcd19be375a74e0c88f2c83` |
| xor | 140017 | 99.02 | 99.61 | 89.06 | 88.67 | `583fde4040d7b669d2f2387005aa1d5655e1581af323d18f41f8d5ae38903f60` |
| xor | 140023 | 99.61 | 99.80 | 90.82 | 91.60 | `38ac7b264a031ad15fe19881923ee31d9d2e244a75b9e667f7aa3c7e3549830e` |
| distractor_binding | 140011 | 54.88 | 56.64 | 52.73 | 53.32 | `0d4841b52610fac44742c0e010f1ebc3083ab06dcca1a4e564bcc946d172d24b` |
| distractor_binding | 140017 | 47.66 | 53.32 | 48.44 | 50.39 | `1cba09746b0885c8fdeb14109c4bce80370640ab0b7b4a603c72f3f27cf41739` |
| distractor_binding | 140023 | 62.50 | 64.26 | 58.40 | 63.87 | `9069df8c624b13f750b8263308f081bb9fd468e62c1e5a5a9085987893c645be` |
| order | 140011 | 99.80 | 100.00 | 100.00 | 100.00 | `0582afee3675b0f733b84652bcc72aed00ba74d6bf7c9f6d538a588dc7cc79e8` |
| order | 140017 | 100.00 | 100.00 | 99.80 | 100.00 | `ea4ff17c34345e6cf703155566dc9f5a414ebade5748b1ddf38172d2be31e9ce` |
| order | 140023 | 100.00 | 100.00 | 97.46 | 94.92 | `e2621954d15697db4e895f0a4558e223374cf8726560f587968730be96d05dcc` |

### braintrace

| Task | Seed | Direct | Probe | OODdirect | OODprobe | ResultSHA256 |
|---|---:|---:|---:|---:|---:|---|
| strict_binding | 140011 | 72.27 | 79.88 | 72.46 | 79.30 | `3489bad04f01bb9791406ab950d3b4be26308a188cef2b01c89da24ce8f79770` |
| strict_binding | 140017 | 66.80 | 68.95 | 64.65 | 61.91 | `13e6cd1cd5feef5c073d0e80db7ba86d9b94f720c6657a87d6451ab40dc7bebf` |
| strict_binding | 140023 | 66.80 | 73.83 | 66.80 | 73.63 | `88150bf887c1cb1963dc24c0ab04cf7a2eb20d75ac05a0f7124ac75529751a20` |
| xor | 140011 | 100.00 | 98.24 | 95.12 | 91.80 | `fa6c5f1f490e1887da1d7ddeeede2085ab98e55fac958dc2600cd9d84840eefc` |
| xor | 140017 | 99.41 | 99.41 | 87.50 | 87.50 | `2588ea8d0933609bddc6d8c97f00e0f151cebe15472567647641f061e2f598ba` |
| xor | 140023 | 98.24 | 99.22 | 84.77 | 86.13 | `a24b95d381c611f415768e046010b77aaffbce6c40e688323fcae580c7ab06a3` |
| distractor_binding | 140011 | 57.03 | 56.45 | 51.95 | 53.32 | `33466eef4924e89eb3ac6bceaf03621cfd603d6b4e3c39521d1b1cb2d941defe` |
| distractor_binding | 140017 | 49.61 | 55.08 | 52.93 | 54.10 | `940b2498d1bd967f487acfda49df0a0b64040bb4501ec88211df4ce6defde3d5` |
| distractor_binding | 140023 | 53.32 | 55.86 | 54.10 | 56.64 | `18a520c1875be1e5efe9d2cfe28b945f8ba489ad6123c0af53dee217036d5470` |
| order | 140011 | 100.00 | 100.00 | 100.00 | 100.00 | `cf2c48eb20e545f9ced40034797d28fe39b547784c34fabce9c89f078c55bbc8` |
| order | 140017 | 99.80 | 100.00 | 100.00 | 100.00 | `645efaf9ec0779ff126a44c1d103f143f64a4e4c85341fa11af14f6edca328f9` |
| order | 140023 | 100.00 | 99.80 | 99.02 | 98.24 | `a7dd24ef43f9a95aaedcdbb02f554ce3d1fe4bb715cc0da91289d7cc30ec4c6c` |
