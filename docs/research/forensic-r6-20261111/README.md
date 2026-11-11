# SCAR-X forensic handoff

- [Review cập nhật R7 DeepSeek/Gemini](REVIEW_R7_PROGRESS_VI.md): hidden positive task, retention trade-off và phần nên giữ nghiên cứu.
- [Bổ sung oracle event/nhãn sau bàn giao24453bd](RAW_EVENT_LABEL_ADDENDUM.md):135.168lượt row,0nhãn sai, không forward/training.
- [Bản gửi ChatGPT web / vấn đề còn mở / 10nhóm task production](CHATGPT_WEB_REVIEW_BRIEF_VI.md)
- [Kết quả và lệnh tái lập](RESULTS.md)
- [Review Gemini mới8f24a5a](REVIEW_GEMINI_8f24a5a.md)
- [Audit tổng hợp](FORENSIC_AUDIT.md), [23claim verdicts](CLAIM_LEDGER.json)
- [Inventory](BASELINE_INVENTORY.md), [data lineage](DATA_LINEAGE.json)
- [R6D source audit](R6D_SOURCE_AUDIT.md), [raw recompute](R6D_CLAIM_RECOMPUTE.json), [gates](R6D_GATE_INTEGRATION.json)
- [R5B/Grok reproduction](R5B_R6B_REPRODUCTION.md), [Grok effective-source normalization](GROK_NORMALIZATION.json)
- [Older campaign verification scope](OLDER_CAMPAIGN_STATUS.md), [related work/product gap](RELATED_WORK_AND_PRODUCT_GAP.md)
- [Audit protocol locked](PROTOCOL_LOCKED.md), [Gemini historical-only addendum](GEMINI_AUDIT_ADDENDUM_LOCKED.md)
- [R6E draft](PROTOCOL_R6E_DRAFT.md), [R7 draft](PROTOCOL_R7_DRAFT.md): **chờ duyệt, chưa chạy**
- [Deviations and limits](DEVIATIONS.md), [delivery verification](DELIVERY_VERIFICATION.json)

Audit code: scripts/forensic_*.py and src/scarx/evaluation/forensic/.
Tests: tests/test_forensic_evaluator.py and tests/test_forensic_mutations.py.
Immutable execution evidence: runs/*forensic*/; receipts: artifacts/forensic-r6/.
No serving/main/weights migration is part of this handoff.
