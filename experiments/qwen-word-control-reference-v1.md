# Actual Qwen reference for missing public word-control cases

Registered BEFORE calls. Same existing pinned Qwen3-4B-Instruct-2507 Q4_K_M
SHA8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc.
No new download, APIpayment, training, SCAR-Xweights/proxy or Hermesconfigchange.
Do NOT start until all6isolatedTorchcontrols/coldchecks complete and their
coordinator is verified terminal; no overlapping Torch/Qwen memory benchmark.

Fixed95publiccases=dev31+prior48+train16 from exact same reviewed/authored
source. Existing48prior single-prompt Qwenoutputs are preserved, not rerun.
The47missing UNIQUEprompts (31dev+16train) are selected solely by absent exact
promptreference, never by modelcorrectness. Freeze case snapshot before calls.
SCAR comparison source fixed densecontrol seed28001 (firstpairedseed), not
selected by performance. It is ARCHIVED BPTT CONTROL, NOT mainlocalSCAR-X.
No private model, transcript, guard or personalmemory is loaded/sent.

Match existingreference settings: CPU2threads,context1024,batch64,ubatch64,
ngl0,no-mmap,oneparallelrequest; exact single userprompt,no system/history/tools.
temperature0,seed919,max_tokens64. Keep truncations/empty/errors and timestamps;
no retries with moretokens or choosing a better answer. Raw Qwenoutputs are
NOT truth labels or automatic trainingdata; no exact-style qualityscore.
Inputtext matches, but training/tokenizer/RAM/decoder budgets differ; cannot
claim equivalentmodelquality, knowledgeequalto4B or resource superiority.

Existing llama-server must be absent; launch an owned loopback dynamicport,
verify pinnedmodelSHA, process-tree4.5GB monitored/per-processcap, below-normal
priority,360s round/30s request/90sstartup. Reference explicitly aboveSCAR-X's
2GBtarget. Stop only ownedserver in finally, keep rawresults partial iferror.
Publish each actual response with original archivedALIFoutput and source label.
Save separate receipt; preserve old48reference immutable. Passive reader may
merge only exactprompt matches from these explicit source receipts, keeping
individual source timestamps/finishreasons. No fakeQwenoutputs for missingcases.
