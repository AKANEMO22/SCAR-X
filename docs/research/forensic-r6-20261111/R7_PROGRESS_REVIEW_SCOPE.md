# Bounded review of newly published R7 branches

User requested latest DeepSeek/Gemini progress and reusable findings. Pin:
DeepSeek `aacde3fa0983b9cebc585e19980022a52a45ccad`, Gemini R7
`64441fb9be162864cf2e1252638d101a2031f2e4`, Gemini R6 erratum
`df683923722409ab00ee40213e7cda29b2003900`.

Source/raw arithmetic review,5existing Gemini P0tests, and at most one exact
historical positive-task replay: strict_binding, first published DEV seed140011,
unchanged128calibration/1024train/256fit/512ID/512OOD and all existing conditions.
Selection is deliberate verification of a claimed positive task, not an unbiased
fresh overall evaluation. No new seed, task, algorithm, confirmation or P2 run.
No modification to peer branches, main/live model or prior audit artifacts.

Recompute all15published Gemini P1rows,18continual paired cells, DeepSeek90three-
task and36corrected-six-task cells. Keep task-order cells grouped by model seed
for primary uncertainty reporting. Distinguish changed ALIF hidden weights from
trained nonlinear rate-based readout. Inspect task-ID input, update counts,
boundary knowledge, forgetting definitions and retracted benchmarks before
comparing reported performance. CPU one thread,2GB monitored tree/capped child,
120sP0,180sone replay. Do not rerun37research rounds or reopen failed gates.
