# Audit deviations and unresolved limitations

1. No fresh confirmation or new candidate was executed. Historical confirmation
   banks are labeled seen, including Gemini121011…121043.
2. R6D matrix and first control replay save predictions but the first controls
   tap did not save random-label predictions. A second historical controls replay
   saved those predictions, train-label/order hashes and trained head arrays.
   Both runs remain immutable; their original trial/summary outputs match history.
3. Full pytest in one process crossed sampled tree commit2,070,958,080B and was
   stopped. A two-part retry still crossed2,032,844,800B in its tail. Diagnosis:
   test_word_control spawns a Torch child when parent pytest already loaded Torch.
   Keep failures. Successful prefix, separate tail groups, new evaluator tests
   and separately built native parity jointly cover the full test set. This is
   a testing execution partition, not a relaxed runtime budget or a product claim.
4. No live server was started/restarted or taught. No Hermes/provider/SQLite/live
   checkpoint read/write was needed. No browser production acceptance was claimed.
5. Gemini R6C was still uncommitted/local at P0, then pushed8f24a5a during the
   audit. A committed addendum192c5f0 registered its historical-only replay before
   execution. Its supplementary review does not rewrite the original P0 snapshot.
6. Gemini CKCER core accuracy/both_A reproduces; strict entire-row equality does
   not.220margin fields differ numerically, and one R5 reversed-write prediction
   at121037 differs. Exact arrays/limits are reported; not fixed by tuning.
7. Gemini's344prediction-group arithmetic is independently checked. The raw event
   oracle for every auxiliary/noise sample was not independently implemented in
   that supplementary tap; labels are not upgraded into verified source truth.
8. Older R1–R5 claims are separated as source/aggregate-row checks or reported;
   no statement that every past training campaign was independently rerun.
9. No external SNN framework or pretrained model was imported as a candidate.
   Literature discussion gives context; no matched superiority claim is made.
10. The initial generic Gemini-engineering receipt used the historical-reproduction
    class label. Its actual command ran only5unit tests. Treat it as ENGINEERING_CHECK;
    future supervisor labels are corrected. The original receipt is not overwritten.
11. Limitation7 was addressed after delivery by a separate data-only follow-up.
    RAW_EVENT_LABEL_ADDENDUM.md records135,168row checks,0mismatched latent labels,
    raw marker ambiguities and cue/pairing limitations. It does not change the
    scope of the already delivered24453bdpackage or authorize new training.
