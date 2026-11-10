# Pre-neural-measurement36B: repair one language-invalid teacher row

2026-11-10 BEFORE prototype implementation/calibration/training/qualityscores.
Manualread of ALL128generatedprompts identified vi_greet_11="Hello bạn trợ lý!"
violates registered language=vi/no mixedlanguages. Original candidate immutable.

Quarantine exactly this row. One additional authorized AGYCLI prompt-only call
asks for ONEall-Vietnamese greeting vi_greet_11 within originallength/schema.
No source/main/privateinput, no targets/modelscore or split transmitted. Same
tool/timeout/quarantine policy. Keep repair source/response SHA and review it.
No other row replacement or model-based selection. This is explicit correction
of language-schema violation before labels/splits measured, not easierdataafterfail.

Protocol36 initially requires complete128reviews and describes only schema repairs;
this amendment extends that premeasurementrepair to this one demonstrably invalid
language row. If replacement invalid/duplicate, STOP_DATA_REVIEW_INCOMPLETE;
no retrygrid. Gates/modelseeds/algorithm/learningbudget/splitRNG unchanged.
