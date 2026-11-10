# Product run audit — selection tie-break deviation

Recorded after run,2026-11-09. Registered primary development score was equal
mean of text exact and digit accuracy, with lower text loss for ties.
Implementation ranked `(mean_dev_score, dev_text_exact)` and did not compute
devteacherforcingloss. Thus tie-break was not implemented as stated. Primary
score,seeds,100epochs,thresholds and testopeningorder remained unchanged.
Source script retained as executed for reproduction.

Both seeds failed80%greetingtarget (50%,n=6). No criterion changed after result.
Productdiagnostic with protocoldeviation, not conforming scientificconfirmation.
No tuning on acceptance prompts. Fresh protocol/confirmation material required
for follow-up; digits dataset also used before, so remains diagnostic.
