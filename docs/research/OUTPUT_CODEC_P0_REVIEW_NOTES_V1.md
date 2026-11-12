# Output codec P0 — prototype review history

This work is authorized only by the user's explicit **isolated codec P0** approval.
No P1/P2, dataset training, release migration or new learning rule is included.
`ponytail` tools/context were searched for and unavailable in this session.

## Initial engineering receipt, preserved

Run `20261111T170938Z-output-codec-p0-v1-e3f834` passed its initial24tests and
roundtrip/parity/cold-continuation fixtures. Its source snapshot and result are
preserved. It is preliminary evidence, not certification of the later review
cases. Its `engineering_fixture_synaptic_updates=11` is a logical fixture count;
it omits duplicate cold-verification update calls and updates in unit tests.
The final runner reports those counts separately rather than rewriting history.

## Five independently reproduced gaps before hardening

All five added regression tests initially failed (`DID NOT RAISE ValueError`):

1. Session JSON could contain an unknown importance field that was dropped.
2. Inner core metadata could contain an unknown importance field that was dropped.
3. Capacity history of33events was silently truncated by the inherited loader.
4. Empty/invalid-Unicode prompts were not validated before beginning a fixture.
5. Inherited in-place vocabulary growth / legacy train_pair / image training could
   bypass the research codec layout or its bounded fixture controller.

The isolated prototype now rejects these paths. Additional tests cover unknown
legacy-wrapper metadata and pending state exceeding the32update fixture budget.
31targeted tests passed after hardening. The final unique run, regression receipt
and handoff report supersede the preliminary24test receipt for current code.

The naive full-width GEMV float-order difference remains separately disclosed in
`experiments/output-codec-p0-numerical-note.md`; exact compatibility still requires
zero error. No algorithmic, neural-quality, retention or production claim follows.
