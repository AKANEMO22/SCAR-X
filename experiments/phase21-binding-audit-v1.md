# Post-hoc task audit: binding shortcut, registered before challenge run

Original binding permits independent values: guessing from the bag of values succeeds75% in the noiseless case (both equal gives correct; different gives chance). This undermines interpreting ~73% as binding. Original results stay immutable and are not relabeled as success.

New challenge, no hyperparameter tuning: values at the two keys always opposite; key/value presentation order and query key randomized. Counts of key0/key1/value0/value1 identical across labels. Hidden learner gets no labels in input. Same48neurons, score selected earlier, readout-only, random-sign modulation;512train/128probe/256test. New independent seeds307,331,353,379,401,431,457,479. Train/probe/test seed offsets200000/210000/220000. Add bag-of-events ridge control to expose shortcuts.

Same representation criterion: lowerCI>=5pp gain vs initial, end-to-end>=65%, paired exact signflip p<=.05 (one challenge primary contrast). No change to previous failure or absoluteforgetting thresholds. This is a task-validity audit after inspecting earlier results, not a new attempt to select a winning rule. Timing/persistence benchmarks are unchanged.
