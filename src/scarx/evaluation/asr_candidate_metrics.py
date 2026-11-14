"""Fixed utterance-level paired ASR measures; no generated labels."""

import re
import unicodedata
import numpy as np


def word_error(hypothesis, reference):
    tokens = lambda s: re.findall(r"\w+", unicodedata.normalize("NFC", s).casefold())
    hyp, ref = tokens(hypothesis), tokens(reference)
    if not ref:
        raise ValueError("Reference must contain independently annotated words")
    previous = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        row = [i]
        for j, other in enumerate(hyp, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1,
                           previous[j - 1] + (token != other)))
        previous = row
    return {"word_edits": previous[-1], "reference_words": len(ref),
            "wer": previous[-1] / len(ref)}


def paired_language(rows, bootstrap=10000, seed=17011):
    if not rows:
        raise ValueError("No paired utterances")
    lengths = np.array([r["base"]["reference_words"] for r in rows], dtype=np.float64)
    if any(r["small"]["reference_words"] != r["base"]["reference_words"] for r in rows):
        raise ValueError("Unmatched reference denominators")
    base = np.array([r["base"]["word_edits"] for r in rows], dtype=np.float64)
    small = np.array([r["small"]["word_edits"] for r in rows], dtype=np.float64)
    if np.any(lengths <= 0):
        raise ValueError("Invalid word denominators")
    base_wer, small_wer = float(base.sum() / lengths.sum()), float(small.sum() / lengths.sum())
    indices = np.random.default_rng(seed).integers(0, len(rows), size=(bootstrap, len(rows)))
    differences = ((small[indices] - base[indices]).sum(axis=1) /
                   lengths[indices].sum(axis=1))
    return {
        "utterance_pairs": len(rows), "reference_words": int(lengths.sum()),
        "base_edits": int(base.sum()), "small_edits": int(small.sum()),
        "base_micro_WER": base_wer, "small_micro_WER": small_wer,
        "relative_error_reduction": (base_wer - small_wer) / base_wer if base_wer else 0.0,
        "delta_small_minus_base": small_wer - base_wer,
        "paired_utterance_delta_CI95": np.percentile(differences, [2.5, 97.5]).tolist(),
        "bootstrap_seed": seed, "bootstrap_replicates": bootstrap,
        "scope": "Utterances in this language, not independent word or learning-seed replicates",
    }
