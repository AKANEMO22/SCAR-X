"""Research-only causal incident observations; no serving import or learning."""

import numpy as np
from scarx.hybrid.learner import features

FORMAT = "scarx-incident-observation-sequence-v1"
FIELDS = {
    "tool", "reported_status", "error_type", "empty_result", "spillover",
    "streak", "probe",
}
PROBE_FIELDS = {"read_ok", "reported_not_found", "original_path_differs", "kind"}


def observation_sequence(question, observations):
    """Only observations before a decision; no scope IDs, labels or future data."""
    if not isinstance(question, str) or not 2 <= len(question) <= 1000:
        raise ValueError("Invalid question")
    if not isinstance(observations, (list, tuple)) or not 1 <= len(observations) <= 8:
        raise ValueError("Expected bounded causal observation history")
    chunks = []
    for row in observations:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Unexpected observation fields")
        if row["reported_status"] not in ("ok", "error", "unknown"):
            raise ValueError("Invalid reported status")
        if (not isinstance(row["tool"], str) or not 1 <= len(row["tool"]) <= 120
                or not isinstance(row["error_type"], str) or len(row["error_type"]) > 120
                or type(row["streak"]) is not int or not 1 <= row["streak"] <= 100):
            raise ValueError("Invalid bounded observation")
        if any(type(row[k]) is not bool for k in ("empty_result", "spillover")):
            raise ValueError("Expected flags")
        probe = row["probe"]
        if probe is not None:
            if not isinstance(probe, dict) or set(probe) != PROBE_FIELDS:
                raise ValueError("Unexpected probe fields")
            if probe["kind"] not in ("requested_path_binding", "local_source_check"):
                raise ValueError("Invalid probe kind")
            if any(type(probe[k]) is not bool for k in PROBE_FIELDS - {"kind"}):
                raise ValueError("Expected probe flags")
        p = probe or {}
        numbers = (
            row["reported_status"] == "ok", row["reported_status"] == "error",
            row["empty_result"], row["spillover"], min(row["streak"] / 3, 1),
            probe is not None, p.get("read_ok", False),
            p.get("reported_not_found", False), p.get("original_path_differs", False),
            p.get("kind") == "requested_path_binding",
        )
        chunks.append(features(
            question + " [observed tool] " + row["tool"] + " " + row["error_type"],
            numbers,
        ))
    return np.concatenate(chunks).astype(np.float32)
