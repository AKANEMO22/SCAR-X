"""Strict, source-typed reporting rubric; never used as inference or training."""
from decimal import Decimal
import unicodedata

from .knowledge_factory import UNKNOWN, numeric_value


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def unquote_literal(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) == 3 and lines[0] == "```" and lines[-1] == "```":
            return lines[1].strip()
    if text.startswith("`") and text.endswith("`") and text.count("`") == 2:
        return text[1:-1].strip()
    return text


def grade(answer, row, fact):
    """Return bounded observations, not a broad semantic/fluency judgment."""
    exact = normalized(answer) == normalized(row["canonical_answer"])
    unknown = normalized(answer) == normalized(UNKNOWN[row["language"]])
    semantic = None
    if fact["known_domain"]:
        literal = unquote_literal(answer)
        if fact["value_type"] == "number":
            try:
                semantic = numeric_value(literal) == Decimal(str(fact["value"]))
            except ValueError:
                semantic = False
        else:
            semantic = normalized(literal).removesuffix(".") == normalized(str(fact["value"]))
    return {"exact_normalized_correct": exact, "typed_semantic_correct": semantic,
            "registered_abstention": unknown,
            "rubric_scope": "Strict literal-only source value; broader prose semantics and fluency are not automatically graded"}


def reference_case(row):
    facts = {"config_inputs", "config_hidden", "config_neuron", "config_leak", "config_ram_limit",
             "g1_hidden", "g1_feature_decay", "g1_vocabulary_cap",
             "api_model_id", "api_port", "api_host", "api_models_route"}
    return row["fact_id"] in facts and row["id"].endswith(("_p05", "_p06")) and row["split"] in ("development", "unknown_development")


def summarize(records):
    known = [r for r in records if r["split"] == "development"]
    unknown = [r for r in records if r["split"] == "unknown_development"]
    def accuracy(rows, key):
        return sum(r["grade"][key] is True for r in rows) / len(rows) if rows else None
    return {"known_questions": len(known), "unknown_questions": len(unknown),
            "known_exact_accuracy": accuracy(known, "exact_normalized_correct"),
            "known_typed_semantic_accuracy": accuracy(known, "typed_semantic_correct"),
            "unknown_registered_abstention_rate": accuracy(unknown, "registered_abstention"),
            "generalization_gate": "NOT_MEASURED_WITH_REQUIRED_TRAINED_CONTROLS",
            "retention_gate": "NOT_MEASURED", "causal_grounding_gate": "NOT_MEASURED",
            "production_ready": False}
