"""Bounded post-turn review state; a generated critique is never a true label."""

import copy
import json
import re
from ..learning.verifiers import parse_integer_expression, integer_answer


def initial_review():
    return {
        "format": "scarx-review-v1",
        "pending": [],
        "reviewed": [],
        "completed": 0,
        "rejected": 0,
        "replay_updates": 0,
        "last_result": None,
    }


def upgrade_controller(state):
    value = copy.deepcopy(state)
    # Explicit additive migration from the shipped pre-review controller.
    if "review" not in value:
        value["review"] = initial_review()
    review = value["review"]
    if (
        not isinstance(review, dict)
        or set(review) != set(initial_review())
        or review["format"] != "scarx-review-v1"
    ):
        raise ValueError("Incompatible review controller")
    for key, limit in (("pending", 32), ("reviewed", 128)):
        rows = review[key]
        if (
            not isinstance(rows, list)
            or len(rows) > limit
            or any(not isinstance(s, str) or not 1 <= len(s) <= 80 for s in rows)
            or len(rows) != len(set(rows))
        ):
            raise ValueError("Invalid bounded review queue")
    for key in ("completed", "rejected", "replay_updates"):
        if type(review[key]) is not int or review[key] < 0:
            raise ValueError("Invalid review counter")
    if review["last_result"] is not None and (
        not isinstance(review["last_result"], dict)
        or len(json.dumps(review["last_result"]).encode()) > 16000
    ):
        raise ValueError("Invalid review result")
    return value


def is_auxiliary_answer(question, answer):
    # Old receipts accidentally logged title-generation as a dialogue answer.
    # Do not turn a helper's {"title":...} output into a learning experience.
    try:
        parsed = json.loads(answer)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(parsed, dict)
        and set(parsed) == {"title"}
        and isinstance(parsed["title"], str)
    )


def arithmetic_check(question, answer):
    expression = parse_integer_expression(question)
    if expression is None:
        return None
    observed = re.fullmatch(
        r"\s*(?:Đáp án\s*:?\s*|Answer\s*:?\s*)?(-?\d+)\s*[.!]?\s*", answer, re.I
    )
    if not observed:
        return None
    expected = integer_answer(expression)
    value = int(observed[1])
    return {
        "expected": expected,
        "observed": value,
        "correct": expected == value,
        "verifier": "integer-arithmetic-v1",
        "scope": "exact numeric answer only",
    }


def review_format(topics):
    identifiers = [item["id"] for item in topics]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "scarx_post_turn_review",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "maxLength": 160},
                    "reason": {"type": "string", "maxLength": 120},
                    "public_topic_id": (
                        {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "string", "enum": identifiers},
                            ]
                        }
                        if identifiers
                        else {"type": "null"}
                    ),
                },
                "required": ["question", "reason", "public_topic_id"],
                "additionalProperties": False,
            },
        },
    }


def parse_review(text, topics):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {
        "question",
        "reason",
        "public_topic_id",
    }:
        raise ValueError("Invalid review fields")
    if (
        not isinstance(value["question"], str)
        or len(value["question"]) > 160
        or not isinstance(value["reason"], str)
        or len(value["reason"]) > 120
    ):
        raise ValueError("Review output exceeds registered bound")
    if value["public_topic_id"] is not None and value["public_topic_id"] not in {
        t["id"] for t in topics
    }:
        raise ValueError("Unapproved review topic")
    if value["question"] and len(value["question"].strip()) < 2:
        raise ValueError("Invalid follow-up question")
    return value
