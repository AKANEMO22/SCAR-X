"""Bounded, persistent mechanical symptoms; no unverified self-training labels."""

import hashlib
import re
import threading
import time


class IncidentLedger:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()

    def observe(self, payload):
        allowed = {
            "scope",
            "call_id",
            "tool",
            "args_sha256",
            "result_sha256",
            "reported_status",
            "error_type",
            "empty_result",
            "spillover",
            "duration_ms",
        }
        if not isinstance(payload, dict) or set(payload) not in (
            allowed,
            allowed | {"probe"},
        ):
            raise ValueError("Unexpected tool observation fields")
        for name in ["scope", "call_id", "args_sha256", "result_sha256"]:
            if not isinstance(payload[name], str) or not re.fullmatch(
                "[0-9a-f]{64}", payload[name]
            ):
                raise ValueError("Invalid observation digest")
        if not isinstance(payload["tool"], str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,120}", payload["tool"]
        ):
            raise ValueError("Invalid tool identity")
        if (
            payload["reported_status"] not in ("ok", "error", "unknown")
            or not isinstance(payload["error_type"], str)
            or len(payload["error_type"]) > 120
        ):
            raise ValueError("Invalid reported tool outcome")
        if any(type(payload[k]) is not bool for k in ["empty_result", "spillover"]):
            raise ValueError("Expected observation flags")
        if (
            type(payload["duration_ms"]) not in (int, float)
            or not 0 <= payload["duration_ms"] <= 86_400_000
        ):
            raise ValueError("Invalid tool duration")
        probe = payload.get("probe")
        if probe is not None:
            fields = {
                "kind",
                "source_name",
                "project_label",
                "original_path_differs",
                "read_ok",
                "reported_not_found",
                "result_sha256",
                "scope",
                "semantic_resolution_verified",
            }
            if (
                not isinstance(probe, dict)
                or set(probe) != fields
                or probe["kind"] not in ("requested_path_binding", "local_source_check")
                or probe["scope"] != "recorded_project_file_tool"
            ):
                raise ValueError("Unsupported diagnostic probe")
            if (
                any(
                    type(probe[k]) is not bool
                    for k in ("original_path_differs", "read_ok", "reported_not_found")
                )
                or probe["semantic_resolution_verified"] is not False
            ):
                raise ValueError("Probe cannot self-verify semantic resolution")
            if (
                not re.fullmatch("[0-9a-f]{64}", str(probe["result_sha256"]))
                or not isinstance(probe["source_name"], str)
                or len(probe["source_name"]) > 100
                or not isinstance(probe["project_label"], str)
                or len(probe["project_label"]) > 80
            ):
                raise ValueError("Invalid bounded probe metadata")
        with self.lock:
            turns = self.store.get("tool_experience", "bounded_turns") or {}
            bucket = turns.setdefault(
                payload["scope"],
                {
                    "recent": [],
                    "seen_calls": [],
                    "previous": None,
                    "streak": 0,
                    "incident": None,
                    "updated": time.time(),
                },
            )
            if payload["call_id"] in bucket["seen_calls"]:
                return {"duplicate": True, "incident": None}
            bucket["seen_calls"] = (bucket["seen_calls"] + [payload["call_id"]])[-64:]
            signature = hashlib.sha256(
                (
                    payload["tool"] + payload["args_sha256"] + payload["result_sha256"]
                ).encode()
            ).hexdigest()
            repeatable = payload["tool"] in ("process_manage", "process") or payload[
                "tool"
            ].endswith(("_poll", "_get_result"))
            bucket["streak"] = (
                0
                if repeatable
                else bucket["streak"] + 1 if bucket["previous"] == signature else 1
            )
            bucket["previous"] = signature
            bucket["updated"] = time.time()
            record = {
                k: payload[k]
                for k in [
                    "tool",
                    "reported_status",
                    "error_type",
                    "empty_result",
                    "spillover",
                    "duration_ms",
                ]
            }
            record.update(streak=bucket["streak"], created=time.time())
            bucket["recent"] = (bucket["recent"] + [record])[-16:]
            incident = None
            if bucket["streak"] >= 3 and bucket["incident"] is None:
                ident = hashlib.sha256(
                    (payload["scope"] + signature).encode()
                ).hexdigest()[:24]
                incident = {
                    "id": ident,
                    "kind": "identical_nonprogress",
                    "tool": payload["tool"],
                    "observed_repeats": bucket["streak"],
                    "created": time.time(),
                    "reported_status": payload["reported_status"],
                    "reported_error_type": payload["error_type"],
                    "observed_empty_result": payload["empty_result"],
                    "observed_spillover": payload["spillover"],
                    "root_cause": "unresolved",
                    "status": "needs_verification",
                    "facts": "Same observed tool/arguments/result repeated; not proof of a semantic cause.",
                    "candidate_probes": [
                        "Check actual source/workspace binding",
                        "Check missing evidence, denied access or rate limits",
                        "Check output delivery and model interpretation",
                    ],
                    "observation_is_training_label": False,
                    "goal_id": None,
                }
                bucket["incident"] = ident
                if probe:
                    incident["diagnostic_probe"] = probe
                    if probe["read_ok"]:
                        incident["root_cause"] = (
                            "path_or_source_binding_hypothesis_supported"
                        )
                        incident["status"] = "probe_supported_pending_owner"
                self.store.put("tool_incident", ident, incident)
                with self.store.lock, self.store.db:
                    self.store.db.execute(
                        "DELETE FROM objects WHERE kind='tool_incident' AND id NOT IN (SELECT id FROM objects WHERE kind='tool_incident' ORDER BY rowid DESC LIMIT32)".replace(
                            "LIMIT32", "LIMIT 32"
                        )
                    )
                self.store.event("tool_incident_detected", incident, "runtime")
            while len(turns) > 64:
                oldest = min(turns, key=lambda k: turns[k]["updated"])
                del turns[oldest]
            self.store.put("tool_experience", "bounded_turns", turns)
            if (
                payload["reported_status"] == "error"
                or payload["empty_result"]
                or bucket["streak"] >= 2
            ):
                self.store.event(
                    "tool_experience_observed",
                    record
                    | {
                        "scope": payload["scope"],
                        "mechanical_only": True,
                        "new_training_label": False,
                    },
                    "runtime",
                )
            return {
                "duplicate": False,
                "streak": bucket["streak"],
                "incident": incident,
            }

    def attach_goal(self, incident, goal):
        with self.lock:
            row = self.store.get("tool_incident", incident)
            if row:
                row["goal_id"] = goal
                self.store.put("tool_incident", incident, row)

    def decision(self, incident, decision):
        with self.lock:
            row = self.store.get("tool_incident", incident)
            if row:
                row["last_policy_action"] = {
                    k: decision[k]
                    for k in (
                        "action",
                        "proposed_action",
                        "selector",
                        "trained_updates",
                        "created",
                    )
                }
                self.store.put("tool_incident", incident, row)
                self.store.event(
                    "tool_incident_policy_selection",
                    {
                        "incident": incident,
                        **row["last_policy_action"],
                        "neural_policy_qualified": False,
                    },
                    (
                        "neural_policy"
                        if decision["selector"] == "neural_policy"
                        else "runtime"
                    ),
                )

    def summary(self):
        return {
            "recent": self.store.list("tool_incident", limit=8),
            "automatic_weight_updates_from_observation": False,
            "semantic_root_cause_qualified": False,
            "audit_survives_neural_rollback": True,
        }

    def feedback(self, incident, helpful, learning=None, skipped=False):
        with self.lock:
            row = self.store.get("tool_incident", incident)
            if row:
                row["status"] = (
                    "owner_skipped"
                    if skipped
                    else (
                        "owner_guidance_received"
                        if helpful is True
                        else (
                            "deferred_after_negative_feedback"
                            if helpful is False
                            else "owner_guidance_received_unrated"
                        )
                    )
                )
                row["last_feedback"] = {
                    "kind": "owner_skipped" if skipped else "explicit_owner_feedback",
                    "helpful": helpful,
                    "learning": learning,
                    "factual_cause_verified": False,
                }
                self.store.put("tool_incident", incident, row)
                self.store.event(
                    "tool_incident_feedback",
                    {"incident": incident, **row["last_feedback"]},
                    "user",
                )
