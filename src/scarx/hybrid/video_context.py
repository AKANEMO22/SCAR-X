"""Out-of-band video metadata; never encode routing in a human message.

Known chats bind by their Hermes session identity. A first-turn draft has no
runtime identity yet: consume exactly one short-lived matching upload ticket.
Ambiguous equal-text drafts fail closed rather than choosing another video's
context. Legacy renderer storage and scarx:// links never create a binding.
"""

import hashlib
import re
import threading
import time
import uuid


def text_digest(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def last_user_text(messages):
    user = next(
        (
            m
            for m in reversed(messages)
            if isinstance(m, dict) and m.get("role") == "user"
        ),
        {},
    )
    content = user.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


class VideoContext:
    def __init__(self, store, clock=time.time):
        self.store = store
        self.clock = clock
        self.pending = {}
        self.lock = threading.RLock()

    def queue(self, stream_id, text_sha256, session_ids=(), ticket_id=None):
        if not isinstance(stream_id, str) or not re.fullmatch(
            "[a-f0-9]{32}", stream_id
        ):
            raise ValueError("Invalid stream identity")
        stream = self.store.get("stream", stream_id)
        if not stream or stream.get("kind") == "screen":
            raise ValueError("An explicit uploaded video is required")
        if not isinstance(text_sha256, str) or not re.fullmatch(
            "[a-f0-9]{64}", text_sha256
        ):
            raise ValueError("Invalid turn digest")
        if (
            not isinstance(session_ids, (list, tuple))
            or len(session_ids) > 2
            or any(
                not isinstance(s, str) or not 1 <= len(s) <= 160 or s == "draft"
                for s in session_ids
            )
        ):
            raise ValueError("Invalid session metadata")
        ticket_id = ticket_id or uuid.uuid4().hex
        if not isinstance(ticket_id, str) or not re.fullmatch(
            "[a-f0-9-]{16,64}", ticket_id
        ):
            raise ValueError("Invalid upload ticket")
        item = {
            "id": ticket_id,
            "stream_id": stream_id,
            "text_sha256": text_sha256,
            "session_ids": list(session_ids),
            "expires": self.clock() + 300,
        }
        with self.lock:
            self._prune()
            old = self.pending.get(ticket_id)
            if old:
                if any(
                    old[k] != item[k]
                    for k in ("stream_id", "text_sha256", "session_ids")
                ):
                    raise ValueError("Upload ticket changed")
                return {"ticket_id": ticket_id, "queued": True}
            # A resubmitted edited draft supersedes the unconsumed upload
            # metadata for this stream; its abandoned text cannot bind later.
            self.pending = {
                key: row
                for key, row in self.pending.items()
                if row["stream_id"] != stream_id
            }
            if len(self.pending) >= 16:
                raise ValueError("Pending video context is full")
            self.pending[ticket_id] = item
        self.store.event(
            "video_context_queued",
            {
                "ticket_id": ticket_id,
                "stream_id": stream_id,
                "session_known": bool(session_ids),
                "user_text_modified": False,
            },
        )
        return {"ticket_id": ticket_id, "queued": True}

    def _prune(self):
        now = self.clock()
        self.pending = {
            key: item for key, item in self.pending.items() if item["expires"] > now
        }

    def resolve(self, messages, session_id):
        if (
            not isinstance(session_id, str)
            or not session_id
            or len(session_id) > 160
            or session_id == "local"
        ):
            return None
        digest = text_digest(last_user_text(messages))
        with self.lock:
            self._prune()
            matched = [
                item
                for item in self.pending.values()
                if item["text_sha256"] == digest
                and (not item["session_ids"] or session_id in item["session_ids"])
            ]
            if len(matched) > 1:
                raise ValueError(
                    "Ambiguous video uploads for equal-text drafts; please attach again in the intended chat"
                )
            if matched:
                item = matched[0]
                binding = {
                    "stream_id": item["stream_id"],
                    "session_id": session_id,
                    "expires": self.clock() + 12 * 3600,
                    "origin": "explicit_upload_metadata_v2",
                }
                self.store.put("video_context_v2", session_id, binding)
                del self.pending[item["id"]]
                self.store.event(
                    "video_context_bound",
                    {
                        "stream_id": binding["stream_id"],
                        "session_id": session_id,
                        "origin": binding["origin"],
                        "user_text_modified": False,
                    },
                )
            else:
                binding = self.store.get("video_context_v2", session_id)
            if (
                not binding
                or binding.get("origin") != "explicit_upload_metadata_v2"
                or binding["expires"] <= self.clock()
            ):
                return None
            if not self.store.get("stream", binding["stream_id"]):
                return None
            return binding["stream_id"]
