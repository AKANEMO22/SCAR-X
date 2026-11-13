"""Persistent scheduling, bounded real work and preemptible wake/sleep lifecycle."""

import copy
import json
import threading
import time

PHASES = (
    "starting",
    "serving",
    "observing",
    "reflecting",
    "verifying",
    "exploring",
    "sleep_consolidating",
    "resting",
    "awake_idle",
    "paused",
    "waking",
)


def initial_cognition(last_activity=0.0):
    return {
        "format": "scarx-cognition-v1",
        "intent": "auto",
        "last_activity": float(last_activity),
        "last_quantum": 0.0,
        "next_quantum": 0.0,
        "next_replay": 0.0,
        "replay_revision": 0,
        "replay_used": 0,
        "cycles": 0,
        "reflections": 0,
        "consolidations": 0,
        "explorations": 0,
        "wakeups": 0,
        "rests": 0,
        "last_outcome": None,
    }


def validate_cognition(value):
    if (
        not isinstance(value, dict)
        or set(value) != set(initial_cognition())
        or value["format"] != "scarx-cognition-v1"
        or value["intent"] not in ("auto", "awake", "sleep")
    ):
        raise ValueError("Incompatible cognitive scheduler state")
    for key in ("last_activity", "last_quantum", "next_quantum", "next_replay"):
        if type(value[key]) not in (int, float) or not 0 <= value[key] < 1e13:
            raise ValueError("Invalid cognitive clock")
    for key in (
        "replay_revision",
        "replay_used",
        "cycles",
        "reflections",
        "consolidations",
        "explorations",
        "wakeups",
        "rests",
    ):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("Invalid cognitive counter")
    if value["last_outcome"] is not None and (
        not isinstance(value["last_outcome"], dict)
        or len(json.dumps(value["last_outcome"]).encode()) > 16000
    ):
        raise ValueError("Invalid bounded cognitive outcome")


class CognitiveLoop:
    def __init__(self, engine):
        self.engine = engine
        self.phase = "starting"
        self.reason = "startup; execution leases are not resumed from disk"
        self.since = time.time()
        self.request_pending = threading.Event()
        self.wake_event = threading.Event()
        self.request_lock = threading.Lock()
        self.requests = 0
        self.work_active = threading.Event()

    @property
    def data(self):
        return self.engine.state["cognition"]

    def transition(self, phase, reason):
        if phase not in PHASES:
            raise ValueError("Unknown cognitive phase")
        if (phase, reason) == (self.phase, self.reason):
            return
        previous = self.phase
        self.phase, self.reason, self.since = phase, reason, time.time()
        self.engine.store.event(
            "cognitive_phase",
            {
                "previous": previous,
                "phase": phase,
                "reason": reason,
                "intent": self.data["intent"],
                "selector": "runtime scheduler, not a learned ACT policy",
            },
        )

    def begin_foreground(self):
        e = self.engine
        # Signal BEFORE waiting for a micro-update to finish, then claim its
        # exclusive learner boundary. Multiple parallel requests keep the lease.
        with self.request_lock:
            self.requests += 1
            self.request_pending.set()
            if hasattr(e.backend, "cancel_idle"):
                e.backend.cancel_idle.set()
        self.wake_event.set()
        try:
            with e.lock:
                e.foreground.set()
                self.data["last_activity"] = time.time()
                if self.data["intent"] == "sleep":
                    self.data["intent"] = "auto"
                self.data["wakeups"] += 1
                self.transition(
                    "serving", "foreground request; background updates blocked"
                )
        except Exception:
            with self.request_lock:
                self.requests -= 1
                if not self.requests:
                    e.foreground.clear()
                    self.request_pending.clear()
                    if hasattr(e.backend, "cancel_idle"):
                        e.backend.cancel_idle.clear()
            raise

    def end_foreground(self):
        e = self.engine
        with e.lock:
            with self.request_lock:
                if self.requests < 1:
                    raise RuntimeError("Unbalanced foreground lease")
                self.requests -= 1
                if not self.requests:
                    e.foreground.clear()
                    self.request_pending.clear()
                    if hasattr(e.backend, "cancel_idle"):
                        e.backend.cancel_idle.clear()
                    self.data["last_activity"] = time.time()
                    self.transition(
                        "awake_idle", "foreground completed; quiet timer restarted"
                    )
            e.save()
        self.wake_event.set()

    def command(self, intent):
        if intent not in ("auto", "awake", "sleep"):
            raise ValueError("Expected auto, awake or sleep")
        with self.engine.lock:
            self.data["intent"] = intent
            if intent != "sleep":
                self.data["last_activity"] = time.time()
            self.engine.save()
            self.engine.store.event(
                "cognitive_intent",
                {
                    "intent": intent,
                    "source": "explicit local control",
                    "queued_behind_foreground": self.request_pending.is_set(),
                },
                "user",
            )
        self.wake_event.set()
        return self.summary()

    def capture_active(self):
        callback = getattr(self.engine, "capture_active", None)
        return bool(callback and callback())

    def resident(self):
        process = getattr(self.engine.backend, "vlm", None)
        return bool(process is not None and process.poll() is None)

    def sleep_permitted(self):
        e = self.engine
        limit = e.config.effective_soft_ram_limit
        resource = e.monitor.sample() if limit is not None else {}
        pressure = (
            limit is not None
            and resource.get("conservative_ram_bytes", resource.get("rss_bytes", 0))
            > limit
        )
        return (
            not self.request_pending.is_set()
            and not e.foreground.is_set()
            and not self.capture_active()
            and (self.data["intent"] != "awake" or pressure)
            and (
                self.data["intent"] == "sleep"
                or pressure
                or time.time() - self.data["last_activity"]
                >= e.config.review_quiet_seconds
            )
        )

    def replay_ready(self, now):
        e = self.engine
        d = self.data
        revision = e.model.replay_seen
        used = d["replay_used"] if revision == d["replay_revision"] else 0
        return bool(
            e.model.replay
            and used < e.config.sleep_replay_budget
            and now >= d["next_replay"]
        )

    def consolidate_once(self, now=None):
        now = time.time() if now is None else now
        e = self.engine
        with e.lock:
            if (
                self.request_pending.is_set()
                or e.foreground.is_set()
                or not e.state["learning_enabled"]
                or not self.replay_ready(now)
            ):
                return None
            previous = e.model.dumps(
                e.state
            )  # BEFORE choosing replay, so RNG rolls back too.
            row = copy.deepcopy(
                e.model.replay[int(e.model.rng.integers(len(e.model.replay)))]
            )

            def commit(result):
                d = self.data
                if d["replay_revision"] != e.model.replay_seen:
                    d["replay_used"] = 0
                d["replay_revision"] = e.model.replay_seen
                d["replay_used"] += 1
                d["next_replay"] = now + e.config.consolidation_interval_seconds
                d["consolidations"] += 1
                d["last_outcome"] = {
                    "kind": "verified_replay",
                    "accepted": result.get("accepted", False),
                    "source": row["evidence"],
                }
                e.state["review"]["replay_updates"] += int(
                    result.get("accepted", False)
                )
                e.state["last_review"] = now

            result = e._guarded_learn(
                row["sequence"],
                row["target"],
                row["head"],
                row["reward"],
                row["evidence"],
                remember=False,
                transaction_snapshot=previous,
                update_controller=commit,
            )
            e.store.event(
                "sleep_replay_quantum",
                {
                    "source": row["evidence"],
                    "learning": result,
                    "budget_used": self.data["replay_used"],
                    "budget_max": e.config.sleep_replay_budget,
                    "new_self_label": False,
                },
            )
            return result

    def choose_work(self, now):
        e = self.engine
        d = self.data
        if self.request_pending.is_set() or e.foreground.is_set():
            return "serving"
        if self.capture_active():
            return "observing"
        if e.config.effective_soft_ram_limit is not None:
            r = e.monitor.sample()
            if (
                r.get("conservative_ram_bytes", r["rss_bytes"])
                > e.config.effective_soft_ram_limit
            ):
                return "resting"
        quiet = now - d["last_activity"]
        if d["intent"] == "sleep":
            return (
                "sleep_consolidating"
                if e.state["learning_enabled"] and self.replay_ready(now)
                else "resting"
            )
        if d["intent"] == "awake" and not self.resident():
            return "waking"
        if not e.state["learning_enabled"]:
            return "paused"
        if quiet < e.config.review_quiet_seconds:
            return "awake_idle"
        if now < d["next_quantum"]:
            return "awake_idle"
        if e.state["review"]["pending"] and e.review_blocker(now) is None:
            return "reflecting"
        if any(g["status"] == "open" for g in e.state["goals"].values()):
            return "exploring"
        if quiet >= e.config.consolidation_after_seconds and self.replay_ready(now):
            return "sleep_consolidating"
        if d["intent"] == "auto" and quiet >= e.config.rest_after_seconds:
            return "resting"
        return "awake_idle"

    def quantum(self, now=None):
        e = self.engine
        now = time.time() if now is None else now
        with e.lock:
            work = self.choose_work(now)
            reason = {
                "serving": "foreground owns the learner",
                "observing": "video/screen has priority",
                "paused": "learning disabled",
                "resting": "sleep requested or no actionable verified work",
                "awake_idle": "quiet/cadence/budget wait; no invented training task",
                "reflecting": "review an actual stored answer",
                "exploring": "advance an existing unresolved goal",
                "sleep_consolidating": "bounded replay of accepted sources",
                "waking": "explicit awake intent",
            }[work]
            self.transition(work, reason)
        if work in ("serving", "observing", "paused", "awake_idle"):
            return None
        self.work_active.set()
        try:
            return self.execute(work, now)
        finally:
            self.work_active.clear()

    def execute(self, work, now):
        e = self.engine
        started = time.monotonic()
        result = None
        if work == "resting":
            if (
                e.config.release_vlm_on_rest
                and self.resident()
                and hasattr(e.backend, "hibernate")
            ):
                result = e.backend.hibernate(
                    "cognitive rest; ALIF/evidence/checkpoint retained",
                    permitted=self.sleep_permitted,
                )
                with e.lock:
                    self.data["rests"] += int(bool(result))
                    e.save()
            return result
        if work == "waking":
            e.backend.wake()
        elif work == "reflecting":
            result = e.review_once()
        elif work == "exploring":
            result = e.step()
        else:
            if (
                e.config.release_vlm_on_rest
                and self.resident()
                and hasattr(e.backend, "hibernate")
            ):
                if self.data["intent"] != "awake":
                    released = e.backend.hibernate(
                        "sleep consolidation uses ALIF only",
                        permitted=self.sleep_permitted,
                    )
                    with e.lock:
                        self.data["rests"] += int(bool(released))
                        e.save()
            result = self.consolidate_once(now)
        with e.lock:
            self.data["cycles"] += 1
            self.data["last_quantum"] = time.time()
            # Preserve the wall-duty bound even when UI events wake the scheduler.
            self.data["next_quantum"] = time.time() + max(
                1.0, (time.monotonic() - started) * (1 / e.config.idle_cpu_fraction - 1)
            )
            if work == "reflecting":
                self.data["reflections"] += int(result is not None)
            if work == "exploring":
                self.data["explorations"] += int(result is not None)
            e.save()
        return result

    def start(self):
        def loop():
            while not self.engine.stop_event.is_set():
                try:
                    self.engine.monitor.check()
                    self.quantum()
                except InterruptedError:
                    self.engine.store.event(
                        "cognitive_quantum_preempted",
                        {"foreground": self.request_pending.is_set()},
                    )
                except Exception as exc:
                    self.engine.store.event(
                        "cognitive_quantum_failed",
                        {"error": type(exc).__name__ + ": " + str(exc)},
                    )
                self.wake_event.wait(1)
                self.wake_event.clear()

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        self.engine.thread = thread

    def summary(self):
        return {
            "phase": self.phase,
            "reason": self.reason,
            "since": self.since,
            "foreground_requests": self.requests,
            "VLM_resident": self.resident(),
            "capture_active": self.capture_active(),
            "state": copy.deepcopy(self.data),
            "general_ACT": "not implemented/qualified",
            "LoRA_auto": "not qualified; disabled",
        }
