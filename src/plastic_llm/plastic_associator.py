"""Plastic Memory: Online Associative Synaptic Memory with Eligibility Traces.

Research Component:
- Extracts deterministic sparse context features x in R^D (D=128).
- Maintains plastic synaptic association matrix W in R^{K x D} between task features and skills.
- Eligibility traces: e(t) = lambda * e(t-1) + x(t).
- Online three-factor update: W(t+1) = clip(W(t) + eta * verified_reward * e(t), -W_max, W_max).
- Weight decay and L2-norm bounding prevents catastrophic growth.
- Zero mutation of Qwen weights; modifies external associative plasticity only.
"""
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import json
import numpy as np


class ContextFeatureExtractor:
    """Maps task prompts and query strings into a normalized sparse feature vector."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def extract(self, text: str) -> np.ndarray:
        clean = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
        tokens = [t for t in clean.split() if len(t) > 1]
        vec = np.zeros(self.dim, dtype=np.float32)

        for tok in tokens:
            # 1-gram hash
            h1 = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h1] += 1.0

        # Bi-gram hashes for sequential context
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]}_{tokens[i+1]}"
            h2 = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h2] += 1.5

        # L2 normalize
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec


class PlasticAssociativeMemory:
    """Biologically-inspired online associative learner for skill and memory selection."""

    def __init__(
        self,
        skill_ids: List[str],
        feature_dim: int = 128,
        eta: float = 0.20,           # Fast online learning rate
        trace_decay: float = 0.85,    # Eligibility trace decay (lambda)
        weight_decay: float = 0.995,  # Experience staleness decay (gamma)
        max_weight: float = 2.5,     # Hard clipping limit
        seed: int = 42,
    ):
        self.skill_ids = sorted(list(skill_ids))
        self.skill_to_idx = {s: i for i, s in enumerate(self.skill_ids)}
        self.k = len(self.skill_ids)
        self.dim = feature_dim
        self.eta = eta
        self.trace_decay = trace_decay
        self.weight_decay = weight_decay
        self.max_weight = max_weight
        self.seed = seed

        # Synaptic association matrix W in R^{K x D}
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02, (self.k, self.dim)).astype(np.float32)

        # Eligibility traces E in R^{K x D}
        self.traces = np.zeros((self.k, self.dim), dtype=np.float32)

        self.extractor = ContextFeatureExtractor(dim=feature_dim)
        self.total_updates = 0
        self.last_active_skill: Optional[str] = None
        self.last_features: Optional[np.ndarray] = None

    def get_skill_affinities(self, prompt: str) -> Dict[str, float]:
        """Compute associative affinity bias for each skill based on current context."""
        x = self.extractor.extract(prompt)
        # Association logits = W @ x
        scores = self.w @ x
        return {skill: float(scores[i]) for i, skill in enumerate(self.skill_ids)}

    def record_decision(self, prompt: str, selected_skill: str) -> None:
        """Step eligibility trace forward for the chosen skill."""
        if selected_skill not in self.skill_to_idx:
            return
        x = self.extractor.extract(prompt)
        idx = self.skill_to_idx[selected_skill]

        # Decay all traces
        self.traces *= self.trace_decay
        # Accumulate active features onto chosen skill's eligibility trace
        self.traces[idx] += x

        self.last_active_skill = selected_skill
        self.last_features = x

    def update_from_feedback(self, verified_reward: float) -> Dict[str, Any]:
        """Three-factor learning update: Delta W = eta * reward * traces."""
        r = float(np.clip(verified_reward, -1.0, 1.0))
        
        # Hebbian trace update
        delta = self.eta * r * self.traces
        self.w = self.w * self.weight_decay + delta
        np.clip(self.w, -self.max_weight, self.max_weight, out=self.w)

        # Trace reset on definitive outcome
        self.traces *= 0.5
        self.total_updates += 1

        return {
            "reward": r,
            "norm_w": float(np.linalg.norm(self.w)),
            "max_abs_w": float(np.max(np.abs(self.w))),
            "total_updates": self.total_updates,
        }

    def reset_traces(self):
        self.traces.fill(0.0)

    def save_state(self) -> Dict[str, Any]:
        return {
            "skill_ids": self.skill_ids,
            "dim": self.dim,
            "eta": self.eta,
            "total_updates": self.total_updates,
            "w": self.w.tolist(),
        }

    def load_state(self, state: Dict[str, Any]):
        self.skill_ids = state["skill_ids"]
        self.skill_to_idx = {s: i for i, s in enumerate(self.skill_ids)}
        self.k = len(self.skill_ids)
        self.dim = state["dim"]
        self.eta = state["eta"]
        self.total_updates = state["total_updates"]
        self.w = np.array(state["w"], dtype=np.float32)
        self.traces = np.zeros_like(self.w)


class SimpleFeedbackRanker:
    """Variant F: Standard Exponential Moving Average (EMA) success-rate tracker (Non-neural bandit baseline)."""

    def __init__(self, skill_ids: List[str], alpha: float = 0.15):
        self.skill_ids = sorted(list(skill_ids))
        self.alpha = alpha
        # Success rate tracker initialized at 0.5 neutral prior
        self.success_rates = {s: 0.5 for s in self.skill_ids}
        self.trial_counts = {s: 0 for s in self.skill_ids}

    def get_skill_affinities(self, prompt: str) -> Dict[str, float]:
        # Return success rate as priority bias
        return {s: float(self.success_rates[s] - 0.5) for s in self.skill_ids}

    def record_decision(self, prompt: str, selected_skill: str) -> None:
        if selected_skill in self.trial_counts:
            self.trial_counts[selected_skill] += 1

    def update_from_feedback(self, selected_skill: str, verified_reward: float) -> None:
        if selected_skill in self.success_rates:
            success_val = 1.0 if verified_reward > 0 else 0.0
            self.success_rates[selected_skill] = (
                (1.0 - self.alpha) * self.success_rates[selected_skill]
                + self.alpha * success_val
            )
