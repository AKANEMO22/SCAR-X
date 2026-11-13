"""Contextual Linear Classifier / Contextual Bandit Baseline (Variant G).

Serves as the fair, classical machine learning baseline:
- Uses the EXACT same sparse D=128 bi-gram context feature extractor as PlasticAssociativeMemory.
- Online regularized Multi-Class Logistic / Softmax Policy Gradient update.
- Enables direct empirical comparison: Does biologically-inspired eligibility-trace plasticity
  offer any tangible benefit over standard contextual linear routing?
"""
from typing import Dict, Any, List, Optional
import numpy as np

from plastic_llm.plastic_associator import ContextFeatureExtractor


class ContextualLinearRouter:
    """Standard Online Contextual Softmax Classifier / Linear Bandit."""

    def __init__(
        self,
        skill_ids: List[str],
        feature_dim: int = 128,
        learning_rate: float = 0.20,
        l2_reg: float = 0.005,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        self.skill_ids = sorted(list(skill_ids))
        self.skill_to_idx = {s: i for i, s in enumerate(self.skill_ids)}
        self.k = len(self.skill_ids)
        self.dim = feature_dim
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.temperature = temperature

        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02, (self.k, self.dim)).astype(np.float32)
        self.extractor = ContextFeatureExtractor(dim=feature_dim)
        self.last_x: Optional[np.ndarray] = None
        self.last_action_idx: Optional[int] = None
        self.total_updates = 0

    def get_skill_affinities(self, prompt: str) -> Dict[str, float]:
        """Compute linear logits for each skill."""
        x = self.extractor.extract(prompt)
        logits = self.w @ x
        return {skill: float(logits[i]) for i, skill in enumerate(self.skill_ids)}

    def record_decision(self, prompt: str, selected_skill: str) -> None:
        """Cache current features and decision for online gradient step."""
        if selected_skill in self.skill_to_idx:
            self.last_x = self.extractor.extract(prompt)
            self.last_action_idx = self.skill_to_idx[selected_skill]

    def update_from_feedback(self, selected_skill: str, verified_reward: float) -> None:
        """Online softmax policy-gradient / logistic update."""
        if self.last_x is None or self.last_action_idx is None:
            return

        x = self.last_x
        action_idx = self.last_action_idx

        # Compute softmax probabilities
        logits = (self.w @ x) / max(1e-4, self.temperature)
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        # Target reward: +1.0 for success, -1.0 for failure
        r = np.clip(verified_reward, -1.0, 1.0)

        # Gradient update for selected action vs non-selected actions
        grad = np.zeros_like(self.w)
        for i in range(self.k):
            if i == action_idx:
                grad[i] = r * (1.0 - probs[i]) * x
            else:
                grad[i] = -r * probs[i] * x

        # Step with L2 weight decay
        self.w = (1.0 - self.l2_reg) * self.w + self.lr * grad
        self.total_updates += 1
