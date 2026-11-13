"""Strict Task Data Contracts: Enforces separation between public inputs and private evaluation data.

Prevents benchmark answer leakage (P0-A):
- PublicTaskInput: Sent to agent. Contains ONLY prompt and user-facing context.
  Zero ground truth, zero expected skill, zero reference parameters, zero fix patches.
- PrivateEvaluationMetadata: Kept exclusively by the evaluator.
  Contains ground truth, expected skill, oracle verification function.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List


@dataclass(frozen=True)
class PublicTaskInput:
    """Public task input visible to the agent and model planner."""
    task_id: str
    prompt: str
    category: str
    public_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "category": self.category,
            "public_context": self.public_context,
        }


@dataclass(frozen=True)
class PrivateEvaluationMetadata:
    """Private benchmark ground truth, strictly isolated from the agent."""
    task_id: str
    category: str
    expected_skill: str
    ground_truth: Any
    reference_params: Optional[Dict[str, Any]] = None
    reference_solution: Optional[str] = None
    test_assertions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "expected_skill": self.expected_skill,
            "ground_truth": self.ground_truth,
            "reference_params": self.reference_params,
            "reference_solution": self.reference_solution,
            "test_assertions": self.test_assertions,
        }
