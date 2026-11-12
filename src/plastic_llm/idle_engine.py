"""Autonomous Idle Learning Engine: Self-directed practice, verification, and memory consolidation."""
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import copy
from pathlib import Path

from plastic_llm.memory_store import MemoryStore
from plastic_llm.skill_registry import SkillRegistry
from plastic_llm.plastic_associator import PlasticAssociativeMemory


class IdleLearningEngine:
    """Manages background self-rehearsal, skill synthesis, and knowledge consolidation."""

    def __init__(
        self,
        memory_store: MemoryStore,
        skill_registry: SkillRegistry,
        plastic_memory: Optional[PlasticAssociativeMemory] = None,
        max_practice_tasks_per_cycle: int = 5,
    ):
        self.memory = memory_store
        self.skills = skill_registry
        self.plastic = plastic_memory
        self.max_practice = max_practice_tasks_per_cycle
        self.total_idle_cycles = 0
        self.consolidated_skills = 0
        self.total_idle_duration_sec = 0.0

    def run_idle_cycle(
        self,
        practice_curriculum: List[Dict[str, Any]],
        oracle_validator: Callable[[str, Dict[str, Any]], bool],
    ) -> Dict[str, Any]:
        """Execute one autonomous idle rehearsal cycle."""
        t0 = time.perf_counter()
        self.total_idle_cycles += 1

        verified_rehearsals = 0
        failed_rehearsals = 0
        learned_items = []

        # Rehearse up to max_practice tasks from curriculum
        candidates = practice_curriculum[:self.max_practice]

        for item in candidates:
            task_id = item.get("task_id", "idle_task")
            prompt = item.get("prompt", "")
            target_skill = item.get("expected_skill", "")
            params = item.get("params", {})
            ground_truth = item.get("ground_truth", None)

            skill = self.skills.get(target_skill)
            if not skill:
                continue

            # Execute in sandbox
            res = skill.execute(params)
            is_valid = oracle_validator(prompt, res)

            reward = 1.0 if is_valid else -1.0

            # Update episodic store
            self.memory.add_episode(
                task_id=f"idle_{task_id}",
                prompt=prompt,
                selected_skill=target_skill,
                params=params,
                result=res,
                verified_reward=reward,
                error=res.get("error", None),
            )

            # If plastic memory is present, provide reinforcement
            if self.plastic is not None:
                self.plastic.record_decision(prompt, target_skill)
                self.plastic.update_from_feedback(reward)

            if is_valid:
                verified_rehearsals += 1
                # Consolidate into semantic knowledge
                rule_desc = f"For task '{prompt}', skill '{target_skill}' produces valid verified outcome: {res.get('result', res.get('stdout', ''))}"
                self.memory.add_semantic_knowledge(
                    key=f"skill_usage_{target_skill}_{task_id}",
                    category="rehearsed_skill_rule",
                    content=rule_desc,
                    confidence=1.0,
                    verified=True,
                )
                learned_items.append({"task_id": task_id, "skill": target_skill, "success": True})
            else:
                failed_rehearsals += 1

        duration = time.perf_counter() - t0
        self.total_idle_duration_sec += duration
        self.consolidated_skills += verified_rehearsals

        return {
            "cycle": self.total_idle_cycles,
            "tasks_practiced": len(candidates),
            "verified_rehearsals": verified_rehearsals,
            "failed_rehearsals": failed_rehearsals,
            "duration_sec": duration,
            "learned_items": learned_items,
        }

    def run_idle_session(
        self,
        num_cycles: int = 2,
        curriculum: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run multiple autonomous idle rehearsal cycles with synthetic self-generated drills."""
        cycle_summaries = []

        # Build self-directed practice curriculum if none provided
        if not curriculum:
            curriculum = self._generate_synthetic_drills()

        def default_validator(prompt: str, res: Dict[str, Any]) -> bool:
            return res.get("success", False) is True

        for c in range(num_cycles):
            start_idx = (c * self.max_practice) % max(1, len(curriculum))
            cycle_tasks = curriculum[start_idx : start_idx + self.max_practice]
            res = self.run_idle_cycle(cycle_tasks, default_validator)
            cycle_summaries.append(res)

        return {
            "total_cycles_run": num_cycles,
            "total_practiced": sum(c["tasks_practiced"] for c in cycle_summaries),
            "total_verified": sum(c["verified_rehearsals"] for c in cycle_summaries),
            "total_consolidated": self.consolidated_skills,
            "cycle_details": cycle_summaries,
        }

    def _generate_synthetic_drills(self) -> List[Dict[str, Any]]:
        """Synthesize verifiable practice tasks across skill categories for self-rehearsal."""
        return [
            {
                "task_id": "idle_math_01",
                "prompt": "Evaluate mathematical expression 12 * 8",
                "expected_skill": "code_execution",
                "params": {"code": "result = 12 * 8"},
            },
            {
                "task_id": "idle_stat_01",
                "prompt": "Analyze numeric series for dataset: values [10, 20, 30]. Compute arithmetic mean.",
                "expected_skill": "data_analysis",
                "params": {"rows": [{"value": 10}, {"value": 20}, {"value": 30}], "key": "value"},
            },
            {
                "task_id": "idle_debug_01",
                "prompt": "Fix bug in Python function: 'def sub(a, b): return a + b' such that result is 5.",
                "expected_skill": "python_debug",
                "params": {"broken_code": "def sub(a, b): return a + b", "fix_patch": "def sub(a, b): return a - b\nresult = sub(10, 5)"},
            },
            {
                "task_id": "idle_unit_01",
                "prompt": "Execute test suite verification: check if square of 4 is 16.",
                "expected_skill": "unit_test",
                "params": {"function_code": "def sq(x): return x * x", "test_code": "assert sq(4) == 16"},
            },
            {
                "task_id": "idle_doc_01",
                "prompt": "Read documentation to retrieve and state the verified endpoint for service API_ENDPOINT_V0.",
                "expected_skill": "document_read",
                "params": {"path": "docs/api_info.txt"},
            },
        ]
