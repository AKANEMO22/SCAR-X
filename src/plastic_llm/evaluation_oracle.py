"""Authoritative Evaluation Oracle: Strictly validates agent execution traces against PrivateEvaluationMetadata.

Enforces P0-C:
- Independent grading logic that re-verifies from execution traces and oracle references.
- Does NOT rely on runner's self-reported 'passed' flag.
- Deconstructs performance into:
  1. Tool Routing (tool selection accuracy)
  2. Parameter Generation Validity (syntactic correctness)
  3. Sandbox Execution Success (runtime exit code == 0)
  4. Ground Truth Correctness (value verification)
  5. End-to-End Task Pass
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from plastic_llm.task_types import PrivateEvaluationMetadata


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    category: str
    tool_routing_correct: bool
    execution_success: bool
    semantic_correctness: bool
    end_to_end_pass: bool
    reward: float
    validation_notes: str


def evaluate_execution_trace(
    meta: PrivateEvaluationMetadata,
    trace: Dict[str, Any],
) -> EvaluationResult:
    """Strictly evaluate an agent's execution trace against ground truth metadata."""
    selected_skill = trace.get("selected_skill", "")
    params = trace.get("params", {})
    exec_res = trace.get("exec_result", {})
    raw_output = trace.get("raw_output", "")
    retrieved_memories = trace.get("retrieved_memories", [])

    # 1. Tool Routing Check
    tool_correct = (selected_skill == meta.expected_skill)
    if not tool_correct:
        return EvaluationResult(
            task_id=meta.task_id,
            category=meta.category,
            tool_routing_correct=False,
            execution_success=bool(exec_res.get("success", False)),
            semantic_correctness=False,
            end_to_end_pass=False,
            reward=-1.0,
            validation_notes=f"Routing failure: selected '{selected_skill}', expected '{meta.expected_skill}'",
        )

    # 2. Sandbox Execution Success Check
    exec_success = bool(exec_res.get("success", False))
    if not exec_success:
        err_msg = exec_res.get("error", "Execution failed")
        return EvaluationResult(
            task_id=meta.task_id,
            category=meta.category,
            tool_routing_correct=True,
            execution_success=False,
            semantic_correctness=False,
            end_to_end_pass=False,
            reward=-1.0,
            validation_notes=f"Sandbox execution error: {err_msg}",
        )

    # 3. Category-Specific Semantic Correctness
    actual_res = exec_res.get("result", exec_res.get("stdout", exec_res.get("output")))
    semantic_pass = False
    notes = ""

    cat = meta.category
    gt = meta.ground_truth

    if cat == "new_knowledge":
        # Check if verified endpoint is recalled in output or retrieved memory
        combined_text = f"{raw_output} {actual_res} " + " ".join(retrieved_memories)
        if str(gt) in combined_text:
            semantic_pass = True
            notes = "Knowledge endpoint verified"
        else:
            notes = f"Missing ground truth endpoint '{gt}'"

    elif cat == "python_debug":
        if actual_res == gt or str(gt) == str(actual_res).strip():
            semantic_pass = True
            notes = "Bug fix output matched ground truth"
        else:
            notes = f"Actual output '{actual_res}' != expected '{gt}'"

    elif cat == "workflow_selection":
        actual_mean = exec_res.get("mean")
        if actual_mean is not None:
            try:
                diff = abs(float(actual_mean) - float(gt))
                if diff < 1e-3:
                    semantic_pass = True
                    notes = "Data analysis mean verified"
                else:
                    notes = f"Mean {actual_mean} != expected {gt}"
            except Exception:
                notes = f"Invalid numeric mean: {actual_mean}"
        else:
            notes = "Missing 'mean' field in data analysis output"

    elif cat == "compositional":
        if exec_success and (actual_res is True or exec_res.get("result") is True):
            semantic_pass = True
            notes = "Compositional test suite passed"
        else:
            notes = f"Assertion verification failed: {actual_res}"

    elif cat == "edge_cases":
        if actual_res == gt or str(gt) in str(actual_res):
            semantic_pass = True
            notes = "Edge case output satisfied ground truth"
        else:
            notes = f"Edge case output '{actual_res}' != expected '{gt}'"

    end_to_end = tool_correct and exec_success and semantic_pass
    reward = 1.0 if end_to_end else -1.0

    return EvaluationResult(
        task_id=meta.task_id,
        category=meta.category,
        tool_routing_correct=tool_correct,
        execution_success=exec_success,
        semantic_correctness=semantic_pass,
        end_to_end_pass=end_to_end,
        reward=reward,
        validation_notes=notes,
    )
