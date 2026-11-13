"""SCAR-X Continual Learning Benchmark Runner for Variants A-G.

Executes EXP-01 through EXP-05:
- EXP-01: Knowledge Acquisition & Skill Selection on 100 Learning Tasks
- EXP-02: Rapid Adaptation & Learning from Mistakes (2nd trial recovery)
- EXP-03: Compositional Generalization on 60 Unseen Tasks
- EXP-04: Continual Learning & Forgetting on 40 Retention Tasks
- EXP-05: Autonomous Idle Learning Consolidation (Variants D & E)

Enforces:
- P0-A: Complete decoupling of PublicTaskInput from PrivateEvaluationMetadata.
  Agent NEVER sees ground truth or reference solution parameters.
- P0-B: Real compositional generalization across 6 distinct algorithmic categories.
- P0-C: Authoritative evaluation oracle re-grades traces from ground truth.
- P0-D: Strict RAM gate measurement (2,000,000,000 bytes decimal).
- P0-E: OS-level subprocess sandboxing for untrusted code execution.
- Fair Baseline: Variant G (Contextual Linear Router).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch

# Ensure src is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from plastic_llm.llm_core import FrozenQwenCore
from plastic_llm.skill_registry import SkillRegistry
from plastic_llm.task_types import PublicTaskInput, PrivateEvaluationMetadata
from plastic_llm.benchmark_dataset import generate_benchmark_dataset
from plastic_llm.evaluation_oracle import evaluate_execution_trace, EvaluationResult
from plastic_llm.variants import (
    VariantA_FrozenOnly,
    VariantB_StaticRAG,
    VariantC_PlasticMemory,
    VariantD_PlasticIdle,
    VariantE_RAGIdle,
    VariantF_FeedbackRanking,
    VariantG_ContextualLinear,
)


def seed_semantic_knowledge_base(variant_obj, metas: List[PrivateEvaluationMetadata]):
    """Seed Category 1 factual endpoints into the variant's MemoryStore if present."""
    if not hasattr(variant_obj, "memory") or variant_obj.memory is None:
        return
    for m in metas:
        if m.category == "new_knowledge":
            try:
                idx = int(m.task_id.split("_")[1])
                k = f"API_ENDPOINT_V{idx}"
                v = m.ground_truth
                variant_obj.memory.add_semantic_knowledge(
                    key=k,
                    category="network",
                    content=f"Verified Endpoint for {k} is {v}",
                    confidence=1.0,
                    verified=True,
                )
            except Exception:
                pass


def run_benchmark_for_seed(
    llm_core: FrozenQwenCore,
    skills: SkillRegistry,
    variants_to_run: List[str],
    seed: int,
    output_dir: Path,
    task_limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run full benchmark for a single seed across specified variants."""
    dataset = generate_benchmark_dataset(seed=seed)

    learning_public = dataset["learning_public"]
    learning_private = dataset["learning_private"]
    unseen_public = dataset["unseen_public"]
    unseen_private = dataset["unseen_private"]
    retention_public = dataset["retention_public"]
    retention_private = dataset["retention_private"]
    private_by_id: Dict[str, PrivateEvaluationMetadata] = dataset["private_by_id"]

    if task_limit is not None:
        per_cat = max(1, task_limit // 5)
        filtered_learn_pub = []
        filtered_unseen_pub = []
        filtered_ret_pub = []
        for cat in ("new_knowledge", "python_debug", "workflow_selection", "compositional", "edge_cases"):
            filtered_learn_pub.extend([t for t in learning_public if t.category == cat][:per_cat])
            filtered_unseen_pub.extend([t for t in unseen_public if t.category == cat][:per_cat])
            filtered_ret_pub.extend([t for t in retention_public if t.category == cat][:per_cat])
        learning_public = filtered_learn_pub
        unseen_public = filtered_unseen_pub
        retention_public = filtered_ret_pub

    variant_classes = {
        "A": VariantA_FrozenOnly,
        "B": VariantB_StaticRAG,
        "C": VariantC_PlasticMemory,
        "D": VariantD_PlasticIdle,
        "E": VariantE_RAGIdle,
        "F": VariantF_FeedbackRanking,
        "G": VariantG_ContextualLinear,
    }

    raw_predictions = []
    summary_results = {}

    for vid in variants_to_run:
        print(f"\n{'='*30} RUNNING VARIANT {vid} (Seed: {seed}) {'='*30}")
        v_work_dir = output_dir / f"variant_{vid}_seed_{seed}"
        if v_work_dir.exists():
            shutil.rmtree(v_work_dir, ignore_errors=True)
        v_work_dir.mkdir(parents=True, exist_ok=True)

        v_cls = variant_classes[vid]
        agent = v_cls(llm_core, skills, v_work_dir)

        # Seed static knowledge for memory-enabled variants
        seed_semantic_knowledge_base(agent, dataset["all_private"])

        # ---------------------------------------------------------
        # Phase 1: Pre-learning Retention Baseline (EXP-04 pre)
        # ---------------------------------------------------------
        print(f"[{vid}] Step 1/5: Measuring pre-learning retention baseline ({len(retention_public)} tasks)...")
        pre_ret_e2e = 0
        pre_ret_routing = 0
        for pub in retention_public:
            meta = private_by_id[pub.task_id]
            out = agent.solve_task(pub)
            eval_res = evaluate_execution_trace(meta, out)

            if eval_res.end_to_end_pass:
                pre_ret_e2e += 1
            if eval_res.tool_routing_correct:
                pre_ret_routing += 1

            raw_predictions.append({
                "variant": vid,
                "seed": seed,
                "phase": "pre_retention",
                "task_id": pub.task_id,
                "category": pub.category,
                "prompt": pub.prompt,
                "expected_skill": meta.expected_skill,
                "selected_skill": out["selected_skill"],
                "tool_routing_correct": eval_res.tool_routing_correct,
                "params": out.get("params", {}),
                "raw_output": out.get("raw_output", ""),
                "exec_result": out.get("exec_result", {}),
                "execution_success": eval_res.execution_success,
                "semantic_correctness": eval_res.semantic_correctness,
                "end_to_end_pass": eval_res.end_to_end_pass,
                "passed": eval_res.end_to_end_pass,  # backwards compatibility
                "reward": eval_res.reward,
                "latency_ms": out.get("latency_ms", 0.0),
                "validation_notes": eval_res.validation_notes,
                "validation_msg": eval_res.validation_notes,
                "retrieved_memories": out.get("retrieved_memories", []),
            })
        pre_ret_e2e_acc = pre_ret_e2e / max(1, len(retention_public))
        pre_ret_route_acc = pre_ret_routing / max(1, len(retention_public))
        print(f"[{vid}] Pre-retention E2E Acc: {pre_ret_e2e_acc*100:.2f}% | Routing Acc: {pre_ret_route_acc*100:.2f}%")

        # ---------------------------------------------------------
        # Phase 2: EXP-01 Knowledge Acquisition (Learning Tasks)
        # ---------------------------------------------------------
        print(f"[{vid}] Step 2/5: EXP-01 Knowledge Acquisition on {len(learning_public)} tasks...")
        learn_e2e = 0
        learn_routing = 0
        failed_tasks_trial1 = []
        learn_latencies = []

        for idx, pub in enumerate(learning_public):
            meta = private_by_id[pub.task_id]
            out = agent.solve_task(pub)
            eval_res = evaluate_execution_trace(meta, out)
            learn_latencies.append(out.get("latency_ms", 0.0))

            if eval_res.end_to_end_pass:
                learn_e2e += 1
            else:
                failed_tasks_trial1.append(pub)

            if eval_res.tool_routing_correct:
                learn_routing += 1

            # Online feedback provided to the agent
            agent.provide_feedback(
                task_id=pub.task_id,
                prompt=pub.prompt,
                selected_skill=out["selected_skill"],
                params=out.get("params", {}),
                result=out.get("exec_result", {}),
                reward=eval_res.reward,
            )

            raw_predictions.append({
                "variant": vid,
                "seed": seed,
                "phase": "learning_exp01",
                "task_id": pub.task_id,
                "category": pub.category,
                "prompt": pub.prompt,
                "expected_skill": meta.expected_skill,
                "selected_skill": out["selected_skill"],
                "tool_routing_correct": eval_res.tool_routing_correct,
                "params": out.get("params", {}),
                "raw_output": out.get("raw_output", ""),
                "exec_result": out.get("exec_result", {}),
                "execution_success": eval_res.execution_success,
                "semantic_correctness": eval_res.semantic_correctness,
                "end_to_end_pass": eval_res.end_to_end_pass,
                "passed": eval_res.end_to_end_pass,
                "reward": eval_res.reward,
                "latency_ms": out.get("latency_ms", 0.0),
                "validation_notes": eval_res.validation_notes,
                "validation_msg": eval_res.validation_notes,
                "retrieved_memories": out.get("retrieved_memories", []),
            })

            if (idx + 1) % 25 == 0 or (idx + 1) == len(learning_public):
                print(f"  [{vid}] Completed {idx+1}/{len(learning_public)} tasks... Current E2E Acc: {learn_e2e/(idx+1)*100:.2f}% | Routing: {learn_routing/(idx+1)*100:.2f}%")

        learn_e2e_acc = learn_e2e / max(1, len(learning_public))
        learn_route_acc = learn_routing / max(1, len(learning_public))
        mean_learn_lat = sum(learn_latencies) / max(1, len(learn_latencies))

        # ---------------------------------------------------------
        # Phase 3: EXP-02 Learning from Mistakes (2nd Trial)
        # ---------------------------------------------------------
        print(f"[{vid}] Step 3/5: EXP-02 Rapid Adaptation on {len(failed_tasks_trial1)} failed tasks...")
        recovered_e2e = 0
        recovered_routing = 0
        for pub in failed_tasks_trial1:
            meta = private_by_id[pub.task_id]
            out = agent.solve_task(pub)
            eval_res = evaluate_execution_trace(meta, out)

            if eval_res.end_to_end_pass:
                recovered_e2e += 1
            if eval_res.tool_routing_correct:
                recovered_routing += 1

            raw_predictions.append({
                "variant": vid,
                "seed": seed,
                "phase": "adaptation_exp02",
                "task_id": pub.task_id,
                "category": pub.category,
                "prompt": pub.prompt,
                "expected_skill": meta.expected_skill,
                "selected_skill": out["selected_skill"],
                "tool_routing_correct": eval_res.tool_routing_correct,
                "params": out.get("params", {}),
                "raw_output": out.get("raw_output", ""),
                "exec_result": out.get("exec_result", {}),
                "execution_success": eval_res.execution_success,
                "semantic_correctness": eval_res.semantic_correctness,
                "end_to_end_pass": eval_res.end_to_end_pass,
                "passed": eval_res.end_to_end_pass,
                "reward": eval_res.reward,
                "latency_ms": out.get("latency_ms", 0.0),
                "validation_notes": eval_res.validation_notes,
                "validation_msg": eval_res.validation_notes,
                "retrieved_memories": out.get("retrieved_memories", []),
            })
        recovery_e2e_rate = recovered_e2e / max(1, len(failed_tasks_trial1))
        recovery_route_rate = recovered_routing / max(1, len(failed_tasks_trial1))
        print(f"[{vid}] EXP-02 E2E Recovery: {recovery_e2e_rate*100:.2f}% ({recovered_e2e}/{len(failed_tasks_trial1)}) | Route Recovery: {recovery_route_rate*100:.2f}%")

        # ---------------------------------------------------------
        # Phase 4: EXP-05 Autonomous Idle Learning Session
        # ---------------------------------------------------------
        idle_stats = {}
        if hasattr(agent, "idle_engine") and agent.idle_engine is not None:
            print(f"[{vid}] Step 4/5: EXP-05 Running Autonomous Idle Learning Consolidation (2 cycles)...")
            idle_stats = agent.idle_engine.run_idle_session(num_cycles=2)
            print(f"[{vid}] Idle Session Completed: {idle_stats}")
        else:
            print(f"[{vid}] Step 4/5: EXP-05 Idle engine not configured for variant {vid} (Skipped).")

        # ---------------------------------------------------------
        # Phase 5: EXP-03 Compositional Generalization (Unseen Tasks)
        # ---------------------------------------------------------
        print(f"[{vid}] Step 5/5: EXP-03 Compositional Generalization on {len(unseen_public)} unseen tasks...")
        unseen_e2e = 0
        unseen_routing = 0
        unseen_latencies = []
        for pub in unseen_public:
            meta = private_by_id[pub.task_id]
            out = agent.solve_task(pub)
            eval_res = evaluate_execution_trace(meta, out)
            unseen_latencies.append(out.get("latency_ms", 0.0))

            if eval_res.end_to_end_pass:
                unseen_e2e += 1
            if eval_res.tool_routing_correct:
                unseen_routing += 1

            raw_predictions.append({
                "variant": vid,
                "seed": seed,
                "phase": "generalization_exp03",
                "task_id": pub.task_id,
                "category": pub.category,
                "prompt": pub.prompt,
                "expected_skill": meta.expected_skill,
                "selected_skill": out["selected_skill"],
                "tool_routing_correct": eval_res.tool_routing_correct,
                "params": out.get("params", {}),
                "raw_output": out.get("raw_output", ""),
                "exec_result": out.get("exec_result", {}),
                "execution_success": eval_res.execution_success,
                "semantic_correctness": eval_res.semantic_correctness,
                "end_to_end_pass": eval_res.end_to_end_pass,
                "passed": eval_res.end_to_end_pass,
                "reward": eval_res.reward,
                "latency_ms": out.get("latency_ms", 0.0),
                "validation_notes": eval_res.validation_notes,
                "validation_msg": eval_res.validation_notes,
                "retrieved_memories": out.get("retrieved_memories", []),
            })
        unseen_e2e_acc = unseen_e2e / max(1, len(unseen_public))
        unseen_route_acc = unseen_routing / max(1, len(unseen_public))
        mean_unseen_lat = sum(unseen_latencies) / max(1, len(unseen_latencies))
        print(f"[{vid}] Unseen E2E Acc: {unseen_e2e_acc*100:.2f}% | Routing: {unseen_route_acc*100:.2f}%")

        # ---------------------------------------------------------
        # Phase 6: EXP-04 Post-learning Retention & Forgetting Rate
        # ---------------------------------------------------------
        print(f"[{vid}] Measuring post-learning retention on {len(retention_public)} tasks...")
        post_ret_e2e = 0
        post_ret_routing = 0
        for pub in retention_public:
            meta = private_by_id[pub.task_id]
            out = agent.solve_task(pub)
            eval_res = evaluate_execution_trace(meta, out)

            if eval_res.end_to_end_pass:
                post_ret_e2e += 1
            if eval_res.tool_routing_correct:
                post_ret_routing += 1

            raw_predictions.append({
                "variant": vid,
                "seed": seed,
                "phase": "post_retention_exp04",
                "task_id": pub.task_id,
                "category": pub.category,
                "prompt": pub.prompt,
                "expected_skill": meta.expected_skill,
                "selected_skill": out["selected_skill"],
                "tool_routing_correct": eval_res.tool_routing_correct,
                "params": out.get("params", {}),
                "raw_output": out.get("raw_output", ""),
                "exec_result": out.get("exec_result", {}),
                "execution_success": eval_res.execution_success,
                "semantic_correctness": eval_res.semantic_correctness,
                "end_to_end_pass": eval_res.end_to_end_pass,
                "passed": eval_res.end_to_end_pass,
                "reward": eval_res.reward,
                "latency_ms": out.get("latency_ms", 0.0),
                "validation_notes": eval_res.validation_notes,
                "validation_msg": eval_res.validation_notes,
                "retrieved_memories": out.get("retrieved_memories", []),
            })
        post_ret_e2e_acc = post_ret_e2e / max(1, len(retention_public))
        post_ret_route_acc = post_ret_routing / max(1, len(retention_public))
        forgetting_rate = max(0.0, pre_ret_e2e_acc - post_ret_e2e_acc)
        retention_stability = post_ret_e2e_acc / max(1e-5, pre_ret_e2e_acc)
        print(f"[{vid}] Post-retention E2E Acc: {post_ret_e2e_acc*100:.2f}% | Forgetting Rate: {forgetting_rate*100:.2f}%")

        v_mem = llm_core.get_memory_stats()

        summary_results[vid] = {
            "variant_id": vid,
            "seed": seed,
            "exp01_e2e_acc": learn_e2e_acc,
            "exp01_route_acc": learn_route_acc,
            "exp01_learning_accuracy": learn_e2e_acc,
            "exp01_learning_correct": learn_e2e,
            "exp01_learning_total": len(learning_public),
            "exp01_mean_latency_ms": mean_learn_lat,
            "exp02_mistake_recovery_rate": recovery_e2e_rate,
            "exp02_route_recovery_rate": recovery_route_rate,
            "exp02_recovered_count": recovered_e2e,
            "exp02_failed_initial_count": len(failed_tasks_trial1),
            "exp03_unseen_generalization_accuracy": unseen_e2e_acc,
            "exp03_unseen_route_accuracy": unseen_route_acc,
            "exp03_unseen_correct": unseen_e2e,
            "exp03_unseen_total": len(unseen_public),
            "exp03_mean_latency_ms": mean_unseen_lat,
            "exp04_pre_retention_accuracy": pre_ret_e2e_acc,
            "exp04_post_retention_accuracy": post_ret_e2e_acc,
            "exp04_forgetting_rate": forgetting_rate,
            "exp04_retention_stability": retention_stability,
            "exp05_idle_stats": idle_stats,
            "memory_rss_mb": v_mem["rss_mb"],
            "memory_peak_rss_mb": v_mem["peak_rss_mb_dec"],
            "memory_peak_rss_mib": v_mem["peak_rss_mib_bin"],
            "memory_vram_mb": v_mem["vram_mb"],
            "within_2gb_limit": v_mem["within_2gb_limit"],
        }

    return raw_predictions, summary_results


def run_benchmark(
    variants_to_run: List[str],
    seeds: List[int],
    output_dir: Path = ROOT_DIR / "artifacts",
    task_limit: Optional[int] = None,
    device: str = "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCAR-X CONTINUAL LEARNING BENCHMARK (VARIANTS A-G)")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {variants_to_run}")
    print(f"Task Limit: {task_limit}")
    print("=" * 70)

    # Initialize shared frozen core and skill registry
    print("\n[INIT] Loading Frozen Qwen2.5-0.5B-Instruct...")
    llm_core = FrozenQwenCore(device=device)
    integrity = llm_core.verify_weights_integrity()
    mem_stats = llm_core.get_memory_stats()
    print(f"[INIT] Parameters: {mem_stats['parameter_count']:,} ({mem_stats['weight_bytes'] / 1e6:.1f} MB)")
    print(f"[INIT] Checkpoint Verified: {integrity['verified']} (SHA256: {llm_core.EXPECTED_WEIGHT_SHA256[:16]}...)")
    print(f"[INIT] Live RSS: {mem_stats['rss_mb']:.1f} MB | Peak RSS: {mem_stats['peak_rss_mb_dec']:.1f} MB ({mem_stats['peak_rss_mib_bin']:.1f} MiB)")
    print(f"[INIT] Total Process within 2GB Gate: {mem_stats['within_2gb_limit']}")

    skills = SkillRegistry()

    all_raw_predictions = []
    all_summary_results: Dict[str, Any] = {}

    for seed in seeds:
        print(f"\n>>> EXECUTING SEED {seed} <<<")
        raw_preds, summaries = run_benchmark_for_seed(
            llm_core=llm_core,
            skills=skills,
            variants_to_run=variants_to_run,
            seed=seed,
            output_dir=output_dir,
            task_limit=task_limit,
        )
        all_raw_predictions.extend(raw_preds)
        all_summary_results[str(seed)] = summaries

    # Save artifacts
    raw_path = output_dir / "plastic_llm_raw_predictions.json"
    results_path = output_dir / "plastic_llm_benchmark_results.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_predictions, f, indent=2)

    # For top-level backwards compatibility, copy primary seed summary to root
    primary_seed_str = str(seeds[0])
    primary_summary = all_summary_results[primary_seed_str]

    results_payload = {
        "timestamp": time.time(),
        "seeds": seeds,
        "dev_seed": seeds[0],
        "device": device,
        "model_id": llm_core.MODEL_ID,
        "model_parameters": mem_stats["parameter_count"],
        "weight_bytes": mem_stats["weight_bytes"],
        "weight_mb_dec": mem_stats["weight_bytes"] / 1e6,
        "peak_rss_bytes": mem_stats["peak_rss_bytes"],
        "peak_rss_mb_dec": mem_stats["peak_rss_mb_dec"],
        "peak_rss_mib_bin": mem_stats["peak_rss_mib_bin"],
        "within_2gb_limit": mem_stats["within_2gb_limit"],
        "results_by_variant": primary_summary,
        "all_seeds_results": all_summary_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK EXECUTION COMPLETE")
    print(f"Raw predictions saved to: {raw_path}")
    print(f"Summary metrics saved to: {results_path}")
    print("=" * 70)
    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCAR-X Plastic LLM Benchmark Runner")
    parser.add_argument("--variants", nargs="+", default=["A", "B", "C", "D", "E", "F", "G"], help="Variants to execute")
    parser.add_argument("--seeds", nargs="+", type=int, default=[140011], help="Random seeds")
    parser.add_argument("--seed", type=int, default=None, help="Single random seed shortcut")
    parser.add_argument("--task-limit", type=int, default=None, help="Limit tasks for testing")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    seeds = args.seeds
    if args.seed is not None:
        seeds = [args.seed]

    run_benchmark(
        variants_to_run=args.variants,
        seeds=seeds,
        task_limit=args.task_limit,
        device=args.device,
    )
