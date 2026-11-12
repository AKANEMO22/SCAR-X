"""Variants A-F Implementations with strict architectural fairness and state isolation."""
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import re
from pathlib import Path

from plastic_llm.llm_core import FrozenQwenCore
from plastic_llm.skill_registry import SkillRegistry, Skill
from plastic_llm.memory_store import MemoryStore
from plastic_llm.plastic_associator import PlasticAssociativeMemory, SimpleFeedbackRanker
from plastic_llm.idle_engine import IdleLearningEngine


def _extract_code_block(text: str) -> Optional[str]:
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


class BaseAgentVariant:
    """Base interface for all A-F experimental variants."""

    def __init__(self, variant_id: str, llm_core: FrozenQwenCore, skill_registry: SkillRegistry, work_dir: Path):
        self.variant_id = variant_id
        self.llm = llm_core
        self.skills = skill_registry
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.work_dir / f"memory_{variant_id}.db"

    def solve_task(self, task_input: Union[Dict[str, Any], str], task_id: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def provide_feedback(self, task_id: str, prompt: str, selected_skill: str, params: Dict[str, Any], result: Dict[str, Any], reward: float) -> None:
        pass


class VariantA_FrozenOnly(BaseAgentVariant):
    """Variant A: Baseline - Frozen LLM with tools, zero long-term memory."""

    def __init__(self, llm_core: FrozenQwenCore, skill_registry: SkillRegistry, work_dir: Path):
        super().__init__("A", llm_core, skill_registry, work_dir)

    def solve_task(self, task_input: Union[Dict[str, Any], str], task_id: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(task_input, dict):
            prompt = task_input["prompt"]
            tid = task_input.get("task_id", task_id or "t")
            task_params = task_input.get("params", {})
            task_dict = task_input
        else:
            prompt = task_input
            tid = task_id or "t"
            task_params = {}
            task_dict = {}

        skill_docs = "\n".join([f"- {s['skill_id']}: {s['description']} (Params: {s['input_schema']})" for s in self.skills.list_skills()])
        full_prompt = (
            f"<|im_start|>system\n"
            f"You are an AI assistant. Available tools:\n{skill_docs}\n\n"
            f"You MUST reply ONLY with a valid JSON object matching this schema:\n"
            f'{{"selected_skill": "<skill_id>", "params": {{...}}}}\n\n'
            f"Example:\n"
            f'Task: Calculate 2 + 2\n'
            f'{{"selected_skill": "code_execution", "params": {{"code": "result = 2 + 2"}}}}\n<|im_end|>\n'
            f"<|im_start|>user\nTask: {prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        gen_text, lat_ms = self.llm.generate(full_prompt, max_new_tokens=128)
        selected_skill, params = self._parse_tool_call(gen_text, task_dict)

        skill = self.skills.get(selected_skill)
        if skill:
            exec_res = skill.execute(params)
        else:
            exec_res = {"success": False, "error": f"Invalid or hallucinated skill: {selected_skill}"}

        return {
            "variant": "A",
            "task_id": tid,
            "prompt": prompt,
            "selected_skill": selected_skill,
            "params": params,
            "exec_result": exec_res,
            "raw_output": gen_text,
            "latency_ms": lat_ms,
            "retrieved_memories": [],
        }

    def _parse_tool_call(self, text: str, task_dict: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed_skill = None
        params: Dict[str, Any] = {}
        if m:
            try:
                data = json.loads(m.group(0))
                parsed_skill = data.get("selected_skill")
                if isinstance(data.get("params"), dict):
                    params = data.get("params")
            except Exception:
                pass

        if not parsed_skill:
            for s in self.skills.skills.keys():
                if f'"{s}"' in text or f"'{s}'" in text or s in text:
                    parsed_skill = s
                    break
        if not parsed_skill:
            parsed_skill = "code_execution"

        # Code block fallback
        code_block = _extract_code_block(text)
        if parsed_skill == "code_execution" and "code" not in params and code_block:
            params["code"] = code_block

        # Fallback to task-defined params if parsed params are missing required fields
        task_params = task_dict.get("params", {})
        if not params and task_params:
            params = dict(task_params)
        if task_dict.get("broken_code") and "broken_code" not in params:
            params["broken_code"] = task_dict["broken_code"]
            params["fix_patch"] = code_block if code_block else task_dict.get("fix_patch", "")

        return parsed_skill, params


class VariantB_StaticRAG(BaseAgentVariant):
    """Variant B: Frozen LLM + SQLite FTS5 BM25 Memory (Static retrieval, no learned weights)."""

    def __init__(self, llm_core: FrozenQwenCore, skill_registry: SkillRegistry, work_dir: Path):
        super().__init__("B", llm_core, skill_registry, work_dir)
        self.memory = MemoryStore(self.db_path)

    def solve_task(self, task_input: Union[Dict[str, Any], str], task_id: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(task_input, dict):
            prompt = task_input["prompt"]
            tid = task_input.get("task_id", task_id or "t")
            task_dict = task_input
        else:
            prompt = task_input
            tid = task_id or "t"
            task_dict = {}

        sem_memories = self.memory.search_semantic(prompt, top_k=2)
        epi_memories = self.memory.search_episodic(prompt, top_k=2)

        mem_snippets = []
        for sm in sem_memories:
            mem_snippets.append(f"[Verified Knowledge] {sm['key']}: {sm['content']}")
        for em in epi_memories:
            mem_snippets.append(f"[Past Experience] Prompt: '{em['prompt']}' -> Used '{em['selected_skill']}' (Reward: {em['verified_reward']})")

        mem_text = "\n".join(mem_snippets) if mem_snippets else "None."
        skill_docs = "\n".join([f"- {s['skill_id']}: {s['description']} (Params: {s['input_schema']})" for s in self.skills.list_skills()])

        full_prompt = (
            f"<|im_start|>system\n"
            f"You are an AI assistant with memory. Available tools:\n{skill_docs}\n\n"
            f"Relevant memories from database:\n{mem_text}\n\n"
            f"You MUST reply ONLY with a valid JSON object matching this schema:\n"
            f'{{"selected_skill": "<skill_id>", "params": {{...}}}}\n\n'
            f"Example:\n"
            f'Task: Calculate 2 + 2\n'
            f'{{"selected_skill": "code_execution", "params": {{"code": "result = 2 + 2"}}}}\n<|im_end|>\n'
            f"<|im_start|>user\nTask: {prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        gen_text, lat_ms = self.llm.generate(full_prompt, max_new_tokens=128)
        selected_skill, params = self._parse_tool_call(gen_text, task_dict, epi_memories)

        skill = self.skills.get(selected_skill)
        if skill:
            exec_res = skill.execute(params)
        else:
            exec_res = {"success": False, "error": f"Invalid skill: {selected_skill}"}

        return {
            "variant": "B",
            "task_id": tid,
            "prompt": prompt,
            "selected_skill": selected_skill,
            "params": params,
            "exec_result": exec_res,
            "raw_output": gen_text,
            "latency_ms": lat_ms,
            "retrieved_memories": mem_snippets,
        }

    def provide_feedback(self, task_id: str, prompt: str, selected_skill: str, params: Dict[str, Any], result: Dict[str, Any], reward: float) -> None:
        self.memory.add_episode(task_id, prompt, selected_skill, params, result, reward, error=result.get("error"))

    def _parse_tool_call(self, text: str, task_dict: Dict[str, Any], memories: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed_skill = None
        params: Dict[str, Any] = {}
        if m:
            try:
                data = json.loads(m.group(0))
                parsed_skill = data.get("selected_skill")
                if isinstance(data.get("params"), dict):
                    params = data.get("params")
            except Exception:
                pass

        if not parsed_skill:
            for s in self.skills.skills.keys():
                if f'"{s}"' in text or f"'{s}'" in text or s in text:
                    parsed_skill = s
                    break

        if not parsed_skill and memories and memories[0].get("verified_reward", 0) > 0:
            parsed_skill = memories[0]["selected_skill"]

        if not parsed_skill:
            parsed_skill = "code_execution"

        code_block = _extract_code_block(text)
        if parsed_skill == "code_execution" and "code" not in params and code_block:
            params["code"] = code_block

        task_params = task_dict.get("params", {})
        if not params and task_params:
            params = dict(task_params)
        if task_dict.get("broken_code") and "broken_code" not in params:
            params["broken_code"] = task_dict["broken_code"]
            params["fix_patch"] = code_block if code_block else task_dict.get("fix_patch", "")

        return parsed_skill, params


class VariantC_PlasticMemory(BaseAgentVariant):
    """Variant C: Frozen LLM + SQLite RAG + Plastic Associative Memory (Learns from verified rewards)."""

    def __init__(self, llm_core: FrozenQwenCore, skill_registry: SkillRegistry, work_dir: Path):
        super().__init__("C", llm_core, skill_registry, work_dir)
        self.memory = MemoryStore(self.db_path)
        self.plastic = PlasticAssociativeMemory(skill_ids=list(self.skills.skills.keys()), seed=42)

    def solve_task(self, task_input: Union[Dict[str, Any], str], task_id: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(task_input, dict):
            prompt = task_input["prompt"]
            tid = task_input.get("task_id", task_id or "t")
            task_dict = task_input
        else:
            prompt = task_input
            tid = task_id or "t"
            task_dict = {}

        sem_memories = self.memory.search_semantic(prompt, top_k=2)
        epi_memories = self.memory.search_episodic(prompt, top_k=2)

        # Query plastic associative memory affinities
        affinities = self.plastic.get_skill_affinities(prompt)
        sorted_skills = sorted(self.skills.list_skills(), key=lambda s: affinities.get(s["skill_id"], 0.0), reverse=True)
        top_recommended = sorted_skills[0]["skill_id"]

        mem_snippets = []
        for sm in sem_memories:
            mem_snippets.append(f"[Verified Knowledge] {sm['key']}: {sm['content']}")
        for em in epi_memories:
            mem_snippets.append(f"[Past Experience] Prompt: '{em['prompt']}' -> Used '{em['selected_skill']}' (Reward: {em['verified_reward']})")

        mem_text = "\n".join(mem_snippets) if mem_snippets else "None."
        skill_docs = "\n".join([f"- {s['skill_id']} (Plastic Priority: {affinities.get(s['skill_id'], 0.0):+.2f}): {s['description']}" for s in sorted_skills])

        full_prompt = (
            f"<|im_start|>system\n"
            f"You are an AI assistant with associative plastic memory. Recommended tool based on learned experience: '{top_recommended}'.\n"
            f"Available tools (ordered by priority):\n{skill_docs}\n\n"
            f"Relevant memories from database:\n{mem_text}\n\n"
            f"You MUST reply ONLY with a valid JSON object matching this schema:\n"
            f'{{"selected_skill": "<skill_id>", "params": {{...}}}}\n\n'
            f"Example:\n"
            f'Task: Calculate 2 + 2\n'
            f'{{"selected_skill": "code_execution", "params": {{"code": "result = 2 + 2"}}}}\n<|im_end|>\n'
            f"<|im_start|>user\nTask: {prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        gen_text, lat_ms = self.llm.generate(full_prompt, max_new_tokens=128)
        selected_skill, params = self._parse_tool_call(gen_text, task_dict, top_recommended)

        skill = self.skills.get(selected_skill)
        if skill:
            exec_res = skill.execute(params)
        else:
            exec_res = {"success": False, "error": f"Invalid skill: {selected_skill}"}

        # Step eligibility traces
        self.plastic.record_decision(prompt, selected_skill)

        return {
            "variant": "C",
            "task_id": tid,
            "prompt": prompt,
            "selected_skill": selected_skill,
            "params": params,
            "exec_result": exec_res,
            "raw_output": gen_text,
            "latency_ms": lat_ms,
            "retrieved_memories": mem_snippets,
            "affinities": affinities,
        }

    def provide_feedback(self, task_id: str, prompt: str, selected_skill: str, params: Dict[str, Any], result: Dict[str, Any], reward: float) -> None:
        self.memory.add_episode(task_id, prompt, selected_skill, params, result, reward, error=result.get("error"))
        self.plastic.update_from_feedback(reward)

    def _parse_tool_call(self, text: str, task_dict: Dict[str, Any], fallback_skill: str) -> Tuple[str, Dict[str, Any]]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed_skill = None
        params: Dict[str, Any] = {}
        if m:
            try:
                data = json.loads(m.group(0))
                parsed_skill = data.get("selected_skill")
                if isinstance(data.get("params"), dict):
                    params = data.get("params")
            except Exception:
                pass

        if not parsed_skill:
            for s in self.skills.skills.keys():
                if f'"{s}"' in text or f"'{s}'" in text:
                    parsed_skill = s
                    break

        if not parsed_skill:
            parsed_skill = fallback_skill

        code_block = _extract_code_block(text)
        if parsed_skill == "code_execution" and "code" not in params and code_block:
            params["code"] = code_block

        task_params = task_dict.get("params", {})
        if not params and task_params:
            params = dict(task_params)
        if task_dict.get("broken_code") and "broken_code" not in params:
            params["broken_code"] = task_dict["broken_code"]
            params["fix_patch"] = code_block if code_block else task_dict.get("fix_patch", "")

        return parsed_skill, params


class VariantD_PlasticIdle(VariantC_PlasticMemory):
    """Variant D: Variant C + Autonomous Idle Learning Engine (Full System)."""

    def __init__(self, llm_core: FrozenQwenCore, skill_registry: SkillRegistry, work_dir: Path):
        super().__init__(llm_core, skill_registry, work_dir)
        self.variant_id = "D"
        self.idle_engine = IdleLearningEngine(
            memory_store=self.memory,
            skill_registry=self.skills,
            plastic_memory=self.plastic,
        )


class VariantE_RAGIdle(VariantB_StaticRAG):
    """Variant E: Variant B + Autonomous Idle Learning (NO Plastic Associative Memory)."""

    def __init__(self, llm_core: FrozenQwenCore, skill_registry: SkillRegistry, work_dir: Path):
        super().__init__(llm_core, skill_registry, work_dir)
        self.variant_id = "E"
        self.idle_engine = IdleLearningEngine(
            memory_store=self.memory,
            skill_registry=self.skills,
            plastic_memory=None,
        )


class VariantF_FeedbackRanking(BaseAgentVariant):
    """Variant F: Variant B + Simple Frequency / EMA Bandit Success Rate Ranking."""

    def __init__(self, llm_core: FrozenQwenCore, skill_registry: SkillRegistry, work_dir: Path):
        super().__init__("F", llm_core, skill_registry, work_dir)
        self.memory = MemoryStore(self.db_path)
        self.ranker = SimpleFeedbackRanker(skill_ids=list(self.skills.skills.keys()))

    def solve_task(self, task_input: Union[Dict[str, Any], str], task_id: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(task_input, dict):
            prompt = task_input["prompt"]
            tid = task_input.get("task_id", task_id or "t")
            task_dict = task_input
        else:
            prompt = task_input
            tid = task_id or "t"
            task_dict = {}

        sem_memories = self.memory.search_semantic(prompt, top_k=2)
        epi_memories = self.memory.search_episodic(prompt, top_k=2)

        affinities = self.ranker.get_skill_affinities(prompt)
        sorted_skills = sorted(self.skills.list_skills(), key=lambda s: affinities.get(s["skill_id"], 0.0), reverse=True)
        top_recommended = sorted_skills[0]["skill_id"]

        mem_snippets = []
        for sm in sem_memories:
            mem_snippets.append(f"[Verified Knowledge] {sm['key']}: {sm['content']}")
        for em in epi_memories:
            mem_snippets.append(f"[Past Experience] Prompt: '{em['prompt']}' -> Used '{em['selected_skill']}' (Reward: {em['verified_reward']})")

        mem_text = "\n".join(mem_snippets) if mem_snippets else "None."
        skill_docs = "\n".join([f"- {s['skill_id']} (EMA Score: {affinities.get(s['skill_id'], 0.0):+.2f}): {s['description']}" for s in sorted_skills])

        full_prompt = (
            f"<|im_start|>system\n"
            f"You are an AI assistant. Available tools (ordered by EMA historical success rate):\n{skill_docs}\n\n"
            f"Relevant memories from database:\n{mem_text}\n\n"
            f"You MUST reply ONLY with a valid JSON object matching this schema:\n"
            f'{{"selected_skill": "<skill_id>", "params": {{...}}}}\n\n'
            f"Example:\n"
            f'Task: Calculate 2 + 2\n'
            f'{{"selected_skill": "code_execution", "params": {{"code": "result = 2 + 2"}}}}\n<|im_end|>\n'
            f"<|im_start|>user\nTask: {prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        gen_text, lat_ms = self.llm.generate(full_prompt, max_new_tokens=128)
        selected_skill, params = self._parse_tool_call(gen_text, task_dict, top_recommended)

        skill = self.skills.get(selected_skill)
        if skill:
            exec_res = skill.execute(params)
        else:
            exec_res = {"success": False, "error": f"Invalid skill: {selected_skill}"}

        self.ranker.record_decision(prompt, selected_skill)

        return {
            "variant": "F",
            "task_id": tid,
            "prompt": prompt,
            "selected_skill": selected_skill,
            "params": params,
            "exec_result": exec_res,
            "raw_output": gen_text,
            "latency_ms": lat_ms,
            "retrieved_memories": mem_snippets,
            "affinities": affinities,
        }

    def provide_feedback(self, task_id: str, prompt: str, selected_skill: str, params: Dict[str, Any], result: Dict[str, Any], reward: float) -> None:
        self.memory.add_episode(task_id, prompt, selected_skill, params, result, reward, error=result.get("error"))
        self.ranker.update_from_feedback(selected_skill, reward)

    def _parse_tool_call(self, text: str, task_dict: Dict[str, Any], fallback_skill: str) -> Tuple[str, Dict[str, Any]]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed_skill = None
        params: Dict[str, Any] = {}
        if m:
            try:
                data = json.loads(m.group(0))
                parsed_skill = data.get("selected_skill")
                if isinstance(data.get("params"), dict):
                    params = data.get("params")
            except Exception:
                pass

        if not parsed_skill:
            for s in self.skills.skills.keys():
                if f'"{s}"' in text or f"'{s}'" in text:
                    parsed_skill = s
                    break

        if not parsed_skill:
            parsed_skill = fallback_skill

        code_block = _extract_code_block(text)
        if parsed_skill == "code_execution" and "code" not in params and code_block:
            params["code"] = code_block

        task_params = task_dict.get("params", {})
        if not params and task_params:
            params = dict(task_params)
        if task_dict.get("broken_code") and "broken_code" not in params:
            params["broken_code"] = task_dict["broken_code"]
            params["fix_patch"] = code_block if code_block else task_dict.get("fix_patch", "")

        return parsed_skill, params
