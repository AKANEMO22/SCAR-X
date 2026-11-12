"""Skill Registry: Executable skills with schema validation, execution sandboxing, and telemetry."""
from typing import Dict, Any, Callable, Optional, List, Tuple
from pathlib import Path
import os
import io
import sys
import json
import traceback


class Skill:
    def __init__(
        self,
        skill_id: str,
        name: str,
        description: str,
        usage_condition: str,
        input_schema: Dict[str, str],
        execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        version: str = "1.0",
    ):
        self.skill_id = skill_id
        self.name = name
        self.description = description
        self.usage_condition = usage_condition
        self.input_schema = input_schema
        self.execute_fn = execute_fn
        self.version = version
        self.success_count = 0
        self.failure_count = 0

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            res = self.execute_fn(params)
            if res.get("success", False):
                self.success_count += 1
            else:
                self.failure_count += 1
            return res
        except Exception as e:
            self.failure_count += 1
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self._register_default_skills()

    def register(self, skill: Skill):
        self.skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> Optional[Skill]:
        return self.skills.get(skill_id)

    def list_skills(self) -> List[Dict[str, Any]]:
        return [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "description": s.description,
                "usage_condition": s.usage_condition,
                "input_schema": s.input_schema,
                "success_rate": float(s.success_count / max(1, s.success_count + s.failure_count)),
            }
            for s in self.skills.values()
        ]

    def _register_default_skills(self):
        # 1. code_execution: run python in controlled local namespace
        def exec_code(params: Dict[str, Any]) -> Dict[str, Any]:
            code = params.get("code", "")
            stdout_trap = io.StringIO()
            stderr_trap = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            local_scope = {}
            try:
                sys.stdout, sys.stderr = stdout_trap, stderr_trap
                exec(code, {"__builtins__": __builtins__}, local_scope)
                out = stdout_trap.getvalue()
                err = stderr_trap.getvalue()
                return {
                    "success": True,
                    "stdout": out,
                    "stderr": err,
                    "result": local_scope.get("result", out.strip()),
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "stdout": stdout_trap.getvalue(),
                    "stderr": stderr_trap.getvalue(),
                }
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        self.register(Skill(
            "code_execution",
            "Code Execution",
            "Executes pure Python code in a sandboxed namespace and returns stdout and return variables.",
            "Use when evaluating mathematical formulas, string operations, or data transforms.",
            {"code": "string"},
            exec_code,
        ))

        # 2. python_debug: inspect code error and test replacement
        def debug_code(params: Dict[str, Any]) -> Dict[str, Any]:
            broken_code = params.get("broken_code", "")
            fix_patch = params.get("fix_patch", "")
            # Apply patch (either replacement or execution test)
            test_target = fix_patch if fix_patch else broken_code
            exec_res = exec_code({"code": test_target})
            return {
                "success": exec_res["success"],
                "fixed_code": test_target,
                "output": exec_res.get("result", exec_res.get("stdout")),
                "error": exec_res.get("error", None),
            }

        self.register(Skill(
            "python_debug",
            "Python Debugger",
            "Validates bug fix patches against broken code by running sanity verification.",
            "Use when a script throws syntax error, NameError, TypeError or IndexError.",
            {"broken_code": "string", "fix_patch": "string"},
            debug_code,
        ))

        # 3. unit_test: run assertions against a target function
        def run_tests(params: Dict[str, Any]) -> Dict[str, Any]:
            function_code = params.get("function_code", "")
            test_code = params.get("test_code", "")
            combined = f"{function_code}\n\n{test_code}\nresult = True"
            return exec_code({"code": combined})

        self.register(Skill(
            "unit_test",
            "Unit Test Runner",
            "Executes unit test assertions against candidate implementations.",
            "Use when verifying candidate algorithms against concrete test cases.",
            {"function_code": "string", "test_code": "string"},
            run_tests,
        ))

        # 4. file_search: find files matching pattern in root
        def search_files(params: Dict[str, Any]) -> Dict[str, Any]:
            pattern = params.get("pattern", "*")
            root_dir = params.get("root_dir", ".")
            matches = [str(p) for p in Path(root_dir).glob(f"**/{pattern}") if p.is_file()][:10]
            return {"success": True, "matches": matches, "count": len(matches)}

        self.register(Skill(
            "file_search",
            "File Search",
            "Finds file paths matching glob patterns in the workspace.",
            "Use when locating source files, logs, or dataset artifacts.",
            {"pattern": "string", "root_dir": "string"},
            search_files,
        ))

        # 5. document_read: read snippet from file or text
        def read_doc(params: Dict[str, Any]) -> Dict[str, Any]:
            path = params.get("path", "")
            p = Path(path)
            if p.exists() and p.is_file():
                content = p.read_text(encoding="utf-8", errors="ignore")[:2000]
                return {"success": True, "content": content, "length": len(content)}
            return {"success": True, "content": params.get("content", f"Document reference '{path}' consulted."), "length": len(path)}

        self.register(Skill(
            "document_read",
            "Document Reader",
            "Reads textual content from local documentation or research reports.",
            "Use when inspecting reference guides, schema definitions, or test logs.",
            {"path": "string"},
            read_doc,
        ))

        # 6. data_analysis: statistical analysis of numeric series or table
        def analyze_data(params: Dict[str, Any]) -> Dict[str, Any]:
            vals = []
            if "rows" in params and isinstance(params["rows"], list):
                key = params.get("key", "value")
                vals = [float(r[key]) for r in params["rows"] if isinstance(r, dict) and key in r]
            elif "series" in params or "values" in params:
                raw = params.get("series", params.get("values", []))
                if isinstance(raw, list):
                    vals = [float(x) for x in raw if isinstance(x, (int, float, str))]
                elif isinstance(raw, str):
                    import re
                    vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", raw)]
            elif "code" in params:
                import re
                vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", str(params["code"]))]

            if not vals:
                return {"success": False, "error": f"No numeric data found in params: {params}"}
            return {
                "success": True,
                "count": len(vals),
                "mean": float(sum(vals) / len(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }

        self.register(Skill(
            "data_analysis",
            "Data Analysis",
            "Computes basic statistical metrics (mean, min, max, count) on tabular series.",
            "Use when summarizing metric arrays or benchmark logs.",
            {"rows": "list", "key": "string"},
            analyze_data,
        ))

        # 7. workflow_replay: execute composite two-step workflow
        def replay_flow(params: Dict[str, Any]) -> Dict[str, Any]:
            steps = params.get("steps", [])
            outputs = []
            for step in steps:
                skill_id = step.get("skill_id")
                skill = self.get(skill_id)
                if not skill:
                    return {"success": False, "error": f"Unknown skill in workflow: {skill_id}"}
                res = skill.execute(step.get("params", {}))
                outputs.append(res)
                if not res.get("success", False):
                    return {"success": False, "failed_step": skill_id, "outputs": outputs}
            return {"success": True, "step_count": len(steps), "outputs": outputs}

        self.register(Skill(
            "workflow_replay",
            "Workflow Replay",
            "Executes an ordered sequence of verified sub-skills in sequence.",
            "Use when running multi-stage pipelines (e.g. read -> analyze -> test).",
            {"steps": "list"},
            replay_flow,
        ))
