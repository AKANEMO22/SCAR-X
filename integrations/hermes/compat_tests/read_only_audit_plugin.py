"""Installed only in the isolated test home, never the user's Hermes profile."""

import json
from pathlib import Path


def register(ctx):
    config = json.loads(
        Path(__file__).with_name("scope.json").read_text(encoding="utf-8")
    )
    root = Path(config["cwd"]).resolve()
    spillover = Path(config["home"]).resolve() / "cache/spillover"

    def before(tool_name, args, **kwargs):
        if (
            tool_name == "terminal"
            and args.get("command") == config["terminal_command"]
        ):
            if Path(args.get("workdir", root)).resolve() == root:
                return None
        if tool_name in ("read_file", "search_files"):
            p = Path(args.get("path", ".")).expanduser()
            p = (p if p.is_absolute() else root / p).resolve()
            if p.is_relative_to(root) or (
                tool_name == "read_file" and p.is_relative_to(spillover)
            ):
                return None
        return {
            "action": "block",
            "message": "This engineering run permits only the bounded workspace prelude and read_file/search_files inside its declared project. No mutations, code execution, network or other workspace reads.",
        }

    ctx.register_hook("pre_tool_call", before)
