"""Locate pinned research inputs from main or an isolated Git worktree.

Read-only lookup. It never creates, moves or rewrites an active experiment.
"""
from pathlib import Path
import subprocess

def main_repository(root):
    root=Path(root).resolve()
    common=subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=root,text=True).strip()
    path=Path(common)
    if not path.is_absolute():path=root/path
    return path.resolve().parent

def peer_checkout(root,name):
    if name not in ('review-gemini-r8-607ffca','review-gemini-r8-fe472b3'):
        raise ValueError('Only explicitly pinned R8 source checkouts supported')
    root=Path(root).resolve()
    candidates=(root.parent/name,main_repository(root)/'local/worktrees'/name)
    for path in candidates:
        if path.is_dir():return path
    raise FileNotFoundError('Pinned peer source missing; run scripts/research/prepare_pinned_peers.py: '+name)

def references_directory(root):return main_repository(root)/'local/references'
