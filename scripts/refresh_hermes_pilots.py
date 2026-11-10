"""Retired entrypoint. Install the single Generation1 provider instead."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('install_generation1_provider.py')),run_name='__main__')
