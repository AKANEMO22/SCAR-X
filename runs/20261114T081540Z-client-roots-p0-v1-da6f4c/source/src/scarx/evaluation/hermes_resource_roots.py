"""Audit-only read-only identity of the explicitly installed local Hermes app."""

import json
import math
from pathlib import Path
import psutil


def verified_gateway_pid(home,process_factory=psutil.Process):
    """A descriptor plus matching process identity, never an argv substring."""
    home=Path(home).resolve()
    try:
        path=home/'gateway.pid'
        if path.stat().st_size>16384:return None
        row=json.loads(path.read_text(encoding='utf-8'))
        pid,stamp=row.get('pid'),row.get('start_time')
        if (row.get('kind')!='hermes-gateway' or type(pid) is not int or pid<=0
                or type(stamp) is not int or stamp<=0):return None
        process=process_factory(pid)
        created=process.create_time()
        if not math.isfinite(created) or abs(created-stamp/100)>.1:return None
        if not process.is_running():return None
        if not Path(process.exe()).resolve().is_relative_to(home):return None
        if process.create_time()!=created:return None
        return pid
    except (OSError,ValueError,TypeError,AttributeError,psutil.Error):
        return None
