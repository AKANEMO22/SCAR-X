"""Bounded read-only identities for the explicitly installed local Hermes app."""

import json
import math
import os
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


def discover_hermes_roots(home=None,processes=None,process_factory=psutil.Process):
    """Current default local gateway plus the installed desktop main process."""
    if home is None:
        appdata=os.environ.get('LOCALAPPDATA')
        if not appdata:return set()
        home=Path(appdata)/'hermes'
    home=Path(home).resolve()
    executable=(home/'hermes-agent/apps/desktop/release/win-unpacked/Hermes.exe').resolve()
    roots=set()
    gateway=verified_gateway_pid(home,process_factory)
    if gateway is not None:roots.add(gateway)
    inventory=processes if processes is not None else psutil.process_iter(['pid','name','exe','cmdline'])
    for process in inventory:
        try:
            info=process.info
            if ((info.get('name') or '').lower()!='hermes.exe'
                    or any(str(arg).startswith('--type=') for arg in (info.get('cmdline') or []))):continue
            if Path(info.get('exe') or '').resolve()!=executable:continue
            if process.is_running():roots.add(process.pid)
        except (OSError,ValueError,psutil.Error):continue
    return roots
