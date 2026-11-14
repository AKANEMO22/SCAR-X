"""Exact-source overlay recipe; do not apply while an endurance manifest is live."""

import hashlib

SOURCE_SHA256='4c7ae85431dade7ef8918ba737974e0ef2eb8bda0a783f3abda09e54bd6395d8'


def patch_bytes(original):
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Resource source changed; review before client-root patch')
    eol=b'\r\n' if b'\r\n' in original else b'\n'
    text=original.decode('utf-8').replace('\r\n','\n')
    anchor='        self.extra_pids = set(extra_pids)\n'
    if text.count(anchor)!=1:raise ValueError('Resource explicit-root anchor changed')
    text=text.replace(anchor,'        self.fixed_extra_pids = set(extra_pids)\n'
        '        self.hermes_client_roots = set()\n        self.extra_pids = set(extra_pids)\n',1)
    start=text.index('        if self.include_hermes and time.monotonic() - self.last_client_scan > 10:')
    stop=text.index('        processes = {self.root.pid: self.root}',start)
    text=text[:start]+('        if self.include_hermes and time.monotonic() - self.last_client_scan > 10:\n'
        '            from ..runtime.hermes_resource_roots import discover_hermes_roots\n'
        '            self.last_client_scan = time.monotonic()\n'
        '            self.hermes_client_roots = discover_hermes_roots()\n'
        '        self.extra_pids = self.fixed_extra_pids | self.hermes_client_roots\n')+text[stop:]
    anchor='            "included_client_pids": sorted(self.extra_pids),\n'
    if text.count(anchor)!=1:raise ValueError('Resource telemetry anchor changed')
    text=text.replace(anchor,anchor+'            "explicit_root_pids": sorted(self.fixed_extra_pids),\n'
        '            "discovered_hermes_roots": sorted(self.hermes_client_roots),\n',1)
    raw=text.encode().replace(b'\n',eol)
    compile(raw,'resources_client_roots_candidate.py','exec')
    return raw
