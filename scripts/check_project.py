"""Check navigation and run/source integrity without editing research results."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

root=Path(__file__).resolve().parents[1]
errors=[]
for p in [root/"README.md",*sorted((root/"docs").rglob("*.md")),root/"runs/INDEX.md"]:
    for link in re.findall(r"\]\(([^)]+)\)",p.read_text(encoding="utf-8")):
        if "://" not in link and not link.startswith("#") and not (p.parent/link.split("#")[0]).exists():
            errors.append(f"Broken link {p.relative_to(root)}: {link}")
project=json.loads((root/"project.json").read_text())
run=root/project["primary_run"]
manifest=json.loads((run/"manifest.json").read_text())
with zipfile.ZipFile(run/"source.zip") as z:
    for name,digest in manifest["source_sha256"].items():
        if hashlib.sha256(z.read(name)).hexdigest()!=digest:
            errors.append("Snapshot hash mismatch: "+name)
first=json.loads((root/"runs/20261108T151927Z-phase2-v1-5e495f/result.json").read_text())
last=json.loads((root/"runs/20261108T152637Z-phase2-v1-7a0790/result.json").read_text())
if first["trials"]!=last["trials"]:
    errors.append("Repeated experiment is not exactly reproducible")
if errors:
    raise SystemExit("\n".join(errors))
print("Navigation links, source snapshot checksums and repeated experimental results: OK")
