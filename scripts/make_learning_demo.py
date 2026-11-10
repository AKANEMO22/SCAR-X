"""Create small deterministic numeric fixtures for the public learning CLI."""
import json
from pathlib import Path
from scarx import Config, SpikingCore
from scarx.evaluation.tasks import episodes

out=Path(__file__).resolve().parents[1]/"local/demo"
out.mkdir(parents=True,exist_ok=True)
SpikingCore(Config(inputs=5)).save(out/"before.npz")
for name,seed,count in (("examples",202,64),("guard",303,32)):
    records=[{"sequence":x.tolist(),"target":y,"group":"association","source":f"synthetic:v1:seed-{seed}","verified":True,"kind":"test_result"} for x,y in episodes(seed,count)]
    (out/(name+".json")).write_text(json.dumps(records),encoding="utf-8")
print(out)
