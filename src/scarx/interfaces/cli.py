import argparse
import json
from pathlib import Path
from ..neural.core import Config, SpikingCore
from ..memory.store import EpisodeStore
from ..evaluation.benchmark import benchmark
from .text import TextAgent
from ..learning.controller import LearningController
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SCAR-X: research prototype, not a trained assistant")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("benchmark")
    p.add_argument("--config",default="configs/phase2.json")
    p.add_argument("--root",default=".")
    p = sub.add_parser("init")
    p.add_argument("--checkpoint",default="local/byte-initial.npz")
    p.add_argument("--seed",type=int,default=17)
    p = sub.add_parser("generate")
    p.add_argument("prompt")
    p.add_argument("--checkpoint",default="local/byte-initial.npz")
    p.add_argument("--max-tokens",type=int,default=64)
    p = sub.add_parser("learn",help="Learn bounded, verified numeric episodes using a development guard")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--examples",required=True)
    p.add_argument("--guard",required=True)
    p.add_argument("--output",required=True)
    p = sub.add_parser("rollback")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    for cmd in ("observe","search"):
        p = sub.add_parser(cmd)
        p.add_argument("text")
        p.add_argument("--db",default="local/episodes.sqlite")
        if cmd == "observe":
            p.add_argument("--source",required=True)
            p.add_argument("--verified",action="store_true")
            p.add_argument("--kind",choices=["document","correction","test_result","generated"],default="document")
    args = parser.parse_args()
    if args.command == "benchmark":
        settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
        run,result = benchmark(Path(args.root).resolve(),settings)
        print(json.dumps({"run":str(run),"verdict":result["verdict"],"gates":result["gates"]},indent=2))
    elif args.command == "init":
        SpikingCore(Config(inputs=256,outputs=256,seed=args.seed)).save(args.checkpoint)
        print("Initialized random byte model. Language quality is untrained.")
    elif args.command == "generate":
        agent = TextAgent(SpikingCore.load(args.checkpoint),None)
        print(json.dumps(agent.generate(args.prompt,args.max_tokens),ensure_ascii=True))
    elif args.command == "rollback":
        SpikingCore.load(args.checkpoint).save(args.output)
        print(json.dumps({"restored_from":args.checkpoint,"output":args.output}))
    elif args.command == "learn":
        model = SpikingCore.load(args.checkpoint)
        def read_episodes(path,guard=False):
            if Path(path).stat().st_size > 8_000_000:
                raise ValueError("Episode file exceeds 8 MB")
            rows = json.loads(Path(path).read_text(encoding="utf-8"))
            if not 1 <= len(rows) <= 512:
                raise ValueError("Require 1..512 episodes")
            result = []
            for row in rows:
                if not row.get("source") or row.get("verified") is not True or row.get("kind") == "generated":
                    raise ValueError("Each lesson needs a source and external verification")
                x = np.asarray(row["sequence"],dtype=np.float32)
                if x.ndim != 2 or x.shape[1] != model.config.inputs or not 1 <= len(x) <= 4096 or not np.isfinite(x).all():
                    raise ValueError("Invalid episode dimensions/values")
                y = row["target"]
                if not isinstance(y,int) or not 0 <= y < model.config.outputs:
                    raise ValueError("Invalid target")
                result.append((x,y,row["group"]) if guard else (x,y))
            return result
        examples,guard = read_episodes(args.examples),read_episodes(args.guard,True)
        controller = LearningController(model,Path(args.output).parent/"rollback")
        result = controller.learn(examples,guard=guard)
        controller.model.save(args.output)
        print(json.dumps(result,indent=2))
    else:
        store = EpisodeStore(args.db)
        try:
            value = store.observe(args.text,args.source,verified=args.verified,kind=args.kind) if args.command=="observe" else store.search(args.text)
            print(json.dumps(value,ensure_ascii=True,indent=2))
        finally:
            store.close()


if __name__ == "__main__":
    main()
