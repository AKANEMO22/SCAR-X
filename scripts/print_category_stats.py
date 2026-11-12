import json
from collections import defaultdict
from pathlib import Path

raw_path = Path("artifacts/plastic_llm_raw_predictions.json")
with open(raw_path, "r", encoding="utf-8") as f:
    preds = json.load(f)

cat_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

for p in preds:
    if p["phase"] == "learning_exp01":
        v = p["variant"]
        c = p["category"]
        cat_stats[v][c]["total"] += 1
        if p["passed"]:
            cat_stats[v][c]["correct"] += 1

print("| Variant | new_knowledge | python_debug | workflow_sel | compositional | edge_cases | Total Learning |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for v in sorted(cat_stats.keys()):
    cats = cat_stats[v]
    nk = f"{cats['new_knowledge']['correct']}/{cats['new_knowledge']['total']} ({cats['new_knowledge']['correct']/max(1, cats['new_knowledge']['total'])*100:.0f}%)"
    pd = f"{cats['python_debug']['correct']}/{cats['python_debug']['total']} ({cats['python_debug']['correct']/max(1, cats['python_debug']['total'])*100:.0f}%)"
    ws = f"{cats['workflow_selection']['correct']}/{cats['workflow_selection']['total']} ({cats['workflow_selection']['correct']/max(1, cats['workflow_selection']['total'])*100:.0f}%)"
    cp = f"{cats['compositional']['correct']}/{cats['compositional']['total']} ({cats['compositional']['correct']/max(1, cats['compositional']['total'])*100:.0f}%)"
    ec = f"{cats['edge_cases']['correct']}/{cats['edge_cases']['total']} ({cats['edge_cases']['correct']/max(1, cats['edge_cases']['total'])*100:.0f}%)"
    tot = sum(c["correct"] for c in cats.values())
    print(f"| **{v}** | {nk} | {pd} | {ws} | {cp} | {ec} | **{tot}/100 ({tot:.1f}%)** |")
