import json
from collections import defaultdict
from pathlib import Path

raw_path = Path("artifacts/plastic_llm_raw_predictions.json")
with open(raw_path, "r", encoding="utf-8") as f:
    preds = json.load(f)

skill_acc = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

for p in preds:
    if p["phase"] == "learning_exp01":
        v = p["variant"]
        c = p["category"]
        skill_acc[v][c]["total"] += 1
        if p["selected_skill"] == p["expected_skill"]:
            skill_acc[v][c]["correct"] += 1

print("\n| Variant | new_knowledge | python_debug | workflow_sel | compositional | edge_cases | Total Skill Selection (Learning) |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for v in sorted(skill_acc.keys()):
    cats = skill_acc[v]
    nk = f"{cats['new_knowledge']['correct']}/{cats['new_knowledge']['total']} ({cats['new_knowledge']['correct']/max(1, cats['new_knowledge']['total'])*100:.0f}%)"
    pd = f"{cats['python_debug']['correct']}/{cats['python_debug']['total']} ({cats['python_debug']['correct']/max(1, cats['python_debug']['total'])*100:.0f}%)"
    ws = f"{cats['workflow_selection']['correct']}/{cats['workflow_selection']['total']} ({cats['workflow_selection']['correct']/max(1, cats['workflow_selection']['total'])*100:.0f}%)"
    cp = f"{cats['compositional']['correct']}/{cats['compositional']['total']} ({cats['compositional']['correct']/max(1, cats['compositional']['total'])*100:.0f}%)"
    ec = f"{cats['edge_cases']['correct']}/{cats['edge_cases']['total']} ({cats['edge_cases']['correct']/max(1, cats['edge_cases']['total'])*100:.0f}%)"
    tot = sum(c["correct"] for c in cats.values())
    print(f"| **{v}** | {nk} | {pd} | {ws} | {cp} | {ec} | **{tot}/100 ({tot:.1f}%)** |")

# Unseen Generalization
unseen_acc = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
for p in preds:
    if p["phase"] == "generalization_exp03":
        v = p["variant"]
        c = p["category"]
        unseen_acc[v][c]["total"] += 1
        if p["selected_skill"] == p["expected_skill"]:
            unseen_acc[v][c]["correct"] += 1

print("\n| Variant | new_knowledge | python_debug | workflow_sel | compositional | edge_cases | Total Skill Selection (Unseen 60) |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for v in sorted(unseen_acc.keys()):
    cats = unseen_acc[v]
    nk = f"{cats['new_knowledge']['correct']}/{cats['new_knowledge']['total']}"
    pd = f"{cats['python_debug']['correct']}/{cats['python_debug']['total']}"
    ws = f"{cats['workflow_selection']['correct']}/{cats['workflow_selection']['total']}"
    cp = f"{cats['compositional']['correct']}/{cats['compositional']['total']}"
    ec = f"{cats['edge_cases']['correct']}/{cats['edge_cases']['total']}"
    tot = sum(c["correct"] for c in cats.values())
    print(f"| **{v}** | {nk} | {pd} | {ws} | {cp} | {ec} | **{tot}/60 ({tot/60*100:.1f}%)** |")

# Retention
pre_ret_acc = defaultdict(lambda: {"correct": 0, "total": 0})
post_ret_acc = defaultdict(lambda: {"correct": 0, "total": 0})
for p in preds:
    v = p["variant"]
    if p["phase"] == "pre_retention":
        pre_ret_acc[v]["total"] += 1
        if p["selected_skill"] == p["expected_skill"]:
            pre_ret_acc[v]["correct"] += 1
    elif p["phase"] == "post_retention_exp04":
        post_ret_acc[v]["total"] += 1
        if p["selected_skill"] == p["expected_skill"]:
            post_ret_acc[v]["correct"] += 1

print("\n| Variant | Pre-Retention Skill (40) | Post-Retention Skill (40) | Retention Delta |")
print("| :--- | :--- | :--- | :--- |")
for v in sorted(pre_ret_acc.keys()):
    pre_c = pre_ret_acc[v]["correct"]
    post_c = post_ret_acc[v]["correct"]
    print(f"| **{v}** | {pre_c}/40 ({pre_c/40*100:.1f}%) | {post_c}/40 ({post_c/40*100:.1f}%) | {(post_c - pre_c)/40*100:+.1f} pp |")
