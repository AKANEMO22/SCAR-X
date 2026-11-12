import json
with open('artifacts/r10_replication_and_diagnosis_results.json') as f:
    d = json.load(f)

print('=== PHASE A1 EXACT REPRODUCTION (SEEDS 140011, 140017, 140023) ===')
for m, tasks in d['phase_a1_reproduction'].items():
    print(f'Method: {m}')
    for t, rows in tasks.items():
        directs = [r['id_direct'] for r in rows]
        probes = [r['id_probe'] for r in rows]
        oods = [r['ood_direct'] for r in rows]
        shufs = [r['shuffled_contrast'] for r in rows]
        print(f'  {t:18s}: Direct {sum(directs)/len(directs)*100:5.2f}% | Probe {sum(probes)/len(probes)*100:5.2f}% | OOD {sum(oods)/len(oods)*100:5.2f}% | Shuf {sum(shufs)/len(shufs)*100:+5.2f}%')

print('\n=== PHASE A2 FALSIFICATION (EXPLORATORY SEEDS 140031, 140037, 140043) ===')
for m, tasks in d['phase_a2_falsification'].items():
    print(f'Method: {m}')
    for t, rows in tasks.items():
        directs = [r['id_direct'] for r in rows]
        probes = [r['id_probe'] for r in rows]
        oods = [r['ood_direct'] for r in rows]
        shufs = [r['shuffled_contrast'] for r in rows]
        print(f'  {t:18s}: Direct {sum(directs)/len(directs)*100:5.2f}% | Probe {sum(probes)/len(probes)*100:5.2f}% | OOD {sum(oods)/len(oods)*100:5.2f}% | Shuf {sum(shufs)/len(shufs)*100:+5.2f}%')

print('\n=== PHASE C ACR (SEEDS 140011, 140017, 140023) ===')
for m, tasks in d['phase_c_acr'].items():
    print(f'Method: {m}')
    for t, rows in tasks.items():
        directs = [r['id_direct'] for r in rows]
        probes = [r['id_probe'] for r in rows]
        oods = [r['ood_direct'] for r in rows]
        shufs = [r['shuffled_contrast'] for r in rows]
        print(f'  {t:18s}: Direct {sum(directs)/len(directs)*100:5.2f}% | Probe {sum(probes)/len(probes)*100:5.2f}% | OOD {sum(oods)/len(oods)*100:5.2f}% | Shuf {sum(shufs)/len(shufs)*100:+5.2f}%')

print('\n=== PHASE B DIAGNOSTICS ===')
for m, res in d['phase_b_diagnostics'].items():
    print(f'{m:14s}: Mean Opt Cosine: {res["mean_optimizer_cosine"]:.4f} | Norm Ratio: {res["mean_optimizer_norm_ratio"]:.4f}')
    traj = res['trajectory']
    print(f'   Step 0: Cosine={traj[0]["oracle_cosine"]:.4f}, Direct={traj[0]["id_direct"]*100:.2f}%, Probe={traj[0]["id_probe"]*100:.2f}%, Gap={traj[0]["probe_direct_gap"]*100:+.2f}pp')
    print(f'   Step 1024: Cosine={traj[-1]["oracle_cosine"]:.4f}, Direct={traj[-1]["id_direct"]*100:.2f}%, Probe={traj[-1]["id_probe"]*100:.2f}%, Gap={traj[-1]["probe_direct_gap"]*100:+.2f}pp')
