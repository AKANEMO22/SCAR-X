"""Vietnamese report from complete capacity evidence; never trains or selects tasks."""
from pathlib import Path
import argparse,json,statistics
ROOT=Path(__file__).resolve().parents[1]
TASKS=('strict_binding','xor','distractor_binding','order')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}đ'
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=(ROOT/a.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run only')
    audit=read(run/'audit.json');result=read(run/'result.json');manifest=read(run/'manifest.json')
    output=ROOT/'docs/research/CAPACITY_H64_H128_RESULTS_20261112.md'
    if output.exists():raise ValueError('Do not overwrite a published report')
    rows=[read(p) for p in (run/'jobs').glob('*/result.json')]+[read(p) for p in (run/'reused-h48').glob('*/capacity-result.json')]
    means=audit['means'];cpu=audit['equal_cpu_means']
    lines=['# SCAR-X — kết quả H64/H128, 12/11/2026','',
      f"**{audit['decision']}**, integrity **{audit['integrity']}**. Đây là phép thử capacity DEV đã duyệt, không phải P1/P2/production hoặc năng lực chat/vision.",
      'Nhãn capacity không phủ nhận hiddenlearning hiện có: nó phân biệt directgain khi thêmneuron với việc learned-vsfrozen advantage tăng đủ ởít nhất2nhóm. Phải đọc cả bảngdirect và ablation; không diễn giải CAPACITY_ONLY_SIGNAL thành “model không học synapse”.',
      'Giữ ALIF hardspikes/sparsefan8/D8/O2, phương trình localeligibility và source readout/optimizer. Không Transformer/pretrained/globalBPTT/RTRLcandidate. Main/Hermes/live không thay.','',
      '## Phạm vi, số đo và tính đúng','',
      f"Run `{run.name}`, source `{manifest['source_commit']}`, base22bb9781. [Protocol đã duyệt](../../experiments/capacity-h64-h128-approved-v1.md).",
      f"Đủ {audit['record_count']} records:204newjobs và48H48primary reuse; primary144records1024updates/cell, secondary108equalCPU0,5s. Không train lại H48primary hoặc chọn lại seed sau scores.",
      'Ba DEVseeds140011/140017/140023, cùng frozenTRAIN1024/refit256/ID512/OOD512, calibration32TRAIN. HeldoutID/OOD là DEV đã mở, không freshconfirmation.',
      'Mỗi task/seed được fit trên checkpoint riêng. Bảng không chứng minh một checkpoint duy nhất giữ đồng thời tất cả kỹ năng; không ghép những model trial này thành bộ não sản phẩm.',
      f"Audit chấm lại {audit['predictions_rescored']:,} prediction entries, {audit['labels_oracle_checked']:,} labels, {audit['checkpoints_roundtripped']} checkpoint roundtrips; {audit['forward_replay_predictions']:,} direct replays và {audit['independent_eigenprobe_predictions']:,} probe bằng eigensolver độc lập. Không learningupdates trong audit.",
      'Reviewer thực độc lập còn kiểm literalALIF replay, thresholdlabeloracle và SVDprobe; xem [review cuối](CAPACITY_RESULTS_INDEPENDENT_AUDIT_20261112.md).',
      'H48 adapter parity exact qua losses/weights/traces/optimizer/RNG; H64/H128 coldprocess tiếp đuôi episode rồi nextupdate exact. Native parity của sharedcore đã kiểm; không claim có kernel ACRnative mới.',
      f"Worker wall {result['wall_seconds']:.2f}s; audit {audit['audit_wall_seconds']:.2f}s. Peak ownedtree RSS {result['resources']['peak_rss_bytes']/1e6:.2f}MB / commit {result['resources']['peak_commit_or_vms_bytes']/1e6:.2f}MB, dưới2.000MB. Windowscap `{result['cap']}`; không GPU/iGPU/điện năng đo được.",
      'ResourceMonitor flag hard_cap_enforced chưa nối receipt; dùng capreceipt thật cùng RSS/commit đo được. Committed memory không phải lượng swap. RAM includeobserver theo PID; arraybytes riêng bên dưới.','',
      '## Direct accuracy — cùng1.024updates','',
      '| H | Learner | Binding | XOR | Distractor | Order | Mảng adapter |',
      '|---:|---|---:|---:|---:|---:|---:|']
    for h in ('48','64','128'):
        for arm in ('own','groups_up','readout_only','groups_shuffle'):
            mem=(944 if arm.startswith('groups') else 596)*int(h)+864
            lines.append('| '+h+' | '+arm+' | '+' | '.join(pct(means[h][arm][t]['id_direct']) for t in TASKS)+f' | {mem:,}B |')
    lines+=['','Mảng adapter H48Own/readout hiện tại khác số mảng của olddualtrace checkpoint được reuse; rawlegacy giữ nguyên và được ghi `reused_source_array_bytes`. Chỉscores/parity được reuse, không dùng CPU cũ làm singletracecost.','',
      f'![Direct và OOD theo số neuron](../../runs/{run.name}/figures/capacity-direct-ood.png)','',
      '## Probe và OOD — không thay direct bằng probe','',
      '| H / learner | Binding probe | Binding OOD direct/probe | XOR probe | XOR OOD direct/probe | Distractor probe | Distractor OOD direct/probe |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for h in ('48','64','128'):
        for arm in ('own','groups_up','readout_only'):
            cells=[]
            for t in TASKS[:3]:
                m=means[h][arm][t];cells.extend([pct(m['id_probe']),pct(m['ood_direct'])+'/'+pct(m['ood_probe'])])
            lines.append('| '+h+' / '+arm+' | '+' | '.join(cells)+' |')
    lines+=['','## Capacity gates theo cùng learner, không chọn riêng mỗi task','',
      '| H / learner | Direct gain>=2đ: số nhóm | Hidden advantage growth>=2đ: số nhóm | ID/OOD preserved | Order protected | Causal guard | Signal |',
      '|---|---:|---:|---|---|---|---|']
    for h in ('64','128'):
        for arm in ('own','groups_up','readout_only'):
            g=audit['gates'][h][arm]
            lines.append(f"| {h} / {arm} | {g['direct_improved_families']}/3 | {g['hidden_advantage_improved_families']}/3 | {g['all_primary_preserved']} | {g['protected_order']} | {g['causality_on_improving_families']} | {g['capacity_dev_signal']} |")
    lines+=['','Hidden advantage growth = (probe learned−probe frozen) tạiH mới trừ cùngcontrastH48. Frozenrandomfeatures tốt hơn khi thêmneuron chưa phải hidden synapses học tốt hơn. Causalguard giữlearnedhead vàrevert hiddenweights; fullrevertrefitrecover frozenprobe phải exact.','',
      'Giới hạn trần của chính contrast: H48OwnXOR hidden advantage≈13,41đ; H128frozenprobe≈91,34% nên ngay cảlearnedprobe100% chỉ còn advantage≤8,66đ, tăng thêm≤−4,75đ. Gate yêu cầu+2đ growth vì thế không thể đạt trên XOR khibaseline này đã tốt lên. Giữ nguyên gate/decision đã đăng ký, nhưng không dùng thất bại đó để phủ nhận hiddenlearning hoặc lợi ích capacity. Futuremetric/task tránh trần cần prereg riêng, không sửa tiêu chí sau scores.',
      f'![Hidden advantage theo số neuron](../../runs/{run.name}/figures/capacity-hidden-advantage.png)','',
      '| H / learner / task | Δdirect vsH48 [95% mô tả] | Δhidden advantage | Direct causal drop |',
      '|---|---:|---:|---:|']
    for h in ('64','128'):
        for arm in ('own','groups_up'):
            for t in TASKS[:3]:
                c=audit['contrasts'][h][arm][t];g=audit['gates'][h][arm]['by_task'][t];s=c['id_direct']
                lines.append(f"| {h} / {arm} / {t} | {pp(s['mean'])} [{pp(s['lower95'])};{pp(s['upper95'])}] | {pp(g['hidden_advantage_growth'])} | {pp(g['causal_direct_drop'])} |")
    lines+=['','Ba seed là3cluster, không252independenttrials. CI mô tả với3seed, exactsign p nhỏ nhất0,25; không significance/confirmation hoặc HolmPASS từ vòng này. Gates giữ nguyên trước scores.','',
      '## Equal-CPU secondary —0,5s học','',
      '| H / learner | Binding | XOR | Distractor | Order | Episodes min–max |',
      '|---|---:|---:|---:|---:|---:|']
    for h in ('48','64','128'):
        for arm in ('own','groups_up','readout_only'):
            g=[r for r in rows if r['height']==int(h) and r['arm']==arm and r['budget']=='equal_cpu']
            counts=[r['training']['episodes'] for r in g]
            lines.append('| '+h+' / '+arm+' | '+' | '.join(pct(cpu[h][arm][t]['id_direct']) for t in TASKS)+f' | {min(counts)}–{max(counts)} |')
    lines+=['','Cutoff kiểm cuối episode, overshoot/CPU/counts đều ở raw. EqualCPU dùng108newcells, cảH48đo cùng hiện tại; không lấy số CPU của cacheddualtrace làm Owncost. Secondary không thay primary khi score bất lợi.','',
      '## Điều có thể và chưa thể kết luận','',
      f"Own-reset H48→H128: Binding {pct(means['48']['own']['strict_binding']['id_direct'])}→{pct(means['128']['own']['strict_binding']['id_direct'])}, XOR {pct(means['48']['own']['xor']['id_direct'])}→{pct(means['128']['own']['xor']['id_direct'])}, Distractor {pct(means['48']['own']['distractor_binding']['id_direct'])}→{pct(means['128']['own']['distractor_binding']['id_direct'])}. Đây là kết quả actual trên cùng dữ liệu/budget, không suy từ RAM trống.",
      '- Đây là kiểm capacity hữu hạn, không kết luận “càng nhiều neuron càng thông minh”. Tăngneuron tạo thêm khả năng biểu diễn; chỉ learning/heldout/OOD/causalcontrols mới đo khai thác được nó.',
      '- SameHinit giữa arms exact. CrossH chỉ pairedseed và cùngphânphối/initrule; weights/topology khácshape, RNGconsumption khác, không cùngprefixweights.',
      '- Fan8/H làm mật độ recurrent đổi; gamma linspace được resample. Source Wout initstd0,03 và learningrate/per-coordinateclips giữ cố định nên aggregate logitvariance/updatevectornorm cũng có thể đổi khiH tăng. Thử nghiệm đo cùng sourcelearner ở cácsize, không tách mọi covariance thành purecapacitylaw; không rescale sau scores.',
      '- Core R12 không có nghĩa chỉ12neuron học: inheritedresearchoptimizer cập nhật toàn bộH×16 hiddenweights. H64/128 cấp cố định mỗi trial, không tựgrow theo độ khó.',
      '- P2 chưa chạy: chưa bằng chứngretention20context, acquisition/final65%,meanforget3đ/worst5đ. Không dùng score2nhãn để tuyên bố hiểu ngôn ngữ/ảnh hoặc ngangLLM.',
      '- Nếu largerH chưa đạt vẫn không chứng minh architectureceiling: learning signal, readout/optimizer và data/updatebudget còn là nguyên nhân cần phân biệt.',
      '', '## Bàn giao và tái lập','',
      'Branch `codex/scarx-capacity-h64-h128-20261112`, main giữ `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`. Prototype/researchonly; không merge/deploy. Không mởH256, freshconfirmation, P2 hoặc language từ lượt này.',
      '```powershell','python scripts/build_native.py',
      'python -m pytest -q -p no:cacheprovider tests/test_capacity.py tests/test_capacity_independent.py tests/test_native.py',
      'python scripts/run_capacity_dev.py',
      f'python scripts/audit_capacity_dev.py --run runs/{run.name}',
      f'python scripts/verify_capacity_independent.py --run runs/{run.name} --require-complete',
      '```',
      'Runner/auditors từ chối overwrite; reproduction dùng checkout/output mới rõ ràng, không xóa reservation/raw hiện tại. Peer607ffca phải có ở siblingworktree review-gemini-r8-607ffca; oldH48evidence ở sibling scarx-acr-20261112 theo manifest.',
      f'Raw `runs/{run.name}/` có252records+CSV từngseed+source/data/reusehashes. Per-cell checkpoint/history/RNG, labels/predictions/interventions đầy đủ. H48cachedCPU được gắn scope riêng. Source-byte-bundle tái dựng exactbody từ pinnedGit+EOLhash, chỉ overlay trong checkoutreproduction mới; không nới checksum hoặc sửa nguồn hiện tại.',
      'Trace http://127.0.0.1:8773 tự đọc actualloss/update/output/H/resources; không Qwen/proxy hoặc lời nghĩ giả.']
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(str(output))
if __name__=='__main__':main()
