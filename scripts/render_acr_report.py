"""Summarize complete raw ACR data, including negative/invalid controls."""
from pathlib import Path
import argparse,csv,json,statistics
ROOT=Path(__file__).resolve().parents[1]
TASKS=('strict_binding','xor','distractor_binding','order')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}đ'
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=(ROOT/a.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned research run required')
    audit=read(run/'audit-v1.json');result=read(run/'result.json');manifest=read(run/'manifest.json');nominee=audit['nominee']
    mechanism=read(run/'mechanism-audit-v1.json')
    router=read(run/'router-analysis-v1.json')
    rows=[read(p) for p in sorted((run/'jobs').glob('*/result-*.json'))]
    path=ROOT/'docs/research/ACR_DEV_RESULTS_20261112.md'
    if path.exists() or (run/'per-seed-results-v2.csv').exists():raise ValueError('Never overwrite published report')
    def frozen(task,seed):return next(r for r in rows if r['task']==task and r['seed']==seed and r['profile']=='readout_only' and r['budget']=='equal_updates')['metrics']['id_probe']
    with (run/'per-seed-results-v2.csv').open('x',encoding='utf-8',newline='') as stream:
        fields=['path','model_index','profile','condition','task','seed','budget','id_direct','id_probe','ood_direct','ood_probe',
          'probe_gain_vs_initial','causal_direct_drop','mean_physical_update_norm','training_episodes','train_cpu_seconds','pair_half_cpu_allocation','array_bytes','segment_peak_rss_so_far','segment_peak_commit_so_far']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            m=r['metrics'];t=r['training']
            writer.writerow({k:r[k] for k in fields[:7]}|{k:m[k] for k in fields[7:11]}|{
               'probe_gain_vs_initial':m['id_probe']-frozen(r['task'],r['seed']),
               'causal_direct_drop':m['id_direct']-m['revert_both_direct'],
               'mean_physical_update_norm':t['mean_update_norm'],'training_episodes':t['episodes'],
               'train_cpu_seconds':t['train_cpu_seconds'],'pair_half_cpu_allocation':t['cpu_is_pair_half_allocation'],'array_bytes':r['array_bytes'],
               'segment_peak_rss_so_far':r['resources']['peak_rss_bytes'],'segment_peak_commit_so_far':r['resources']['peak_commit_or_vms_bytes']})
    m=audit['profile_means'];lines=['# ACR DEV — kết quả và phản biện, 12/11/2026','',
      f"**{audit['decision']}**; integrity `{audit['integrity']}`. Một nominee toàn cục `{nominee}`, không chọn alpha theo task.",
      'Đây là nghiên cứu ALIF H48 D8/fan8, không Transformer/BPTT/pretrained, không main/Hermes. Không gọi kết quả này là lifelong intelligence hay năng lực ngôn ngữ/vision.','',
      '## Provenance và continuation','',
      f"Run `{run.name}`, source `{manifest['source_commit']}`; base93b5b96, equationsbc686500/peer607ffca. [Protocol gốc](../../experiments/acr-dev-v1.md) và [continuation](../../experiments/acr-resume-20261112.md).",
      f"Reused cells kiểm hash: {audit.get('reused_cells_verified',0)}; recursivechain `{audit.get('reuse_chain',[])}`.55cell đầu bị gián đoạn do credits; lượt sau hoàn tất203cell rồi mất worker khi tiếp tục phiên. Một preflight staleobserver dừng trước0newupdates và được giữ nguyên, ngân sách trừ thêm2s.",
      'V3 copy203completecells, chỉ replay đúng một cell mất state từ initialcheckpoint byte-exact. Ít nhất128 và tối đa1.024lượt học của attempt cũ không được lưu là compute thừa, không seed độc lập. Đây là ngoại lệ phục hồi hành chính được ghi trước khi chạy, không đổi thuật toán/criteria. Remainingbudget trừ snapshotelapsed và120s uncertainty; không chứng nhận exact tổngwall của quãng worker mất giám sát.',
      f"Đủ {audit['models']} model records/{audit['expected_models']} expected, missing={len(audit['missing'])}; invalid controls={len(audit['invalid_cells'])}.",
      f"Accepted training trajectories có {sum(r['training']['episodes'] for r in rows):,}updatecalls; cộng144diagnosticcloneupdates trong worker và128..1.024uncheckpointedupdates đã mất. Audit/postprocess không tạo learningupdates. Reusedcells không được tính thêm lần nữa.",
      f"Chấm lại {audit['prediction_rows_rescored']:,} prediction entries; {audit['checkpoints_roundtripped']} checkpoint roundtrip; {audit['direct_replay_predictions']:,} direct replay và {audit['independent_eigen_ridge_predictions']:,} probe qua eigensolver độc lập. Không có cập nhật học trong audit.",
      'Ba seed DEV140011/140017/140023, không dùng lại12confirmationseeds hoặc bankR9. CI chỉ mô tả3seed; exact sign-test có p tối thiểu0,25. Không P1 certification, không fresh confirmation/P2.','',
      '## Direct và hidden representation — equal1.024updates','',
      '| Profile | Binding direct/probe | XOR direct/probe | Distractor direct/probe | Order direct |',
      '|---|---:|---:|---:|---:|']
    for profile,bytask in m.items():
        values=[pct(bytask[t]['id_direct'])+'/'+pct(bytask[t]['id_probe']) for t in TASKS[:3]]
        lines.append('| '+profile+' | '+' | '.join(values)+' | '+pct(bytask['order']['id_direct'])+' |')
    lines+=['','Dữ liệu từng seed, OOD, norms/CPU và source path nằm trong `per-seed-results-v2.csv`; cột model_index/path phân biệt hai trajectory norm-paired, không gộp chúng thành baseline độc lập. Probe gain dùng frozen initial representation; readout budget không đổi frozen probe. RAM columns là segment study peaks tại thời điểm ghi row, không phải per-model isolated peak. V1CSV giữ nguyên, V2 thêm scope RAM.','',
      f"![Static credit screen](../../runs/{run.name}/figures/static-credit-screen.png)",
      'Vùng tô là min–max ba seed, không phải confidence interval. Order ceiling không được tính là representation success.','',
      '## Nominee và điều kiện bắt buộc','',
      '| Task | Probe gain vs readout [95% mô tả] | Direct causal drop | OOD preserved | Direct65 |',
      '|---|---:|---:|---|---|']
    for t in TASKS[:3]:
        c=audit['broad_checks'][t];h=audit['contrasts'][t]['hidden']
        lines.append(f"| {t} | {pp(h['mean'])} [{pp(h['lower95'])};{pp(h['upper95'])}] | {pp(c['causal_direct_drop'])} | {c['ood_preserved']} | {c['direct65']} |")
    lines += ['',f"Joint shared-checkpoint guards: `{audit['broad_checks']['joint_preserved']}`; order guard `{audit['broad_checks']['order_preserved']}`. Pooled mixed training chưa là continual learning/P2, dù chỉ có một shared head trong joint environment.",
      f"Shuffle-margin qua {audit['shuffle_families_pass']}/3; actual-norm margin qua {audit['norm_families_pass']}/3; control regression `{audit['control_regression']}`.",
      f"Norm control kiểm {audit['norm_steps_checked']:,} bước; zero radius {audit['zero_norm_steps']:,}; max relative physical norm error {audit['max_norm_relative_error']:.12g}, tolerance1e-5. Invalid controls phải chặn advance, không loại khỏi bảng để cứu điểm.",
      'Norm radius=.5min(raw physical updates) trên hai trajectory thật; optimizer/RMS/momentum history vẫn riêng. Do đó kiểm amplitude không chứng minh chỉ hướng cập nhật là nguyên nhân duy nhất. EqualCPU so dual-stream ACR endpoints, không phải claim tiết kiệm CPU trước baseline single-stream gốc. Pair CPU được phân bổ một nửa cho mỗi row, không phải đo riêng từng model.','',
      '## Actual diagnostic traces và OOD stress','',
      f"Đã kiểm {mechanism['diagnostic_files']} diagnostic clone tapes và {mechanism['stress_files']} stressfiles, {mechanism['stress_prediction_entries_rescored']:,} stresspredictionentries; không thêm learningupdates. Replay stress dùng cùng pinned probe, không gọi đó là independent ridge proof. Mixture maxerror={mechanism['worst_mixture_error']}, reconstructedphysicaldelta maxerror={mechanism['max_physical_delta_error']}.",
      'Các clone được lấy từ finalcheckpoint với TRAINindices0/63/511/1023; không nhầm chúng với trajectory của đúng update index ban đầu. Full u/theta/spike/adaptation/own-stop-mixed eligibility/alpha/q, signal, delta, loss và prediction nằm trong NPZ.','',
      '| Profile / task | Alpha TB | Alpha std TB/tape | Corr alpha–spike | Cos own/stop signal-credit | Norm own/stop |',
      '|---|---:|---:|---:|---:|---:|']
    for profile,taskgroups in mechanism['mechanism_summary'].items():
        for task,entry in taskgroups.items():
            corr=entry['alpha_spike_correlation'];cos=entry['signal_own_stop_cosine']
            lines.append(f"| {profile} / {task} | {entry['alpha_mean']:.4f} | {entry['alpha_std']:.4f} | {'undefined' if corr is None else format(corr,'.4f')} | {'undefined' if cos is None else format(cos,'.4f')} | {entry['raw_own_norm']:.4f}/{entry['raw_stop_norm']:.4f} |")
    lines+=['','Per-neuron48alpha và per-tickalpha means, adaptation correlations và per-seed rows ở mechanism-audit-v1.json; không suy ra consciousness từ điện thế hoặc spike.','',
      '### State-adaptive router — trace thật từ checkpoint C','',
      f"Bổ sung {router['rollouts']} frozen rollouts trên {router['checkpoint_files']} checkpoint C,0learningupdates và không dùng giá trị labels; độc lập kiểm q/alpha/permutation/mix trên cùng state. Metadata được đọc từ JSON có chứa prediction containers, không nhầm điều đó với đưa labels vào router. Không so phân phối alpha giữa các model đã học khác weights rồi gọi là matched.",'',
      '| Router / task | Alpha TB | Temporal std/neuron | Corr alpha–spike | Corr alpha–adaptation |',
      '|---|---:|---:|---:|---:|']
    for profile,groups in router['summary'].items():
        for task,entry in groups.items():
            cs=entry['alpha_spike_correlation'];ca=entry['alpha_adaptation_correlation']
            lines.append(f"| {profile} / {task} | {entry['alpha_mean']:.4f} | {entry['mean_within_neuron_temporal_std']:.4f} | {'undefined' if cs is None else format(cs,'.4f')} | {'undefined' if ca is None else format(ca,'.4f')} |")
    lines+=['','Đây là fixed heuristic router có alpha biến thiên, không phải router đã học policy tối ưu. Correlation là mô tả; final-episode gating chưa chứng minh online weight updates từng tick. [Phân tích đầy đủ](ACR_ROUTER_ANALYSIS_20261112.md).','',
      '| Task / stress | Own direct/probe | Stop direct/probe | Nominee direct/probe |',
      '|---|---:|---:|---:|']
    for task in TASKS:
        case_names=mechanism['stress_summary'][nominee][task]
        for case in case_names:
            values=[]
            for profile in ('static_0','static_100',nominee):
                items=mechanism['stress_summary'][profile][task][case]
                values.append(pct(statistics.mean(x['direct'] for x in items))+'/'+pct(statistics.mean(x['probe'] for x in items)))
            lines.append('| '+task+' / '+case+' | '+' | '.join(values)+' |')
    lines+=['','Stress transforms giữ finalmarker, nhưng noise-amplification/shift còn biến đổi distractorpulses; không gọi toàn bộ chúng là noise-only. Queryflip là paired intervention cùng writes, không distribution query mới.','',
      '## Cơ chế và giới hạn','',
      '- Own và stopped giữ cùng hard ALIF forward. Khác nhau là reset Jacobian trong eligibility, có thể khác ngay cả khi hard spike chưa xảy ra vì surrogate slope khác0.',
      '- Hai stream voltage/adaptation/filter độc lập; trộn sau filter. Endpoint0/1 đã khớp baseline exact, gồm optimizer/RNG/coldresume.',
      '- Router state là heuristic cố định, không đọc task ID, loss/test label/accuracy. Cập nhật chỉ cuối episode nên alpha cuối mới trực tiếp ảnh hưởng update; alpha(t) không tự chứng minh học online từng tick.',
      '- Random-alpha giữ phân phối alpha của chính state hiện thời rồi hoán vị; hai learner sau học có state khác nhau nên không tự coi marginal across-run là matched.',
      '- Refit probe đo thông tin có thể đọc ra; direct causal drop và OOD mới giúp đánh giá kỹ năng usable. Truy xuất/SSD/LLM không tham gia benchmark.',
      '- Bộ nhớ mảng bổ sung16.704B so baseline:4trace matrices12.288B và router/diagnostic arrays4.416B. Python/runtime/ZIP/buffers tính riêng trong measured process RAM.',
      f"V3 peak RSS {result['resources']['peak_rss_bytes']/1e6:.2f}MB/commit {result['resources']['peak_commit_or_vms_bytes']/1e6:.2f}MB. Peak của các run gián đoạn chỉ là historical snapshot lowerbounds, không finalglobalpeak; xem progress gốc trong recursivechain. Không iGPU/điện năng suy diễn.",
      '', '## Kết luận và đường đi tiếp','',
      'Không thay gateP1/P2. Nếu có specialized/probe-only signal phải giữ đúng nhãn, không suy rộng thành cùng learner giữ kỹ năng lâu dài. Chưa có kiểm quên20context trong vòng này. Không merge/main/Hermes hoặc mở confirmation khi chưa có approval riêng.',
      'Đối chiếu literature nằm ở nhánh codex/scarx-literature-first-20261112: e-prop/BrainTrace adapters và SI helper đã kiểm số học; reference DEV kết luận REFERENCE_MIXED. Không được tuyên bố literature thất bại nói chung từ một adapter/H48/budget duy nhất.',
      '', '## Tái lập','', '```powershell','python -m pytest -q -p no:cacheprovider tests/test_acr.py tests/test_acr_study.py',
      '# Original once-reserved study: scripts/run_acr_dev.py; final continuation: scripts/run_acr_resume_v3.py.',
      '# Reproductions use a new isolated checkout; never delete old reservations or overwrite evidence.',
      f'python scripts/audit_acr_dev.py --run runs/{run.name}', '```',
      'Observer http://127.0.0.1:8771 tự đọc actual progress/answers/loss. Không mô phỏng suy nghĩ thành lời; scientific reports giữ source SHA/protocol/data/hash để truy vấn lại.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(path)
if __name__=='__main__':main()
