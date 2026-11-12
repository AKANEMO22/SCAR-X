"""Render a Vietnamese evidence report from completed immutable research outputs."""
from pathlib import Path
import argparse,json,statistics

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}đ'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=(ROOT/a.run).resolve()
    audit=read(run/'audit-v2.json');result=read(run/'result.json');manifest=read(run/'manifest.json')
    rows=[read(p) for p in sorted((run/'jobs').glob('*/result.json'))]
    methods=('own','stop','eprop_matched','eprop_aligned','braintrace','readout_only')
    tasks=('strict_binding','xor','distractor_binding','order');m=audit['means'];nominee=audit['nominee']
    out=ROOT/'docs/research/LITERATURE_REFERENCE_RESULTS_20261112.md'
    if out.exists():raise ValueError('Do not overwrite published report')
    lines=['# Literature-first — reference ALIF H48, 12/11/2026','',
      f"**{audit['decision']}**, integrity **{audit['integrity']}**. Nominee toàn cục `{nominee}`; chưa P1/P2 hay production.",
      'Không đổi forward ALIF, topology, input D8, readout, optimizer hoặc main/Hermes. Đây là adapter chuyển phương trình về SCAR-X, không tái lập accuracy paper gốc.','',
      '## Lượt thực nghiệm và kiểm toán','',
      f"Run `{run.name}`, source commit `{manifest['source_commit']}`. Protocol [đăng ký](../../experiments/literature-reference-dev-v1.md).",
      'Ba seed DEV đã dùng 140011/140017/140023; 1.024 update mỗi primary cell. TRAIN1.024/refit256/ID512/OOD512; calibration32TRAIN. Không sử dụng confirmation bank để tune.',
      f"Đủ {result['completed_jobs']} jobs:72primary,12nominee-shuffle,48equalCPU. Wall {result['wall_seconds']:.2f}s. Audit {audit['audit_wall_seconds']:.2f}s, không có cập nhật học.",
      f"Chấm lại {audit['predictions_rescored']:,} prediction entries, {audit['checkpoints_roundtripped']} checkpoint round-trip, {audit['forward_replay_predictions']:,} direct replay và {audit['independent_eigenprobe_predictions']:,} probe bằng eigensolver độc lập.",
      f"Peak owned-tree RSS {result['resources']['peak_rss_bytes']/1e6:.2f}MB; commit {result['resources']['peak_commit_or_vms_bytes']/1e6:.2f}MB; trần2.000MB. Windows cap receipt: `{result['cap']}`. Observer process được tính theo PID; không có iGPU/GPU.",
      'ResourceMonitor flag hard_cap_enforced không được truyền từ cap receipt; dùng receipt thực và measured RSS/commit, không đồng nhất committed memory với lượng swap.',
      '','## Direct output và tài nguyên','',
      '| Method | Binding | XOR | Distractor | Order | OOD direct TB3primary | CPU train TB/job | Mảng model |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        group=[r for r in rows if r['method']==method and r['budget']=='equal_updates' and not r['shuffled']]
        cpu=statistics.mean(r['training']['train_cpu_seconds'] for r in group);mem=group[0]['array_bytes']
        lines.append('| '+method+' | '+' | '.join(pct(m[method][t]['id_direct']) for t in tasks)+f" | {pct(statistics.mean(m[method][t]['ood_direct'] for t in tasks[:3]))} | {cpu:.3f}s | {mem:,}B |")
    lines+=['','CPU là actual single-stream cho own/stop/reference; không phải dual-stream ACR endpoint. Mảng model không phải process RAM. Bảng OOD tách từng nhóm bên dưới để không che regression.','',
      '| Method | Binding ID probe | Binding OOD direct/probe | XOR ID probe | XOR OOD direct/probe | Distractor ID probe | Distractor OOD direct/probe |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        cells=[]
        for t in tasks[:3]:cells.extend([pct(m[method][t]['id_probe']),pct(m[method][t]['ood_direct'])+'/'+pct(m[method][t]['ood_probe'])])
        lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines+=['','### Equal-CPU secondary — 0,5s học, không thay primary','',
        '| Method | Binding direct | XOR direct | Distractor direct | Order direct | Số episode min–max |',
        '|---|---:|---:|---:|---:|---:|']
    for method in ('own','stop',nominee,'readout_only'):
        group=[r for r in rows if r['method']==method and r['budget']=='equal_cpu']
        values=[statistics.mean(r['metrics']['id_direct'] for r in group if r['task']==t) for t in tasks]
        counts=[r['training']['episodes'] for r in group]
        lines.append('| '+method+' | '+' | '.join(pct(v) for v in values)+f' | {min(counts)}–{max(counts)} |')
    lines+=['','Các điểm thấp được giữ nguyên. CPU cutoff kiểm ở cuối episode, có overshoot được ghi raw; số update khác nhau nên không thay thế đối chứng1.024update. Không có lợi thế CPU đa năng được chứng minh.',
       'Bảng per-seed v2 tra readout cùng budget. Bản đầu tra readout1.024update cho cả equalCPU; trong thí nghiệm này probe của frozen hidden exact ở hai budget nên không có giá trị gain nào bị đổi. Hai bản được giữ để ghi rõ sửa cách lookup; primary metrics và quyết định audit v1/v2 không đổi.']
    lines+=['','## Hidden learning và các gate không được miễn','',
      '| Task | Nominee direct | Probe gain vs readout [95% mô tả] | Direct causal drop revert-both | Δdirect vs better endpoint | OOD preserved |',
      '|---|---:|---:|---:|---:|---|']
    for t in tasks[:3]:
        c=audit['checks'][t];h=audit['contrasts'][t]['hidden']
        lines.append(f"| {t} | {pct(m[nominee][t]['id_direct'])} | {pp(h['mean'])} [{pp(h['lower95'])};{pp(h['upper95'])}] | {pp(c['causal_direct_drop'])} | {pp(c['direct_gain_vs_better_endpoint'])} | {c['ood_preserved']} |")
    lines+=['',f"Có {audit['improving_families_with_causality']}/3 family đạt cải thiện direct kèm probe/causality theo ruleDEV; {audit['shuffle_families']}/3 đạt probe margin so shuffle. Protected order: `{audit['protected_order']}`.",
      'Ba seed chỉ cho thống kê khám phá: exact sign-test hai phía có p nhỏ nhất0,25. Không thể gọi significance/confirmation. Probe gain không thay direct skill; full-revert recovery được kiểm exact với readout-only.',
      'Chưa có actual-update-norm matched control cho reference ở vòng này: loss, physical weight-change norms và optimizer giữ được ghi lại nhưng không chứng minh lợi ích độc lập với amplitude.','',
      '## Những gì đã kế thừa và điều vẫn thiếu','',
      '- `ALREADY_SOLVED_IN_LITERATURE`: eligibility/Jacobian theo intrinsic multi-state dynamics, phân tích e-prop factorization và SI path importance.',
      '- `REPRODUCED_IN_SCARX`: NumPy adapters, upstream trace AST parity, local/full surrogate oracle và strict checkpoint/cold resume trong phạm vi technical tests.',
      '- `PARTIALLY_SOLVED`: hướng credit trên frozen fixtures; local block vẫn bỏ cross-neuron recurrent sensitivities, learning signal và normalizer còn xấp xỉ.',
      '- `UNRESOLVED`: một checkpoint tích lũy nhiều kỹ năng, readout overwrite, task-agnostic retention, distractor/OOD bền vững, chat/vision đa năng.',
      '- `POTENTIAL_NEW_CONTRIBUTION`: chưa đủ bằng chứng gán nhãn này cho ACR; không tính việc chuyển phương trình có sẵn thành tính mới.',
      '', 'Chi tiết: [e-prop](literature/eprop.md), [BrainTrace](literature/braintrace.md), [reuse matrix/SI/EWC/DECOLLE/DGN/R9b](literature/continual-and-local-signals.md).',
      'SI helper qua10test toán nhưng chưa có retention/P2. Explicit consolidation dùng task boundaries là boundary-assisted; dùng local credit thay loss gradient là biến đổi cần kiểm riêng. EWC/Fisher và replay chưa được chứng minh ở protocol mới. DGN đổi neuron dynamics nên ngoài primary forward đang khóa; DECOLLE pin hiện có điều khoản phi thương mại, không copy code.',
      '', '## ACR và bước tiếp theo','',
      'ACR cũ bị gián đoạn sau55cell hợp lệ; còn110AB cell và chưa C/norm/shuffle/equalCPU. Không gọi partial study là ACR thắng/thua. [Audit gián đoạn](literature/acr-interruption-audit.md) xác nhận12own-reference cells khớp exact với alpha0 cũ. Không vứt bỏ raw hoặc chạy lại confirmation.',
      'Quyết định mở rộng ACR phải dựa cả reference DEV này, direct/representation riêng và matched controls. Không bảo vệ router bằng cách nới gate. P1 vẫn cần2/3primary đủ direct65%, representation CI/matched shuffle, OODNI/protected; P2 vẫn acquisition65%/final65%/mean forgetting3đ/worst5đ theo seed. Chưa mở fresh confirmation/P2.',
      '', '## Tái lập và bàn giao','',
      'Branch `codex/scarx-literature-first-20261112`, base `950718bd410371f8ef15dab2d78d722b04ab506d`. Main trước/sau `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`.',
      '```powershell',
      'python scripts/prepare_literature_references.py',
      'python scripts/build_native.py',
      'python -m pytest -q -p no:cacheprovider tests/test_literature_eprop.py tests/test_literature_braintrace.py tests/test_literature_si.py tests/test_literature_snapshots.py tests/test_literature_oracle.py tests/test_native.py',
      'python scripts/run_literature_reference_dev.py',
      '# Runner reserves once; reproductions use a new isolated checkout, never delete old reservation to overwrite evidence.',
      f'python scripts/audit_literature_reference_dev.py --run runs/{run.name}',
      '```',
      f'Raw `runs/{run.name}/`; per-seed table `per-seed-results-v2.csv`, audit `audit-v2.json`, checkpoint/initial/history ở từng job. Source hashes, upstream commit/license và data hashes đi cùng run. Observer http://127.0.0.1:8772 chỉ đọc outputs thật, không mô phỏng lời nghĩ.',
      'Chưa merge; final delivery commit/push được xác minh riêng. Không đổi production core/checkpoint/SQLite/Hermes.']
    out.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(str(out))
if __name__=='__main__':main()
