"""Additive metric audit, shared-prompt reference, exploratory rote retention."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,sys,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.knowledge_capacity import probe,fraction,retention,frozen_slot
from scarx.evaluation.runs import new_run,write_json,cluster_interval
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
cap=apply_memory_limit();primary=json.loads((root/'artifacts/knowledge-retention-latest.json').read_text(encoding='utf-8'))
reference=json.loads((root/'artifacts/qwen4b-comparison-latest.json').read_text(encoding='utf-8'))
source=root/primary['run'];compare_rows=json.loads((root/reference['run']/'comparison-data.json').read_text(encoding='utf-8'))
rows=json.loads((source/'data.json').read_text(encoding='utf-8'));distractors=rows[160:]
old=json.loads((source/'old-baseline.json').read_text(encoding='utf-8'))
run=new_run(root,'knowledge-retention-audit',{'protocol':'experiments/knowledge-load-audit-v1.md','primary_run':primary['run'],'reference_run':reference['run'],'posthoc_rote_probe':True},data_description='Existing public arithmetic checkpoints; additive correction of net-drop metric and shared-format comparison')
def families(outputs):
    return np.array([fraction([x for x in outputs if x['id'].startswith(f'old-{i}-')]) for i in range(8)])
results=[]
with ResourceMonitor(interval=.1) as monitor:
 for row in primary['results']:
    condition,seed=row['condition'],row['order_seed'];m=UnifiedCore.loads((root/row['checkpoint']).read_bytes())
    peak=families(old);audited=[]
    for stage in row['stages']:
        raw=json.loads((source/f"{condition}-{seed}-stage-{stage['lessons_presented']}.json").read_text(encoding='utf-8'))
        current=families(raw['old']);loss=np.maximum(0,peak-current)
        audited.append({'lessons_presented':stage['lessons_presented'],'mean_forgetting_pp':float(100*loss.mean()),
          'worst_family_forgetting_pp':float(100*loss.max()),'family_forgetting_pp':(loss*100).tolist(),
          'original_mislabeled_net_drop_pp':stage['old_mean_forgetting_pp'],'historical_gate_met':bool(loss.mean()<=.03 and loss.max()<=.05)})
        peak=np.maximum(peak,current)
    shared=probe(m,compare_rows);write_json(run/f'{condition}-{seed}-shared-prompts.json',shared)
    seen=json.loads((source/f'{condition}-{seed}-probe-items.json').read_text(encoding='utf-8'))
    at_start=probe(m,seen,'prompt');branches={}
    for branch in ('quiet','interference','interference_replay'):
        b=UnifiedCore.loads(m.dumps());rng=np.random.default_rng(seed+991);t0=time.monotonic()
        if branch=='quiet':
            for _ in range(10000):b.tick()
        else:
            for item in distractors:
                b.expand_vocabulary(item['target'])
                for k in range(32):
                    item2=seen[int(rng.integers(len(seen)))] if branch=='interference_replay' and k%4==3 else item
                    if condition=='frozen':frozen_slot(b,item2['prompt'],item2['target'])
                    else:b.train_pair(item2['prompt'],item2['target'],'readout' if condition=='readout' else 'full')
        now=probe(b,seen,'prompt');value={'exact':fraction(now),'retention_from_start_of_secondary_probe':retention(at_start,[],now),'seconds':time.monotonic()-t0}
        branches[branch]=value;write_json(run/f'{condition}-{seed}-rote-{branch}.json',{'summary':value,'outputs':now})
    results.append({'condition':condition,'order_seed':seed,'lessons_presented':row['lessons_presented'],
      'audited_forgetting':audited,'primary_paraphrase_exact':row['final_exact'],'trained_wording_exact':fraction(at_start),
      'shared_format_exact_160':fraction(shared),'shared_format_unseen32':fraction(shared[128:160]),
      'rote_probe':branches,'primary_useful_acquisition':row['retention']['acquired'],'cold_process_restart_exact':row['restart_exact']})
    monitor.check()
 groups={}
 for condition in ('frozen','full','readout','full_replay','guarded'):
    rs=[r for r in results if r['condition']==condition]
    groups[condition]={'orders':len(rs),'presented':sorted(set(r['lessons_presented'] for r in rs)),
      'trained_wording_mean':float(np.mean([r['trained_wording_exact'] for r in rs])),
      'primary_paraphrase_mean':float(np.mean([r['primary_paraphrase_exact'] for r in rs])),
      'shared_format_mean':float(np.mean([r['shared_format_exact_160'] for r in rs])),
      'old_mean_forgetting_pp':float(np.mean([r['audited_forgetting'][-1]['mean_forgetting_pp'] for r in rs])),
      'old_worst_family_pp':max(r['audited_forgetting'][-1]['worst_family_forgetting_pp'] for r in rs),
      'rote_remembered_before':sum(r['rote_probe']['quiet']['retention_from_start_of_secondary_probe']['acquired'] for r in rs),
      'rote_retained_after_interference':sum(r['rote_probe']['interference']['retention_from_start_of_secondary_probe']['retained'] for r in rs),
      'rote_retained_with_replay':sum(r['rote_probe']['interference_replay']['retention_from_start_of_secondary_probe']['retained'] for r in rs)}
 hidden=[next(r for r in results if r['condition']=='full' and r['order_seed']==s)['primary_paraphrase_exact']-next(r for r in results if r['condition']=='readout' and r['order_seed']==s)['primary_paraphrase_exact'] for s in (17,29,43)]
 result={'run':run.relative_to(root).as_posix(),'primary_run':primary['run'],'reference_run':reference['run'],'groups':groups,'results':results,
   'qwen_exact':reference['exact'],'qwen_unseen_exact':reference['unseen_exact'],'qwen_completed':reference['completed'],
   'qwen_server_stopped':reference['server_stopped'],'qwen_ram':reference['resources'],'scarx_primary_ram':primary['resources'],
   'audit_ram':monitor.report(),'hard_cap':cap,'hidden_vs_readout_paraphrase':cluster_interval(hidden),
   'stats_scope':'Three curriculum-order seeds, ONE pretrained initialization; bootstrap cannot establish population-level equality.',
   'metric_correction':'Original old_mean_forgetting_pp was NET overall drop. This audit averages per-family forgetting from maximum past performance; thresholds unchanged, no stopping decisions affected.',
   'signed_answer_decoder_limitation_unresolved':True,'primary_retention_undefined_no_useful_acquisition':True,
   'calendar_long_term_retention_tested':False,'four_billion_equivalence':False,'deployed':False,'decision':'PIVOT_NO_SCALE'}
 write_json(run/'result.json',result);write_json(root/'artifacts/knowledge-retention-audit-latest.json',result)
 lines=['# Tăng tải kiến thức và đối chiếu Qwen4B','','## Kết quả chính','',
  '| Cơ chế | Bài đã nạp/order | Đúng nguyên cách hỏi | Đúng cách hỏi mới | Quên kỹ năng cũ TB | Xấu nhất |',
  '|---|---:|---:|---:|---:|---:|']
 for c,g in groups.items():lines.append(f"| {c} | {g['presented']} | {100*g['trained_wording_mean']:.2f}% | {100*g['primary_paraphrase_mean']:.2f}% | {g['old_mean_forgetting_pp']:.2f}pp | {g['old_worst_family_pp']:.2f}pp |")
 lines+=['',f"Qwen3-4B-Instruct-2507 Q4_K_M: {reference['completed']} câu, exact {100*reference['exact']:.2f}%; server đã tắt: {reference['server_stopped']}.",
  'Cùng160 chuỗi câu hỏi số học trong phép đối chiếu phụ; không suy rộng thành toàn bộ kiến thức4B. Qwen có pretrained weights và RAM cao hơn; không là matched-memory comparison.',
  '', '## Nhớ lâu và ablation','',
  'Không có ví dụ mới được học hữu ích theo cách hỏi chưa gặp, nên primary retention không xác định. Bài ghi nhớ nguyên văn dưới đây là phân tích thăm dò thêm sau kết quả, không thay tiêu chí thành công:',
  '', '| Cơ chế | Nhớ nguyên văn trước nhiễu | Giữ sau8bài nhiễu | Giữ khi replay |','|---|---:|---:|---:|']
 for c,g in groups.items():lines.append(f"| {c} | {g['rote_remembered_before']} | {g['rote_retained_after_interference']} | {g['rote_retained_with_replay']} |")
 lines+=['','Cold process restart khớp;10kquiet ticks không kiểm chứng nhớ ngày/tháng. Các nhóm có3orderseed từ1initialization; CI chỉ phản ánh biến thiên thứ tự. Hidden/readout đều fail paraphrase, chưa chứng minh representation.',
  '', '## Audit và giới hạn','',result['metric_correction'],
  'V1 signed arithmetic abort do leading minus bị decoder mask; v1r1 chỉ đáp án không âm. Raw runs/protocol giữ nguyên. Core live và private weights không thay; không mở pretraining/model lớn.',
  '', '**Quyết định: PIVOT_NO_SCALE.** Sửa credit assignment/representation/readout stability trước khi tăng lượng data. Không chọn candidate để thay model người dùng.']
 (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'run':str(run),'groups':groups,'qwen_exact':reference['exact'],'qwen_peak_ram':reference['resources']['peak_rss_bytes']},ensure_ascii=True),flush=True)
