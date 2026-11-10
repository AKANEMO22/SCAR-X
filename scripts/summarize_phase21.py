"""Build a linked report from immutable results, never edit measured runs."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
def load(pattern):
    candidates=sorted(p for p in (root/'runs').glob(pattern) if (p/'result.json').exists())
    if not candidates: raise SystemExit('Missing completed '+pattern)
    p=candidates[-1]
    return p,json.loads((p/'result.json').read_text())

mainp,main=load('*-phase21-v1-*')
lifep,life=load('*-phase21-lifelong-v1-*')
bindp,binding=load('*-phase21-binding-audit-v1-*')
controlp,control=load('*-phase21-bptt-control-v1-*')
replicatep,replicate=load('*-phase21-reproduction-*')
audit=json.loads((mainp/'phase2_audit.json').read_text())
def pct(x): return f'{100*x:.2f}'
def ci(v): return f"{pct(v['mean'])} [{pct(v['lower95'])}; {pct(v['upper95'])}]"
def link(path): return '../../runs/'+path.name+'/report.md'

lines=['# 08 — Phase2.1: Lifelong Adaptive Intelligence','',
       '**Kết luận: PIVOT tiếp ở quy mô nhỏ; NO-GO cho mở rộng quy tắc/cấu hình hiện tại. Chưa có lợi thế continual learning đáng tin cậy so với các đối chứng đã chạy.**','',
       'Cập nhật ngày 2026-11-09. Mục tiêu 1M context đã bị loại bỏ. Giữ 48 neuron, local learning, RAM 2 GB, mean forgetting ≤3 điểm phần trăm và worst ≤5 điểm phần trăm. Không thay ngưỡng sau kết quả.','',
       '## Hồ sơ và thiết kế','',
       f'- [Representation/replay factorial]({link(mainp)}): 3 development seeds, 8 confirmation seeds; lựa chọn cấu hình trước test.',
       f'- [Stream không task boundary]({link(lifep)}): 7 methods×8 seeds,1280 trải nghiệm có noise, periodic consolidation.',
       f'- [Strict binding audit]({link(bindp)}): 8 seed mới, không chọn lại rule.',
       f'- [BPTT control riêng]({link(controlp)}): 24trial; không thuộc nhánh SCAR-X.',
       f'- [Kiểm tra tái lập]({link(replicatep)}): source snapshot, statistics và recorded-update schedule.',
       '- Tổng 303 trial mới, ngoài audit lịch sử. CI bootstrap theo seed; task contrasts có Holm correction. Các condition dùng cùng seed là so sánh ghép cặp, không phải seed độc lập mới.','',
       '## 1. Vì sao Phase2 quên?','',
       'Toàn bộ15trial và checkpoint được kiểm lại. Full: acquisition ban đầu trung bình45.36%, final34.17%. Max-to-final forgetting22.29pp; diagonal-to-final11.20pp. Phần chênh11.09pp bao gồm các đỉnh xuất hiện sau lần học đầu và biến động; không quy hết cho nhiễu thống kê.','',
       '| Seed | Task0 sau học | Cuối stream | Khôi phục head cũ | Khôi phục hidden cũ | Nhóm replay bị thiếu/80 |',
       '|---|---:|---:|---:|---:|---:|']
for r in audit['counterfactuals']:
    if r['mode']=='full': lines.append(f"| {r['seed']} | {pct(r['task0_after_learning'])}% | {pct(r['task0_final'])}% | {pct(r['task0_old_head_final_hidden'])}% | {pct(r['task0_old_hidden_final_head'])}% | {r['missing_groups']} |")
lines+=['','Head overwrite là nguyên nhân nổi bật ở task0 được audit. Replay thiếu coverage; anchor EMA không chứa importance về mức hữu ích. Các context dùng permutation khác trên cùng head, tạo interference. Đây là counterfactual trên task0, không phải phân rã nhân quả cho mọi kỹ năng.','',
        '## 2. Hidden có học representation mới?','',
        '| Bài | Probe gain so với initial, pp [CI95%] | Accuracy trực tiếp | Holm p | Gate |','|---|---:|---:|---:|---|']
for task,c in main['analysis']['representation'].items(): lines.append(f"| {task} | {ci(c['probe_gain'])} | {pct(c['accuracy']['mean'])}% | {c['holm_signflip_p']:.4f} | {c['pass']} |")
lines+=['','Score eligibility có tín hiệu nhỏ ở order, chưa vượt ngưỡng 5 điểm phần trăm và hiệu chỉnh kiểm định nhiều lần. Trọng số thay đổi thật chưa đủ chứng minh representation hữu ích. Không vượt ổn định đối chứng đảo dấu modulation.','',
        '**Lỗi thiết kế được phát hiện:** binding ban đầu có shortcut bag-of-values đạt75% lý thuyết. Giữ kết quả nguyên bản nhưng không tính đó là khả năng binding. Challenge mới buộc hai value đối nhau, giữ cùng số sự kiện/key/value. Đây là audit bổ sung sau khi nhìn kết quả, dùng seed mới và rule cố định.','',
        f"Strict binding: probe={ci(binding['summary']['probe'])}%; gain={ci(binding['summary']['probe_gain'])}pp; accuracy={ci(binding['summary']['accuracy'])}%; bag-only={ci(binding['summary']['bag_accuracy'])}%; exact p={binding['summary']['signflip_p']:.4f}; pass={binding['summary']['pass']}.",
        '', 'Temporal permutation cố định có thể giữ hoặc biến đổi tín hiệu order, nên một lesion tăng điểm không chứng minh model hiểu quan hệ thời gian. Reset-state và OOD chỉ là chẩn đoán phụ.','',
        '## 3. Replay và consolidation với budget đối chiếu','',
        f"Development chọn hidden={main['selection']['hidden_rule']}, replay={main['selection']['replay']}, consolidation={main['selection']['consolidation']}; time budget={main['selection']['matched_stage_seconds']:.6f}s/stage đã khóa trước confirmation.",
        '', '| Budget | Condition | Acquisition | Final | Mean forgetting | Train seconds | Updates |', '|---|---|---:|---:|---:|---:|---:|']
for budget,c in main['analysis']['retention'].items():
    for name,row in c['conditions'].items(): lines.append(f"| {budget} | {name} | {pct(row['acquisition']['mean'])}% | {pct(row['final_accuracy']['mean'])}% | {pct(row['forgetting']['mean'])}pp | {row['train_seconds']['mean']:.3f} | {row['updates']['mean']:.1f} |")
lines+=['','Cùng số cập nhật: bỏ EMA giảm forgetting khoảng 0,22 điểm phần trăm, quá nhỏ so với yêu cầu 5 điểm. Cùng ngân sách thời gian: lợi thế không còn ổn định. Acquisition quanh 30% (mức đoán ngẫu nhiên 25%) chưa thể gọi là giữ kỹ năng thành công. Balanced replay + metaplasticity có forgetting thấp trong development nhưng acquisition yếu; không chọn cấu hình chỉ vì quên ít.','',
        'Không so trực tiếp mức forgetting Phase 2 với Phase 2.1 như một cải thiện thuật toán: forward đã chuyển sang stochastic ALIF, độ dài episode và bộ đánh giá cũng khác. Chỉ các đối chứng trong cùng thiết kế mới hỗ trợ so sánh.','',
        '## 4. Stream không boundary, noisy feedback và plasticity','',
        '| Cơ chế | Final accuracy | Mean forgetting | TB mức quên tệ nhất mỗi seed | New-rule gain | Fresh-network gain |', '|---|---:|---:|---:|---:|---:|']
for mode,c in life['summaries'].items():
    lines.append(f"| {mode} | {pct(c['final_accuracy']['mean'])}% | {pct(c['forgetting']['mean'])}pp | {pct(c['worst_forgetting']['mean'])}pp | {pct(c['new_rule_gain']['mean'])}pp | {pct(c['fresh_new_rule_gain']['mean'])}pp |")
lines+=['','| Primary contrast | Final accuracy gain, pp [CI95%] | Forgetting reduction, pp [CI95%] | Holm p |','|---|---:|---:|---:|']
for name,c in life['comparisons'].items(): lines.append(f"| {name} | {ci(c['accuracy_gain'])} | {ci(c['forgetting_reduction'])} | {c['holm_p']:.4f} |")
lines+=['','Các cột trong bảng là trung bình theo seed; gate tuyệt đối được kiểm riêng trên mọi seed, không chỉ trung bình này. Không condition nào đạt toàn bộ ngưỡng. Metaplasticity/decay giảm số đo forgetting ở một số trường hợp nhưng accuracy gần mức đoán ngẫu nhiên 50%: chưa chứng minh giữ được kỹ năng. Importance dựa độ lớn tín hiệu có thể bảo vệ cả update nhiễu; chưa đo được importance nhân quả.','',
        '![Độ chính xác và forgetting cùng CI95%](../../artifacts/phase21-figures/lifelong_accuracy_forgetting.png)','',
        'Selective learning được thêm metadata chất lượng không hoàn hảo; so với random-rejection để tránh nhầm ít update với học tốt. Kết quả chỉ áp dụng cho giả lập chất lượng nguồn này, không chứng minh hệ tự phân biệt chân lý. SI-readout là adaptation có khả năng học khác, không phải whole-network SI.','',
        'Probe cuối dùng đảo nhãn binding 64 mẫu; đó là thích nghi sang quy tắc xung đột, không thể đòi đồng thời giữ nhãn cũ cho cùng đầu vào. So mạng đã học với mạng mới còn phụ thuộc điểm khởi đầu; CI của chênh lệch rộng. Chưa đủ kết luận mạng mất plasticity lâu dài. Bài này vẫn kế thừa shortcut binding ban đầu, nên không được dùng chứng minh học quan hệ mới.','',
        '## 5. So với recurrent/backprop control','', '| Bài | BPTT accuracy trung bình |', '|---|---:|']
for task in ('xor','order','binding'):
    values=[r['accuracy'] for r in control['trials'] if r['task']==task]
    lines.append(f'| {task} | {pct(float(np.mean(values)))}% |')
lines+=['','BPTT control dùng deterministic ALIF, nhánh local dùng stochastic ALIF; cùng kích thước và 512 mẫu, chi phí tính toán thực khác. Không suy diễn trực tiếp thành lợi thế phần cứng. Bài binding control cũng chịu shortcut đã nêu. BPTT cũng không giải được XOR ở ngân sách này: điều đó giới hạn khả năng phân biệt các cơ chế trên bài XOR, không chứng minh bài toán bất khả thi.','',
        'Đã so với replay reservoir/balanced, readout-only recurrent, EMA, local metaplasticity và output-local SI. Chưa tái lập MESU, BNN metaplasticity hay Continual Backprop đầy đủ trong cùng budget; không có cơ sở tuyên bố thắng các phương pháp hiện đại nói chung.','',
        '## 6. RAM, persistence và phán quyết','', '| Run | Peak RSS MB | Peak commit MB |', '|---|---:|---:|']
for name,data in [('phase21',main['analysis']),('lifelong',life),('binding audit',binding),('BPTT control',control)]:
    r=data['resources']; lines.append(f"| {name} | {r['peak_rss_bytes']/1e6:.2f} | {r['peak_commit_or_vms_bytes']/1e6:.2f} |")
lines+=['','Trọng số, traces, importance, reward/RNG và replay continuation đã được kiểm tra. RAM quan sát trong trần 2 GB; đây là giám sát, chưa phải giới hạn cứng do hệ điều hành áp đặt. Không có kết luận về iGPU hoặc điện năng.','',
        'Các thí nghiệm đo ba thang: cập nhật trọng số nhanh mỗi trải nghiệm; EMA stable anchor mỗi 64 cập nhật; importance tăng hoặc giảm chậm hơn. Việc có ba thang thời gian chưa chứng minh cách phối hợp chúng là đúng. Importance từ độ lớn tín hiệu tăng kể cả khi nhãn nhiễu; protection có thể củng cố một trạng thái chưa học hữu ích. Đây là giả thuyết giải thích cần kiểm tra nhân quả tiếp.','',
        '**Có thể học tiếp không cần BPTT ở phạm vi hữu hạn; chưa chứng minh SCAR-X duy trì kỹ năng phức tạp lâu dài.** Capacity cố định không giữ vô hạn ánh xạ độc lập. Credit assignment scalar và chất lượng representation hiện là nút thắt; memory/replay không tự giải quyết chúng. Xem [phản biện giới hạn](07_LIFELONG_LIMITS.md).','',
        '**Quyết định:** PIVOT hẹp về importance gắn với mức hữu ích, ổn định readout và credit assignment/representation. NO-GO cho tăng model hoặc quảng bá lợi thế Lifelong của rule hiện tại. Không đổi ngưỡng, không huấn luyện ngôn ngữ, không mở rộng tham số.','',
        'Chưa đủ cơ sở nói SCAR-X vượt các phương pháp hiện có. Kết quả âm chỉ bác bỏ ưu thế của các quy tắc và cấu hình đã thử trong ngân sách đã công bố. Hệ hữu hạn có thể tiếp tục học bằng cách chọn lọc, tái sử dụng và quên có chủ đích; không thể bảo đảm giữ chính xác vô hạn kỹ năng độc lập trong dung lượng cố định.']
(root/'docs/research/08_PHASE21_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
summary={'phase21':str(mainp.relative_to(root)),'lifelong':str(lifep.relative_to(root)),'strict_binding':str(bindp.relative_to(root)),
         'control':str(controlp.relative_to(root)),'reproduction':str(replicatep.relative_to(root)),'decision':'PIVOT_NO_SCALE',
         'report':'docs/research/08_PHASE21_RESULTS.md','new_trial_count':len(main['development_representation'])+len(main['development_retention'])+len(main['confirmation_representation'])+len(main['confirmation_retention'])+len(life['trials'])+len(binding['trials'])+len(control['trials'])}
(root/'artifacts/phase21-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
