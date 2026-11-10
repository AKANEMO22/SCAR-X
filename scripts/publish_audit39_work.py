"""Publish the completed audit39 work to the passive localhost observer.

Read-only with respect to neural state: only runtime.work_trace.activity records
are appended. No inference, teaching or training is triggered.
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, json
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from scarx.runtime.work_trace import activity, read
from scarx.runtime.experiment_status import RUN_ID

RUN = 'runs/20261110T080426Z-local-credit-alignment-audit-v1-db01fc'
SUPERSEDED = 'runs/20261110T080003Z-local-credit-alignment-audit-v1-d8de33'

if not RUN_ID.fullmatch(RUN[5:]):
    raise ValueError('Registered run id')

result = json.loads((root / RUN / 'result.json').read_text(encoding='utf-8'))
if result.get('decision') != 'LOCAL_CREDIT_ALIGNMENT_DIAGNOSTIC_ONLY':
    raise ValueError('Unexpected decision')
macro = result['dev_macro']
flags = result['flags']
heldout = result['heldout']
ci = result['dev_gain_vs_signflip_interval']

entries = [
    ('Đọc 15 checkpoint nguồn và kiểm SHA256',
     'Xác minh SHA256 của 15 checkpoint vòng 36 trước khi đo; mọi field được bảo vệ '
     '(weights/stable/topology/head stats/Adam/counters/RNG) được assert không đổi. '
     'Không mở learner, optimizer hay head refit.', {'source_models': 15}),
    ('Phát hiện lỗi trong phép đo của chính mình',
     'Bản chạy đầu chụp norm áp dụng SAU khi cập nhật nên mọi applied_norm đọc 0,0 và '
     'việc khớp độ lớn control không kiểm chứng được. Đã sửa thứ tự chụp, thêm control '
     'hoán vị và test hồi quy bắt đúng lỗi này. Bản lỗi đánh dấu do_not_cite.', None),
    ('Đo ΔNLL thật: hướng local có giảm loss?',
     f'Dev 192 case: đúng hướng {macro["alignment_fraction"]*100:.2f}%, '
     f'ΔNLL {macro["delta_direction"]:+.5f} (âm = loss giảm), đảo dấu {macro["delta_signflip"]:+.5f}. '
     f'Chênh so đảo dấu {macro["gain_vs_signflip"]:+.5f} '
     f'CI95% {ci["lower95"]:+.5f}..{ci["upper95"]:+.5f}.',
     {'alignment_fraction': macro['alignment_fraction'], 'neural_updates': 0}),
    ('Cảnh báo độ lớn control — đổi cách đọc kết quả',
     'Adam chuẩn hoá theo từng phần tử nên control Gaussian bị áp lực lớn hơn thật '
     f'{macro["norm_match_ratio_random"]:.2f}× và control hoán vị '
     f'{macro["norm_match_ratio_permuted"]:.2f}×. Chỉ control đảo dấu khớp chính xác '
     'độ lớn, nên nó là bằng chứng chính; mọi "gain so random" bị nhiễu độ lớn.',
     {'ratio_random': macro['norm_match_ratio_random'],
      'ratio_permuted': macro['norm_match_ratio_permuted']}),
    ('Kết luận trung thực: đúng hướng nhưng không tổng quát hoá',
     f'Held-out (hướng gộp từ TRAIN, áp một lần, chấm dev) ΔNLL '
     f'{heldout["mean_dev_delta"]:+.8f} '
     f'CI95% {heldout["interval"]["lower95"]:+.8f}..{heldout["interval"]["upper95"]:+.8f}. '
     'Tín hiệu đúng dấu trên từng bài, nhưng hướng gộp từ train gần như không chuyển '
     'sang bài khác. Đây là chẩn đoán cơ chế, KHÔNG phải normalchat/vision/retention.',
     flags),
    ('Kiểm thử và tài nguyên',
     '9 test mới cho hình học tín hiệu + 4 test cho accessor UI. Toàn bộ 214 test qua. '
     'Peak RSS 53,40 MB, commit 38,24 MB, cap 2 GB không đổi; 0 neural update, không Torch.',
     {'tests': 214, 'torch_imported': False}),
]

for title, detail, evidence in entries:
    activity(root / RUN, title, detail, status='completed', evidence=evidence)

activity(root / SUPERSEDED, 'Bản chạy này bị thay thế, không được trích dẫn',
         'Lỗi đo: applied_norm chụp sau cập nhật nên luôn đọc 0,0, không kiểm chứng được '
         'khớp độ lớn control. Giữ lại làm lịch sử theo quy tắc không ghi đè run cũ; '
         'dùng bản db01fc thay thế.', status='completed',
         evidence={'superseded_by': RUN, 'do_not_cite': True})

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

recent = [r for r in read(root) if r.get('run_id') == RUN[5:]]
print(json.dumps({'published': len(recent), 'run': RUN,
                  'titles': [r['title'] for r in recent[:len(entries)]],
                  'flags': flags}, ensure_ascii=True, indent=2))
