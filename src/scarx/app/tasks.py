"""Structured MVP tasks. Rules verify labels; never supply neural predictions."""
import numpy as np

TASKS={'A':{'name':'A · Nhóm tín hiệu','rule':'0, 1 → nhãn 0; 2, 3 → nhãn 1'},
       'B':{'name':'B · Tín hiệu chẵn / lẻ','rule':'0, 2 → nhãn 0; 1, 3 → nhãn 1'}}


def normalize_sample(data):
    task=data.get('task'); signal=data.get('signal')
    if task not in TASKS or type(signal) is not int or signal not in range(4):
        raise ValueError('Tác vụ A/B và tín hiệu nguyên từ 0 đến 3 là bắt buộc.')
    amplitude=data.get('amplitude',1.)
    if isinstance(amplitude,bool) or not isinstance(amplitude,(int,float)) or not np.isfinite(amplitude) or not .7<=amplitude<=1.3:
        raise ValueError('Cường độ phải trong khoảng 0,7–1,3.')
    noise_seed=data.get('noise_seed',12345)
    if type(noise_seed) is not int or not 0<=noise_seed<2**31: raise ValueError('Seed không hợp lệ.')
    return {'task':task,'signal':signal,'amplitude':float(amplitude),'noise_seed':noise_seed}


def expected_label(sample):
    return sample['signal']//2 if sample['task']=='A' else sample['signal']%2


def encode(sample):
    """No target in the encoding. Task cue is explicit, observable input."""
    rng=np.random.default_rng(sample['noise_seed'])
    x=rng.normal(0,.025,(12,8)).astype(np.float32)
    x[:3,sample['signal']]+=sample['amplitude']
    x[:,4+(sample['task']=='B')]+=.65
    return x


def examples(task,count,rng):
    if task not in TASKS or type(count) is not int or not 4<=count<=128 or count%4:
        raise ValueError('Số ví dụ phải là bội của 4, từ 4 đến 128.')
    samples=[{'task':task,'signal':i%4,'amplitude':float(rng.uniform(.8,1.2)),
              'noise_seed':int(rng.integers(0,2**31-1))} for i in range(count)]
    rng.shuffle(samples)
    return samples
