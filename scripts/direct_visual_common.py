"""Pinned37sources, unchanged32sampler, no Torch import unless controlrequested."""
from visual_word_control_common import source_data,sample_stream,SEEDS as OLD_SEEDS,DATA_SHA
from scarx.evaluation.direct_visual import DirectVisualWords

SEEDS=(37313,37321,37339);METHODS=('correct','shuffled','zero','frozen','no_image','no_question','bptt_control')


def make(vocab,seed,method,train,images):
    if method not in METHODS:raise ValueError('Registered37method')
    m=DirectVisualWords(vocab,seed,'correct' if method=='bptt_control' else method)
    m.calibrate([{'prompt':r['prompt'],'image':images[r['scene_id']]} for r in train])
    if method=='bptt_control':
        from scarx.evaluation.controls import DirectVisualLearnabilityControl
        return DirectVisualLearnabilityControl(m)
    return m


def reference(m):return getattr(m,'reference',m)


def load(payload,method):
    if method=='bptt_control':
        from scarx.evaluation.controls import DirectVisualLearnabilityControl
        return DirectVisualLearnabilityControl.loads(payload)
    return DirectVisualWords.loads(payload)
