"""Registered numeric proposal30 fixtures/setup; no serving or Torch import."""
from pathlib import Path
from scarx.neural.core import Config
from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.tasks21 import representation
from scarx.evaluation.binding_audit21 import strict_binding

ROOT=Path(__file__).resolve().parents[1]
TASKS=('order','xor','strict_binding');METHODS=('correct','shuffled','zero','frozen','bptt_control')
DEV=(35113,35119,35141);CONFIRM=(35201,35219,35227,35233,35239,35243,35257,35261)


def dataset(task,seed,count,ood=False):
    if task not in TASKS:raise ValueError('Registered temporal task')
    return ((None,None) if ood else strict_binding(seed,count)) if task=='strict_binding' else representation(task,seed,count,ood)


def make(task,seed,method):
    if method not in METHODS:raise ValueError('Registered proposal30 method')
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0)
    m=ReadoutEligibility(c,'correct' if method=='bptt_control' else method);cal,_=dataset(task,seed+1360000,128);m.calibrate(cal)
    if method=='bptt_control':
        from scarx.evaluation.controls import EligibilityLearnabilityControl
        return EligibilityLearnabilityControl(m)
    return m


def load(method,payload):
    if method=='bptt_control':
        from scarx.evaluation.controls import EligibilityLearnabilityControl
        return EligibilityLearnabilityControl.loads(payload)
    return ReadoutEligibility.loads(payload)


def reference(model):return getattr(model,'reference',model)
