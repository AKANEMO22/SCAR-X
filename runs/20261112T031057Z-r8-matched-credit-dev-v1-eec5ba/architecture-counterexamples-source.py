"""Counterexamples to architecture-necessity claims; no neural training."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.neural.core import Config,SpikingCore
from scarx.evaluation.r8_matched_credit import peer_module,PEER_SHA
from scarx.evaluation.forensic.research_gates import require_injective_codebook


def main():
    if len(sys.argv)!=2:raise ValueError('Fresh owned output path required')
    path=Path(sys.argv[1]).resolve()
    if not path.is_relative_to(ROOT/'runs') or path.exists():raise ValueError('Fresh registered run output required')
    cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
    a=SpikingCore(cfg);a.w_in.fill(0);a.w_rec.fill(0);a.w_in[:,0]=1
    b=copy.deepcopy(a);a.voltage.fill(.2);b.voltage.fill(.4)
    x=np.zeros(8,np.float32);x[0]=.9;a.step(x,False);b.step(x,False)
    result={'peer_sha':PEER_SHA,'new_quality_training':False,'subtractive_reset':{
        'both_emit_spike':bool(np.all(a.spikes==1)&np.all(b.spikes==1)),
        'voltage_a_after':float(a.voltage[0]),'voltage_b_after':float(b.voltage[0]),
        'prior_state_difference_after':float(np.max(np.abs(a.voltage-b.voltage))),
        'scope':'Same ALIF equations, subtraction reset does not set all membrane information to zero'}}
    generator=peer_module('r8_benchmark_integrity');data,_=generator.generate_distractor_binding(42,16)
    clamped=SpikingCore(cfg);clamped.w_in[:,6]=0;other=copy.deepcopy(clamped)
    maximum=0.
    for seq in data:
        modified=seq.copy();modified[3,6]+=2
        clamped.reset_state();other.reset_state()
        for left,right in zip(seq,modified):
            clamped.step(left,False);other.step(right,False)
            maximum=max(maximum,float(np.max(np.abs(clamped.voltage-other.voltage))))
    result['selective_synapse_counterexample']={'distractor_channel':6,'added_pulse':2.,'voltage_max_difference':maximum,
        'scope':'Manually clamped weight, diagnostic capacity control only; not a learned result or robustness to noise on relevant channels'}
    codes={k:tuple(1.5 if k&(1<<bit) else -1.5 for bit in range(3)) for k in range(8)}
    result['repaired_8key_codebook_injective']=require_injective_codebook(codes)
    result['context_readout_parameter_count']={'base_readout_weights':2*48,'extra_context_weights':8*2*48,'effective_total':9*2*48,
        'scope':'Current source head has9x base readout weights; not a matched-parameter proof of hidden capacity'}
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
