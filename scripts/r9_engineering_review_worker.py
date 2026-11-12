"""Pinned R9 no-learning counterexamples; never runs its search script."""
from pathlib import Path
import io,json,os,subprocess,sys,zipfile
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np


def main():
    source=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();sha='2c41fdaf6fc491845879d638ac80449caf597471'
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=sha or output.exists():raise ValueError('Pinned source/fresh output required')
    sys.path.insert(0,str(source/'src'))
    from scarx.neural.core import Config
    from scarx.neural.r9_unified_mechanisms import R9UnifiedNeuralModel as M,generate_timescale_vector
    cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
    a=M(cfg,reset_mode='subtractive_exact');b=M(cfg,reset_mode='hard_reset')
    seq=np.full((5,8),.03,np.float32);seq[0,[0,2]]=1.5;seq[3,4]=1.5;seq[4,7]=1
    max_forward=0.
    for row in seq:
        a.step(row,True);b.step(row,True);max_forward=max(max_forward,float(np.max(np.abs(a.core.voltage-b.core.voltage))))
    result={'peer_sha':sha,'new_learning_calls':0,'hard_reset_vs_subtractive':{'forward_voltage_max_difference':max_forward,
        'eligibility_max_difference':float(np.max(np.abs(a.e_v-b.e_v))),
        'interpretation':'Different eligibility recurrences over the same subtractive ALIF forward; not a hard-reset neuron trial'},
        'uniform_mid_actual':float(generate_timescale_vector(48,'uniform_mid')[0,0])}
    m=M(cfg,readout_mode='context_gated');x=np.zeros((12,8),np.float32);x[10,:4]=-1.5;x[11,7]=1
    y=x.copy();y[10,:4]=1.5;_,left=m.run_sequence(x);_,right=m.run_sequence(y)
    result['bipolar_query_context_trace']={'all4query_bits_flipped':True,'context_trace_identical':bool(np.array_equal(left,right)),
        'scope':'Absolute-value raw context trace loses query polarity; hidden features may still differ, so not an impossibility theorem'}
    rng=M(cfg,condition='shuffled_signal');rng.signal_rng.permutation(48);rng.syn_trace.fill(.25)
    payload=rng.dumps();restored=M.loads(payload)
    result['rng_state_preserved']=rng.signal_rng.bit_generator.state==restored.signal_rng.bit_generator.state
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:members={n:archive.read(n) for n in archive.namelist()}
    with np.load(io.BytesIO(members['state.npz']),allow_pickle=False) as archive:arrays={n:archive[n].copy() for n in archive.files}
    arrays.pop('syn_trace');stream=io.BytesIO();np.savez(stream,**arrays);members['state.npz']=stream.getvalue();stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        for name,raw in members.items():archive.writestr(name,raw)
    missing=M.loads(stream.getvalue())
    result['missing_momentum_state_accepted_and_reset']=bool(np.all(missing.syn_trace==0))
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
