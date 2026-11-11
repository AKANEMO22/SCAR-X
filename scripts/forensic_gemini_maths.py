"""Small counterexamples to Gemini attribution claims, not benchmark fitting."""
from pathlib import Path
import json,os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'forensic-source-gemini-8f24a5a/src'))
import numpy as np
from scarx.evaluation import r6_associative as R
from scarx.runtime.limits import apply_memory_limit


def main():
    cap=apply_memory_limit()
    # A shared linear trunk on POST-query states need not be query invariant.
    x,*_=R.G.independent_paired_set_v1(521011,32)
    captured=R.capture_r6(121011,x)
    post=captured['post'];difference=post[::2]-post[1::2]
    row,coordinate=np.unravel_index(np.argmax(np.abs(difference)),difference.shape)
    m=np.stack([np.arange(96)-1,np.arange(96),np.arange(96)+1]).astype(np.float32)
    cue=np.tile(np.array([[1.,0.]],np.float32),(len(m),1));labels=np.zeros(len(m),np.int32)
    ab=R.R5AblationHead(dim=96,seed=5,pre_gating_norm=True)
    ck=R.CKCERHead(dim=96,mode='A',seed=5)
    ab.train_online(m,labels,cue);ck.train_online(m,labels,cue=cue)
    result={'source_sha':'8f24a5abde0598201e558eee559564ccde889145','scope':'Bounded engineering counterexamples, not accuracy/representation/retention experiments',
        'shared_postquery_trunk_counterexample':{'historical_seed':121011,'pair_seed':521011,'pairs':32,
            'max_post_state_difference_between_queries':float(np.max(np.abs(difference))),
            'unit_coordinate_readout_query_margin_difference':float(difference[row,coordinate]),
            'coordinate':int(coordinate),'row':int(row),
            'conclusion':'Original R5 post-query trunk is not necessarily query invariant. A theorem assuming the same s for both queries cannot be applied to it without that condition.'},
        'ablation_optimizer_counterexample':{'all_toy_targets':0,'ablation_weight_max':float(np.abs(ab.w).max()),
            'ckcer_weight_max':float(np.abs(ck.w).max()),
            'conclusion':'Pre-gating ablation uses hard argmax perceptron errors; CKCER uses softmax probability errors. The ladder changes more than just the named feature factor.'},
        'cap':cap,'new_scored_experiment':False}
    p=ROOT/'docs/research/forensic-r6-20261111/GEMINI_MATH_COUNTEREXAMPLES.json'
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
