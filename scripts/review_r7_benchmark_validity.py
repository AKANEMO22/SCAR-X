"""Bounded generator-only verification of DeepSeek's stated benchmark correction."""
from pathlib import Path
import hashlib,importlib.util,itertools,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT.parent/'review-deepseek-r7-aacde3f'
sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(SOURCE/'scripts'))


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def main():
    old=load('historical_r32_generators',SOURCE/'scripts/r7_r32_tier3_six.py')
    new=load('corrected_r37_generators',SOURCE/'scripts/r7_r37_t6_fixed.py')
    seed=420001;n=128
    invalid={}
    for a,b in itertools.combinations(('parity3','and3','maj3'),2):
        xa,ya=old.GENS[a](seed,n);xb,yb=old.GENS[b](seed,n)
        invalid[a+'__'+b]={'entire_inputs_identical':bool(np.array_equal(xa,xb)),
            'same_input_different_label_rows':int(np.sum(np.all(xa==xb,axis=(1,2))&(ya!=yb)))}
    corrected={name:new.GENS[name](seed,n) for name in new.TASKS}
    grouped={}
    for name,(x,y) in corrected.items():
        for row,label in zip(x,y):grouped.setdefault(hashlib.sha256(row.tobytes()).hexdigest(),[]).append((name,int(label)))
    conflicts=[v for v in grouped.values() if len({label for _,label in v})>1]
    result={'source_sha':'aacde3fa0983b9cebc585e19980022a52a45ccad','historical_specimen_seed':seed,'rows_per_task':n,
        'old_invalid_generator_pairs':invalid,'corrected_exact_sample_label_conflicts':len(conflicts),
        'corrected_specimen_scope':'Exact sample conflicts only in this generator specimen; no assertion about all distributions or neural learnability.',
        'xnor_query_marker_channel':2,'neural_forward_calls':0,'learning_updates':0}
    p=ROOT/'docs/research/forensic-r6-20261111/R7_BENCHMARK_VALIDITY.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
