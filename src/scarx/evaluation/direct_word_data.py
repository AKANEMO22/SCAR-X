"""Owned conversational style labels and deterministic review split, no model."""
import numpy as np
from ..neural.language import tokenize

TARGETS={'vi_greet':'Chào bạn.','en_greet':'Hello.','vi_identity':'Tôi là SCAR-X.','en_identity':'I am SCAR-X.',
    'vi_thanks':'Không có gì.','en_thanks':'You are welcome.','vi_bye':'Tạm biệt bạn.','en_bye':'Goodbye.'}
FAMILIES=tuple(TARGETS)


def validate(items):
    if not isinstance(items,list) or len(items)!=128:raise ValueError('Exactly128publicprompts required')
    ids=set();prompts=set()
    for r in items:
        if not isinstance(r,dict) or set(r)!={'id','family','language','prompt'} or r['family'] not in TARGETS or r['language']!=r['family'][:2] or not isinstance(r['prompt'],str) or not 1<=len(r['prompt'])<=180 or not 1<=len(tokenize(r['prompt']))<=32:raise ValueError('Publicprompt schema/budget')
        if r['id'] not in {r['family']+'_'+str(i).zfill(2) for i in range(1,17)}:raise ValueError('Prompt ID')
        normalized=' '.join(tokenize(r['prompt']))
        if r['id'] in ids or normalized in prompts:raise ValueError('Duplicate ID/prompt')
        ids.add(r['id']);prompts.add(normalized)
    if any(sum(r['family']==f for r in items)!=16 for f in FAMILIES):raise ValueError('Family balance')
    return items


def split_reviewed(items):
    validate(items);rng=np.random.default_rng(36101);out=[]
    for family in FAMILIES:
        rows=sorted((r for r in items if r['family']==family),key=lambda r:r['id']);chosen=set(int(i) for i in rng.permutation(16)[:8])
        for i,row in enumerate(rows):out.append({**row,'target':TARGETS[family],'split':'train' if i in chosen else 'development'})
    return out
