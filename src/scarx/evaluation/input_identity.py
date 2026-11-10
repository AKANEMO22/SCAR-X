"""Public train-only lexical identity controls, not a neural answer algorithm."""
from collections import Counter
import hashlib
import numpy as np
from ..neural.language import tokenize
from ..neural.unified import UnifiedCore

VIEWS=('legacy128','dictionary128','hash_same_dictionary128','full_dictionary_reference')


def words(text):return [t for t in tokenize(text) if any(c.isalnum() for c in t)]


def fit(prompts):
    prompts=list(prompts);df=Counter(t for prompt in prompts for t in set(words(prompt)))
    if not df:raise ValueError('No training word types')
    vocabulary=sorted(df,key=lambda t:(-df[t],t));selected=vocabulary[:128]
    return {'train_documents':len(prompts),'vocabulary':vocabulary,'selected':selected,
        'idf':{t:float(np.log((1+len(prompts))/(1+n))+1) for t,n in df.items()},
        'selection_uses_labels':False}


def transform(metadata,prompts,view):
    if view not in VIEWS:raise ValueError('Unknown inputcontrol')
    vocabulary=metadata['vocabulary'] if view=='full_dictionary_reference' else metadata['selected'];ids={t:i for i,t in enumerate(vocabulary)}
    # Numeric encoderadapter only: no neuralcore/weights or answerlabel exists.
    class Context:
        context_size=128
        encoding={'kind':'legacy','surface_normalization':'terminal-punctuation-v1'}
    out=[];coverage=[]
    for prompt in prompts:
        tokens=words(prompt);known=sum(t in ids for t in tokens);coverage.append(known/max(1,len(tokens)))
        if view=='legacy128':out.append(UnifiedCore.context(Context(),prompt));continue
        value=np.zeros(len(vocabulary) if view=='full_dictionary_reference' else 128,np.float32)
        for token in tokens:
            if token not in ids:continue
            weight=metadata['idf'][token]
            if view=='hash_same_dictionary128':
                code=int.from_bytes(hashlib.blake2b(token.encode(),digest_size=8).digest(),'little')
                value[code%128]+=weight*(1 if (code>>8)&1 else -1)
            else:value[ids[token]]+=weight
        value/=max(1.,float(np.linalg.norm(value)));out.append(value)
    return np.stack(out),{'mean_known_word_occurrence_fraction':float(np.mean(coverage)),
        'all_unknown_prompts':int(sum(v==0 for v in coverage)),'dimensions':len(out[0])}


def dual_ridge(a,y,b,ty,classes=8):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);mean=a.mean(0);scale=a.std(0)+1e-4
    x=np.column_stack(((a-mean)/scale,np.ones(len(a))));z=np.column_stack(((b-mean)/scale,np.ones(len(b))))
    dual=np.linalg.solve(x@x.T+.01*np.eye(len(x)),np.eye(classes)[y]);pred=(z@x.T@dual).argmax(1)
    return {'accuracy':float(np.mean(pred==ty)),'predictions':pred.tolist()}
