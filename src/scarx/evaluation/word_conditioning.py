"""Read-only first-word context interventions; no fitted head or training."""
import numpy as np
from .score_word import ScoreWord
from ..neural.language import tokenize


def measure(payload,cases,seed=991000):
    model=ScoreWord.loads(payload);model.mask(0);before=model.dumps()
    events=np.stack([model.sequence(r['prompt'],[]) for r in cases]);start=len(model.vocabulary)
    targets=np.array([model.ids[tokenize(r['target'])[0]] for r in cases])
    families=list(dict.fromkeys(r['intent'] for r in cases));first={name:next(i for i,r in enumerate(cases) if r['intent']==name) for name in families}
    if len(families)!=8:raise ValueError('Expected all8families per split')
    permutation=[first[families[(families.index(r['intent'])+1)%8]] for r in cases]
    inputs={'real':events,'zero':events.copy(),'cross_family':events.copy()}
    inputs['zero'][:,:,start:start+128]=0
    inputs['cross_family'][:,:,start:start+128]=events[permutation,:,start:start+128]
    outputs={};summary={};probabilities={};features={}
    for mode,x in inputs.items():
        f=model.core.features_many(x,seed,samples=8);p=model.core.probabilities(f);pred=p.argmax(1)
        probabilities[mode]=p;features[mode]=f
        summary[mode]={'first_token_accuracy':float(np.mean(pred==targets)),
            'nll':float(-np.log(np.maximum(p[np.arange(len(cases)),targets],1e-12)).mean())}
        outputs[mode]=[{'id':r['id'],'actual_first_token':model.vocabulary[int(pred[i])],
            'one_reference_first_token':model.vocabulary[int(targets[i])],'correct':bool(pred[i]==targets[i])} for i,r in enumerate(cases)]
    for mode in ('zero','cross_family'):
        summary[mode].update(argmax_flip_vs_real=float(np.mean(probabilities[mode].argmax(1)!=probabilities['real'].argmax(1))),
            probability_l1_vs_real=float(np.abs(probabilities[mode]-probabilities['real']).sum(1).mean()),
            hidden_rms_vs_real=float(np.sqrt(np.mean((features[mode]-features['real'])**2))))
    assert model.dumps()==before,'Read-only intervention mutated model'
    return {'summary':summary,'outputs':outputs,'model_unchanged':True,'cases':len(cases),
        'actual_ticks':3*8*len(events)*events.shape[1],
        'input_array_bytes':sum(x.nbytes for x in inputs.values()),'scope':'firsttokenonly, not full utterance or original v1 decoder RNG scores'}
