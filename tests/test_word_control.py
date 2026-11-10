import importlib.util,os,subprocess,sys
from pathlib import Path
import pytest


def _exercise():
    import numpy as np
    from scarx.evaluation.controls import WordLearnabilityControl,WordTapeSurrogate,torch
    from scarx.evaluation.score_word import ScoreWord
    from scarx.evaluation.fixed_word_codes import fit
    from scarx.neural.unified import UnifiedCore
    torch.set_num_threads(1)
    v=torch.tensor([.2,100.],requires_grad=True);threshold=torch.tensor([.1,.1],requires_grad=True)
    observed=torch.tensor([1.,0.]);z=WordTapeSurrogate.apply(v,threshold,observed);z.sum().backward()
    p=1/(1+np.exp(-.25));np.testing.assert_allclose(v.grad.numpy(),[p*(1-p)/.4,0],atol=1e-7,rtol=0)
    np.testing.assert_array_equal(v.grad.numpy(),-threshold.grad.numpy());np.testing.assert_array_equal(z.detach().numpy(),observed.numpy())
    vocab=['<eos>','<bos>','hello','bye','.'];prompts=['hi','bye']
    for encoding in (None,fit(prompts,'known-digest128-v1')):
        a=ScoreWord(vocab,29101,'frozen',encoding);b=ScoreWord(vocab,29101,'frozen',encoding)
        a.calibrate(prompts);b.calibrate(prompts);control=WordLearnabilityControl(b)
        for i in range(8):
            prompt=prompts[i%2];target='hello.' if i%2==0 else 'bye.'
            assert a.learn(prompt,target)==control.learn(prompt,target) and a.dumps()==control.reference.dumps()
        full=ScoreWord(vocab,29101,'full',encoding);full.calibrate(prompts);control=WordLearnabilityControl(full)
        old=full.core.w_in.copy();core=full.core
        control.learn('hi','hello.');assert np.any(core.w_in!=old)
        assert control.backward_passes==4*core.updates and control.surrogate_forward_ticks==core.ticks
        assert control.max_graph_state_error<=3e-5 and control.max_graph_feature_error<=1e-6
        copied=WordLearnabilityControl.loads(control.dumps());assert copied.dumps()==control.dumps()
        assert control.learn('bye','bye.')==copied.learn('bye','bye.') and control.dumps()==copied.dumps()
        assert control.generate('hi')==copied.generate('hi') and control.dumps()==copied.dumps()
        for loader in (ScoreWord.loads,UnifiedCore.loads):
            with pytest.raises(ValueError):loader(control.dumps())


def test_word_control_isolated_forward_localhead_freeze_surrogate_and_codec():
    if importlib.util.find_spec('torch') is None:pytest.skip('Optional isolated control dependency')
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--isolated'],capture_output=True,text=True,
        timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert child.returncode==0,child.stdout+child.stderr


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'));_exercise()
