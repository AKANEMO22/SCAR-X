import io,json,zipfile
import numpy as np
import pytest
from scarx.neural.unified import UnifiedCore
from scarx.app.workload import plan_workload
from scarx.app.generation import Generation


def test_growth_preserves_base_weights_predictions_and_core_identity():
    m=UnifiedCore(['<eos>','<bos>','hello','.']);m.train_pair('hi','hello.')
    base=UnifiedCore.loads(m.dumps());identity=id(m.core)
    arrays={k:v.copy() for k,v in m.core.arrays().items()}
    m.select_capacity(96)
    assert id(m.core)==identity and m.core.config.hidden==96
    for name in ('w_in','w_rec','indices'):np.testing.assert_array_equal(getattr(m.core,name)[:48],arrays[name])
    assert not m.core.w_out[:,48:].any()
    np.testing.assert_allclose(m.probabilities('language'),base.probabilities('language'),atol=1e-7)
    m.select_capacity(48)
    assert m.core.config.hidden==96 and m.active_count==48
    assert m.generate('hi')['text']==base.generate('hi')['text']


def test_dormant_neurons_do_not_execute_dense_rows_and_resume_exactly():
    m=UnifiedCore(['<eos>','<bos>','hello','.']);m.select_capacity(128);m.tick(np.ones(m.core.config.inputs,np.float32))
    m.select_capacity(48);voltage=m.core.voltage[48:].copy();clocks=m.last_active[48:].copy()
    for _ in range(5):m.tick()
    np.testing.assert_array_equal(m.core.voltage[48:],voltage)
    np.testing.assert_array_equal(m.last_active[48:],clocks)
    replica=UnifiedCore.loads(m.dumps())
    m.select_capacity(96);replica.select_capacity(96)
    assert m.generate('hi')==replica.generate('hi')
    for k,v in m.core.arrays().items():np.testing.assert_array_equal(v,replica.core.arrays()[k])
    with pytest.raises(ValueError):m.select_capacity(129)


def test_lexical_idf_is_training_only_and_persisted():
    encoding=UnifiedCore.lexical_encoding(['xin chào bạn','chào bạn','hello'])
    m=UnifiedCore(['<eos>','<bos>','hello','.'],encoding=encoding)
    m.train_pair('hi','hello.')
    loaded=UnifiedCore.loads(m.dumps())
    assert loaded.encoding==encoding and loaded.generate('hi')==m.generate('hi')
    assert loaded.context('never seen').shape==(128,)


def test_new_output_tokens_preserve_old_ids_image_head_and_recurrent_topology():
    model=UnifiedCore(['<eos>','<bos>','hello','.']);model.select_capacity(64)
    old_ids=model.ids.copy();old_rec=model.core.w_rec.copy();old_indices=model.core.indices.copy()
    image_weights=model.core.w_out[-10:].copy();model.expand_vocabulary('2')
    for word,identifier in old_ids.items():assert model.ids[word]==identifier
    np.testing.assert_array_equal(model.core.w_rec,old_rec);np.testing.assert_array_equal(model.core.indices,old_indices)
    np.testing.assert_array_equal(model.core.w_out[-10:],image_weights)
    model.train_pair('1+1','2')
    loaded=UnifiedCore.loads(model.dumps());assert loaded.generate('1+1')==model.generate('1+1')
    assert loaded.core.config.hidden==64 and loaded.active_count==64


def test_task_allocation_and_checkpoint_rollback(tmp_path,monkeypatch):
    app=Generation(tmp_path,background=False)
    try:
        initial=app.checkpoint()['checkpoint'];before=app.model.core.fingerprint()
        result=app.chat('hi',neuron_budget=64,request_id='public-test')
        assert result['workload']['active_neurons']==64 and result['workload']['allocated_neurons']==64
        assert result['workload']['output_connected_neurons']<=48
        app.idle_once();assert app.model.active_count==48 and app.model.core.config.hidden==64
        app.restore(initial);assert app.model.core.config.hidden==48 and app.model.core.fingerprint()==before
        packed=app._pack()
        def fail(*args,**kwargs):raise RuntimeError('commit fault')
        monkeypatch.setattr(app,'_commit',fail)
        with pytest.raises(RuntimeError):app.chat('hi',neuron_budget=96)
        assert app._pack()==packed
    finally:app.close()


def test_workload_is_input_specific_without_retrieval_or_difficulty_claim():
    m=UnifiedCore(['<eos>','<bos>','hello','.'])
    small=plan_workload(m,'hi');long=plan_workload(m,'word '*80)
    image=plan_workload(m,'look',np.zeros((8,8)))
    assert small['input_characters']==2 and small['input_utf8_bytes']==2
    assert small['requested_neurons']==48 and long['requested_neurons']==96 and image['requested_neurons']==64
    assert image['neural_image_pixels']==64 and small['retrieved_episodes']==0
    assert 'not validated task difficulty' in small['allocation_reason']


def test_learned_reserve_not_silently_disabled_for_simple_foreground_task():
    model=UnifiedCore(['<eos>','<bos>','hello','.']);model.select_capacity(64)
    assert plan_workload(model,'hi')['requested_neurons']==48
    model.core.w_out[2,63]=.1
    assert plan_workload(model,'hi')['requested_neurons']==64
    with pytest.raises(ValueError,match='readout'):plan_workload(model,'hi',budget=48)


def test_all_active_kernel_agrees_with_dense_reference_dynamics():
    from scarx.neural.core import SpikingCore
    model=UnifiedCore(['<eos>','<bos>','hello','.']);model.select_capacity(64)
    reference=SpikingCore.loads(model.core.dumps());rng=np.random.default_rng(691)
    for _ in range(25):
        x=rng.normal(0,.2,model.core.config.inputs).astype(np.float32)
        model.tick(x,track=True);reference.step(x,track=True)
    for name in ('voltage','adaptation','spikes','features','elig_in','elig_rec','pre','post'):
        np.testing.assert_allclose(getattr(model.core,name),getattr(reference,name),atol=2e-5,rtol=2e-5)
