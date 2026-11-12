from pathlib import Path
import copy,hashlib,io,json,zipfile
import numpy as np
import pytest

from scarx.evaluation.output_codec_p0 import AnswerCodec,CodecCoreP0,CodecFixtureSession
from scarx.evaluation.knowledge_factory import UNKNOWN
from scarx.neural.unified import UnifiedCore

ROOT=Path(__file__).resolve().parents[1]


def source_model():
    desc=json.loads((ROOT/'artifacts/generation1-release.json').read_text(encoding='utf-8'))
    raw=(ROOT/desc['checkpoint']).read_bytes();assert hashlib.sha256(raw).hexdigest()==desc['sha256']
    return UnifiedCore.loads(raw)


def tokens_for(version):
    receipt=json.loads((ROOT/'artifacts/basic-knowledge-verified-v1.json').read_text(encoding='utf-8'))
    rows=json.loads((ROOT/receipt['data']).read_text(encoding='utf-8'))['items'];codec=AnswerCodec(version)
    return {t for row in rows if row['split']=='train' for t in codec.encode(row['canonical_answer'])}|{t for a in UNKNOWN.values() for t in codec.encode(a)}


def model(version='numeric-atom-v2'):
    return CodecCoreP0.migrate(source_model(),tokens_for(version),version)


@pytest.mark.parametrize('answer',['0.25.','20.0.','-3.125.','+0.50.','2000000000.','Tôi chưa biết.','I do not know.','alif.'])
def test_output_only_numeric_codec_roundtrips_supported_words_and_literals(answer):
    codec=AnswerCodec('numeric-atom-v2');assert codec.roundtrip_matches(answer)
    if answer[0] in '0123456789+-':assert codec.encode(answer)[-1]=='.' and len(codec.encode(answer))==2


def test_invalid_codec_unicode_and_content_are_not_silently_dropped():
    with pytest.raises(ValueError):AnswerCodec('unknown')
    with pytest.raises(ValueError):AnswerCodec('numeric-atom-v2').encode('\ud800')
    c=AnswerCodec('numeric-atom-v2')
    assert 'garbage' in c.encode('0.25 garbage.')
    with pytest.raises(ValueError,match='64'):c.encode('1'*65+'.')


def test_all_frozen_targets_codec_only_no_neural_quality_scoring():
    receipt=json.loads((ROOT/'artifacts/basic-knowledge-verified-v1.json').read_text(encoding='utf-8'));rows=json.loads((ROOT/receipt['data']).read_text(encoding='utf-8'))['items']
    new=AnswerCodec('numeric-atom-v2');old=AnswerCodec('legacy-word-v1')
    assert sum(not old.roundtrip_matches(r['canonical_answer']) for r in rows)==128
    assert all(new.roundtrip_matches(r['canonical_answer']) for r in rows)
    assert len(model().vocabulary)==55 and len(model('legacy-word-v1').vocabulary)==55


def test_context_and_old_word_neuron_probability_trajectory_exact():
    original=source_model();before=original.dumps();migrated=CodecCoreP0.migrate(original,tokens_for('numeric-atom-v2'))
    assert original.dumps()==before
    old_v=len(original.vocabulary)
    for prompt in ['hello','Tôi hỏi 0.25 và 20.0','p0 unknown / unicode α']:
        np.testing.assert_array_equal(original.context(prompt),migrated.context(prompt))
        for position,previous in enumerate([1,2,3]):
            a=original.word_step(previous,original.context(prompt),position,track=True)
            b=migrated.word_step(previous,migrated.context(prompt),position,track=True)
            np.testing.assert_array_equal(a[:old_v],b[:old_v]);assert not b[old_v:len(migrated.vocabulary)].any()
            for name in ('voltage','adaptation','spikes','features','post','elig_rec','rows'):
                np.testing.assert_array_equal(getattr(original.core,name),getattr(migrated.core,name))
            projected=np.r_[np.arange(old_v),np.arange(len(migrated.vocabulary),migrated.core.config.inputs)]
            np.testing.assert_array_equal(original.core.elig_in,migrated.core.elig_in[:,projected])


def test_digit_outputs_remapped_without_changing_old_probabilities_or_hidden():
    a=source_model();b=CodecCoreP0.migrate(a,tokens_for('numeric-atom-v2'))
    for pixels in [np.arange(64,dtype=np.float32)%17,np.zeros(64,np.float32)]:
        va=a.image(pixels);vb=b.image(pixels)
        assert va==vb
        for name in ('voltage','adaptation','spikes','features'):np.testing.assert_array_equal(getattr(a.core,name),getattr(b.core,name))


def test_existing_local_learning_rule_remains_exact_on_a_tiny_legacy_fixture():
    a=source_model();b=CodecCoreP0.migrate(a,tokens_for('numeric-atom-v2'));s=CodecFixtureSession(b)
    old_loss=a.train_pair('p0 old-words fixture','Hello.',mode='full')
    s.begin_fixture('p0 old-words fixture','Hello.',mode='full');losses=[]
    while s.controller['phase']=='teaching':losses.append(s.advance_fixture())
    assert old_loss==sum(losses)/len(losses) and a.core.updates==b.core.updates
    v=len(a.vocabulary);nv=len(b.vocabulary)
    inputs=np.r_[np.arange(v),np.arange(nv,b.core.config.inputs)]
    outputs=np.r_[np.arange(v),np.arange(nv,nv+10)]
    for name in ('w_in','stable_in','elig_in'):np.testing.assert_array_equal(getattr(a.core,name),getattr(b.core,name)[:,inputs])
    for name in ('w_out','stable_out'):np.testing.assert_array_equal(getattr(a.core,name),getattr(b.core,name)[outputs])
    for name in ('bias','stable_bias'):np.testing.assert_array_equal(getattr(a.core,name),getattr(b.core,name)[outputs])
    for name in ('w_rec','stable_rec','voltage','adaptation','spikes','features','post','elig_rec','rows'):
        np.testing.assert_array_equal(getattr(a.core,name),getattr(b.core,name))


def test_new_tokens_are_explicitly_masked_until_enabled():
    m=model();i=m.ids['0.25'];assert m.probabilities('language')[i]==0
    m.enable_tokens(['0.25']);assert m.probabilities('language')[i]>0
    with pytest.raises(ValueError):m.enable_tokens(['not_in_vocab'])
    with pytest.raises(ValueError):m.enable_tokens('0.25')


def test_invalid_fixture_targets_do_not_advance_neural_or_controller_state():
    s=CodecFixtureSession(model());before=s.dumps()
    for answer in ('.','not_in_vocab.',''):
        with pytest.raises(ValueError):s.begin_fixture('p0 invalid',answer)
        assert s.dumps()==before


def test_foreign_capacity_and_extra_learner_state_are_rejected_not_stripped():
    m=source_model();m.core.importance=np.zeros(1)
    with pytest.raises(ValueError,match='discarded'):CodecCoreP0.migrate(m,[])
    m=source_model();m.select_capacity(64)
    with pytest.raises(ValueError,match='H48'):CodecCoreP0.migrate(m,[])
    m=source_model();m.importance=np.ones(1)
    with pytest.raises(ValueError,match='discarded'):CodecCoreP0.migrate(m,[])


def test_unknown_controller_state_and_unsafe_direct_constructor_are_rejected():
    with pytest.raises(ValueError,match='explicit migrate'):CodecCoreP0(['<eos>','<bos>','hello'])
    s=CodecFixtureSession(model());s.untracked_optimizer=np.zeros(1)
    with pytest.raises(ValueError,match='discarded'):s.dumps()
    m=model();m.core.importance=np.zeros(1)
    with pytest.raises(ValueError,match='discarded'):m.dumps()


def test_mid_fixture_model_controller_replay_rng_and_trace_state_exact(tmp_path):
    m=model();m.enable_tokens(['0.25']);s=CodecFixtureSession(m)
    s.add_replay_fixture('p0-a','0.25.');s.add_replay_fixture('p0-b','Tôi chưa biết.')
    s.begin_fixture('p0-decimal-fixture','0.25.',mode='full');s.advance_fixture()
    before=s.dumps();loaded=CodecFixtureSession.loads(before);assert loaded.dumps()==before
    while s.controller['phase']=='teaching':assert s.advance_fixture()==loaded.advance_fixture()
    assert s.next_replay_fixture()==loaded.next_replay_fixture()
    assert s.rng.integers(100000)==loaded.rng.integers(100000)
    assert s.model.capacity_rng.integers(100000)==loaded.model.capacity_rng.integers(100000)
    assert s.dumps()==loaded.dumps()
    path=tmp_path/'codec-fixture.zip';s.save(path);assert CodecFixtureSession.load(path).dumps()==s.dumps()


def test_rejected_or_failed_fixture_restores_full_snapshot():
    m=model();m.enable_tokens(['0.25']);s=CodecFixtureSession(m);s.add_replay_fixture('p0','0.25.')
    before=s.dumps()
    def reject(state):state.next_replay_fixture();state.model.capacity_rng.integers(10000);return False
    assert s.guarded_fixture('p0-reject','0.25.',reject,mode='full') is False
    assert s.dumps()==before
    def fail(state):state.model.enabled_words[2]=False;raise RuntimeError('injected')
    with pytest.raises(RuntimeError):s.guarded_fixture('p0-fail','0.25.',fail)
    assert s.dumps()==before
    with pytest.raises(ValueError):s.restore(before[:-10])
    assert s.dumps()==before


def test_serving_loader_rejects_research_model_and_session_wrappers():
    m=model();s=CodecFixtureSession(m)
    for payload in [m.dumps(),s.dumps()]:
        with pytest.raises(ValueError):UnifiedCore.loads(payload)


def test_missing_codec_metadata_is_legacy_never_auto_numeric_migration():
    original=source_model();loaded=CodecCoreP0.loads(original.dumps())
    assert loaded.codec_version=='legacy-word-v1' and loaded.vocabulary==original.vocabulary
    assert AnswerCodec(loaded.codec_version).encode('0.25.')==['0','.','25','.']
    a=original.generate('p0 legacy');b=loaded.generate('p0 legacy')
    assert a['tokens']==b['tokens'] and a['text']==b['text']


def _rewrite_core_member(payload,name,replacement):
    import scarx.evaluation.output_codec_p0 as C
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        meta=json.loads(outer.read('metadata.json'));core=outer.read('core.npz')
    out=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(core)) as old,zipfile.ZipFile(out,'w') as new:
        for member in old.namelist():new.writestr(zipfile.ZipInfo(member),replacement if member==name else old.read(member))
    modified=out.getvalue();meta.pop('metadata_sha256');meta['sha256']=C._hash(modified)
    meta['metadata_sha256']=C._hash(C._json(meta))
    return C._archive([('core.npz',modified),('metadata.json',C._json(meta))])


def test_huge_declared_array_rejected_before_npz_materialization(monkeypatch):
    m=model();header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,{'descr':'<f4','fortran_order':False,'shape':(10**9,10**9)})
    malformed=_rewrite_core_member(m.dumps(),'w_in.npy',header.getvalue())
    original=np.load
    def limited_load(stream,*args,**kwargs):
        position=stream.tell();magic=stream.read(2);stream.seek(position)
        assert magic!=b'PK','Whole NPZ loader reached before validating declared shapes'
        return original(stream,*args,**kwargs)
    monkeypatch.setattr(np,'load',limited_load)
    with pytest.raises(ValueError,match='shape/type'):CodecCoreP0.loads(malformed)


def test_float64_checkpoint_array_cannot_be_silently_cast_to_float32():
    m=model();stream=io.BytesIO();np.save(stream,m.core.w_in.astype(np.float64),allow_pickle=False)
    malformed=_rewrite_core_member(m.dumps(),'w_in.npy',stream.getvalue())
    with pytest.raises(ValueError,match='shape/type'):CodecCoreP0.loads(malformed)


def test_duplicate_members_and_corrupt_metadata_rejected():
    s=CodecFixtureSession(model());raw=s.dumps()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:model_blob=z.read('model.zip');session=z.read('session.json')
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        z.writestr('model.zip',model_blob);z.writestr('session.json',session);z.writestr('session.json',session)
    with pytest.raises(ValueError):CodecFixtureSession.loads(out.getvalue())
    meta=json.loads(session);meta['fixture_updates']=99
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('model.zip',model_blob);z.writestr('session.json',json.dumps(meta))
    with pytest.raises(ValueError):CodecFixtureSession.loads(out.getvalue())


def _rewrite_json_member(payload,name,change):
    import scarx.evaluation.output_codec_p0 as C
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        members={key:z.read(key) for key in z.namelist()}
    meta=json.loads(members[name]);meta.pop('metadata_sha256',None);change(meta)
    meta['metadata_sha256']=C._hash(C._json(meta));members[name]=C._json(meta)
    return C._archive(list(members.items()))


def test_extra_session_metadata_cannot_be_silently_discarded():
    payload=_rewrite_json_member(CodecFixtureSession(model()).dumps(),'session.json',lambda m:m.update(importance=[1.]))
    with pytest.raises(ValueError,match='state'):CodecFixtureSession.loads(payload)


def test_unknown_core_metadata_cannot_be_silently_discarded():
    m=model()
    with zipfile.ZipFile(io.BytesIO(m.core.dumps())) as z:meta=json.loads(str(np.load(io.BytesIO(z.read('metadata.npy')),allow_pickle=False)))
    meta['importance']=[1.];stream=io.BytesIO();np.save(stream,np.array(json.dumps(meta)),allow_pickle=False)
    payload=_rewrite_core_member(m.dumps(),'metadata.npy',stream.getvalue())
    with pytest.raises(ValueError,match='metadata'):CodecCoreP0.loads(payload)


def test_unbounded_capacity_history_is_rejected_before_truncation():
    payload=_rewrite_json_member(model().dumps(),'metadata.json',lambda m:m.update(capacity_events=[{'tick':0}]*33))
    with pytest.raises(ValueError,match='capacity'):CodecCoreP0.loads(payload)


def test_invalid_prompt_is_rejected_before_neuron_state_changes():
    s=CodecFixtureSession(model());before=s.dumps()
    for prompt in ('','   ','\ud800'):
        with pytest.raises(ValueError):s.begin_fixture(prompt,'Hello.')
        assert s.dumps()==before


def test_legacy_mutating_entrypoints_cannot_bypass_codec_fixture_layout():
    m=model();before=m.dumps()
    for mutation in (lambda:m.expand_vocabulary('0.111.'),lambda:m.train_pair('p0','Hello.'),lambda:m.image(np.zeros(64),train_label=1)):
        with pytest.raises(ValueError,match='P0'):mutation()
        assert m.dumps()==before


def test_legacy_loader_does_not_drop_foreign_metadata():
    with zipfile.ZipFile(io.BytesIO(source_model().dumps())) as z:
        core=z.read('core.npz');meta=json.loads(z.read('metadata.json'))
    meta['importance']=[1.];out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('core.npz',core);z.writestr('metadata.json',json.dumps(meta))
    with pytest.raises(ValueError,match='legacy metadata'):CodecCoreP0.loads(out.getvalue())


def test_pending_state_cannot_exceed_fixture_update_budget():
    m=model();m.enable_tokens(['0.25']);s=CodecFixtureSession(m);s.begin_fixture('p0','0.25.')
    raw=_rewrite_json_member(s.dumps(),'session.json',lambda m:m.update(fixture_updates=32))
    with pytest.raises(ValueError,match='budget'):CodecFixtureSession.loads(raw)
