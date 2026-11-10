import json
from pathlib import Path
import numpy as np
from scarx.neural.language import WordLanguageCore,tokenize


def test_language_updates_and_persistent_generated_sequence(tmp_path):
    pairs=[('xin chào','chào bạn.'),('bạn là ai','tôi là scar-x.')]
    model=WordLanguageCore(WordLanguageCore.vocabulary_from(pairs))
    original=model.core.fingerprint()
    for prompt,answer in pairs*3: model.train_pair(prompt,answer)
    assert model.core.fingerprint()!=original
    model.save(tmp_path); restored=WordLanguageCore.load(tmp_path)
    before=restored.core.fingerprint()
    assert model.generate('xin chào')==restored.generate('xin chào')
    assert restored.core.fingerprint()==before
    metadata=json.loads((tmp_path/'language.json').read_text(encoding='utf-8'))
    assert set(metadata)=={'format','vocabulary','sha256'}
    assert 'xin chào' not in json.dumps(metadata,ensure_ascii=False)


def test_dialogue_splits_are_disjoint_and_inputs_have_no_labels():
    root=Path(__file__).resolve().parents[1]
    corpus=json.loads((root/'data/dialogue/pilot-v1.json').read_text(encoding='utf-8'))
    splits=[{p.casefold() for f in corpus['families'] for p in f[s]} for s in ('train','development','test')]
    assert not splits[0]&splits[1] and not splits[0]&splits[2] and not splits[1]&splits[2]
    model=WordLanguageCore(['<eos>','<bos>','chào','bạn','.'])
    np.testing.assert_array_equal(model.context('CHÀO BẠN'),model.context('chào bạn'))
    assert not np.array_equal(model.context('chào bạn'),model.context('tạm biệt'))


def test_beam_decoder_is_neural_bounded_and_does_not_update(tmp_path):
    pairs=[('xin chào','chào bạn.')]
    model=WordLanguageCore(WordLanguageCore.vocabulary_from(pairs))
    for _ in range(3):model.train_pair(*pairs[0])
    before=model.core.fingerprint()
    result=model.generate_beam('xin chào',max_words=8,width=2)
    assert len(result['tokens'])<=8 and model.core.fingerprint()==before
    model.save(tmp_path)
    assert WordLanguageCore.load(tmp_path).generate_beam('xin chào',max_words=8,width=2)==result
