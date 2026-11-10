"""External wording stays bounded and cannot cause private-source reads."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('adaptive_data',Path(__file__).resolve().parents[1]/'scripts/prepare_adaptive_curriculum.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)


def fixtures():
    return [{'id':f'{mod}_{family}_{lang}_{i:02d}','modality':mod,'family':family,'language':lang,'prompt':f'Public example wording {family} {lang} {i}'}
            for mod,families,n in [('chat',data.CHAT,8),('vision',data.VISION,4)]
            for family in families for lang in ('vi','en') for i in range(1,n+1)]


def test_teacher_payload_is_wording_only_and_rejects_unbounded_or_mislabelled_rows():
    rows=fixtures();assert len(data.validate(rows))==128
    for change in ({'target':'invented automatic label'},{'language':'fr'},{'modality':'tool'},{'prompt':'x'*181}):
        altered=[dict(r) for r in rows];altered[0].update(change)
        with pytest.raises(ValueError):data.validate(altered)
    with pytest.raises(ValueError):data.validate(rows[:-1])


def test_normalization_detects_unicode_and_punctuation_duplicates():
    assert data.normalized('Chào bạn!')==data.normalized('CHÀO bạn ?')
    assert data.normalized('What color?')!=data.normalized('What shape?')


def test_public_overlap_index_excludes_local_private_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'root',tmp_path)
    (tmp_path/'artifacts').mkdir();public=tmp_path/'data/candidates/public';public.mkdir(parents=True)
    (public/'reviewed.json').write_text(json.dumps({'items':[{'prompt':'Public safe wording'}]}),encoding='utf-8')
    private=tmp_path/'local';private.mkdir();(private/'private.json').write_text(json.dumps({'prompt':'NEVER READ OR INDEX'}),encoding='utf-8')
    (tmp_path/'artifacts/direct-word-data-reviewed.json').write_text(json.dumps({'data':'data/candidates/public/reviewed.json'}),encoding='utf-8')
    (tmp_path/'artifacts/vision-questions-reviewed.json').write_text(json.dumps({'data':'local/private.json'}),encoding='utf-8')
    words,paths=data.old_public_wording()
    assert words=={'public safe wording'} and paths==['data/candidates/public/reviewed.json']
