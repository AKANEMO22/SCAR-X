import numpy as np
import pytest
from scarx import Config, SpikingCore
from scarx.memory.store import EpisodeStore
from scarx.learning.controller import LearningController
from scarx.evaluation.tasks import episodes
from scarx.interfaces.text import TextAgent


def test_exact_episodes_and_safe_query(tmp_path):
    s = EpisodeStore(tmp_path/"memory.sqlite")
    text = 'Tôi thích học tiếng Việt.\nBản gốc: ắ ễ ự.'
    key = s.observe(text,"user:correction",kind="correction",verified=True)
    assert s.get(key)["text"] == text
    assert s.search('"Việt" OR 1=1 --')[0]["id"] == key
    assert len(s.lessons()) == 1
    with pytest.raises(ValueError): s.observe("x","self",kind="generated",verified=True)
    s.close()
    s = EpisodeStore(tmp_path/"memory.sqlite")
    assert s.get(key)["text"] == text
    s.close()


def test_controller_rolls_back_exception(tmp_path):
    m = SpikingCore(Config(inputs=5))
    fingerprint = m.fingerprint()
    c = LearningController(m,tmp_path)
    samples = list(episodes(5,4))
    guard = [(x,y,"old") for x,y in samples]
    original = c.model.learn
    count = 0
    def interrupted(*args,**kwargs):
        nonlocal count
        count += 1
        if count==3: raise RuntimeError("simulated interrupted update")
        return original(*args,**kwargs)
    c.model.learn = interrupted
    with pytest.raises(RuntimeError): c.learn(samples,guard=guard)
    assert c.model.fingerprint() == fingerprint
    assert c.model.updates == 0


def test_controller_rejects_regression(tmp_path,monkeypatch):
    c = LearningController(SpikingCore(Config(inputs=5)),tmp_path)
    before = c.model.fingerprint()
    scores = iter([{"old":1.},{"old":.5}])
    monkeypatch.setattr(c,"_score",lambda _:next(scores))
    sample = list(episodes(1,1))
    result = c.learn(sample,guard=[(*sample[0],"old")])
    assert not result["accepted"] and c.model.fingerprint() == before


def test_generation_never_updates_weights():
    m = SpikingCore(Config(inputs=256,outputs=256))
    agent = TextAgent(m,None)
    before = m.fingerprint()
    result = agent.generate("Xin chào",12)
    assert result["tokens"] == 12 and result["status"] == "untrained_research_model"
    assert m.fingerprint() == before and m.updates == 0

