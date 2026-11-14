"""Actual SQLite shadow source; no neural/model transport stubs."""

import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def shadow_store():
    spec = importlib.util.spec_from_file_location("full_query_recipe_test", ROOT/"integrations/hermes/full_query_retrieval_patch.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    scope = {"__name__":"full_query_store_shadow"}
    exec(compile(recipe.patch_bytes((ROOT/"src/scarx/hybrid/store.py").read_bytes()), "shadow-store.py", "exec"), scope)
    return scope["Store"]


def test_last_keyword_is_retrieved_and_stream_time_boundaries_remain_enforced(tmp_path):
    Store = shadow_store()
    store = Store(tmp_path)
    try:
        past = store.observe("lamp color blue",source="fixture",stream="a",start=0,end=1)
        store.observe("lamp color blue",source="fixture",stream="b",start=0,end=1)
        store.observe("lamp color blue",source="fixture",stream="a",start=2,end=3)
        rows = store.search("note "*200+"blue",stream="a",before=1,limit=1)
        assert [row["id"] for row in rows] == [past["id"]]
        assert store.search("note "*200+"blue",stream="missing",limit=32) == []
    finally:
        store.close()


def test_adversarial_long_expression_is_bounded_literal_SQLite_query(tmp_path):
    store = shadow_store()(tmp_path)
    try:
        row = store.observe("specialneedle",source="fixture",stream="a")
        assert store.search("note "*1197+"specialneedle",stream="a")[0]["id"] == row["id"]
        assert store.search(" OR NOT NEAR "+"note "*300+"specialneedle",stream="a")[0]["id"] == row["id"]
        assert store.search("?! @#$",stream="a") == []
        with pytest.raises(ValueError,match="6000-character"):
            store.search("x"*6001)
        with pytest.raises(ValueError):
            store.search(None)
    finally:
        store.close()
