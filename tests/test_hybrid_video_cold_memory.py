import importlib.util
from pathlib import Path
import sqlite3
from scarx.hybrid.store import Store


def test_evidence_digest_survives_sqlite_backup_and_row_factory_with_unicode(tmp_path):
    source=Path(__file__).resolve().parents[1]/'scripts/diagnose_hybrid_video_cold_memory.py'
    spec=importlib.util.spec_from_file_location('owned_cold_memory_diagnostic',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    store=Store(tmp_path/'original')
    store.observe('Màu cuối cùng là vàng; dữ liệu UTF-8.',source='owned-fixture',stream='video',start=1,end=2)
    store.put('stream','video',{'id':'video','status':'complete'})
    before=module.memory_state(store.db)
    with sqlite3.connect(tmp_path/'copy.sqlite') as copied:
        store.db.backup(copied)
        assert module.memory_state(copied)==before
        copied.row_factory=sqlite3.Row
        assert module.memory_state(copied)==before
    store.close()
