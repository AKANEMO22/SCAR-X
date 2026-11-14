import asyncio
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('owned_submission_scope_fixture',ROOT/'integrations/hermes/session_submission.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_concurrent_admissions_and_thread_hops_keep_their_own_request_and_reset():
    async def check():
        ready=asyncio.Event();entered=[]
        async def task(identity):
            with module.submission_scope(identity):
                entered.append(identity)
                if len(entered)==2:ready.set()
                await ready.wait()
                assert module.current_submission_id()==identity
                assert await asyncio.to_thread(module.current_submission_id)==identity
                return module.current_submission_id()
        assert await asyncio.gather(task('draft-A'),task('draft-B'))==['draft-A','draft-B']
        assert module.current_submission_id() is None
    asyncio.run(check())
    assert module.current_submission_id() is None


def test_failed_or_nested_turn_cannot_leave_identity_on_a_reused_agent_thread():
    assert module.current_submission_id() is None
    with module.submission_scope('first'):
        try:
            with module.submission_scope('second'):
                assert module.current_submission_id()=='second'
                raise OSError('controlled failure')
        except OSError:pass
        assert module.current_submission_id()=='first'
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(module.current_submission_id).result() is None
    assert module.current_submission_id() is None
