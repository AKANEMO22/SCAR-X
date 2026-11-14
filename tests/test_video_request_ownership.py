import hashlib
import sqlite3
import pytest
from scarx.hybrid.store import Store
from scarx.hybrid.video_context import VideoContext,text_digest


@pytest.fixture
def context(tmp_path):
    store=Store(tmp_path)
    for stream in ('a'*32,'b'*32):store.put('stream',stream,{'id':stream,'kind':'video','status':'complete'})
    value=VideoContext(store)
    yield value
    store.close()


def turn(text='same question'):
    return [{'role':'user','content':text}]


def test_different_drafts_with_identical_text_cannot_steal_each_others_video(context):
    context.queue('a'*32,text_digest('same question'),['session-a'],submission_id='submission-a')
    context.queue('b'*32,text_digest('same question'),['session-b'],submission_id='submission-b')
    assert context.resolve(turn(),'session-b',submission_id='submission-a') is None
    assert context.resolve(turn(),'session-a',submission_id='submission-b') is None
    assert context.resolve(turn(),'unrelated',submission_id='submission-a') is None
    assert context.resolve(turn(),'session-b',submission_id='submission-b')=='b'*32
    assert context.resolve(turn(),'session-a',submission_id='submission-a')=='a'*32


def test_same_chat_queued_equal_questions_keep_per_submission_sources(context):
    context.queue('a'*32,text_digest('same question'),['owner'],submission_id='first')
    context.queue('b'*32,text_digest('same question'),['owner'],submission_id='second')
    assert context.resolve(turn(),'owner',submission_id='first')=='a'*32
    assert context.resolve(turn(),'owner',submission_id='second')=='b'*32
    assert context.resolve(turn(),'owner',submission_id='first')=='a'*32
    assert context.resolve(turn('follow-up'),'owner')=='b'*32


def test_pending_and_consumed_request_bindings_survive_restart_without_text_matching(context):
    context.queue('a'*32,text_digest('raw input'),['runtime','stored'],ticket_id='1234567890abcdef',submission_id='exact-request')
    cold=VideoContext(context.store)
    assert cold.resolve(turn('host-expanded text'),'runtime',submission_id='exact-request')=='a'*32
    result=cold.queue('a'*32,text_digest('raw input'),['runtime','stored'],ticket_id='1234567890abcdef',submission_id='exact-request')
    assert result['already_bound'] and not cold.pending
    assert VideoContext(context.store).resolve(turn(),'runtime',submission_id='exact-request')=='a'*32


def test_failed_ownership_transaction_consumes_nothing_and_injects_nothing(context):
    context.queue('a'*32,text_digest('question'),['owner'],submission_id='exact-request')
    with context.store.db:
        context.store.db.executescript("CREATE TRIGGER owned_failure BEFORE INSERT ON objects WHEN NEW.kind='media_focus_v1' BEGIN SELECT RAISE(ABORT,'owned metadata write failure'); END;")
    with pytest.raises(sqlite3.IntegrityError):context.resolve(turn('question'),'owner',submission_id='exact-request')
    key=hashlib.sha256(b'owner\0exact-request').hexdigest()
    assert context.store.get('video_turn_v3',key) is None and context.store.get('video_context_v2','owner') is None
    assert len(VideoContext(context.store).pending)==1


@pytest.mark.parametrize('sessions,submission',[([], 'request'),(['draft'],'request'),(['owner'],None),(['owner'],'bad\r\nheader')])
def test_unscoped_or_missing_admission_identity_is_rejected_before_queuing(context,sessions,submission):
    with pytest.raises(ValueError):context.queue('a'*32,text_digest('question'),sessions,submission_id=submission)
    assert not context.pending and context.store.get('video_context_v3','pending') is None


def test_admitted_request_cannot_be_rebound_to_another_video(context):
    context.queue('a'*32,text_digest('question'),['owner'],submission_id='request')
    assert context.resolve(turn(),'owner',submission_id='request')=='a'*32
    with pytest.raises(ValueError,match='different attachment'):
        context.queue('b'*32,text_digest('question'),['owner'],submission_id='request')
    assert context.resolve(turn(),'owner',submission_id='request')=='a'*32


def test_video_readiness_returns_only_analyzed_local_frames_or_explicit_error(context):
    from scarx.hybrid.media import VideoMemory
    from types import SimpleNamespace
    memory=VideoMemory(SimpleNamespace(store=context.store))
    context.store.put('stream','ready',{'id':'ready','status':'complete','vision_frontier':10})
    assert memory.wait_first_observation('ready')['ready']
    context.store.put('stream','failed',{'id':'failed','status':'failed','vision_frontier':0})
    with pytest.raises(RuntimeError,match='no usable visual'):memory.wait_first_observation('failed')
    context.store.put('stream','pending',{'id':'pending','status':'capturing','vision_frontier':0})
    with pytest.raises(TimeoutError,match='draft retained'):memory.wait_first_observation('pending',timeout=0)
