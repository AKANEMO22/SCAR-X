"""Readonly canonical submission identity bound to its execution task.

The client request id belongs to this admitted input, never to a text digest,
cached agent, ambient profile or a later queued input on the same session.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_submission=ContextVar('canonical_submission_identity',default=None)


@contextmanager
def submission_scope(request_id):
    token=_submission.set(request_id if isinstance(request_id,str) else None)
    try:yield
    finally:_submission.reset(token)


def current_submission_id():
    return _submission.get()
