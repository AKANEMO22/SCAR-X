"""Exact-source action authorization fix, not a learner or truth-verifier change."""

import hashlib

EXPECTED="a46f4c2a019520505af4190f59e2803c35d095c7b5b877f1864ebcd34fe48fae"


def patch_bytes(original):
    if hashlib.sha256(original).hexdigest()!=EXPECTED:
        raise ValueError("Engine source changed; review capability guard before applying")
    eol=b'\r\n' if b'\r\n' in original else b'\n'
    text=original.decode('utf-8').replace('\r\n','\n')
    before='''        if not goal["verified_attempted"]:
            result.append("verify")
'''
    after='''        # Authorize only the checker this runtime actually implements.
        # Parser acceptance is not a general factual/user-intent proof.
        if (not goal["verified_attempted"]
                and parse_integer_expression(goal["question"]) is not None):
            result.append("verify")
'''
    if text.count(before)!=1:raise ValueError("Verifier authorization anchor changed")
    raw=text.replace(before,after,1).encode().replace(b'\n',eol)
    compile(raw,'capability-guard-engine.py','exec')
    return raw
