"""Exact-source SQLite query patch; shadow-only until the active run ends."""

import hashlib

EXPECTED = "1ffc3c1673625698df18ecf0b020d397bf1d98ef37fcfde6a477b3ff803267bb"


def patch_bytes(original):
    if hashlib.sha256(original).hexdigest() != EXPECTED:
        raise ValueError("Store source changed; review before applying full-query patch")
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode("utf-8").replace("\r\n", "\n")
    before = '''        words = re.findall(r"[^\\W_]+", query, flags=re.UNICODE)[:16]
        if not words:
            return []
        match = " OR ".join('"' + word + '"' for word in words)
'''
    after = '''        if not isinstance(query, str) or len(query) > 6000:
            raise ValueError("Search query exceeds bounded 6000-character working request")
        words = re.findall(r"[^\\W_]+", query, flags=re.UNICODE)
        if not words:
            return []
        if len(words) <= 16:
            match = " OR ".join('"' + word + '"' for word in words)
        else:
            # Full bounded request, independent of labels or source contents.
            terms = list(dict.fromkeys(word.lower() for word in words))
            nodes = ['"' + word + '"' for word in terms]
            while len(nodes) > 1:
                nodes = ["(" + nodes[i] + " OR " + nodes[i + 1] + ")"
                         if i + 1 < len(nodes) else nodes[i]
                         for i in range(0, len(nodes), 2)]
            match = nodes[0]
'''
    if text.count(before) != 1:
        raise ValueError("Search source anchor mismatch")
    result = text.replace(before, after, 1).encode().replace(b"\n", eol)
    compile(result, "full-query-Store.py", "exec")
    return result
