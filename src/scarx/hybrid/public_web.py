"""Read-only public-topic lookup. Runtime authorizes topics, never model text."""

from html.parser import HTMLParser
import ipaddress
import json
import socket
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener, urlopen


def public_url(url):
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Only public HTTPS sources are allowed")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(item[4][0]).is_global for item in addresses
    ):
        raise ValueError("Non-public destination blocked")
    return url


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_public(url, max_bytes=512_000):
    opener = build_opener(NoRedirect())
    for _ in range(4):
        public_url(url)
        try:
            response = opener.open(
                Request(
                    url,
                    headers={
                        "User-Agent": "SCARXHybrid/1.0 (https://github.com/AKANEMO22/SCAR-X)",
                        "Accept-Encoding": "identity",
                    },
                ),
                timeout=12,
            )
        except HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                url = urljoin(url, exc.headers["Location"])
                continue
            raise
        with response:
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError("Public source exceeds read budget")
            return body.decode("utf-8", errors="replace"), url
    raise ValueError("Too many redirects")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class PublicWeb:
    def __init__(self, store, local_search=False):
        self.store = store
        self.local_search = local_search

    def initialize_public_curriculum(self):
        if self.store.get("runtime", "public_curriculum_initialized"):
            return
        # Non-sensitive public subjects covered by the user's broad public-web
        # approval. These are permissions/curriculum, not factual training labels.
        for topic in (
            "toán học",
            "vật lý",
            "lập trình máy tính",
            "ngữ pháp tiếng Việt",
            "English grammar",
        ):
            self.allow_topic(topic, approval="approved_public_web_policy")
        self.store.put("runtime", "public_curriculum_initialized", {"version": 1})

    def allow_topic(self, topic, approval="user"):
        if not isinstance(topic, str) or not 2 <= len(topic) <= 160 or "\n" in topic:
            raise ValueError(
                "Public topic must be a short explicit owner-approved subject"
            )
        import hashlib

        identifier = hashlib.sha256(topic.encode()).hexdigest()[:24]
        if approval not in ("user", "approved_public_web_policy"):
            raise ValueError("Unknown approval provenance")
        self.store.put(
            "public_topic",
            identifier,
            {
                "id": identifier,
                "topic": topic,
                "approved_by": approval,
                "created": time.time(),
            },
        )
        self.store.event(
            "public_topic_approved",
            {"id": identifier, "topic": topic, "approval": approval},
            "user" if approval == "user" else "orchestrator",
        )
        return identifier

    def search(self, topic_id, language="vi"):
        topic = self.store.get("public_topic", topic_id)
        if not topic or topic.get("approved_by") not in (
            "user",
            "approved_public_web_policy",
        ):
            raise PermissionError(
                "No owner-approved public topic; keep private goal local"
            )
        if language not in ("vi", "en"):
            raise ValueError("Supported search languages vi/en")
        if self.local_search:
            return self.search_local(topic)
        # The exact approved topic is the only outgoing query. No private suffix.
        parameters = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": topic["topic"],
            "gsrlimit": 3,
            "prop": "extracts|info|extlinks",
            "exintro": 1,
            "explaintext": 1,
            "exchars": 3500,
            "inprop": "url",
            "ellimit": 3,
        }
        url = (
            "https://" + language + ".wikipedia.org/w/api.php?" + urlencode(parameters)
        )
        self.store.event(
            "web_search_started",
            {"topic_id": topic_id, "query": topic["topic"], "url": url},
            "runtime",
        )
        text, _ = read_public(url)
        result = json.loads(text)
        pages = list(result.get("query", {}).get("pages", {}).values())[:3]
        evidence = []
        for page in pages:
            if page.get("extract"):
                item = self.store.observe(
                    page["extract"][:12000],
                    source=page.get("fullurl", url),
                    stream="public:" + topic_id,
                    modality="web",
                    public_topic=topic_id,
                )
                item["linked_sources"] = [
                    row.get("*") for row in page.get("extlinks", []) if row.get("*")
                ]
                self.store.put("linked_sources", item["id"], item["linked_sources"])
                evidence.append(item)
        self.store.event(
            "web_search_completed",
            {
                "topic_id": topic_id,
                "evidence_ids": [e["id"] for e in evidence],
                "status": "retrieved, not factual verification",
            },
            "runtime",
        )
        # One bounded linked-source read per public search round. Agreement is
        # still only source support; it never becomes an automatic true label.
        linked = next((item for item in evidence if item.get("linked_sources")), None)
        if linked:
            try:
                primary = self.read_linked(linked["id"], 0)
                evidence.append(primary)
                self.store.event(
                    "linked_public_source_read",
                    {
                        "evidence_id": primary["id"],
                        "source": primary["source"],
                        "verified": False,
                    },
                    "runtime",
                )
            except (ValueError, OSError) as exc:
                self.store.event(
                    "linked_public_source_unavailable",
                    {"from_evidence": linked["id"], "error": str(exc)},
                    "runtime",
                )
        return evidence

    def search_local(self, topic):
        # Fixed owned loopback service, not an arbitrary model-supplied URL.
        url = "http://127.0.0.1:18778/search?" + urlencode(
            {"q": topic["topic"], "format": "json"}
        )
        self.store.event(
            "web_search_started",
            {
                "topic_id": topic["id"],
                "query": topic["topic"],
                "backend": "native_searxng",
                "private_goal_sent": False,
            },
        )
        with urlopen(
            Request(url, headers={"Accept": "application/json"}), timeout=10
        ) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("SearXNG observation bound exceeded")
        data = json.loads(raw)
        evidence = []
        parser = PlainText()
        for row in data.get("results", [])[:3]:
            source = public_url(row["url"])
            parser = PlainText()
            parser.feed(row.get("title", "") + " " + row.get("content", ""))
            text = " ".join(" ".join(parser.parts).split())[:12000]
            if text:
                evidence.append(
                    self.store.observe(
                        text,
                        source=source,
                        stream="public:" + topic["id"],
                        modality="web",
                        public_topic=topic["id"],
                    )
                )
        self.store.event(
            "web_search_completed",
            {
                "topic_id": topic["id"],
                "backend": "native_searxng",
                "evidence_ids": [r["id"] for r in evidence],
                "status": "search snippets observed, not full-page verification",
            },
        )
        return evidence

    def read_linked(self, evidence_id, index=0):
        links = self.store.get("linked_sources", evidence_id) or []
        if type(index) is not int or not 0 <= index < len(links):
            raise ValueError("Not a linked source")
        text, url = read_public(links[index])
        parser = PlainText()
        parser.feed(text)
        source = self.store.evidence(evidence_id)
        return self.store.observe(
            " ".join(" ".join(parser.parts).split())[:12000],
            source=url,
            stream=source["stream"],
            modality="web",
            public_topic=source["public_topic"],
        )
