"""Pinned official source into a separate local Python environment, no Docker."""

import hashlib, json, shutil, subprocess, sys, urllib.request, zipfile, secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "219b63b207c98b85471de5abab773e76432f1961"
SOURCE_ZIP_SHA256 = "f2095a690fff506ac70c1823b370828a1e9e0e927ab4b3f207dc1e921ca145b8"


def configure_native(source, base, python):
    # Upstream's optional Valkey error logger imports Unix pwd at module load.
    # Search and HTTP code stay unchanged; do not fake Unix identities.
    path = source / "searx/valkeydb.py"
    raw = path.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    old = "import pwd"
    new = "if os.name != 'nt':\n    import pwd"
    old_log = '        _pw = pwd.getpwuid(os.getuid())\n        logger.exception("[%s (%s)] can\'t connect valkey DB ...", _pw.pw_name, _pw.pw_uid)'
    new_log = '        if os.name == "nt":\n            logger.exception("can\'t connect valkey DB ...")\n        else:\n            _pw = pwd.getpwuid(os.getuid())\n            logger.exception("[%s (%s)] can\'t connect valkey DB ...", _pw.pw_name, _pw.pw_uid)'
    if new not in text:
        if text.count(old) != 1 or text.count(old_log) != 1:
            raise ValueError("Pinned SearXNG compatibility anchors changed")
        (base / "valkeydb.original.py").write_bytes(raw)
        path.write_text(
            text.replace(old, new, 1).replace(old_log, new_log, 1), encoding="utf-8"
        )
        (base / "windows-compatibility.json").write_text(
            json.dumps(
                {
                    "purpose": "Windows-only optional Valkey logging; real SearXNG search unchanged",
                    "before_sha256": hashlib.sha256(raw).hexdigest(),
                    "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    frozen = source / "searx/version_frozen.py"
    frozen.write_text(
        'VERSION_STRING = "2026.11.13+219b63b20-windows-local"\nVERSION_TAG = "2026.11.13+219b63b20"\nDOCKER_TAG = "unused"\nGIT_URL = "https://github.com/searxng/searxng"\nGIT_BRANCH = "'
        + COMMIT
        + '"\n',
        encoding="utf-8",
    )
    if not (base / "settings.yml").exists():
        settings = {
            "use_default_settings": {
                "engines": {"keep_only": ["bing", "google", "duckduckgo", "wikipedia"]}
            },
            "general": {
                "instance_name": "SCAR-X local search",
                "enable_metrics": False,
                "contact_url": "https://github.com/searxng/searxng",
            },
            "server": {
                "secret_key": secrets.token_hex(32),
                "bind_address": "127.0.0.1",
                "port": 18778,
                "limiter": False,
                "public_instance": False,
                "image_proxy": False,
            },
            "search": {
                "formats": ["html", "json"],
                "autocomplete": "",
                "default_lang": "vi",
                "safe_search": 1,
            },
            "valkey": {"url": False},
            "outgoing": {"request_timeout": 5.0, "max_request_timeout": 8.0},
            "engines": [{"name": "wikipedia", "disabled": False}],
        }
        # Use the isolated environment's own YAML dependency, not global Python.
        subprocess.run(
            [
                str(python),
                "-c",
                "import sys,json,yaml; print(yaml.safe_dump(json.load(sys.stdin),sort_keys=False))",
            ],
            input=json.dumps(settings),
            text=True,
            stdout=(base / "settings.yml").open("w", encoding="utf-8"),
            check=True,
        )
    if not (base / "limiter.toml").exists():
        shutil.copyfile(source / "searx/limiter.toml", base / "limiter.toml")


def main():
    base = ROOT / "local/searxng"
    base.mkdir(exist_ok=True)
    archive = base / (COMMIT + ".zip")
    if not archive.exists():
        with (
            urllib.request.urlopen(
                "https://codeload.github.com/searxng/searxng/zip/" + COMMIT, timeout=45
            ) as r,
            archive.open("xb") as f,
        ):
            shutil.copyfileobj(r, f, 1024 * 1024)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_ZIP_SHA256:
        raise ValueError("Pinned official SearXNG source archive hash mismatch")
    skipped_links = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = (base / info.filename).resolve()
            if not p.is_relative_to(base.resolve()):
                raise ValueError("Unsafe source member")
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                skipped_links.append(info.filename)
                continue
            if info.is_dir():
                p.mkdir(parents=True, exist_ok=True)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_bytes(z.read(info))
    source = base / ("searxng-" + COMMIT)
    env = base / "env"
    if not (env / "Scripts/python.exe").exists():
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    python = env / "Scripts/python.exe"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(source / "requirements.txt"),
            "waitress==3.0.2",
        ],
        check=True,
    )
    configure_native(source, base, python)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (base / "requirements.resolved.txt").write_text(frozen, encoding="utf-8")
    receipt = {
        "commit": COMMIT,
        "source": str(source),
        "python": str(python),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "Docker": False,
        "WSL": False,
        "skipped_symlinks": skipped_links,
        "platform": "native Windows qualification pending",
    }
    (base / "installation.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
