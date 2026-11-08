"""Collect non-English description strings from MCP server packages on npm, as benign examples.

The English npm descriptions in data/npm_benign_descriptions.jsonl contain no text in other
languages, and ShieldLM fine-tuned on them reads Chinese descriptions as injections. This
script finds MCP packages written in other languages and extracts the strings their code
passes as `description` or `.describe(...)`, the same kind of strings as the English set.

    python scripts/collect_npm_descriptions.py -o data/npm_nonen_descriptions.jsonl

Package tarballs are downloaded to --cache and read as text; nothing in them is run.
Strings are assumed benign, like the English set. Packages already in the English set are skipped.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = "https://registry.npmjs.org"
QUERIES = [
    "mcp server", "mcp", "modelcontextprotocol", "mcp-server", "mcp tools",
    "mcp 中文", "mcp 服务", "mcp 服务器", "mcp 工具", "mcp サーバー", "mcp ツール", "mcp 서버", "mcp 도구",
    "mcp сервер", "mcp инструменты", "mcp servidor", "mcp herramientas", "mcp serveur", "mcp outils",
    "mcp für", "mcp werkzeuge", "mcp ferramentas", "mcp strumenti",
]  # fmt: skip
SOURCE_SUFFIXES = (".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".json")

STRING = r"""(?P<q>["'`])(?P<s>(?:\\.|(?!(?P=q))[^\\])*)(?P=q)"""
PATTERNS = [
    re.compile(r"""["']?description["']?\s*:\s*""" + STRING, re.S),
    re.compile(r"""\.describe\(\s*""" + STRING + r"""\s*\)""", re.S),
]

SCRIPTS = {
    "ja": re.compile(r"[\u3040-\u30fa\u30fc-\u30ff]"),  # kana, without the middle dot Korean text also uses
    "ko": re.compile(r"[\uac00-\ud7af]"),
    "zh": re.compile(r"[\u3400-\u9fff]"),
    "ru": re.compile(r"[\u0400-\u04ff]"),
}
STOPWORDS = {
    "de": "der die das und ist nicht mit für ein eine einen den dem von zu auf werden wird oder sie aus bei".split(),
    "es": "el la los las de que y para con una un por del se es al como más o su sus".split(),
    "fr": "le la les des une un et pour est avec dans du au aux sur par ce cette qui ou vous".split(),
    "pt": "o a os as de que e para com uma um não do da dos das em no na por ao ou se".split(),
    "it": "il lo la gli le di che e per con una un non del della dei delle nel nella sono".split(),
}
NON_LATIN = re.compile(r"[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def language(text: str) -> str:
    """Rough language of a short string: by script for CJK and Cyrillic, by stopwords for Latin-script languages."""
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return "unknown"
    non_ascii = sum(not c.isascii() for c in letters) / len(letters)
    for lang in ("ko", "ja", "zh", "ru"):
        if SCRIPTS[lang].search(text) and non_ascii > 0.15:
            return lang
    words = [w for w in re.findall(r"[a-zà-ÿ]+", text.lower()) if len(w) > 1]
    if len(words) < 4:
        return "unknown"
    hits = {lang: sum(w in set(sw) for w in words) for lang, sw in STOPWORDS.items()}
    hits["en"] = sum(w in {"the", "a", "an", "of", "to", "and", "is", "for", "with", "in", "this", "or", "from", "by", "be", "if", "it"} for w in words)
    best = max(hits, key=hits.get)
    return best if hits[best] >= 2 and hits[best] / len(words) >= 0.12 else "unknown"


def unescape(s: str) -> str:
    s = re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), s)
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "r": ""}.get(m.group(1), m.group(1)), s, flags=re.S)


def strings(source: str) -> list[str]:
    out = []
    for pattern in PATTERNS:
        for m in pattern.finditer(source):
            s = m.group("s")
            if m.group("q") == "`" and "${" in s:
                continue
            s = " ".join(unescape(s).split())
            # Chinese and Japanese say in 6 characters what English says in 20.
            long_enough = len(s) >= 12 or len(NON_LATIN.findall(s)) >= 5
            if long_enough and len(s) <= 3000 and not s.startswith(("http://", "https://")):
                out.append(s)
    return out


def get(url: str, cache: Path | None = None, binary: bool = False):
    if cache and cache.exists():
        data = cache.read_bytes()
    else:
        for attempt in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                time.sleep(5 * 2**attempt)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        time.sleep(0.3)
    return data if binary else json.loads(data)


def search(cache: Path) -> dict[str, dict]:
    found = {}
    for q in QUERIES:
        slug = re.sub(r"[^\w]+", "_", q)
        for frm in range(0, 1000, 250):
            url = f"{REGISTRY}/-/v1/search?" + urllib.parse.urlencode({"text": q, "size": 250, "from": frm})
            objs = get(url, cache / "search" / f"{slug}_{frm}.json")["objects"]
            for o in objs:
                found[o["package"]["name"]] = dict(o["package"], monthly=(o.get("downloads") or {}).get("monthly", 0))
            time.sleep(2)
            if len(objs) < 250:
                break
    return found


def package_language(pkg: dict) -> str | None:
    """Language of an MCP package's npm description, or None if it is not an MCP package or is in English."""
    text = " ".join([pkg.get("description") or "", " ".join(pkg.get("keywords") or [])])
    if not re.search(r"\bmcp\b|model ?context ?protocol|modelcontextprotocol", (pkg["name"] + " " + text).lower()):
        return None
    lang = language(text)
    return None if lang in ("en", "unknown") else lang


def package_strings(name: str, cache: Path) -> tuple[str, list[str]]:
    meta = get(f"{REGISTRY}/{urllib.parse.quote(name, safe='@')}/latest", cache / "meta" / f"{name.replace('/', '__')}.json")
    version, tarball = meta["version"], meta["dist"]["tarball"]
    data = get(tarball, cache / "tarballs" / f"{name.replace('/', '__')}-{version}.tgz", binary=True)
    found: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.size > 5_000_000 or not member.name.endswith(SOURCE_SUFFIXES):
                continue
            if "/node_modules/" in member.name or member.name.endswith(("package.json", "package-lock.json", ".d.ts", ".map")):
                continue
            f = tar.extractfile(member)
            if f:
                found += strings(f.read().decode("utf-8", errors="replace"))
    return version, list(dict.fromkeys(found))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "npm_cache")
    ap.add_argument("--english", type=Path, default=ROOT / "data" / "npm_benign_descriptions.jsonl", help="Skip packages in this set")
    ap.add_argument("--min-strings", type=int, default=3, help="Keep packages with at least this many non-English strings")
    ap.add_argument("--per-language", type=int, default=120, help="Scan at most this many packages per language, most downloaded first")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    skip = {json.loads(line)["package"] for line in args.english.read_text(encoding="utf-8").splitlines() if line.strip()}
    found = search(args.cache)
    by_lang = defaultdict(list)
    for name, pkg in found.items():
        lang = package_language(pkg)
        if lang and name not in skip:
            by_lang[lang].append(pkg)
    candidates = sorted(p["name"] for pkgs in by_lang.values() for p in sorted(pkgs, key=lambda p: -p["monthly"])[: args.per_language])
    print(f"{len(found)} packages found; MCP packages with a non-English description: "
          + ", ".join(f"{lang} {len(v)}" for lang, v in sorted(by_lang.items())) + f"; scanning {len(candidates)}", flush=True)  # fmt: skip

    def fetch(name: str):
        try:
            return name, *package_strings(name, args.cache)
        except Exception as e:  # noqa: BLE001 - a broken package should not stop the crawl
            print(f"  skip {name}: {e}", flush=True)
            return name, None, []

    rows, langs = [], Counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(fetch, candidates))
    for name, version, texts in results:
        keep = [(t, language(t)) for t in texts]
        keep = [(t, lang) for t, lang in keep if lang not in ("en", "unknown")]
        if len(keep) >= args.min_strings:
            rows += [{"package": name, "version": version, "lang": lang, "text": t} for t, lang in keep]
            langs.update(lang for _, lang in keep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    per_pkg = defaultdict(set)
    for r in rows:
        per_pkg[r["lang"]].add(r["package"])
    print(f"wrote {len(rows)} strings from {len({r['package'] for r in rows})} packages to {args.out}")
    for lang, n in langs.most_common():
        print(f"  {lang}: {n} strings, {len(per_pkg[lang])} packages")


if __name__ == "__main__":
    main()
