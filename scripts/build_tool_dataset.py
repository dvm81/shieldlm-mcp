"""Put every tool-description source into one labelled file with fixed splits.

    python scripts/build_tool_dataset.py -o data/tool_dataset.jsonl

Inputs, none redistributed here:

- data/mcptox_pure_tool.json: MCPTox (download instructions in scripts/eval_mcptox.py)
- data/npm_benign_descriptions.jsonl: English descriptions from npm packages
- data/npm_nonen_descriptions.jsonl: non-English ones (scripts/collect_npm_descriptions.py)
- data/external.jsonl: MCP-SafetyBench, MSB and mcp-tool-poisoning (scripts/build_external.py)
- tests/corpus/descriptions.jsonl and tests/corpus/holdout.jsonl

Splits come from tests/corpus/eval_split.json; keys missing there are created with a seeded
shuffle and written back. Training sees only `train`, thresholds come from `calib`, runs are
compared on `dev`, and results are reported on `test`:

    split  poisoned                                              benign
    train  MCPTox dev servers, tuning corpus                     npm train, non-English npm train, tuning corpus
    calib                                                        npm calib, non-English npm calib
    dev    external sets, dev half                               external sets, dev half
    test   MCPTox test servers, external test half, held-out     npm test, non-English npm test, external test half, held-out

Rows get `translate_to`, the languages scripts/translate_descriptions.py should translate them
into: two for poisoned training rows, one for the others chosen, in turn. Calibration rows are
not translated, so thresholds rest on real descriptions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "tests" / "corpus" / "eval_split.json"
LANGS = ["zh", "de", "ja", "es", "ko", "fr", "ru", "pt"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fill(groups: dict[str, int], shares: dict[str, float], seed: int = 0) -> dict[str, list[str]]:
    """Assign whole groups to splits so each split gets about its share of rows."""
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = sum(groups.values())
    filled = dict.fromkeys(shares, 0)
    out: dict[str, list[str]] = {s: [] for s in shares}
    for k in keys:
        s = max(shares, key=lambda s: (shares[s] * total - filled[s]) / (shares[s] * total))
        out[s].append(k)
        filled[s] += groups[k]
    return {s: sorted(v) for s, v in out.items()}


def external_key(r: dict) -> str:
    return f"{r['source']}/{r['server'] or r['tool']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--mcptox", type=Path, default=ROOT / "data" / "mcptox_pure_tool.json")
    ap.add_argument("--npm", type=Path, default=ROOT / "data" / "npm_benign_descriptions.jsonl")
    ap.add_argument("--nonen", type=Path, default=ROOT / "data" / "npm_nonen_descriptions.jsonl")
    ap.add_argument("--external", type=Path, default=ROOT / "data" / "external.jsonl")
    ap.add_argument("--translate-npm-train", type=int, default=700, help="English npm train rows to translate")
    ap.add_argument("--translate-npm-test", type=int, default=400, help="English npm test rows to translate")
    ap.add_argument("--max-per-package", type=int, default=60, help="Keep at most this many non-English strings per package")
    args = ap.parse_args()

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tox = [v for group in json.loads(args.mcptox.read_text(encoding="utf-8")) for v in group.values()]
    npm, nonen, external = load_jsonl(args.npm), load_jsonl(args.nonen), load_jsonl(args.external)
    # A few packages carry hundreds of strings; cap them so no single package dominates.
    per_pkg = defaultdict(list)
    for r in nonen:
        per_pkg[r["package"]].append(r)
    nonen = [r for pkg in sorted(per_pkg) for r in random.Random(pkg).sample(per_pkg[pkg], min(args.max_per_package, len(per_pkg[pkg])))]

    if "npm_nonen" not in split:
        # Split packages within each language, so every language reaches train, calib and test when it can.
        pkg_langs = defaultdict(Counter)
        for r in nonen:
            pkg_langs[r["package"]][r["lang"]] += 1
        by_lang = defaultdict(Counter)
        for pkg, langs in pkg_langs.items():
            by_lang[langs.most_common(1)[0][0]][pkg] = sum(langs.values())
        split["npm_nonen"] = {"train": [], "calib": [], "test": []}
        for counts in by_lang.values():
            for s, pkgs in fill(counts, {"train": 0.5, "calib": 0.2, "test": 0.3}).items():
                split["npm_nonen"][s] += pkgs
        split["npm_nonen"] = {s: sorted(v) for s, v in split["npm_nonen"].items()}
    if "external" not in split:
        split["external"] = fill(Counter(external_key(r) for r in external), {"dev": 0.5, "test": 0.5})
    SPLIT.write_text(json.dumps(split, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def where(part: dict[str, list[str]], key: str) -> str:
        return next(s for s, keys in part.items() if key in keys)

    rows: list[dict] = []

    def add(text: str, label: int, s: str, dataset: str, lang: str = "en", **extra) -> None:
        rows.append({"text": text, "label": label, "split": s, "set": dataset, "lang": lang, "translated": False, **extra})

    for r in tox:
        add(r["tool_content"], 1, "train" if r["server_name"] in split["mcptox"]["dev"] else "test", "mcptox", category=r["paradigm"])
    for r in npm:
        add(r["text"], 0, where(split["npm"], r["package"]), "npm", group=r["package"])
    for r in nonen:
        add(r["text"], 0, where(split["npm_nonen"], r["package"]), "npm-nonen", r["lang"], group=r["package"])
    for name, s in (("descriptions", "train"), ("holdout", "test")):
        for r in load_jsonl(ROOT / "tests" / "corpus" / f"{name}.jsonl"):
            add(r["text"], int(r["label"] == "poisoned"), s, "tuning" if s == "train" else "holdout")
    for r in external:
        add(r["text"], int(r["label"] == "poisoned"), where(split["external"], external_key(r)), "external",
            source=r["source"], category=r["category"], instructs=r["instructs"])  # fmt: skip

    # Which rows to translate, and into what.
    rng = random.Random(0)
    npm_rows = {s: [r for r in rows if r["set"] == "npm" and r["split"] == s] for s in ("train", "test")}
    chosen = {id(r) for r in rng.sample(npm_rows["train"], min(args.translate_npm_train, len(npm_rows["train"])))}
    chosen |= {id(r) for r in rng.sample(npm_rows["test"], min(args.translate_npm_test, len(npm_rows["test"])))}
    turn = Counter()
    for r in rows:
        if r["set"] in ("npm-nonen",) or r["split"] == "calib" or (r["set"] == "npm" and id(r) not in chosen):
            continue
        if r["set"] == "external" and r["label"] == 1 and not r["instructs"]:
            continue
        k = (r["split"], r["label"])
        i = turn[k]
        turn[k] += 1
        r["translate_to"] = [LANGS[i % 8], LANGS[(i + 4) % 8]] if r["split"] == "train" and r["label"] == 1 else [LANGS[i % 8]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}; {sum(len(r.get('translate_to', [])) for r in rows)} translations to make")
    counts = Counter((r["split"], r["set"], r["label"]) for r in rows)
    for (s, d, label), n in sorted(counts.items()):
        print(f"  {s:6} {d:10} {'poisoned' if label else 'benign':9} {n}")


if __name__ == "__main__":
    main()
