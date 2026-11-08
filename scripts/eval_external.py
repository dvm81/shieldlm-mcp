"""Score the external test set from scripts/build_external.py with the rules and with ShieldLM models.

Model thresholds are calibrated on the npm calib packages, as in scripts/finetune_tools.py,
so the numbers line up with that experiment. The npm test packages and, if present, the
MCPTox test servers, and this repo's held-out corpus are scored alongside. Without --model
only the rules run.

    python scripts/eval_external.py data/external.jsonl
    python scripts/eval_external.py data/external.jsonl --model dmilush/shieldlm-deberta-base --model models/shieldlm-tools-s0
    python scripts/eval_external.py data/external.jsonl --model ... --latin-calib   # calibrate without CJK or Cyrillic text
    python scripts/eval_external.py data/external.jsonl --misses 5                  # print rule misses per category

Rows marked instructs=false (attacks that give the model no instruction) are reported
separately and left out of the totals.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from finetune_tools import FPR_TARGETS, HIDDEN, ROOT, SPLIT, load_jsonl, margins, threshold_at

# Every model scores the calib split's few Chinese descriptions as certain injections.
INTERNAL = {"npm test", "MCPTox test", "held-out corpus"}
NON_LATIN = re.compile(r"[Ѐ-ӿ぀-ヿ㐀-鿿가-힯]")


def groups(row: dict) -> list[str]:
    if row["label"] == "benign":
        return [f"{row['source']} | benign"] + (["all external | benign"] if row["source"] not in INTERNAL else [])
    if not row["instructs"]:
        return [f"{row['source']} | {row['category']} (no instruction)"]
    if row["source"] in ("MCPTox test", "held-out corpus"):
        return [f"{row['source']} | poisoned"]
    return [f"{row['source']} | {row['category']}", f"{row['source']} | all with instructions", "all external | poisoned, with instructions"]


def score_models(names: list[str], texts: list[str], calib: list[str]) -> dict[str, tuple[list[float], list[float]]]:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from shieldlm_mcp.detection.shieldlm import SELF_CHECK_TEXTS, self_check

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for name in names:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        m = margins(model, tok, list(SELF_CHECK_TEXTS), device)
        self_check([1 / (1 + math.exp(-x)) for x in m], transformers.__version__)
        out[name] = (margins(model, tok, texts, device), margins(model, tok, calib, device))
        del model
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="JSONL from scripts/build_external.py")
    ap.add_argument("--model", action="append", default=[], help="ShieldLM model id or path; repeat for several")
    ap.add_argument("--npm", type=Path, default=ROOT / "data" / "npm_benign_descriptions.jsonl")
    ap.add_argument("--mcptox", type=Path, default=ROOT / "data" / "mcptox_pure_tool.json")
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.01, 0.02], choices=FPR_TARGETS, help="npm calib flag rates to report")
    ap.add_argument("--latin-calib", action="store_true", help="Leave CJK and Cyrillic descriptions out of calibration")
    ap.add_argument("--misses", type=int, default=0, help="Print this many descriptions per category that the rules miss")
    ap.add_argument("--out", type=Path, help="Write per-row scores here")
    args = ap.parse_args()

    from shieldlm_mcp.detection.heuristic import HeuristicDetector

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = load_jsonl(args.data)
    npm = load_jsonl(args.npm) if args.npm.exists() else []
    rows += [{"source": "npm test", "label": "benign", "instructs": False, "text": r["text"]} for r in npm if r["package"] in set(split["npm"]["test"])]
    if args.mcptox.exists():
        tox = [v for group in json.loads(args.mcptox.read_text(encoding="utf-8")) for v in group.values()]
        rows += [
            {"source": "MCPTox test", "label": "poisoned", "instructs": True, "category": r["paradigm"], "tool": r.get("target_tool", ""), "text": r["tool_content"]}
            for r in tox
            if r["server_name"] in set(split["mcptox"]["test"])
        ]
    rows += [
        {"source": "held-out corpus", "label": r["label"], "instructs": r["label"] == "poisoned", "category": "", "tool": "", "text": r["text"]}
        for r in load_jsonl(ROOT / "tests" / "corpus" / "holdout.jsonl")
    ]
    texts = [r["text"] for r in rows]
    rules = [r.label == "injection" and any(s not in HIDDEN for s in r.signals) for r in HeuristicDetector().score_batch(texts)]

    models = {}
    if args.model:
        calib = [r["text"] for r in npm if r["package"] in set(split["npm"]["calib"])]
        if args.latin_calib:
            calib = [t for t in calib if not NON_LATIN.search(t)]
        print(f"calibrating on {len(calib)} npm calib descriptions\n")
        models = score_models(args.model, texts, calib)

    members = defaultdict(list)
    for i, r in enumerate(rows):
        for g in groups(r):
            members[g].append(i)
    order = sorted(members, key=lambda g: (g.split(" | ")[0] in INTERNAL, g.startswith("all"), "benign" in g, "no instruction" in g, g))

    def table(title: str, cols: dict[str, list[bool]]) -> None:
        print(f"{title}\n  {'share flagged (for benign rows, the false-positive rate)':62} {'n':>4} " + " ".join(f"{c:>9}" for c in cols))
        for g in order:
            idx = members[g]
            print(f"  {g:62} {len(idx):>4} " + " ".join(f"{sum(f[i] for i in idx) / len(idx):>9.1%}" for f in cols.values()))
        print()

    if not models:
        table("rules", {"rules": rules})
    for name, (scores, calib_scores) in models.items():
        cols = {"rules": rules}
        for fpr in args.fpr:
            t = threshold_at(calib_scores, fpr)
            cols[f"@{fpr:.1%}"] = [x >= t for x in scores]
            cols["+rules"] = [x >= t or h for x, h in zip(scores, rules, strict=True)]
            cols = {(f"@{fpr:.1%}+rules" if k == "+rules" else k): v for k, v in cols.items()}
        table(name, cols)

    if args.misses:
        by_cat = defaultdict(list)
        for r, hit in zip(rows, rules, strict=True):
            if r["instructs"] and not hit and r["source"] not in INTERNAL:
                by_cat[f"{r['source']} | {r['category']}"].append(r)
        for cat, missed in sorted(by_cat.items()):
            print(f"missed by the rules: {cat} ({len(missed)})")
            for r in missed[: args.misses]:
                print(f"  - {r['tool']}: {' '.join(r['text'].split())[:300]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        scored = [dict(r, rules=h, **{f"margin {n}": s[0][i] for n, s in models.items()}) for i, (r, h) in enumerate(zip(rows, rules, strict=True))]
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), encoding="utf-8")


if __name__ == "__main__":
    main()
