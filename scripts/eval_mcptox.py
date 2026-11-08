"""Measure recall on the MCPTox benchmark of poisoned tool descriptions.

MCPTox (Wang et al., AAAI 2026, arXiv:2508.14925) contains 485 poisoned tool
descriptions written against 45 real MCP servers. The data is not redistributed
here. Download it first:

    curl -o pure_tool.json https://anonymous.4open.science/api/repo/AAAI26-7C02/file/pure_tool.json

Then:

    python scripts/eval_mcptox.py pure_tool.json                      # rules
    python scripts/eval_mcptox.py pure_tool.json --detector shieldlm  # rules + ShieldLM
    python scripts/eval_mcptox.py pure_tool.json --split test         # held-out servers only

The rules were written from the servers in the "dev" half of tests/corpus/eval_split.json,
so report recall on --split test.

Every description in the set is poisoned, so this measures recall only. For false
positives, use scripts/eval_corpus.py and real benign descriptions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from shieldlm_mcp.detection import build_detector

HIDDEN = {"invisible_unicode", "html_comment", "encoded_blob"}
SPLIT = Path(__file__).resolve().parent.parent / "tests" / "corpus" / "eval_split.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="MCPTox pure_tool.json")
    ap.add_argument("--detector", choices=["heuristic", "shieldlm", "shieldlm-tools"], default="heuristic")
    ap.add_argument("--fpr", type=float, help="ML false-positive budget (default: the model's own)")
    ap.add_argument("--misses", type=int, default=0, help="Print this many missed descriptions")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all", help="Servers to include (see tests/corpus/eval_split.json)")
    args = ap.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    rows = [v for group in data for v in group.values()]
    if args.split != "all":
        servers = set(json.loads(SPLIT.read_text(encoding="utf-8"))["mcptox"][args.split])
        rows = [r for r in rows if r["server_name"] in servers]
    det = build_detector(args.detector, **({"fpr_target": args.fpr} if args.detector != "heuristic" else {}))
    results = det.score_batch([r["tool_content"] for r in rows])

    def caught(res) -> bool:
        return res.label == "injection" and any(s not in HIDDEN for s in res.signals)

    total, hit = Counter(), Counter()
    for row, res in zip(rows, results, strict=True):
        for key in ("all", f"paradigm: {row.get('paradigm')}", f"risk: {row.get('security risk')}"):
            total[key] += 1
            hit[key] += caught(res)

    print(f"detector: {det.name}   split: {args.split}   descriptions: {len(rows)}")
    for key in sorted(total, key=lambda k: (k != "all", k)):
        print(f"  {key:38} {hit[key]:4}/{total[key]:<4} {hit[key] / total[key]:6.1%}")
    if args.misses:
        print("\nmissed:")
        missed = [r for r, res in zip(rows, results, strict=True) if not caught(res)]
        for r in missed[: args.misses]:
            print(f"  [{r.get('paradigm')}, {r.get('security risk')}] {r['tool_content'].strip()[:200]!r}")


if __name__ == "__main__":
    main()
