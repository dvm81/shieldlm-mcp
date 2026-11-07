"""Measure a detector on the labelled tool-description corpora.

    python scripts/eval_corpus.py                       # heuristic rules
    python scripts/eval_corpus.py --detector shieldlm   # rules + ShieldLM (needs the [ml] extra)

`descriptions.jsonl` was used to write the rules; `holdout.jsonl` was written
afterwards and never used for tuning, so its numbers are the honest ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shieldlm_mcp.detection import build_detector

CORPORA = Path(__file__).resolve().parent.parent / "tests" / "corpus"


def evaluate(detector, path: Path, show_errors: bool) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = detector.score_batch([r["text"] for r in rows])
    tp = fp = fn = tn = 0
    for row, res in zip(rows, results, strict=True):
        flagged = res.label == "injection"
        if row["label"] == "poisoned":
            tp, fn = tp + flagged, fn + (not flagged)
        else:
            fp, tn = fp + flagged, tn + (not flagged)
        if show_errors and flagged != (row["label"] == "poisoned"):
            kind = "missed" if row["label"] == "poisoned" else "false positive"
            print(f"  {kind:14} p={res.probability:.3f} {row['text'][:100]!r}")
    return {"corpus": path.name, "recall": tp / (tp + fn), "fpr": fp / (fp + tn), "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["heuristic", "shieldlm"], default="heuristic")
    ap.add_argument("--fpr", type=float, default=0.001, help="ShieldLM false-positive budget (0.001, 0.005, 0.01, 0.05)")
    ap.add_argument("--errors", action="store_true", help="Print every miss and false positive")
    args = ap.parse_args()
    det = build_detector(args.detector, **({"fpr_target": args.fpr} if args.detector == "shieldlm" else {}))
    print(f"detector: {det.name}")
    for name in ("descriptions.jsonl", "holdout.jsonl"):
        r = evaluate(det, CORPORA / name, args.errors)
        print(f"{r['corpus']:20} recall {r['tp']}/{r['tp'] + r['fn']} = {r['recall']:.0%}   "
              f"false positives {r['fp']}/{r['fp'] + r['tn']} = {r['fpr']:.0%}")


if __name__ == "__main__":
    main()
