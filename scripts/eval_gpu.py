"""One-pass ShieldLM evaluation on tool descriptions (run on a GPU machine).

Scores every text once with ShieldLM, then reports, at each calibrated threshold,
how often ShieldLM alone and rules + ShieldLM flag:

- MCPTox: 485 poisoned descriptions written against 45 real MCP servers (recall)
- real descriptions from popular npm MCP packages, assumed benign (false-positive rate)
- this repo's own tuning and held-out corpora

    python scripts/eval_gpu.py --benign-npm data/npm_benign_descriptions.jsonl

MCPTox is downloaded on first run unless --mcptox points at a local copy.
Raw scores are saved to results/gpu_eval.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MCPTOX_URL = "https://anonymous.4open.science/api/repo/AAAI26-7C02/file/pure_tool.json"
# ShieldLM thresholds calibrated on its validation split (model card).
THRESHOLDS = {"0.1%": 0.9998, "0.5%": 0.9695, "1%": 0.1239, "5%": 0.0024}
HIDDEN = {"invisible_unicode", "html_comment", "encoded_blob"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shieldlm_scores(texts: list[str], model_name: str, batch_size: int) -> list[float]:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from shieldlm_mcp.detection.shieldlm import SELF_CHECK_TEXTS, self_check

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()
    print(f"ShieldLM on {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)
    with torch.no_grad():
        enc = tok(list(SELF_CHECK_TEXTS), padding=True, return_tensors="pt").to(device)
        self_check(torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].tolist(), transformers.__version__)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted batches run faster
    out = [0.0] * len(texts)
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tok([texts[i] or "" for i in idx], truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].tolist()
        for i, v in zip(idx, p, strict=True):
            out[i] = float(v)
    print(f"scored {len(texts)} texts in {time.time() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcptox", type=Path, default=ROOT / "data" / "mcptox_pure_tool.json")
    ap.add_argument("--benign-npm", type=Path, help="JSONL of real tool descriptions, one {'package','text'} per line")
    ap.add_argument("--model", default="dmilush/shieldlm-deberta-base")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from shieldlm_mcp.detection.heuristic import HeuristicDetector

    if not args.mcptox.exists():
        args.mcptox.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MCPTox to {args.mcptox}", flush=True)
        try:
            args.mcptox.write_bytes(urllib.request.urlopen(MCPTOX_URL, timeout=120).read())
        except urllib.error.URLError as e:
            # The host often refuses scripted downloads (HTTP 403) but serves the same file to a browser.
            raise SystemExit(f"MCPTox download failed ({e}). Save {MCPTOX_URL} from a browser and pass --mcptox <file>.") from e
    mcptox = [v for group in json.loads(args.mcptox.read_text(encoding="utf-8")) for v in group.values()]

    sets: dict[str, tuple[list[str], str]] = {"MCPTox (poisoned)": ([r["tool_content"] for r in mcptox], "recall")}
    if args.benign_npm:
        sets["npm real descriptions (benign)"] = ([r["text"] for r in load_jsonl(args.benign_npm)], "false positives")
    for name, label in (("descriptions.jsonl", "tuning"), ("holdout.jsonl", "held-out")):
        rows = load_jsonl(ROOT / "tests" / "corpus" / name)
        sets[f"own {label} (poisoned)"] = ([r["text"] for r in rows if r["label"] == "poisoned"], "recall")
        sets[f"own {label} (benign)"] = ([r["text"] for r in rows if r["label"] == "benign"], "false positives")

    all_texts = [t for texts, _ in sets.values() for t in texts]
    scores = shieldlm_scores(all_texts, args.model, args.batch_size)
    rules = HeuristicDetector()
    rule_hit = [r.label == "injection" and any(s not in HIDDEN for s in r.signals) for r in rules.score_batch(all_texts)]

    print("\nFlag rate per set. Recall for poisoned sets, false-positive rate for benign sets.")
    header = f"{'set':34} {'n':>5} {'rules':>7}" + "".join(f" {'SLM@' + k:>9} {'R+S@' + k:>9}" for k in THRESHOLDS)
    print(header)
    print("-" * len(header))
    pos = 0
    saved = {}
    for name, (texts, _kind) in sets.items():
        n = len(texts)
        s = scores[pos : pos + n]
        r = rule_hit[pos : pos + n]
        pos += n
        line = f"{name:34} {n:5} {sum(r) / n:7.1%}"
        for t in THRESHOLDS.values():
            slm = [x >= t for x in s]
            line += f" {sum(slm) / n:9.1%} {sum(a or b for a, b in zip(slm, r, strict=True)) / n:9.1%}"
        print(line)
        saved[name] = [{"text": tx[:300], "shieldlm": round(sc, 6), "rules": rh} for tx, sc, rh in zip(texts, s, r, strict=True)]

    # MCPTox recall by template and risk, ShieldLM alone, at 1% FPR.
    m = scores[: len(mcptox)]
    print("\nMCPTox recall at the 1% FPR threshold, ShieldLM alone / rules + ShieldLM:")
    for key in ("paradigm", "security risk"):
        groups: dict[str, list[int]] = {}
        for i, row in enumerate(mcptox):
            groups.setdefault(str(row.get(key)), []).append(i)
        for g, idx in sorted(groups.items()):
            a = sum(m[i] >= THRESHOLDS["1%"] for i in idx)
            b = sum(m[i] >= THRESHOLDS["1%"] or rule_hit[i] for i in idx)
            print(f"  {key}: {g:28} {a:4}/{len(idx):<4} {a / len(idx):6.1%}   {b / len(idx):6.1%}")

    if args.benign_npm:
        npm = saved["npm real descriptions (benign)"]
        print("\nHighest-scoring real (benign) descriptions, to inspect false positives:")
        for row in sorted(npm, key=lambda x: -x["shieldlm"])[:10]:
            print(f"  {row['shieldlm']:.4f}  {row['text'][:160]!r}")

    out = ROOT / "results" / "gpu_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"thresholds": THRESHOLDS, "sets": saved}, indent=1), encoding="utf-8")
    print(f"\nraw scores saved to {out}")


if __name__ == "__main__":
    main()
