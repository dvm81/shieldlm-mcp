"""Fine-tune ShieldLM on tool descriptions and evaluate on held-out servers and packages (GPU machine).

Starts from dmilush/shieldlm-deberta-base and trains on:

- poisoned: MCPTox descriptions from the dev servers, plus this repo's tuning corpus
- benign: real npm descriptions from the train packages, plus this repo's tuning corpus
- replay: a balanced sample of ShieldLM's own training data, so it keeps detecting prompt injections

Thresholds are calibrated on the npm calib packages. Results are reported on the MCPTox test
servers, the npm test packages, this repo's held-out corpus and ShieldLM's own test split.
The split is tests/corpus/eval_split.json.

    python scripts/finetune_tools.py --mcptox data/mcptox_pure_tool.json --npm data/npm_benign_descriptions.jsonl
    python scripts/finetune_tools.py ... --exclude-template Template-2   # attack style never seen in training
    python scripts/finetune_tools.py ... --epochs 0                       # the base model, same protocol
    python scripts/finetune_tools.py --summarize results/finetune/*.json

Scores are logit margins (logit[injection] - logit[benign]), not probabilities, so thresholds
keep their resolution where softmax saturates near 1. A margin m is probability 1 / (1 + e^-m).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "tests" / "corpus" / "eval_split.json"
SHIELDLM_DATA = "dmilush/shieldlm-prompt-injection"
HIDDEN = {"invisible_unicode", "html_comment", "encoded_blob"}
FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shieldlm_split(name: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    return pd.read_parquet(hf_hub_download(SHIELDLM_DATA, f"data/{name}-00000-of-00001.parquet", repo_type="dataset"))


def threshold_at(benign: list[float], fpr: float) -> float:
    """Smallest margin that flags at most `fpr` of these benign scores."""
    s = sorted(benign, reverse=True)
    k = int(fpr * len(s))
    return s[k] + 1e-6 if k < len(s) else -math.inf


def rate(scores: list[float], t: float) -> float:
    return sum(x >= t for x in scores) / len(scores) if scores else float("nan")


def margins(model, tok, texts: list[str], device: str, batch_size: int = 64, max_length: int = 512) -> list[float]:
    import torch

    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [0.0] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tok([texts[i] or "" for i in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(**enc).logits.float()
        for i, m in zip(idx, (logits[:, 1] - logits[:, 0]).tolist(), strict=True):
            out[i] = m
    return out


def train(model, tok, texts: list[str], labels: list[int], args, device: str) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup

    rng = random.Random(args.seed)
    order = list(range(len(texts)))
    steps = args.epochs * math.ceil(len(order) / args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for epoch in range(args.epochs):
        rng.shuffle(order)
        total, t0 = 0.0, time.time()
        for s in range(0, len(order), args.batch_size):
            idx = order[s : s + args.batch_size]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            y = torch.tensor([labels[i] for i in idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(**enc).logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(order):.4f}  {time.time() - t0:.0f}s", flush=True)


def run(args) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from shieldlm_mcp.detection.heuristic import HeuristicDetector
    from shieldlm_mcp.detection.shieldlm import SELF_CHECK_TEXTS, self_check

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tox = [v for group in json.loads(args.mcptox.read_text(encoding="utf-8")) for v in group.values()]
    npm = load_jsonl(args.npm)
    corpus = {name: load_jsonl(ROOT / "tests" / "corpus" / f"{name}.jsonl") for name in ("descriptions", "holdout")}

    def tox_rows(part: str) -> list[dict]:
        return [r for r in tox if r["server_name"] in set(split["mcptox"][part])]

    def npm_texts(part: str) -> list[str]:
        return [r["text"] for r in npm if r["package"] in set(split["npm"][part])]

    # Training data.
    poisoned = [r["tool_content"] for r in tox_rows("dev") if r["paradigm"] != args.exclude_template]
    poisoned += [r["text"] for r in corpus["descriptions"] if r["label"] == "poisoned"]
    benign = npm_texts("train") + [r["text"] for r in corpus["descriptions"] if r["label"] == "benign"]
    texts = poisoned * args.oversample + benign
    labels = [1] * (len(poisoned) * args.oversample) + [0] * len(benign)
    if args.replay:
        df = shieldlm_split("train")
        for label in (0, 1):
            rows = df[df["label_binary"] == label].sample(n=args.replay // 2, random_state=args.seed)
            texts += rows["text"].tolist()
            labels += [label] * len(rows)
    print(f"train: {len(poisoned)} poisoned x{args.oversample}, {len(benign)} benign, {args.replay} replay -> {len(texts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    m = margins(model, tok, list(SELF_CHECK_TEXTS), device)
    self_check([1 / (1 + math.exp(-x)) for x in m], transformers.__version__)
    if args.epochs:
        train(model, tok, texts, labels, args, device)
    if args.save_model:
        model.save_pretrained(args.save_model)
        tok.save_pretrained(args.save_model)

    # Evaluation.
    rules = HeuristicDetector()
    test_tox = tox_rows("test")
    sets = {
        "calib_benign": (npm_texts("calib"), None),
        "mcptox_test": ([r["tool_content"] for r in test_tox], [r["paradigm"] for r in test_tox]),
        "npm_test": (npm_texts("test"), None),
        "holdout_poisoned": ([r["text"] for r in corpus["holdout"] if r["label"] == "poisoned"], None),
        "holdout_benign": ([r["text"] for r in corpus["holdout"] if r["label"] == "benign"], None),
    }
    scores = {k: margins(model, tok, t, device) for k, (t, _) in sets.items()}
    rule_hits = {
        k: [r.label == "injection" and any(s not in HIDDEN for s in r.signals) for r in rules.score_batch(t)]
        for k, (t, _) in sets.items()
    }
    out: dict = {"config": {k: str(v) for k, v in vars(args).items()}, "transformers": transformers.__version__, "tools": {}}
    for fpr in FPR_TARGETS:
        t = threshold_at(scores["calib_benign"], fpr)
        row = {"threshold_margin": t}
        for k in ("mcptox_test", "npm_test", "holdout_poisoned", "holdout_benign"):
            row[k] = rate(scores[k], t)
            row[k + "+rules"] = sum((x >= t) or h for x, h in zip(scores[k], rule_hits[k], strict=True)) / len(scores[k])
        by_template = defaultdict(list)
        for x, p in zip(scores["mcptox_test"], sets["mcptox_test"][1], strict=True):
            by_template[p].append(x)
        row["mcptox_test_by_template"] = {p: rate(v, t) for p, v in sorted(by_template.items())}
        out["tools"][f"{fpr:.1%}"] = row

    # Did it forget prompt injections? TPR at 0.1% and 1% FPR on ShieldLM's own test split.
    df = shieldlm_split("test")
    s = margins(model, tok, df["text"].tolist(), device)
    pos = [x for x, y in zip(s, df["label_binary"], strict=True) if y == 1]
    neg = [x for x, y in zip(s, df["label_binary"], strict=True) if y == 0]
    out["shieldlm_test"] = {f"tpr_at_{f:.1%}_fpr": rate(pos, threshold_at(neg, f)) for f in (0.001, 0.01)}
    return out


def summarize(paths: list[Path]) -> None:
    runs = defaultdict(list)
    for p in paths:
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["config"]
        key = "base model" if c["epochs"] == "0" else ("all templates" if c["exclude_template"] == "None" else f"without {c['exclude_template']}")
        runs[key].append(r)

    def cell(vals: list[float]) -> str:
        if len(vals) == 1:
            return f"{vals[0]:6.1%}"
        return f"{statistics.mean(vals):6.1%} ({min(vals):.0%}-{max(vals):.0%})"

    for fpr in (f"{f:.1%}" for f in FPR_TARGETS):
        print(f"\nthreshold: {fpr} of npm calib descriptions flagged")
        print(f"  {'run':24} {'n':>2} {'MCPTox test':>18} {'+rules':>18} {'npm test FP':>18} {'holdout P':>18} {'holdout FP':>18} {'unseen template':>18}")
        for key, rs in sorted(runs.items(), key=lambda kv: (kv[0] != "base model", kv[0])):
            g = [r["tools"][fpr] for r in rs]
            tpl = key.removeprefix("without ") if key.startswith("without") else None
            unseen = cell([x["mcptox_test_by_template"][tpl] for x in g]) if tpl else ""
            print(f"  {key:24} {len(rs):>2} {cell([x['mcptox_test'] for x in g]):>18} {cell([x['mcptox_test+rules'] for x in g]):>18} "
                  f"{cell([x['npm_test'] for x in g]):>18} {cell([x['holdout_poisoned'] for x in g]):>18} "
                  f"{cell([x['holdout_benign'] for x in g]):>18} {unseen:>18}")
    base = runs.get("base model", [])
    if base:
        print("\nbase model, recall per template on MCPTox test:")
        for fpr in (f"{f:.1%}" for f in FPR_TARGETS):
            print(f"  {fpr}: {base[0]['tools'][fpr]['mcptox_test_by_template']}")
    print("\nShieldLM's own test split (did it forget prompt injections?):")
    for key, rs in sorted(runs.items(), key=lambda kv: (kv[0] != "base model", kv[0])):
        print(f"  {key:24} TPR@0.1%FPR {cell([r['shieldlm_test']['tpr_at_0.1%_fpr'] for r in rs]):>18}   "
              f"TPR@1%FPR {cell([r['shieldlm_test']['tpr_at_1.0%_fpr'] for r in rs]):>18}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcptox", type=Path, default=ROOT / "data" / "mcptox_pure_tool.json")
    ap.add_argument("--npm", type=Path, default=ROOT / "data" / "npm_benign_descriptions.jsonl")
    ap.add_argument("--model", default="dmilush/shieldlm-deberta-base")
    ap.add_argument("--exclude-template", help="Leave this MCPTox paradigm out of training, e.g. Template-2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--oversample", type=int, default=3, help="Repeat poisoned tool descriptions this many times")
    ap.add_argument("--replay", type=int, default=3000, help="ShieldLM training rows to mix in, half of each label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summarize", type=Path, nargs="+", help="Print a table from earlier --out files and exit")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    result = run(args)
    print(json.dumps({"tools": result["tools"], "shieldlm_test": result["shieldlm_test"]}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
