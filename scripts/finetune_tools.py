"""Fine-tune ShieldLM on tool descriptions in English and eight other languages (GPU machine).

Reads data/tool_dataset.jsonl (scripts/build_tool_dataset.py) and its translations
(scripts/translate_descriptions.py). Trains on the `train` split plus a sample of ShieldLM's
own training data (replay, so it keeps detecting prompt injections), sets thresholds on
`calib`, and scores `dev` and `test`. A threshold "at 1%" flags at most 1% of the English and at
most 1% of the non-English calib descriptions.

    python scripts/finetune_tools.py --seed 0 --save-model models/run --out results/finetune2/run.json
    python scripts/finetune_tools.py --epochs 0 --out results/finetune2/base.json   # the base model, same protocol
    python scripts/finetune_tools.py --select results/finetune2/*.json               # choose a run on dev, print test

Runs are compared on dev only. Among runs that keep ShieldLM's validation recall at 0.1% FPR
within 3 points of the base model, the chosen run has the highest recall on the dev attacks at
the 1% threshold. Its test numbers are the result.

Scores are logit margins (logit[injection] - logit[benign]), not probabilities, so thresholds
keep their resolution where softmax saturates near 1. A margin m is probability 1 / (1 + e^-m).

The first version of this experiment (English only, no dev split) is at commit aacc8b5.
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
SELECT_FPR = "1.0%"


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


def group_names(r: dict) -> list[str]:
    """Report groups a row belongs to, e.g. 'test | MCPTox | translated'."""
    if r["split"] == "train":
        return []
    if r["split"] == "calib":
        return ["calib | English" if r["lang"] == "en" else "calib | non-English"]
    if r["set"] == "external" and r["label"] == 1 and not r.get("instructs", True):
        return [f"{r['split']} | external, no instruction"]
    name = {"mcptox": "MCPTox", "holdout": "held-out corpus", "external": "external", "npm": "npm", "npm-nonen": "npm non-English"}[r["set"]]
    kind = "attacks" if r["label"] else "benign"
    variant = "translated" if r["translated"] else ("original" if r["lang"] == "en" or r["set"] != "npm-nonen" else "real")
    out = [f"{r['split']} | {name} {kind} | {variant}", f"{r['split']} | all {kind} | all"]
    if r["translated"] or r["set"] == "npm-nonen":
        out.append(f"{r['split']} | {kind} by language | {r['lang']}")
    return out


def run(args) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from shieldlm_mcp.detection.heuristic import HeuristicDetector
    from shieldlm_mcp.detection.shieldlm import SELF_CHECK_TEXTS, self_check

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_jsonl(args.dataset) + [r for r in load_jsonl(args.translations) if r["ok"]]

    # Training data.
    train_rows = [r for r in rows if r["split"] == "train" and r["set"] != "npm-nonen"]
    nonen = [r for r in rows if r["split"] == "train" and r["set"] == "npm-nonen"]
    train_rows += random.Random(args.seed).sample(nonen, min(args.nonen_train, len(nonen)))
    poisoned = [r["text"] for r in train_rows if r["label"] == 1]
    benign = [r["text"] for r in train_rows if r["label"] == 0]
    texts = poisoned * args.oversample + benign
    labels = [1] * (len(poisoned) * args.oversample) + [0] * len(benign)
    if args.replay:
        df = shieldlm_split("train")
        for label in (0, 1):
            sample = df[df["label_binary"] == label].sample(n=args.replay // 2, random_state=args.seed)
            texts += sample["text"].tolist()
            labels += [label] * len(sample)
    print(f"train: {len(poisoned)} poisoned x{args.oversample}, {len(benign)} benign, {args.replay} replay -> {len(texts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    self_check([1 / (1 + math.exp(-x)) for x in margins(model, tok, list(SELF_CHECK_TEXTS), device)], transformers.__version__)
    if args.epochs:
        train(model, tok, texts, labels, args, device)
    if args.save_model:
        model.save_pretrained(args.save_model)
        tok.save_pretrained(args.save_model)

    # Evaluation.
    evals = [r for r in rows if r["split"] != "train"]
    scores = margins(model, tok, [r["text"] for r in evals], device)
    rule_hits = [r.label == "injection" and any(s not in HIDDEN for s in r.signals) for r in HeuristicDetector().score_batch([r["text"] for r in evals])]
    members = defaultdict(list)
    for i, r in enumerate(evals):
        for g in group_names(r):
            members[g].append(i)
    calib = {k: [scores[i] for i in members[f"calib | {k}"]] for k in ("English", "non-English")}
    out: dict = {"config": {k: str(v) for k, v in vars(args).items()}, "transformers": transformers.__version__, "thresholds": {}, "groups": {}}
    for fpr in FPR_TARGETS:
        # The stricter of the two, so neither English nor other languages get more than their share of flags.
        t = max(threshold_at(calib["English"], fpr), threshold_at(calib["non-English"], fpr))
        out["thresholds"][f"{fpr:.1%}"] = {"margin": t, "probability": 1 / (1 + math.exp(-t)) if math.isfinite(t) else 0.0}
        for g, idx in members.items():
            entry = out["groups"].setdefault(g, {"n": len(idx), "rules": sum(rule_hits[i] for i in idx) / len(idx)})
            entry[f"{fpr:.1%}"] = sum(scores[i] >= t for i in idx) / len(idx)
            entry[f"{fpr:.1%}+rules"] = sum(scores[i] >= t or rule_hits[i] for i in idx) / len(idx)

    # Did it forget prompt injections? Validation for choosing runs, test for reporting.
    for name in ("validation", "test"):
        df = shieldlm_split(name)
        s = margins(model, tok, df["text"].tolist(), device)
        pos = [x for x, y in zip(s, df["label_binary"], strict=True) if y == 1]
        neg = [x for x, y in zip(s, df["label_binary"], strict=True) if y == 0]
        out[f"shieldlm_{name}"] = {f"tpr_at_{f:.1%}_fpr": rate(pos, threshold_at(neg, f)) for f in (0.001, 0.01)}
    return out


def config_key(r: dict) -> str:
    c = r["config"]
    return "base model" if c["epochs"] == "0" else f"replay {c['replay']}, oversample {c['oversample']}, {c['epochs']} epochs"


def select(paths: list[Path]) -> None:
    runs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    base = next((r for _, r in runs if r["config"]["epochs"] == "0"), None)
    floor = base["shieldlm_validation"]["tpr_at_0.1%_fpr"] - 0.03 if base else 0.0

    def dev_recall(r: dict) -> float:
        return r["groups"]["dev | all attacks | all"][SELECT_FPR]

    print(f"dev: recall on dev attacks at {SELECT_FPR} of calib flagged; ShieldLM validation recall at 0.1% FPR must be >= {floor:.1%}\n")
    print(f"  {'run':34} {'dev recall':>10} {'dev benign':>10} {'SL val':>7}")
    for p, r in sorted(runs, key=lambda pr: -dev_recall(pr[1])):
        val = r["shieldlm_validation"]["tpr_at_0.1%_fpr"]
        print(f"  {p.stem:34} {dev_recall(r):>10.1%} {r['groups']['dev | all benign | all'][SELECT_FPR]:>10.1%} {val:>7.1%}{'' if val >= floor else '  (forgets)'}")
    eligible = [(p, r) for p, r in runs if r["config"]["epochs"] != "0" and r["shieldlm_validation"]["tpr_at_0.1%_fpr"] >= floor]
    best_path, best = max(eligible, key=lambda pr: dev_recall(pr[1]))
    print(f"\nchosen: {best_path.stem}  (model: {best['config']['save_model']})")

    by_config = defaultdict(list)
    for _, r in runs:
        by_config[config_key(r)].append(r)

    def cell(vals: list[float]) -> str:
        if len(vals) == 1:
            return f"{vals[0]:.1%}"
        return f"{statistics.mean(vals):.1%} ({min(vals):.0%}-{max(vals):.0%})"

    groups = sorted(g for g in best["groups"] if g.startswith("test"))
    for fpr in (f"{f:.1%}" for f in FPR_TARGETS):
        print(f"\ntest, threshold flags {fpr} of calib  (columns: rules | model | model+rules)")
        print(f"  {'group':48} {'n':>5} {'rules':>7} | {'chosen run':>18} | " + " | ".join(f"{k:>30}" for k in sorted(by_config)))
        for g in groups:
            e = best["groups"][g]
            cols = [f"{cell([x['groups'][g][fpr] for x in rs])} / {cell([x['groups'][g][fpr + '+rules'] for x in rs])}" for _, rs in sorted(by_config.items())]
            print(f"  {g:48} {e['n']:>5} {e['rules']:>7.1%} | {e[fpr]:>7.1%} / {e[fpr + '+rules']:>7.1%} | " + " | ".join(f"{c:>30}" for c in cols))
    print("\nShieldLM test split, recall at 0.1% / 1% FPR")
    print(f"  chosen run   {best['shieldlm_test']['tpr_at_0.1%_fpr']:.1%} / {best['shieldlm_test']['tpr_at_1.0%_fpr']:.1%}")
    for k, rs in sorted(by_config.items()):
        print(f"  {k:34} {cell([x['shieldlm_test']['tpr_at_0.1%_fpr'] for x in rs])} / {cell([x['shieldlm_test']['tpr_at_1.0%_fpr'] for x in rs])}")
    print("\nthresholds of the chosen run:", json.dumps(best["thresholds"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "tool_dataset.jsonl")
    ap.add_argument("--translations", type=Path, default=ROOT / "data" / "translations.jsonl")
    ap.add_argument("--model", default="dmilush/shieldlm-deberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--oversample", type=int, default=3, help="Repeat poisoned tool descriptions this many times")
    ap.add_argument("--replay", type=int, default=3000, help="ShieldLM training rows to mix in, half of each label")
    ap.add_argument("--nonen-train", type=int, default=1400, help="Non-English npm training rows to sample, about as many as the English ones")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--select", type=Path, nargs="+", help="Choose a run on dev from earlier --out files, print its test results, and exit")
    args = ap.parse_args()
    if args.select:
        select(args.select)
        return
    result = run(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for g, e in sorted(result["groups"].items()):
        print(f"  {g:48} n={e['n']:>5}  rules {e['rules']:.1%}  @1%: {e['1.0%']:.1%}  +rules {e['1.0%+rules']:.1%}")


if __name__ == "__main__":
    main()
