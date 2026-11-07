# Fine-tuning ShieldLM on tool descriptions

ShieldLM was trained on prompts. On real tool descriptions it misfires: at the threshold where it flags 1% of real descriptions, it catches only 10% of MCPTox. This experiment fine-tunes it on tool descriptions and measures what that buys. The shipped detector still uses the original model.

Reproduce on a GPU machine with `scripts/finetune_tools.py` (needs the `[ml]` extra plus `pandas` and `pyarrow`, MCPTox in `data/mcptox_pure_tool.json`, and the npm descriptions in `data/npm_benign_descriptions.jsonl`). Each run takes about 3 minutes on an RTX 3090.

## Setup

Data is split by server and package in `tests/corpus/eval_split.json`, so nothing in a test set shares a server or package with training.

| | Poisoned | Benign |
| --- | --- | --- |
| Training | MCPTox, 22 dev servers (239), repeated 3×; tuning corpus (28) | npm, 40 train packages (1,369); tuning corpus (50) |
| Replay, to keep prompt-injection skills | 1,500 from ShieldLM's training data | 1,500 from ShieldLM's training data |
| Threshold calibration | | npm, 9 calib packages (606) |
| Test | MCPTox, 23 test servers (246); held-out corpus (20) | npm, 28 test packages (876); held-out corpus (30) |

Training starts from `dmilush/shieldlm-deberta-base`: 3 epochs, learning rate 2e-5, batch 16, bf16, transformers 5.3.0. Every configuration ran with seeds 0, 1 and 2; tables show the mean and, in brackets, the range. Thresholds are set on logit margins so they keep their resolution where probabilities round to 0.9999.

## Results

At the threshold that flags 1% of the calibration descriptions:

| Model | MCPTox test | npm test flagged | Held-out corpus caught | With rules |
| --- | --- | --- | --- | --- |
| Base ShieldLM | 10.2% | 0.2% | 10% | 60% |
| Fine-tuned | 97.2% (96–98) | 0.0% | 32% (30–35) | 62% |

At 2%, the fine-tuned model catches 99.5% of MCPTox test and, with the rules, 85% of the held-out corpus, while flagging 1.3% of npm test descriptions and 1 of 30 held-out benign ones.

**Unseen attack styles.** MCPTox has three templates, and the test servers share them with training, so 97% overstates recall on attacks written differently. Leaving one template out of training entirely and testing on it, at the 1% threshold:

| Template left out | Recall on that template | Base model on it |
| --- | --- | --- |
| Template-1 | 56% (23–74) | 10% |
| Template-2 | 89% (79–100) | 7% |
| Template-3 | 72% (53–89) | 12% |

**Prompt injections.** On ShieldLM's own test split, recall at 1% FPR barely moves (98.6% → 98.2%), but at 0.1% FPR it drops from 96.0% to 93.0%. One run without Template-3 fell to 19% there, so the replay mix is too small to make this reliable.

## Reading

- Fine-tuning fixes the false-positive problem: real descriptions no longer score as injections, so a strict threshold becomes usable.
- It learns MCPTox's attacks very well and transfers in part to unseen templates, with large variation between seeds.
- On the held-out corpus, written independently of MCPTox, it adds almost nothing to the rules at 1% (60% → 62%). Most of the gain is specific to how MCPTox phrases attacks.

Before shipping a model trained this way, it needs poisoned training data that does not all come from one benchmark, a larger independent test set, and more replay to keep strict-threshold recall on prompt injections.
