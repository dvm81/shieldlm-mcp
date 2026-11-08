# Fine-tuning ShieldLM on tool descriptions

ShieldLM was trained on prompts. On real tool descriptions it misfires: at the threshold where it flags 1% of real descriptions, it catches only 10% of MCPTox. This experiment fine-tunes it on tool descriptions and measures what that buys. The shipped detector still uses the original model.

Reproduce on a GPU machine with `scripts/finetune_tools.py` (needs the `[ml]` extra plus `pandas` and `pyarrow`, MCPTox in `data/mcptox_pure_tool.json`, and the npm descriptions in `data/npm_benign_descriptions.jsonl`). Each run takes about 3 minutes on an RTX 3090. The independent test sets below are built with `scripts/build_external.py` and scored with `scripts/eval_external.py`.

## Setup

Data is split by server and package in `tests/corpus/eval_split.json`, so nothing in a test set shares a server or package with training.

| | Poisoned | Benign |
| --- | --- | --- |
| Training | MCPTox, 22 dev servers (239), repeated 3×; tuning corpus (28) | npm, 40 train packages (1,369); tuning corpus (50) |
| Replay, to keep prompt-injection skills | 1,500 from ShieldLM's training data | 1,500 from ShieldLM's training data |
| Threshold calibration | | npm, 9 calib packages (606) |
| Test | MCPTox, 23 test servers (246); held-out corpus (20) | npm, 28 test packages (876); held-out corpus (30) |
| Independent test, never used for rules or training | 244 from three other public sets (below) | 186 from the same sets |

Training starts from `dmilush/shieldlm-deberta-base`: 3 epochs, learning rate 2e-5, batch 16, bf16, transformers 5.3.0. Every configuration ran with seeds 0, 1 and 2; tables show the mean and, in brackets, the range. Thresholds are set on logit margins so they keep their resolution where probabilities round to 0.9999.

## Results on MCPTox

At the threshold that flags 1% of the calibration descriptions:

| Model | MCPTox test | npm test flagged | Held-out corpus caught | With rules |
| --- | --- | --- | --- | --- |
| Base ShieldLM | 10.2% | 0.2% | 10% | 60% |
| Fine-tuned | 97.2% (96–98) | 0.0% | 32% (30–35) | 62% |

These thresholds turned out to be set by nine Chinese descriptions; see [Non-English text](#non-english-text). Calibrated on English text only, the fine-tuned models catch 99.6–100% of MCPTox test and, with the rules, 85% of the held-out corpus.

**Unseen attack styles.** MCPTox has three templates, and the test servers share them with training. Leaving one template out of training entirely and testing on it, at the 1% threshold:

| Template left out | Recall on that template | Base model on it |
| --- | --- | --- |
| Template-1 | 56% (23–74) | 10% |
| Template-2 | 89% (79–100) | 7% |
| Template-3 | 72% (53–89) | 12% |

**Prompt injections.** On ShieldLM's own test split, recall at 1% FPR barely moves (98.6% → 98.2%), but at 0.1% FPR it drops from 96.0% to 93.0%. One run without Template-3 fell to 19% there, so the replay mix is too small to make this reliable.

## Independent test sets

Three public sets written without reference to MCPTox:

| Set | Poisoned, with instructions | Benign |
| --- | --- | --- |
| [MCP-SafetyBench](https://github.com/xjzzzzzzzz/MCPSafety) (ICLR 2026): attacks written per task, in 7 categories such as command injection, redirection to another tool, and shadowing another tool | 114 | 155, from its servers |
| [MSB](https://github.com/dongsenzhang/MSB) (ICLR 2026): one template that impersonates the user, with 5 goals, on 21 tools | 105 | 21 |
| [mcp-tool-poisoning](https://huggingface.co/datasets/fevziegeyurtsevenler/mcp-tool-poisoning): 25 short pairs | 25 | 10 |

They also hold 70 attacks that give the model no instruction: a tool advertised as "the best tool in the world", a copy of another tool's description, or an extra parameter asking for the model's name. No detector flags any of them, and they are left out of the numbers below. Catching them needs checks on names and schemas, not on text.

The seed 1 and 2 models here were retrained to save them. Training is not bit-for-bit reproducible on GPU: the retrained seed 1 caught 90% of MCPTox test at the original 1% threshold, against 96% the first time.

Thresholds calibrated on the English npm calib descriptions (`--latin-calib`):

| Threshold | Model | External attacks | With rules | External benign flagged | npm test flagged | Held-out corpus, with rules |
| --- | --- | --- | --- | --- | --- | --- |
| – | Rules alone | 29.9% | | 0% | 0% | 55% |
| 1% | Base ShieldLM | 0.4% | 30.3% | 0% | 0.1% | 60% |
| 0.5% | Fine-tuned | 63% (56–69) | 75% (69–80) | 1.2% (0.5–2.7) | 1.0% (0.8–1.3) | 80% (75–85) |
| 1% | Fine-tuned | 77% (60–92) | 84% (72–94) | 2.5% (0.5–4.3) | 2.0% (1.7–2.6) | 85% |
| 2% | Fine-tuned | 92% (86–96) | 94% (91–96) | 4.5% (3.8–5.4) | 3.0% (2.5–3.3) | 87% (85–90) |

By set, at the 1% threshold with rules: MCP-SafetyBench 42% → 82% (68–94), MSB 0% → 82% (70–93), mcp-tool-poisoning 100% → 100%. The base model adds less than one point at any of these thresholds, because real descriptions already fill its top scores.

Rules alone catch 55% of the held-out corpus here, not the README's 60%, because one of its attacks is caught only through hidden content, which this count leaves out.

## Non-English text

The npm calib packages include nine Chinese descriptions, all from `@snowluma/mcp`; the train and test packages have none. All three fine-tuned models rank all nine among their ten highest-scoring calib descriptions, as near-certain injections. Nine of 606 is 1.5%, so thresholds at 0.5% and 1% sit above them and above most attacks. With the original calibration, the fine-tuned models catch only 12% (9–15) of the external attacks at 1%, and 36% (34–40) with rules.

The fine-tuned models therefore treat Chinese tool descriptions as attacks. Training had no non-English benign descriptions, and ShieldLM's own data is mostly English.

## Reading

- Fine-tuning makes a strict threshold usable on English descriptions, and it carries over to attacks written independently of MCPTox. At a threshold that flags about 2% of real English descriptions, the fine-tuned model with the rules catches 84% of the external attacks, against 30% for the rules alone.
- An earlier version of this page concluded that fine-tuning added almost nothing on the held-out corpus (60% → 62%). That came from thresholds set by the Chinese descriptions; with English calibration the held-out corpus goes from 60% to 85%.
- The models read non-English text as injection. A detector that flags every Chinese description cannot ship.
- Seeds differ widely (60–92% at 1%), and the same seed does not retrain to the same model.

Before shipping a model trained this way, it needs non-English benign descriptions in training and test, a way to choose between runs on a dev set rather than a test set, more replay to keep strict-threshold recall on prompt injections, and calibration on a larger set of real descriptions.
