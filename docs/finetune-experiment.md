# Fine-tuning ShieldLM on tool descriptions

ShieldLM was trained on prompts. On real tool descriptions it misfires: they are full of instructions by design, and at thresholds that keep false alarms on real descriptions near 1% it catches almost no attacks. This experiment fine-tunes it on tool descriptions in English and eight other languages. The result is published as [dmilush/shieldlm-deberta-mcp-tools](https://huggingface.co/dmilush/shieldlm-deberta-mcp-tools) and used by `--detector shieldlm-tools`. It is experimental; the default detector is still the rules.

## Data and splits

Every source is split by MCP server or npm package (`tests/corpus/eval_split.json`), so no test text shares a server or package with training text. Training sees only `train`, thresholds come from `calib`, runs are compared on `dev`, and results are reported on `test`.

| Split | Poisoned | Benign |
| --- | --- | --- |
| train | MCPTox, 22 servers (239); hand-written tuning set (28) | npm English, 40 packages (1,369); npm non-English, 216 packages (1,400 sampled from 6,806); tuning set (50) |
| calib | | npm English, 9 packages (606); npm non-English, 86 packages (2,824) |
| dev | Independent sets, half of their servers (115) | Same (102) |
| test | MCPTox, other 23 servers (246); independent sets, other half (129); held-out set (20) | npm English, 28 packages (876); npm non-English, 136 packages (4,108); independent sets (84); held-out set (30) |

- **Independent sets:** MCP-SafetyBench, MSB and mcp-tool-poisoning (`scripts/build_external.py`), written without reference to MCPTox and never trained on. Attacks that give the model no instruction (self-promotion, copied descriptions, extra parameters) are reported separately; nothing flags them.
- **Non-English benign descriptions:** strings passed as `description` or `.describe(...)` in npm packages of MCP servers described in other languages (`scripts/collect_npm_descriptions.py`): 68,141 strings from 438 packages, capped at 60 per package. By language after the cap: Korean, Chinese, Russian, Japanese, Spanish, Portuguese, French, a little German and Italian.
- **Translations:** poisoned and benign rows of train, dev and test were machine-translated into Chinese, Japanese, Korean, Russian, German, Spanish, French and Portuguese (`scripts/translate_descriptions.py`): two languages for each poisoned training row, one otherwise. Translating both classes the same way keeps translation artifacts from predicting the label. Calibration rows are not translated. The translator was Qwen3.6-35B-A3B (Apache-2.0) in llama.cpp; 2,397 of 2,410 translations passed the automatic checks (target language, not cut off, code, paths and tool names unchanged).

`scripts/build_tool_dataset.py` puts it all in one file.

## Procedure

- **Training:** from dmilush/shieldlm-deberta-base; poisoned rows repeated 3×; plus a replay sample of ShieldLM's own training data, half injections. 3 epochs, learning rate 2e-5, batch 16, bf16, transformers 5.3.0.
- **Thresholds:** each budget (0.5%, 1%, 2%, 5%) takes the stricter of two thresholds, one that flags that share of the English calib descriptions and one for the non-English ones. Scores are logit margins, so thresholds keep their resolution where probabilities round to 1.
- **Choosing a run:** ten runs, replay 3,000 or 8,000 and seeds 0–4. Runs whose ShieldLM validation recall at 0.1% FPR fell more than 3 points below the base model's were excluded (seven of ten). Among the rest, the run with the highest dev recall at the 1% threshold was chosen: replay 8,000, seed 0. Its test numbers are the result; the spread over seeds is reported too.

## Results

Test split, 1% threshold:

| Test set | Rules | Base model | Chosen model | Chosen + rules |
| --- | --- | --- | --- | --- |
| Independent attacks (129) | 24.8% | 0% | 89.1% | 93.0% |
| Same, translated (128) | 7.8% | 0% | 71.9% | 71.9% |
| MCPTox (246) | 48.0% | 1.2% | 100% | 100% |
| MCPTox, translated (245) | 14.7% | 0.4% | 100% | 100% |
| Held-out set (20) | 55% | 0% | 70% | 80% |
| Held-out set, translated (20) | 20% | 0% | 65% | 65% |

Benign descriptions flagged by the chosen model:

| Test set | at 0.5% | at 1% | at 2% |
| --- | --- | --- | --- |
| npm English (876) | 1.3% | 1.7% | 3.2% |
| npm non-English (4,108) | 1.2% | 1.9% | 3.9% |
| npm English, translated (399) | 2.3% | 4.0% | 8.0% |
| Independent sets (84 / 84 translated) | 0% / 0% | 0% / 1.2% | 0% / 2.4% |

Rules flag none of them.

**By language,** real non-English test descriptions flagged at 1%: Japanese 0.0% (508), Korean 0.7% (992), Russian 1.0% (794), Chinese 1.6% (834), Spanish 3.2% (442), French 4.6% (173), Portuguese 7.3% (328), German 12% (33, two packages). The non-English calibration set is mostly Korean, Chinese and Russian, so Latin-script languages other than English go over budget.

**Seeds.** Over the five runs with replay 8,000, at 1%: independent attacks 71.9% on average (60–89%), 78.1% (67–93%) with rules; translated 70.5% (62–77%); npm English flagged 1.3% (1–2%); npm non-English 1.9%. The chosen run is at the top of the range on test, as it was on dev. Expect a retrained model anywhere in it.

**Prompt injections.** On ShieldLM's test split, recall at 0.1% FPR is 94.8% (base 96.0%) and at 1% FPR 98.2% (base 98.6%). The check on that excluded four of five runs with replay 3,000 and three of five with replay 8,000.

## The first version

An earlier version trained on English data only (commit aacc8b5). It caught 97% of held-out MCPTox servers but only 12% of the independent attacks, which looked like it had learned MCPTox's style. It hadn't: the English npm calib packages include nine Chinese descriptions, all from `@snowluma/mcp`, and the model, which had never seen Chinese, scored all nine as certain injections. They are 1.5% of the calib set, so the 1% threshold sat above them and above most attacks. Calibrated on English text alone, the same model caught 77% of the independent attacks. That model would have flagged every Chinese description, which is why this version adds real and translated text in other languages. The chosen model scores those nine descriptions at a margin of −8.5, far below its 1% threshold of −5.45.

## Limitations

- **Translated attacks are not native attacks.** Non-English recall is measured on machine translations.
- **Latin-script languages other than English** are flagged more often than the budget (French 4.6%, Portuguese 7.3% at 1%).
- **Training attacks are narrow:** one benchmark plus a small hand-written set. Recall on the independent sets is the best guide to new attack styles, and it varies between runs.
- **No instruction, no detection.** Self-promotion, copied descriptions and extra parameters are not flagged.
- **False positives of 1–2%** on real descriptions. Findings from the model alone are reported as `medium`, below the default `--fail-on high`.

## Reproduce

```bash
python scripts/build_external.py --safetybench MCPSafety --msb MSB --mtp mtp.jsonl -o data/external.jsonl
python scripts/collect_npm_descriptions.py -o data/npm_nonen_descriptions.jsonl
python scripts/build_tool_dataset.py -o data/tool_dataset.jsonl
python scripts/translate_descriptions.py data/tool_dataset.jsonl -o data/translations.jsonl --endpoint http://localhost:8080/v1
python scripts/finetune_tools.py --epochs 0 --out results/finetune2/base.json
python scripts/finetune_tools.py --replay 8000 --seed 0 --save-model models/r8000_s0 --out results/finetune2/r8000_s0.json   # and the other runs
python scripts/finetune_tools.py --select results/finetune2/*.json
```

Each training run takes 5–10 minutes on an RTX 3090. The scripts' docstrings say where to get MCPTox, the independent sets and the English npm descriptions.
