#!/usr/bin/env bash
# Run the ShieldLM evaluation on a GPU machine. Reuses an existing torch install if there is one.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -c "import torch" 2>/dev/null || pip install --quiet torch
pip install --quiet -e ".[ml]"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python scripts/eval_gpu.py --benign-npm data/npm_benign_descriptions.jsonl 2>&1 | tee gpu_eval.txt
echo; echo "Done. Paste the contents of gpu_eval.txt back into the chat."
