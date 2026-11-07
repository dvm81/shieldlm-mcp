"""ShieldLM backend: the dmilush/shieldlm-deberta-base classifier, run locally.

Optional: install with `pip install "shieldlm-mcp[ml]"`. Nothing is sent anywhere;
the model is downloaded once from Hugging Face and runs on this machine.
"""

from __future__ import annotations

from .base import DetectionResult

DEFAULT_MODEL = "dmilush/shieldlm-deberta-base"

# Thresholds calibrated on the ShieldLM validation split (see the model card).
# Tool descriptions differ from that data, so treat these as starting points.
THRESHOLDS = {0.001: 0.9998, 0.005: 0.9695, 0.01: 0.1239, 0.05: 0.0024}


class ShieldLMDetector:
    name = "shieldlm"

    def __init__(self, model: str = DEFAULT_MODEL, fpr_target: float = 0.001, threshold: float | None = None,
                 device: str | None = None, batch_size: int = 16):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - exercised only without the extra installed
            raise RuntimeError('The ShieldLM backend needs: pip install "shieldlm-mcp[ml]"') from e
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        if threshold is None:
            if fpr_target not in THRESHOLDS:
                raise ValueError(f"fpr_target must be one of {sorted(THRESHOLDS)}, or pass threshold=")
            threshold = THRESHOLDS[fpr_target]
        self.threshold = threshold
        self.batch_size = batch_size

    def score_batch(self, texts: list[str]) -> list[DetectionResult]:
        out: list[DetectionResult] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t or "" for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].tolist()
            for p in probs:
                out.append(
                    DetectionResult(
                        label="injection" if p >= self.threshold else "benign",
                        probability=round(float(p), 4),
                        backend=self.name,
                        signals=["shieldlm"] if p >= self.threshold else [],
                    )
                )
        return out

    def score(self, text: str) -> DetectionResult:
        return self.score_batch([text])[0]
