from __future__ import annotations

from .base import DetectionResult, Detector
from .heuristic import HeuristicDetector


class EnsembleDetector:
    """Heuristic rules plus an optional ML detector: flag if either fires."""

    def __init__(self, heuristic: HeuristicDetector, ml: Detector | None = None):
        self.heuristic = heuristic
        self.ml = ml
        self.name = "heuristic+shieldlm" if ml else "heuristic"

    def score_batch(self, texts: list[str]) -> list[DetectionResult]:
        h = self.heuristic.score_batch(texts)
        if not self.ml:
            return h
        m = self.ml.score_batch(texts)
        merged = []
        for a, b in zip(h, m, strict=True):
            injection = a.label == "injection" or b.label == "injection"
            merged.append(
                DetectionResult(
                    label="injection" if injection else "benign",
                    probability=max(a.probability, b.probability),
                    backend=self.name,
                    signals=a.signals + b.signals,
                    sensitive_target=a.sensitive_target,
                )
            )
        return merged

    def score(self, text: str) -> DetectionResult:
        return self.score_batch([text])[0]


def build_detector(kind: str = "heuristic", **kwargs) -> EnsembleDetector:
    heuristic = HeuristicDetector()
    if kind == "heuristic":
        return EnsembleDetector(heuristic)
    if kind in ("shieldlm", "shieldlm-tools"):
        from .shieldlm import TOOLS_MODEL, ShieldLMDetector

        if kind == "shieldlm-tools":
            kwargs.setdefault("model", TOOLS_MODEL)
        return EnsembleDetector(heuristic, ShieldLMDetector(**kwargs))
    raise ValueError(f"unknown detector: {kind}")


__all__ = ["DetectionResult", "Detector", "EnsembleDetector", "HeuristicDetector", "build_detector"]
