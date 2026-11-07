from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    label: Literal["benign", "injection"]
    probability: float
    backend: str
    signals: list[str] = Field(default_factory=list)  # rule ids that fired (heuristic) or "shieldlm"
    sensitive_target: bool = False  # instruction points at credentials or secrets


class Detector(Protocol):
    name: str

    def score(self, text: str) -> DetectionResult: ...

    def score_batch(self, texts: list[str]) -> list[DetectionResult]: ...
