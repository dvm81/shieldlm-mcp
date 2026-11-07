"""The bundled indicator feed: known-malicious packages and a known-good registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Feed:
    version: str
    malicious: list[dict]
    known_good: list[dict]


def load_feed(path: Path | None = None) -> Feed:
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        data = json.loads(resources.files(__package__).joinpath("seed.json").read_text(encoding="utf-8"))
    return Feed(
        version=str(data.get("feed_version", "unknown")),
        malicious=list(data.get("malicious", [])),
        known_good=list(data.get("known_good", [])),
    )
