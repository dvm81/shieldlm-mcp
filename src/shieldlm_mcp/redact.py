"""Keep secrets out of every report and state file."""

from __future__ import annotations

import re

SECRET_KEY = re.compile(r"(key|token|secret|password|passwd|pwd|auth|credential|cookie|session|bearer)", re.IGNORECASE)
REFERENCE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_:\-]*\}?$|^\$\{(env|input):[^}]+\}$")
TOKEN_SHAPES = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[abpr]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30,}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,})"
)


def is_reference(value: str) -> bool:
    """True when a config value points at a variable instead of holding the secret itself."""
    return bool(REFERENCE.match(value.strip()))


def looks_like_secret(name: str, value: str) -> bool:
    if not value or is_reference(value):
        return False
    return bool(TOKEN_SHAPES.search(value)) or (bool(SECRET_KEY.search(name)) and len(value) >= 8)


def mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 8:
        return "****"
    return v[:4] + "…" + f"({len(v)} chars)"


def scrub(text: str) -> str:
    """Mask token-shaped strings inside free text."""
    return TOKEN_SHAPES.sub(lambda m: mask(m.group(0)), text)
