"""Keep secrets out of every report and state file."""

from __future__ import annotations

import re

SECRET_KEY = re.compile(r"(key|token|secret|password|passwd|pwd|auth|credential|cookie|session|bearer)", re.IGNORECASE)
REFERENCE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_:\-]*\}?$|^\$\{(env|input):[^}]+\}$")
TOKEN_SHAPES = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[abpr]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30,}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,})"
)

# Values people leave for the reader to fill in: YOUR_API_KEY, <token>, sk-xxxx, replace_me, GITHUB_TOKEN_HERE.
PLACEHOLDER = re.compile(
    r"your|here$|replace|change[-_ ]?me|placeholder|example|dummy|fake|fixture|redacted|<[^>]*>|\{\{|x{4,}|\*{3,}|\.\.\."
    r"|paste|seu_|sua_|substitu|reemplaz|remplac|ersetz|votre",  # the same in Portuguese, Spanish, French, German
    re.I,
)
# Values that point at a secret instead of holding it: templates, secret-manager references, file paths, plain URLs.
INDIRECT = re.compile(r"\$\{|\{\{|\{env:|^env:|^op://|^ENC\[|^vault:|^~?/|^\.\.?/|^[A-Za-z]:[\\/]|^https?://[^/@]*(/|$)")


def is_reference(value: str) -> bool:
    """True when a config value points at a variable instead of holding the secret itself."""
    return bool(REFERENCE.match(value.strip()))


def is_placeholder(name: str, value: str) -> bool:
    """True for a value left to be filled in, such as YOUR_API_KEY or the variable's own name."""
    v = value.strip()
    dummy = len(v) >= 8 and len(set(v.lower())) <= 3  # aaaaaaaa, 11111111-1111-...
    return bool(PLACEHOLDER.search(v)) or v.upper() == name.upper() or dummy


def is_indirect(value: str) -> bool:
    """True when the value names where a secret lives (a template, op:// reference, path or URL) rather than holding it."""
    return bool(INDIRECT.search(value.strip()))


def looks_like_secret(name: str, value: str) -> bool:
    if not value or is_reference(value) or is_placeholder(name, value):
        return False
    if TOKEN_SHAPES.search(value):
        return True
    return bool(SECRET_KEY.search(name)) and len(value) >= 8 and not is_indirect(value)


def mask(value: str) -> str:
    v = value.strip()
    if len(v) <= 8:
        return "****"
    return v[:4] + "…" + f"({len(v)} chars)"


def scrub(text: str) -> str:
    """Mask token-shaped strings inside free text."""
    return TOKEN_SHAPES.sub(lambda m: mask(m.group(0)), text)
