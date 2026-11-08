"""C4: known-malicious packages, likely typosquats of well-known servers, and unpinned installs."""

from __future__ import annotations

from ..models import Finding, Inventory, PackageRef
from . import ScanContext, make_finding

HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "_": "-", ".": "-"})


def damerau_levenshtein(a: str, b: str, limit: int = 3) -> int:
    """Optimal string alignment distance, stopping early above `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def _norm(name: str) -> str:
    return name.lower().replace("rn", "m").translate(HOMOGLYPHS)


def _closest_known(pkg: PackageRef, known: list[dict]) -> tuple[str, int] | None:
    same_eco = [k["name"] for k in known if k.get("ecosystem") == pkg.ecosystem]
    if pkg.name.lower() in {n.lower() for n in same_eco}:
        return None
    p_scope = pkg.name.split("/", 1)[0].lower() if pkg.name.startswith("@") and "/" in pkg.name else None
    best: tuple[str, int] | None = None
    for name in same_eco:
        if len(name) < 8:
            continue
        # Only a scope's owner can publish in it, so a near-name in the same scope is not a squat.
        if p_scope and name.lower().startswith(p_scope + "/"):
            continue
        d = min(damerau_levenshtein(pkg.name.lower(), name.lower()), damerau_levenshtein(_norm(pkg.name), _norm(name)))
        if d <= 2 and (best is None or d < best[1]):
            best = (name, d)
        # Same package under a look-alike scope, e.g. @model-context-protocol/server-filesystem.
        if "/" in name and "/" in pkg.name:
            k_scope, k_base = name.split("/", 1)
            p_scope, p_base = pkg.name.split("/", 1)
            if k_base.lower() == p_base.lower() and k_scope.lower() != p_scope.lower() \
                    and damerau_levenshtein(_norm(k_scope.replace("-", "")), _norm(p_scope.replace("-", ""))) <= 2:
                best = (name, 1)
    return best


def run(inventory: Inventory, ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    for s in inventory.servers:
        pkg = s.package
        if not pkg:
            continue
        label = f"{pkg.ecosystem}:{pkg.name}" + (f"@{pkg.version}" if pkg.version else "")
        bad = next(
            (m for m in ctx.feed.malicious if m.get("ecosystem") == pkg.ecosystem and m.get("name", "").lower() == pkg.name.lower()),
            None,
        )
        if bad:
            findings.append(
                make_finding(
                    "MCP-C4-MALICIOUS",
                    title=f"Server '{s.name}' installs known-malicious package {pkg.name}",
                    detail=f"{bad.get('summary', '')} Indicator {bad.get('id')} (feed {ctx.feed.version}). "
                           f"References: {', '.join(bad.get('references', []))}",
                    server=s.name,
                    evidence=label,
                    location=s.source_ref,
                    remediation="Remove the server now, rotate every credential it could reach, and check outgoing mail and logs.",
                )
            )
            continue
        near = _closest_known(pkg, ctx.feed.known_good)
        if near:
            findings.append(
                make_finding(
                    "MCP-C4-TYPOSQUAT",
                    title=f"Package {pkg.name} looks like a misspelling of {near[0]}",
                    detail=f"The name is {near[1]} edit(s) away from the well-known server {near[0]}. "
                           "Typosquats copy a real server and add one malicious line.",
                    server=s.name,
                    evidence=label,
                    location=s.source_ref,
                    remediation=f"Check the publisher; if you meant {near[0]}, fix the name.",
                )
            )
        if pkg.ecosystem in {"npm", "pypi"} and (not pkg.version or pkg.version.lower() == "latest"):
            findings.append(
                make_finding(
                    "MCP-C4-UNPINNED",
                    title=f"Package {pkg.name} is not pinned to a version",
                    detail="The agent installs whatever is newest each time it starts, so a compromised release "
                           "reaches you without any review.",
                    server=s.name,
                    evidence=label,
                    location=s.source_ref,
                    remediation=f"Pin a reviewed version, for example {pkg.name}@1.2.3 (npm) or {pkg.name}==1.2.3 (PyPI).",
                )
            )
        elif pkg.ecosystem == "docker" and (not pkg.version or pkg.version == "latest"):
            findings.append(
                make_finding(
                    "MCP-C4-UNPINNED",
                    title=f"Image {pkg.name} is not pinned",
                    detail="The ':latest' tag (or no tag) changes under you; a compromised push reaches you on the next start.",
                    server=s.name,
                    evidence=label,
                    location=s.source_ref,
                    remediation="Pin the image by digest (image@sha256:...).",
                )
            )
    return findings
