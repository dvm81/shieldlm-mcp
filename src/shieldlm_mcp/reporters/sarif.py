"""SARIF 2.1.0 output for GitHub code scanning and other SARIF viewers."""

from __future__ import annotations

from pathlib import Path

from .. import __version__
from ..checks import RULES
from ..models import Finding, Severity

DOCS = "https://github.com/dvm81/shieldlm-mcp/blob/main/docs/rules.md"
SECURITY_SEVERITY = {Severity.CRITICAL: "9.5", Severity.HIGH: "8.0", Severity.MEDIUM: "5.5", Severity.LOW: "3.0", Severity.INFO: "1.0"}


def _uri(path: str, base: Path | None) -> str:
    """Repository-relative path when possible (what code scanning needs), else a file URI."""
    p = Path(path)
    base = (base or Path.cwd()).resolve()
    try:
        return p.resolve().relative_to(base).as_posix()
    except ValueError:
        return p.resolve().as_uri()


def to_sarif(findings: list[Finding], *, feed_version: str, detector: str, base: Path | None = None) -> dict:
    rule_ids = list(RULES)
    rules = []
    for rid in rule_ids:
        info = RULES[rid]
        rules.append(
            {
                "id": rid,
                "name": info.name,
                "shortDescription": {"text": info.summary},
                "fullDescription": {"text": info.summary},
                "helpUri": f"{DOCS}#{rid.lower()}",
                "help": {"text": info.summary},
                "defaultConfiguration": {"level": info.severity.sarif_level},
                "properties": {
                    "tags": ["security", *info.owasp, *[c.lower() for c in info.cwe]],
                    "security-severity": SECURITY_SEVERITY[info.severity],
                },
            }
        )
    results = []
    for f in findings:
        region = {"startLine": f.location.line} if f.location.line else None
        physical = {"artifactLocation": {"uri": _uri(f.location.path, base)}}
        if region:
            physical["region"] = region
        logical = f"{f.server}/{f.tool}" if f.tool else f.server
        results.append(
            {
                "ruleId": f.rule_id,
                "ruleIndex": rule_ids.index(f.rule_id),
                "level": f.severity.sarif_level,
                "message": {"text": f"{f.title}. {f.detail} Fix: {f.remediation}"},
                "locations": [
                    {
                        "physicalLocation": physical,
                        "logicalLocations": [{"fullyQualifiedName": logical, "kind": "member" if f.tool else "module"}],
                    }
                ],
                "partialFingerprints": {"shieldlmMcp/v1": f.fingerprint},
                "properties": {
                    "severity": f.severity.value,
                    "security-severity": SECURITY_SEVERITY[f.severity],
                    "evidence": f.evidence,
                    **({"confidence": f.confidence} if f.confidence is not None else {}),
                },
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "shieldlm-mcp",
                        "semanticVersion": __version__,
                        "informationUri": "https://github.com/dvm81/shieldlm-mcp",
                        "rules": rules,
                    }
                },
                "results": results,
                "properties": {"feedVersion": feed_version, "detector": detector},
            }
        ],
    }
