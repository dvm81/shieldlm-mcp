"""C1: adversarial instructions and hidden content in tool metadata, plus description drift."""

from __future__ import annotations

from ..detection.heuristic import INVISIBLE, HeuristicDetector, hidden_content
from ..models import Finding, Inventory, Server, Severity, SourceRef, Tool
from ..redact import scrub
from . import ScanContext, make_finding

HIDDEN_SIGNALS = {"invisible_unicode", "html_comment", "encoded_blob"}


def evidence(text: str, limit: int = 220) -> str:
    shown = INVISIBLE.sub(lambda m: f"<U+{ord(m.group(0)):04X}>", text)
    shown = " ".join(shown.split())
    shown = scrub(shown)
    return shown if len(shown) <= limit else shown[: limit - 1] + "…"


def _schema_descriptions(schema: dict | None, prefix: str = "") -> list[tuple[str, str]]:
    """Every description string inside a JSON Schema, with its path."""
    out: list[tuple[str, str]] = []
    if not isinstance(schema, dict):
        return out
    if isinstance(schema.get("description"), str) and prefix:
        out.append((prefix, schema["description"]))
    for key in ("properties", "$defs", "definitions"):
        for name, sub in (schema.get(key) or {}).items():
            out.extend(_schema_descriptions(sub, f"{prefix}.{name}" if prefix else name))
    if isinstance(schema.get("items"), dict):
        out.extend(_schema_descriptions(schema["items"], prefix + "[]"))
    return out


def _texts(server: Server) -> list[tuple[Tool | None, str, str, SourceRef]]:
    items: list[tuple[Tool | None, str, str, SourceRef]] = []
    if server.instructions:
        items.append((None, "server instructions", server.instructions, server.source_ref))
    for tool in server.tools:
        if tool.description:
            items.append((tool, "description", tool.description, tool.source_ref))
        title = (tool.annotations or {}).get("title")
        if isinstance(title, str) and title:
            items.append((tool, "annotation title", title, tool.source_ref))
        for path, desc in _schema_descriptions(tool.input_schema):
            items.append((tool, f"parameter '{path}' description", desc, tool.source_ref))
    return items


def run(inventory: Inventory, ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    for server in inventory.servers:
        items = _texts(server)
        if items:
            results = ctx.detector.score_batch([t for _, _, t, _ in items])
            for (tool, field, text, ref), res in zip(items, results, strict=True):
                hidden = hidden_content(text)
                attack_signals = [s for s in res.signals if s not in HIDDEN_SIGNALS]
                if res.label == "injection" and attack_signals:
                    if res.sensitive_target:
                        sev = Severity.CRITICAL
                    elif res.probability >= 0.8:
                        sev = Severity.HIGH
                    else:
                        sev = Severity.MEDIUM
                    if hidden and sev != Severity.CRITICAL:
                        sev = Severity.HIGH if sev == Severity.MEDIUM else Severity.CRITICAL
                    reasons = "; ".join(dict.fromkeys(HeuristicDetector.explain(s) for s in res.signals))
                    where = f"tool '{tool.name}'" if tool else f"server '{server.name}'"
                    findings.append(
                        make_finding(
                            "MCP-C1-INJECTION",
                            severity=sev,
                            title=f"Instructions aimed at the model in the {field} of {where}",
                            detail=f"The {field} {reasons}. An agent reads tool metadata as trusted context, "
                                   f"so these instructions can redirect it.",
                            server=server.name,
                            tool=tool.name if tool else None,
                            evidence=evidence(text),
                            location=ref,
                            confidence=res.probability,
                            remediation="Remove this server or tool, or pin a reviewed version. Report the "
                                        "description to the server's maintainer.",
                        )
                    )
                elif hidden:
                    findings.append(
                        make_finding(
                            "MCP-C1-HIDDEN",
                            title=f"Hidden content in the {field} of "
                                  + (f"tool '{tool.name}'" if tool else f"server '{server.name}'"),
                            detail="The text " + "; ".join(HeuristicDetector.explain(s) for s in hidden)
                                   + ". The model sees it; a reviewer reading the description usually does not.",
                            server=server.name,
                            tool=tool.name if tool else None,
                            evidence=evidence(text),
                            location=ref,
                            remediation="Inspect the raw description bytes and remove hidden content before approving.",
                        )
                    )
        for tool in server.tools:
            key = f"{server.name}/{tool.name}"
            old = ctx.previous_hashes.get(key)
            if old and old != tool.description_hash:
                findings.append(
                    make_finding(
                        "MCP-C1-DRIFT",
                        title=f"Description of tool '{tool.name}' changed since the last approved scan",
                        detail="A tool that changes its description after approval can turn malicious later "
                               "(a 'rug pull'). Review the new text before the agent uses it.",
                        server=server.name,
                        tool=tool.name,
                        evidence=evidence(tool.description or ""),
                        location=tool.source_ref,
                        remediation="Review the new description, then re-approve with `shieldlm-mcp scan --update-state`.",
                    )
                )
    return findings
