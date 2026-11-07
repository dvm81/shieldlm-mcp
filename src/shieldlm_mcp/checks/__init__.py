"""Checks: each takes the Inventory and returns Findings.

Rule ids, OWASP Top 10 for Agentic Applications (2026) ids and CWE ids
are listed in RULES so reporters can describe every rule, not only the ones that fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..detection import EnsembleDetector
from ..feed import Feed
from ..models import Finding, Inventory, Severity


@dataclass
class ScanContext:
    detector: EnsembleDetector
    feed: Feed
    previous_hashes: dict[str, str] = field(default_factory=dict)  # "server/tool" -> description hash


@dataclass(frozen=True)
class RuleInfo:
    id: str
    name: str
    severity: Severity
    summary: str
    owasp: tuple[str, ...]
    cwe: tuple[str, ...]


RULES: dict[str, RuleInfo] = {
    r.id: r
    for r in [
        RuleInfo("MCP-C1-INJECTION", "ToolDescriptionInjection", Severity.HIGH,
                 "Tool metadata contains instructions aimed at the model.", ("ASI01", "ASI04"), ("CWE-1427",)),
        RuleInfo("MCP-C1-HIDDEN", "HiddenContentInToolMetadata", Severity.MEDIUM,
                 "Tool metadata hides text from human reviewers.", ("ASI01",), ("CWE-1427",)),
        RuleInfo("MCP-C1-DRIFT", "ToolDescriptionChanged", Severity.MEDIUM,
                 "A tool description changed since the last approved scan.", ("ASI04",), ("CWE-494",)),
        RuleInfo("MCP-C2-BROAD-FS", "BroadFilesystemAccess", Severity.HIGH,
                 "Server is given the filesystem root or a whole home directory.", ("ASI03",), ("CWE-269",)),
        RuleInfo("MCP-C2-PLAINTEXT-SECRET", "PlaintextSecretInConfig", Severity.HIGH,
                 "A secret is written directly into the config.", ("ASI03",), ("CWE-798",)),
        RuleInfo("MCP-C2-CONTAINER-PRIVILEGE", "PrivilegedContainer", Severity.HIGH,
                 "Container server runs with host-level privileges.", ("ASI03", "ASI05"), ("CWE-250",)),
        RuleInfo("MCP-C2-REMOTE-SCRIPT", "PipedRemoteScript", Severity.CRITICAL,
                 "Server start-up downloads and runs a script.", ("ASI04", "ASI05"), ("CWE-494",)),
        RuleInfo("MCP-C2-ARBITRARY-EXEC", "ArbitraryCommandTool", Severity.MEDIUM,
                 "A tool accepts arbitrary commands or code.", ("ASI02", "ASI05"), ("CWE-78",)),
        RuleInfo("MCP-C3-PLAINTEXT-HTTP", "UnencryptedRemoteServer", Severity.HIGH,
                 "Remote server is reached over unencrypted HTTP.", (), ("CWE-319",)),
        RuleInfo("MCP-C3-CREDENTIAL-IN-URL", "CredentialInUrl", Severity.MEDIUM,
                 "A credential is passed in the server URL.", ("ASI03",), ("CWE-598",)),
        RuleInfo("MCP-C4-MALICIOUS", "KnownMaliciousPackage", Severity.CRITICAL,
                 "Package matches a known-malicious indicator.", ("ASI04",), ("CWE-506",)),
        RuleInfo("MCP-C4-TYPOSQUAT", "PossibleTyposquat", Severity.HIGH,
                 "Package name is one or two edits from a well-known MCP server.", ("ASI04",), ("CWE-829",)),
        RuleInfo("MCP-C4-UNPINNED", "UnpinnedPackage", Severity.LOW,
                 "Package is installed at whatever version is latest when the agent starts.", ("ASI04",), ("CWE-829",)),
    ]
}


def make_finding(rule_id: str, *, severity: Severity | None = None, **kwargs) -> Finding:
    info = RULES[rule_id]
    return Finding(
        rule_id=rule_id,
        severity=severity or info.severity,
        owasp=list(info.owasp),
        cwe=list(info.cwe),
        **kwargs,
    )


def run_all(inventory: Inventory, ctx: ScanContext) -> list[Finding]:
    from . import injection, privilege, registry, transport

    findings: list[Finding] = []
    for module in (injection, privilege, transport, registry):
        findings.extend(module.run(inventory, ctx))
    order = {s: i for i, s in enumerate(["critical", "high", "medium", "low", "info"])}
    findings.sort(key=lambda f: (order[f.severity.value], f.server, f.tool or "", f.rule_id))
    return findings
