"""Reporters: terminal table, JSON, SARIF and a gateway allowlist policy."""

from __future__ import annotations

import json
from collections import Counter

from rich.console import Console
from rich.table import Table

from ..models import Finding, Inventory, Severity

STYLE = {"critical": "bold red", "high": "red", "medium": "yellow", "low": "cyan", "info": "dim"}


def to_json(findings: list[Finding], inventory: Inventory, *, feed_version: str, detector: str) -> str:
    return json.dumps(
        {
            "tool": "shieldlm-mcp",
            "feed_version": feed_version,
            "detector": detector,
            "scanned_at": inventory.scanned_at.isoformat(),
            "sources": inventory.source_paths,
            "servers": [s.name for s in inventory.servers],
            "findings": [{**f.model_dump(mode="json"), "fingerprint": f.fingerprint} for f in findings],
        },
        indent=2,
    )


def print_table(findings: list[Finding], inventory: Inventory, console: Console, *, verbose: bool = False) -> None:
    n_tools = sum(len(s.tools) for s in inventory.servers)
    console.print(
        f"[bold]shieldlm-mcp[/bold]  {len(inventory.servers)} server(s), {n_tools} tool(s) from "
        f"{len(inventory.source_paths)} file(s)"
    )
    if not findings:
        console.print("[green]No findings.[/green]")
        return
    table = Table(show_lines=False, header_style="bold")
    table.add_column("Severity")
    table.add_column("Rule")
    table.add_column("Server / tool")
    table.add_column("Finding", overflow="fold")
    for f in findings:
        where = f"{f.server}/{f.tool}" if f.tool else f.server
        table.add_row(f"[{STYLE[f.severity.value]}]{f.severity.value.upper()}[/]", f.rule_id, where, f.title)
    console.print(table)
    if verbose:
        for f in findings:
            console.print(f"\n[{STYLE[f.severity.value]}]{f.rule_id}[/] {f.title}")
            console.print(f"  {f.detail}")
            if f.evidence:
                console.print(f"  [dim]evidence:[/dim] {f.evidence}", markup=False, highlight=False)
            console.print(f"  [dim]fix:[/dim] {f.remediation}")
            console.print(f"  [dim]location:[/dim] {f.location.path}:{f.location.line or '?'}  "
                          f"[dim]fingerprint:[/dim] {f.fingerprint}")
    counts = Counter(f.severity.value for f in findings)
    summary = ", ".join(f"{counts[s.value]} {s.value}" for s in Severity if counts[s.value])
    console.print(f"\n{len(findings)} finding(s): {summary}")


def build_policy(inventory: Inventory, findings: list[Finding], fail_on: Severity) -> dict:
    """Allowlist of servers and tools with no finding at or above `fail_on`."""
    blocked_servers = {f.server for f in findings if f.tool is None and f.severity.rank >= fail_on.rank}
    blocked_tools = {(f.server, f.tool) for f in findings if f.tool and f.severity.rank >= fail_on.rank}
    allowed = []
    for s in inventory.servers:
        if s.name in blocked_servers:
            continue
        entry: dict = {"name": s.name, "transport": s.transport.value}
        if s.package:
            entry["package"] = s.package.model_dump(exclude_none=True)
        if s.url:
            entry["url"] = s.url.split("?", 1)[0]
        if s.tools:
            entry["tools"] = [
                {"name": t.name, "description_sha256": t.description_hash}
                for t in s.tools
                if (s.name, t.name) not in blocked_tools
            ]
        allowed.append(entry)
    return {
        "policy_version": 1,
        "generated_by": "shieldlm-mcp",
        "blocked_at_or_above": fail_on.value,
        "allow": allowed,
        "blocked_servers": sorted(blocked_servers),
        "blocked_tools": sorted(f"{a}/{b}" for a, b in blocked_tools),
    }
