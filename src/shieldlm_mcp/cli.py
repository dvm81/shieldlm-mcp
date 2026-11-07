"""Command-line interface: `shieldlm-mcp scan | discover | rules`."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .checks import RULES, ScanContext, run_all
from .detection import build_detector
from .discovery import DiscoveryError, default_paths, discover
from .feed import load_feed
from .models import Inventory, Severity
from .redact import scrub
from .reporters import build_policy, print_table, to_json
from .reporters.sarif import to_sarif

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Static security scanner for MCP server configurations.")
err = Console(stderr=True)


class Format(str, Enum):
    table = "table"
    json = "json"
    sarif = "sarif"


class DetectorKind(str, Enum):
    heuristic = "heuristic"
    shieldlm = "shieldlm"


def _version(value: bool) -> None:
    if value:
        typer.echo(f"shieldlm-mcp {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool | None, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")] = None,
) -> None:
    """Never starts, connects to or executes the servers it scans."""


def _inventory(paths: list[Path]) -> Inventory:
    files: list[Path] = []
    for p in paths or default_paths():
        if p.is_dir():
            files.extend(q for q in (p / ".mcp.json", p / ".cursor" / "mcp.json", p / ".vscode" / "mcp.json") if q.is_file())
            files.extend(sorted(p.glob("*.tools.json")))
        elif p.is_file():
            files.append(p)
        else:
            raise DiscoveryError(f"not found: {p}")
    if not files:
        raise DiscoveryError("no MCP config found; pass a path, for example: shieldlm-mcp scan .mcp.json")
    return discover(files)


def _load_state(path: Path) -> dict[str, str]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")).get("descriptions", {})
    return {}


def _load_suppressions(values: list[str], allowlist: Path | None) -> set[str]:
    out = set(values)
    if allowlist:
        text = allowlist.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            out.update(data if isinstance(data, list) else data.get("suppress", []))
        except json.JSONDecodeError:
            out.update(line.split("#", 1)[0].strip() for line in text.splitlines() if line.split("#", 1)[0].strip())
    return out


@app.command()
def scan(
    paths: Annotated[list[Path] | None, typer.Argument(help="Config files, *.tools.json snapshots or directories. Default: the usual client config locations.")] = None,
    fmt: Annotated[Format, typer.Option("--format", "-f", help="Output format.")] = Format.table,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write the report to this file.")] = None,
    fail_on: Annotated[Severity, typer.Option("--fail-on", help="Exit 1 if any finding is at or above this severity.")] = Severity.HIGH,
    detector: Annotated[DetectorKind, typer.Option("--detector", help="heuristic (no dependencies) or shieldlm (local ML model, needs the [ml] extra).")] = DetectorKind.heuristic,
    fpr: Annotated[float, typer.Option("--fpr", help="ShieldLM false-positive budget: 0.001, 0.005, 0.01 or 0.05.")] = 0.001,
    suppress: Annotated[list[str] | None, typer.Option("--suppress", help="Fingerprint of a reviewed finding to ignore (repeatable).")] = None,
    allowlist: Annotated[Path | None, typer.Option("--allowlist", help="File of reviewed fingerprints to ignore.")] = None,
    state: Annotated[Path, typer.Option("--state", help="Approved tool-description hashes, for drift detection.")] = Path(".shieldlm-mcp/state.json"),
    update_state: Annotated[bool, typer.Option("--update-state", help="Record current tool descriptions as approved.")] = False,
    policy: Annotated[Path | None, typer.Option("--policy", help="Also write a gateway allowlist of servers and tools that passed.")] = None,
    feed: Annotated[Path | None, typer.Option("--feed", help="Indicator feed JSON (default: the bundled seed).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detail, evidence and fixes in the table output.")] = False,
) -> None:
    """Scan MCP configs and tool snapshots and report findings."""
    try:
        inventory = _inventory(paths or [])
        fd = load_feed(feed)
        kwargs = {"fpr_target": fpr} if detector == DetectorKind.shieldlm else {}
        det = build_detector(detector.value, **kwargs)
        ctx = ScanContext(detector=det, feed=fd, previous_hashes=_load_state(state))
        findings = run_all(inventory, ctx)
    except (DiscoveryError, RuntimeError, ValueError, OSError) as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2) from e

    skip = _load_suppressions(suppress or [], allowlist)
    findings = [f for f in findings if f.fingerprint not in skip]

    if fmt == Format.table:
        console = Console(file=out.open("w", encoding="utf-8") if out else sys.stdout, width=140 if out else None)
        print_table(findings, inventory, console, verbose=verbose)
    else:
        text = (
            json.dumps(to_sarif(findings, feed_version=fd.version, detector=det.name), indent=2)
            if fmt == Format.sarif
            else to_json(findings, inventory, feed_version=fd.version, detector=det.name)
        )
        if out:
            out.write_text(text + "\n", encoding="utf-8")
        else:
            typer.echo(text)

    if policy:
        policy.write_text(json.dumps(build_policy(inventory, findings, fail_on), indent=2) + "\n", encoding="utf-8")
    if update_state:
        state.parent.mkdir(parents=True, exist_ok=True)
        hashes = {f"{s.name}/{t.name}": t.description_hash for s in inventory.servers for t in s.tools}
        state.write_text(json.dumps({"descriptions": hashes}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if any(f.severity.rank >= fail_on.rank for f in findings):
        raise typer.Exit(1)


@app.command("discover")
def discover_cmd(
    paths: Annotated[list[Path] | None, typer.Argument(help="Config files, snapshots or directories.")] = None,
) -> None:
    """Print the normalised inventory of servers and tools, without running checks."""
    try:
        inventory = _inventory(paths or [])
    except DiscoveryError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2) from e
    data = inventory.model_dump(mode="json")
    for s in data["servers"]:
        s["env"] = {k: "<redacted>" for k in s["env"]}
        s["headers"] = {k: "<redacted>" for k in s["headers"]}
        if s.get("url") and "?" in s["url"]:
            s["url"] = s["url"].split("?", 1)[0] + "?<redacted>"
        s["args"] = [scrub(a) for a in s["args"]]
    typer.echo(json.dumps(data, indent=2))


@app.command("rules")
def rules_cmd() -> None:
    """List every rule with its default severity, OWASP Agentic and CWE ids."""
    for r in RULES.values():
        typer.echo(f"{r.id:28} {r.severity.value:9} {','.join(r.owasp) or '-':12} {','.join(r.cwe):10} {r.summary}")


if __name__ == "__main__":  # pragma: no cover
    app()
