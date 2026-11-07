"""C2: servers and tools that get more access than their job needs."""

from __future__ import annotations

import re

from ..models import Finding, Inventory, Transport
from ..redact import looks_like_secret, mask
from . import ScanContext, make_finding

BROAD_ROOTS = re.compile(
    r"^(/|~|~/|\$HOME/?|\$\{HOME\}/?|%USERPROFILE%\\?|[A-Za-z]:[\\/]?|/Users/[^/]+/?|/home/[^/]+/?|/root/?)$"
)
SHELLS = {"bash", "sh", "zsh", "dash", "fish", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh"}
REMOTE_PIPE = re.compile(r"\b(curl|wget|iwr|invoke-webrequest|irm|invoke-restmethod)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|iex|python3?|node)\b", re.I)
EXEC_TOOL_NAME = re.compile(r"(^|[_\-.])(exec|execute|shell|run_command|run_shell|terminal|bash|powershell|eval|run_code|run_script|system)($|[_\-.])", re.I)
EXEC_PARAM = {"command", "cmd", "script", "shell_command", "code", "bash_command"}


def _docker_issues(args: list[str]) -> list[str]:
    issues = []
    joined = " ".join(args)
    if "--privileged" in args:
        issues.append("--privileged")
    for flag in ("--network", "--net", "--pid", "--ipc"):
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args) and args[i + 1] == "host":
                issues.append(f"{flag} host")
            elif a == f"{flag}=host":
                issues.append(f"{flag}=host")
    mounts = []
    for i, a in enumerate(args):
        if a in {"-v", "--volume", "--mount"} and i + 1 < len(args):
            mounts.append(args[i + 1])
        elif a.startswith(("--volume=", "--mount=")):
            mounts.append(a.split("=", 1)[1])
    for m in mounts:
        if m.startswith("/:") or "source=/," in m or m.rstrip(",") == "source=/":
            issues.append("host root mounted")
        if "/var/run/docker.sock" in m:
            issues.append("Docker socket mounted")
    if re.search(r"--cap-add[\s=](SYS_ADMIN|ALL)\b", joined):
        issues.append("--cap-add " + re.search(r"--cap-add[\s=](\w+)", joined).group(1))
    return issues


def run(inventory: Inventory, ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    for s in inventory.servers:
        if s.transport == Transport.STDIO and s.command:
            exe = s.command.replace("\\", "/").rsplit("/", 1)[-1].lower()
            is_container = exe in {"docker", "podman"}
            # Broad filesystem roots passed as arguments (filesystem servers take roots this way).
            for a in s.args:
                value = a.split("=", 1)[1] if a.startswith("-") and "=" in a else a
                if BROAD_ROOTS.match(value.strip()) and not (is_container and value.strip() in {"/"}):
                    findings.append(
                        make_finding(
                            "MCP-C2-BROAD-FS",
                            title=f"Server '{s.name}' can reach '{value}'",
                            detail="Passing the filesystem root or a whole home directory gives the server, "
                                   "and any agent steered by injected text, access to SSH keys, cloud credentials "
                                   "and browser data.",
                            server=s.name,
                            evidence=f"{s.command} {' '.join(s.args)}",
                            location=s.source_ref,
                            remediation="Pass only the project directories the server needs.",
                        )
                    )
                    break
            if is_container:
                issues = _docker_issues(s.args)
                if issues:
                    findings.append(
                        make_finding(
                            "MCP-C2-CONTAINER-PRIVILEGE",
                            title=f"Container for server '{s.name}' runs with host privileges",
                            detail="Flags found: " + ", ".join(issues) + ". These remove most of the isolation the container gives.",
                            server=s.name,
                            evidence=f"{s.command} {' '.join(s.args)}",
                            location=s.source_ref,
                            remediation="Drop privileged flags and host mounts; mount only the directories the server needs, read-only where possible.",
                        )
                    )
            if exe in SHELLS:
                script = " ".join(s.args)
                if REMOTE_PIPE.search(script):
                    findings.append(
                        make_finding(
                            "MCP-C2-REMOTE-SCRIPT",
                            title=f"Server '{s.name}' downloads and runs a script at start-up",
                            detail="Whatever the URL serves on the day the agent starts is executed with your user's rights.",
                            server=s.name,
                            evidence=script[:220],
                            location=s.source_ref,
                            remediation="Install a pinned, reviewed package instead of piping a download into a shell.",
                        )
                    )
        # Secrets written into env or headers.
        for kind, mapping in (("env", s.env), ("header", s.headers)):
            for k, v in mapping.items():
                value = v[7:] if v.lower().startswith("bearer ") else v
                if looks_like_secret(k, value):
                    findings.append(
                        make_finding(
                            "MCP-C2-PLAINTEXT-SECRET",
                            title=f"Plaintext secret in {kind} '{k}' of server '{s.name}'",
                            detail="Config files get committed, synced and read by other tools, and an injected "
                                   "agent with file access can read them.",
                            server=s.name,
                            evidence=f"{k}={mask(v)}",
                            location=s.source_ref,
                            remediation="Reference an environment variable or the client's secret input "
                                        "(for example ${env:NAME} or ${input:name}) and rotate this secret.",
                        )
                    )
        # Tools that take arbitrary commands or code.
        for t in s.tools:
            props = set(((t.input_schema or {}).get("properties") or {}).keys())
            hits = props & EXEC_PARAM
            if EXEC_TOOL_NAME.search(t.name) or hits:
                what = f"parameter '{sorted(hits)[0]}'" if hits else "its name"
                findings.append(
                    make_finding(
                        "MCP-C2-ARBITRARY-EXEC",
                        title=f"Tool '{t.name}' can run arbitrary commands",
                        detail=f"Judging by {what}, this tool executes commands or code. One injected instruction "
                               "anywhere in the agent's context can then become code execution.",
                        server=s.name,
                        tool=t.name,
                        evidence=t.name + (f" ({', '.join(sorted(hits))})" if hits else ""),
                        location=t.source_ref,
                        remediation="Require human approval for this tool, or run the server in a sandbox with no network and no secrets.",
                    )
                )
    return findings
