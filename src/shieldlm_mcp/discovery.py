"""Locate and parse MCP client configs and tool snapshots into an Inventory.

Static only: nothing here starts, connects to or executes an MCP server.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .models import Inventory, PackageRef, Server, SourceRef, Tool, Transport


class DiscoveryError(Exception):
    pass


def default_paths(cwd: Path | None = None) -> list[Path]:
    """Config files that exist in the usual places for this project and user."""
    cwd = cwd or Path.cwd()
    home = Path.home()
    candidates = [
        cwd / ".mcp.json",
        cwd / ".cursor" / "mcp.json",
        cwd / ".vscode" / "mcp.json",
        home / ".cursor" / "mcp.json",
    ]
    if sys.platform == "darwin":
        candidates.append(home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
    elif sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
    else:
        candidates.append(home / ".config" / "Claude" / "claude_desktop_config.json")
    return [p for p in candidates if p.is_file()]


def _line_of(text: str, pattern: str, start: int = 0) -> int | None:
    m = re.compile(pattern).search(text, start)
    return text.count("\n", 0, m.start()) + 1 if m else None


def _key_line(text: str, key: str) -> int | None:
    return _line_of(text, r'"' + re.escape(key) + r'"\s*:')


def resolve_package(command: str | None, args: list[str]) -> PackageRef | None:
    """Best-effort package identity from how a stdio server is launched."""
    if not command:
        return None
    exe = Path(command).name.lower()
    rest = list(args)

    def first_positional(items: list[str], skip_values: set[str] = frozenset()) -> str | None:
        it = iter(items)
        for a in it:
            if a in skip_values:
                next(it, None)
                continue
            if a.startswith("-"):
                continue
            return a
        return None

    if exe in {"npx", "bunx"} or (exe in {"pnpm", "yarn"} and rest[:1] == ["dlx"]):
        if exe in {"pnpm", "yarn"}:
            rest = rest[1:]
        spec = first_positional(rest, {"-p", "--package"})
        if not spec:
            return None
        if spec.startswith("@"):
            name, _, ver = spec[1:].partition("@")
            name = "@" + name
        else:
            name, _, ver = spec.partition("@")
        return PackageRef(ecosystem="npm", name=name, version=ver or None)

    if exe in {"uvx", "pipx"} or (exe == "uv" and rest[:2] == ["tool", "run"]):
        if exe == "pipx" and rest[:1] == ["run"]:
            rest = rest[1:]
        if exe == "uv":
            rest = rest[2:]
        if "--from" in rest:
            spec = rest[rest.index("--from") + 1] if rest.index("--from") + 1 < len(rest) else None
        else:
            spec = first_positional(rest, {"--python", "--with"})
        if not spec:
            return None
        m = re.match(r"^([A-Za-z0-9_.\-\[\]]+?)(?:(==|@)([^\s]+))?$", spec)
        if not m:
            return None
        return PackageRef(ecosystem="pypi", name=m.group(1), version=m.group(3))

    if exe in {"docker", "podman"} and rest[:1] == ["run"]:
        image = first_positional(
            rest[1:],
            {"-e", "--env", "-v", "--volume", "--name", "--network", "-p", "--publish", "--user", "-u", "--mount",
             "--env-file", "-w", "--workdir", "--entrypoint"},
        )
        if not image:
            return None
        if "@sha256:" in image:
            name, _, digest = image.partition("@")
            return PackageRef(ecosystem="docker", name=name, version="sha256:" + digest.split(":", 1)[1])
        last = image.rsplit("/", 1)[-1]
        if ":" in last:
            name, _, tag = image.rpartition(":")
            return PackageRef(ecosystem="docker", name=name, version=tag)
        return PackageRef(ecosystem="docker", name=image, version=None)
    return None


def _servers_map(data: dict) -> dict | None:
    if isinstance(data.get("mcpServers"), dict):
        return data["mcpServers"]
    if isinstance(data.get("servers"), dict):
        return data["servers"]
    mcp = data.get("mcp")
    if isinstance(mcp, dict) and isinstance(mcp.get("servers"), dict):
        return mcp["servers"]
    return None


def _as_dict(value) -> dict:
    """env and headers as a mapping; real configs also write them as ["KEY=VALUE", ...]."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return dict(str(item).split("=", 1) for item in value if "=" in str(item))
    return {}


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(a) for a in value]
    return [value] if isinstance(value, str) else []


def _parse_server(name: str, spec: dict, path: str, text: str) -> Server:
    ref = SourceRef(path=path, line=_key_line(text, name), pointer=f"/{name}")
    kind = str(spec.get("type") or spec.get("transport") or "").lower()
    url = spec.get("url") or spec.get("serverUrl")
    if url and kind not in {"stdio"}:
        transport = Transport.SSE if kind == "sse" or str(url).rstrip("/").endswith("/sse") else Transport.HTTP
        return Server(
            name=name,
            transport=transport,
            url=str(url),
            headers={str(k): str(v) for k, v in _as_dict(spec.get("headers")).items()},
            source_ref=ref,
        )
    command = spec.get("command")
    args = _as_list(spec.get("args"))
    if isinstance(command, list):  # ["python", "server.py"] instead of command + args
        command, args = (str(command[0]) if command else None), [str(a) for a in command[1:]] + args
    elif not isinstance(command, str):
        command = None
    return Server(
        name=name,
        transport=Transport.STDIO,
        command=command or None,
        args=args,
        env={str(k): str(v) for k, v in _as_dict(spec.get("env")).items()},
        package=resolve_package(command, args),
        source_ref=ref,
    )


def _parse_snapshot(data: dict, path: Path, text: str) -> tuple[str, str | None, list[Tool]]:
    """A tool snapshot: {"server": name, "tools": [...]} or a raw tools/list response."""
    server = data.get("server")
    tools_raw = data.get("tools")
    if tools_raw is None and isinstance(data.get("result"), dict):
        tools_raw = data["result"].get("tools")
    if not server:
        server = path.name.split(".")[0]
    tools: list[Tool] = []
    for i, t in enumerate(tools_raw if isinstance(tools_raw, list) else []):
        if not isinstance(t, dict) or "name" not in t:
            continue
        tname = str(t["name"])
        tools.append(
            Tool(
                name=tname,
                description=t.get("description"),
                input_schema=t.get("inputSchema") or t.get("input_schema"),
                annotations=t.get("annotations"),
                source_ref=SourceRef(
                    path=str(path),
                    line=_line_of(text, r'"name"\s*:\s*"' + re.escape(tname) + r'"'),
                    pointer=f"/tools/{i}",
                ),
            )
        )
    return str(server), data.get("instructions"), tools


def discover(paths: list[Path]) -> Inventory:
    servers: dict[str, Server] = {}
    snapshots: list[tuple[str, str | None, list[Tool], Path]] = []
    used: list[str] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, UnicodeDecodeError) as e:
            raise DiscoveryError(f"cannot read {p}: {e}") from e
        except json.JSONDecodeError as e:
            raise DiscoveryError(f"{p} is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise DiscoveryError(f"{p}: expected a JSON object")
        used.append(str(p))
        smap = _servers_map(data)
        if smap is not None:
            for name, spec in smap.items():
                if isinstance(spec, dict):
                    srv = _parse_server(str(name), spec, str(p), text)
                    servers[f"{p}::{name}"] = srv
            continue
        if "tools" in data or "result" in data:
            snapshots.append((*_parse_snapshot(data, p, text), p))
            continue
        raise DiscoveryError(f"{p}: no mcpServers/servers map and no tools list found")

    by_name: dict[str, list[Server]] = {}
    for srv in servers.values():
        by_name.setdefault(srv.name, []).append(srv)
    for name, instructions, tools, p in snapshots:
        targets = by_name.get(name)
        if not targets:
            # Snapshot for a server not in any scanned config: keep it as its own entry.
            srv = Server(name=name, transport=Transport.STDIO, source_ref=SourceRef(path=str(p), line=1))
            servers[f"{p}::{name}"] = srv
            by_name[name] = [srv]
            targets = by_name[name]
        for srv in targets:
            srv.tools.extend(tools)
            if instructions:
                srv.instructions = instructions
    return Inventory(servers=list(servers.values()), source_paths=used)
