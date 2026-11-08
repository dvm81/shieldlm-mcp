"""C3 (static part): how remote servers are reached. No network probing is done."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit

from ..models import Finding, Inventory, Transport
from ..redact import mask
from . import ScanContext, make_finding

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}
CRED_PARAM = re.compile(r"(token|key|secret|password|passwd|auth|sig|signature|session)", re.I)


def run(inventory: Inventory, ctx: ScanContext) -> list[Finding]:
    findings: list[Finding] = []
    for s in inventory.servers:
        if s.transport == Transport.STDIO or not s.url:
            continue
        try:
            parts = urlsplit(s.url)
            host = (parts.hostname or "").lower()
        except ValueError:  # e.g. http://[SERVER_IP]:8080, a placeholder in brackets
            continue
        if parts.scheme == "http" and host not in LOCAL_HOSTS and not host.endswith(".localhost"):
            findings.append(
                make_finding(
                    "MCP-C3-PLAINTEXT-HTTP",
                    title=f"Server '{s.name}' is reached over unencrypted HTTP",
                    detail="Tool calls, results and any auth headers cross the network in clear text, and a "
                           "network attacker can rewrite tool descriptions in transit.",
                    server=s.name,
                    evidence=f"{parts.scheme}://{parts.netloc}{parts.path}",
                    location=s.source_ref,
                    remediation="Use the server's https:// endpoint.",
                )
            )
        secret_params = [(k, v) for k, v in parse_qsl(parts.query) if CRED_PARAM.search(k) and v]
        if secret_params or parts.password:
            shown = ", ".join(f"{k}={mask(v)}" for k, v in secret_params) or "password in URL"
            findings.append(
                make_finding(
                    "MCP-C3-CREDENTIAL-IN-URL",
                    title=f"Credential in the URL of server '{s.name}'",
                    detail="URLs end up in logs, proxies and browser history.",
                    server=s.name,
                    evidence=shown,
                    location=s.source_ref,
                    remediation="Send the credential in an Authorization header or use the server's OAuth flow, and rotate it.",
                )
            )
    return findings
