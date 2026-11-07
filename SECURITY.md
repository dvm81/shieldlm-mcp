# Security policy

## Reporting a vulnerability in shieldlm-mcp

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/dvm81/shieldlm-mcp/security/advisories/new)
rather than in a public issue. You should get a reply within five working days.

## Findings in third-party MCP servers

If shieldlm-mcp finds a poisoned description or a malicious package in someone else's
MCP server, report it to that server's maintainer (and the registry, for example npm or
PyPI) before publishing details. Allow a reasonable time for a fix, typically 30 to 90 days.
Indicators added to the bundled feed must cite a public source.

## What the scanner itself guarantees

- It never starts, connects to or executes the servers it scans. It only reads JSON files.
- It makes no network calls, except downloading the ShieldLM model from Hugging Face the
  first time you use `--detector shieldlm`.
- Secrets found in configs are masked in every output format.
