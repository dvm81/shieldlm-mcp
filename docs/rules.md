# Rules

Every rule shipped in shieldlm-mcp, with its default severity, the [OWASP Top 10 for Agentic Applications (2026)](https://genai.owasp.org/) category and the CWE it maps to. List them from the CLI with `shieldlm-mcp rules`.

## MCP-C1-INJECTION

**ToolDescriptionInjection** · default severity **high** · OWASP ASI01, ASI04 · CWE-1427

Free text the model reads as trusted (tool descriptions, parameter descriptions, annotation titles, server instructions) contains orders aimed at the model: ignore earlier instructions, hide actions from the user, read credential files, send data to a URL or email address, move secrets into a tool argument, take priority over the user's request, rewrite the user's arguments, or call another tool first under a made-up threat.

Critical when the text targets credentials or secrets; high when the rules are confident; medium otherwise, and always medium when only ShieldLM fired. Hidden content raises the level by one.

**Fix:** Remove the server or tool, or pin a reviewed version. Report the description to the maintainer.

## MCP-C1-HIDDEN

**HiddenContentInToolMetadata** · default severity **medium** · OWASP ASI01 · CWE-1427

The text contains invisible Unicode (zero-width, bidirectional overrides, tag characters), an HTML comment or a long encoded blob. The model reads these; a person reviewing the description usually does not.

**Fix:** Inspect the raw bytes and remove the hidden part before approving.

## MCP-C1-DRIFT

**ToolDescriptionChanged** · default severity **medium** · OWASP ASI04 · CWE-494

A tool's description hash differs from the one recorded with `--update-state`. A server that changes its descriptions after approval can turn malicious later (a rug pull).

**Fix:** Review the new text, then record it again with `--update-state`.

## MCP-C2-BROAD-FS

**BroadFilesystemAccess** · default severity **high** · OWASP ASI03 · CWE-269

An argument is the filesystem root, a home directory or a drive root. Filesystem servers take their allowed roots this way, so an injected agent could read SSH keys, cloud credentials and browser data.

**Fix:** Pass only the project directories the server needs.

## MCP-C2-PLAINTEXT-SECRET

**PlaintextSecretInConfig** · default severity **high** · OWASP ASI03 · CWE-798

An `env` value or header holds a literal secret instead of a reference such as `${env:NAME}` or `${input:name}`. Values are masked in every report.

**Fix:** Move the secret to an environment variable or the client's secret input, and rotate it.

## MCP-C2-CONTAINER-PRIVILEGE

**PrivilegedContainer** · default severity **high** · OWASP ASI03, ASI05 · CWE-250

A `docker run` or `podman run` server uses `--privileged`, host networking, PID or IPC namespaces, `--cap-add SYS_ADMIN`/`ALL`, or mounts the host root or the Docker socket.

**Fix:** Drop those flags and mount only what the server needs, read-only where possible.

## MCP-C2-REMOTE-SCRIPT

**PipedRemoteScript** · default severity **critical** · OWASP ASI04, ASI05 · CWE-494

The server is started through a shell that pipes a download into an interpreter (`curl … | sh`). Whatever the URL serves on the day the agent starts runs with your rights.

**Fix:** Install a pinned, reviewed package instead.

## MCP-C2-ARBITRARY-EXEC

**ArbitraryCommandTool** · default severity **medium** · OWASP ASI02, ASI05 · CWE-78

A tool's name (exec, shell, run_command and similar) or a parameter (`command`, `cmd`, `script`, `code`) shows it runs arbitrary commands or code. One injected instruction anywhere in context can become code execution.

**Fix:** Require human approval for the tool, or run the server in a sandbox with no network and no secrets.

## MCP-C3-PLAINTEXT-HTTP

**UnencryptedRemoteServer** · default severity **high** · OWASP none · CWE-319

A remote server URL uses `http://` to a non-local host. Tool calls, results and auth headers travel in clear text, and tool descriptions can be rewritten in transit.

**Fix:** Use the https:// endpoint.

## MCP-C3-CREDENTIAL-IN-URL

**CredentialInUrl** · default severity **medium** · OWASP ASI03 · CWE-598

The server URL carries a credential in its query string or user-info part. URLs are logged by proxies and servers.

**Fix:** Send the credential in an Authorization header or use OAuth, and rotate it.

## MCP-C4-MALICIOUS

**KnownMaliciousPackage** · default severity **critical** · OWASP ASI04 · CWE-506

The package matches an entry in the indicator feed. Every entry cites public sources.

**Fix:** Remove the server, rotate every credential it could reach, and review logs and outgoing mail.

## MCP-C4-TYPOSQUAT

**PossibleTyposquat** · default severity **high** · OWASP ASI04 · CWE-829

The package is not in the known-good registry but is within two edits (after folding look-alike characters) of a well-known MCP server, or uses a look-alike scope for the same package name.

**Fix:** Check the publisher. If you meant the well-known package, fix the name.

## MCP-C4-UNPINNED

**UnpinnedPackage** · default severity **low** · OWASP ASI04 · CWE-829

An npm or PyPI package has no version (or `latest`), or a container image has no tag, `:latest`, or no digest. A compromised release reaches you on the next start without review.

**Fix:** Pin a reviewed version, or an image digest.
