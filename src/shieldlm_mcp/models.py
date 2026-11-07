"""Normalised data model: every client config becomes one Inventory."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Transport(str, Enum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        return ["info", "low", "medium", "high", "critical"].index(self.value)

    @property
    def sarif_level(self) -> str:
        return {"critical": "error", "high": "error", "medium": "warning"}.get(self.value, "note")


class SourceRef(BaseModel):
    path: str
    line: int | None = None
    pointer: str | None = None  # JSON pointer inside the file


class PackageRef(BaseModel):
    ecosystem: str  # "npm" | "pypi" | "docker"
    name: str
    version: str | None = None


class Tool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict | None = None
    annotations: dict | None = None
    source_ref: SourceRef

    @property
    def description_hash(self) -> str:
        return hashlib.sha256((self.description or "").encode()).hexdigest()


class Server(BaseModel):
    name: str
    transport: Transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    package: PackageRef | None = None
    instructions: str | None = None
    tools: list[Tool] = Field(default_factory=list)
    source_ref: SourceRef


class Inventory(BaseModel):
    servers: list[Server]
    source_paths: list[str]
    scanned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Finding(BaseModel):
    rule_id: str
    severity: Severity
    title: str
    detail: str
    server: str
    tool: str | None = None
    evidence: str = ""
    location: SourceRef
    owasp: list[str] = Field(default_factory=list)
    cwe: list[str] = Field(default_factory=list)
    remediation: str = ""
    confidence: float | None = None

    @property
    def fingerprint(self) -> str:
        key = "|".join([self.rule_id, self.server, self.tool or "", self.evidence])
        return hashlib.sha256(key.encode()).hexdigest()[:24]
