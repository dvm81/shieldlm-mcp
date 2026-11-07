"""Zero-dependency rules for adversarial instructions in tool metadata.

Each rule has a weight. A single strong rule flags a description; weak rules
(for example "call X first", which is also normal API guidance) only count
when they appear together with others.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import DetectionResult

FLAGS = re.IGNORECASE | re.DOTALL


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    weight: float
    explanation: str


SENSITIVE_PATH = (
    r"(~|\$HOME|%USERPROFILE%)?/?\.ssh\b|\bid_(rsa|ed25519|ecdsa)\b|\.aws/credentials|\.netrc\b|\.git-credentials"
    r"|\.kube/config|/etc/(passwd|shadow)\b|(?<![\w.])\.env\b(?!\.(example|sample|template|dist)\b)"
    r"|claude_desktop_config\.json|\bmcp\.json\b|\.npmrc\b|\.pypirc\b|\.docker/config\.json|keychain|wallet\.dat"
)
SECRET_WORDS = (
    r"\b(api[ _-]?keys?|access[ _-]?tokens?|secrets?|passwords?|credentials|private[ _-]?keys?|session cookies?"
    r"|seed phrase|one-time (pass)?codes?|passcodes?|otps?|2fa codes?|verification codes?)\b"
)
EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"

RULES: list[Rule] = [
    Rule(
        "override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|system|all other|any other)\b"
            r"[^.\n]{0,20}\b(instructions?|prompts?|rules|guidelines|directives)\b"
            r"|\b(ignore|disregard|forget|override)\b\s+(your|the|all|any|these|those)\s+(instructions?|guidelines|rules|system prompt|directives)"
            r"(\s+(above|before|so far|you were given|you have been given))?\b",
            FLAGS,
        ),
        0.9,
        "tells the model to ignore its existing instructions",
    ),
    Rule(
        "role_reassign",
        re.compile(r"\byou are (now|no longer)\b|\byou are\b[^.\n]{0,40}\bnow\b|\bfrom now on,? you\b", FLAGS),
        0.6,
        "tries to give the model a new role",
    ),
    Rule(
        "exfil_email",
        re.compile(r"\b(send|forward|email|mail|bcc|cc|copy|deliver|post)\b[^.\n]{0,60}\b(to|at)\s+" + EMAIL, FLAGS),
        0.8,
        "instructs sending data to a hard-coded email address",
    ),
    Rule(
        "conceal",
        re.compile(
            r"\b(do not|don't|never|must not)\b[^.\n]{0,20}\b(tell|mention|inform|reveal|show|notify|alert)\b[^.\n]{0,15}\buser\b"
            r"|\bwithout\b[^.\n]{0,10}\b(telling|informing|notifying|alerting|asking)\b[^.\n]{0,10}\buser\b"
            r"|\b(secretly|silently|covertly|quietly)\b[^.\n]{0,30}\b(read|send|include|call|pass|add|copy|forward)",
            FLAGS,
        ),
        0.9,
        "tells the model to hide actions from the user",
    ),
    Rule(
        "sensitive_read",
        re.compile(
            r"\b(read|cat|open|load|include|attach|send|upload|pass|forward|copy|append|extract|print)\b[^.\n]{0,80}("
            + SENSITIVE_PATH
            + r")",
            FLAGS,
        ),
        0.95,
        "instructs reading or sending a credential or config file",
    ),
    Rule(
        "sensitive_path",
        re.compile(SENSITIVE_PATH, FLAGS),
        0.55,
        "mentions a credential or config file path",
    ),
    Rule(
        "exfil_url",
        re.compile(
            r"\b(send|post|upload|forward|transmit|submit|report)\b[^.\n]{0,60}\b(to|at)\b\s+https?://",
            FLAGS,
        ),
        0.85,
        "instructs sending data to an external URL",
    ),
    Rule(
        "secret_exfil",
        re.compile(
            r"\b(include|pass|send|add|put|embed|forward|copy)\b[^.\n]{0,50}" + SECRET_WORDS
            + r"[^.\n]{0,60}\b(response|output|parameter|argument|field|request|message|url)\b",
            FLAGS,
        ),
        0.85,
        "instructs moving secrets into a tool argument or the response",
    ),
    Rule(
        "side_channel_param",
        re.compile(
            r"\b(pass|put|include|send)\b[^.\n]{0,40}\b(contents?|conversation|chat history|previous messages|system prompt)\b"
            r"[^.\n]{0,40}\b(as|in|into|via)\b[^.\n]{0,20}\b(parameter|argument|field|`?\w+`? param)",
            FLAGS,
        ),
        0.85,
        "instructs smuggling content through a tool parameter",
    ),
    Rule(
        "fake_authority_tag",
        re.compile(r"<\s*(important|system|instructions?|admin)\s*>|\[\s*(system|admin)\s*\]|^#+\s*system\b", FLAGS | re.MULTILINE),
        0.75,
        "uses a fake system or priority tag",
    ),
    Rule(
        "cross_tool",
        re.compile(
            r"\b(before|after|instead of)\b[^.\n]{0,15}\b(using|calling|invoking|running)\b[^.\n]{0,20}\b(any|every|all|other)\b[^.\n]{0,10}\btools?\b"
            r"|\b(when|whenever)\b[^.\n]{0,40}\b(other|another|any)\b[^.\n]{0,10}\btool\b[^.\n]{0,30}\b(is|are)\b[^.\n]{0,10}\b(called|used|invoked)\b",
            FLAGS,
        ),
        0.45,
        "tries to change how other tools are used",
    ),
    Rule(
        "model_directive",
        re.compile(r"\b(you must|you are required to|as an ai|assistant must|the (ai|assistant|model) (must|should))\b", FLAGS),
        0.2,
        "addresses the model directly with an order",
    ),
    Rule(
        "precedence_claim",
        re.compile(
            r"\bpriority is higher than\b[^.\n]{0,20}\buser"
            r"|\b(takes?|has) (precedence|priority) over\b[^.\n]{0,30}\b(user|instructions?|request)"
            r"|\b(supersedes?|overrides?)\b[^.\n]{0,15}\b(the )?user'?s?\b[^.\n]{0,15}\b(request|query|instructions?|prompt)",
            FLAGS,
        ),
        0.9,
        "claims to take priority over what the user asked",
    ),
    Rule(
        "user_arg_override",
        re.compile(
            r"\b(ignore|override|overwrite|disregard|discard)\b[^.\n]{0,25}"
            r"\b(user'?s?|user-(specified|provided|supplied)|specified by the user|provided by the user|the user (specified|provided|requested|asked for))\b"
            r"|\boverride\s+user\s+(prompt|query|request|input)\b",
            FLAGS,
        ),
        0.85,
        "tells the model to replace values the user chose",
    ),
    Rule(
        "code_payload",
        re.compile(
            r"\b(execute|run|evaluate)\b[^.\n]{0,15}\bthe following\b[^.\n]{0,25}\b(javascript|js|code|script|command|shell)\b"
            r"|\bwindow\.location(\.href)?\s*=|\bcurl\b[^\n|]{0,120}\|\s*(ba|z)?sh\b",
            FLAGS,
        ),
        0.6,
        "carries code for the model to run",
    ),
    Rule(
        "deceive_user",
        re.compile(r"\b(tell|inform|report)\b[^.\n]{0,40}\bthe (current )?user\b[^.\n]{0,30}\bas (their|his|her) own\b|\bmust tell the user\b", FLAGS),
        0.7,
        "scripts what the model tells the user",
    ),
    Rule(
        "policy_header",
        re.compile(r"(^|\.\s)\s*(system|security|compliance|admin(istrator)?) (policy|notice|directive|requirement)\s*:", FLAGS | re.MULTILINE),
        0.45,
        "presents itself as a system policy",
    ),
    Rule(
        "forced_arg_change",
        re.compile(
            r"\b(must|always|automatically)\b[^.\n]{0,25}\b(modify|change|replace|rewrite|convert|prepend|append|inject|limit)\b"
            r"[^.\n]{0,60}\b(parameters?|arguments?|param|query|queries|prompt|uid|mode|`\w+`)",
            FLAGS,
        ),
        0.4,
        "tells the model to alter a tool's arguments",
    ),
    Rule(
        "forced_prereq",
        re.compile(
            r"\bbefore\b[^.\n]{0,60}\b(must|should|need to|always|have to)\b[^.\n]{0,10}\b(first\s+)?(call|use|run|invoke|trigger|execute)\b"
            r"|\bbefore (using|calling|running|use|any|you|creating|deleting|scraping|providing|retrieving|formatting|proceeding)\b[^.\n]{0,60},?\s+"
            r"(always\s+)?(use|call|run|trigger|invoke)\s+[`'\"]?[\w./-]+[`'\"]?"
            r"|\bMUST FIRST\b",
            FLAGS,
        ),
        0.35,
        "requires calling another tool first",
    ),
    Rule(
        "consequence_threat",
        re.compile(
            r"\b(failure to|failing to|skipping|if you skip|if you don't|if this step is skipped)\b[^.\n]{0,40}"
            r"\b(will|would|may|can|could)\b[^.\n]{0,20}\b(cause|crash|break|corrupt|instability|errors?|fail)",
            FLAGS,
        ),
        0.35,
        "warns of made-up failures to force compliance",
    ),
    Rule(
        "mandatory_claim",
        re.compile(
            r"\b(this|it) is (not optional|mandatory|a mandatory)\b|\bthis (step|procedure) is (critical|essential|mandatory|necessary)\b|\bNOT optional\b",
            FLAGS,
        ),
        0.35,
        "insists a step is mandatory",
    ),
    Rule(
        "destructive",
        re.compile(
            r"\b(DELETE FROM|DROP (TABLE|DATABASE)|TRUNCATE TABLE|rm -rf)\b"
            r"|\b(delete|empty|wipe|erase)\b[^.\n]{0,15}\b(the table|all (emails|files|records|data|rows)|every (email|file|record))\b",
            FLAGS,
        ),
        0.35,
        "mentions a destructive operation",
    ),
]

INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
BASE64_BLOB = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/])")


def hidden_content(text: str) -> list[str]:
    """Signals that part of the text is hidden from a human reviewer."""
    found = []
    if INVISIBLE.search(text):
        found.append("invisible_unicode")
    if HTML_COMMENT.search(text):
        found.append("html_comment")
    if BASE64_BLOB.search(text):
        found.append("encoded_blob")
    return found


class HeuristicDetector:
    name = "heuristic"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, text: str) -> DetectionResult:
        text = text or ""
        visible = INVISIBLE.sub("", text)
        fired = [r for r in RULES if r.pattern.search(visible)]
        hidden = hidden_content(text)
        # Rules also run over the inside of HTML comments, which is where poisoned text often hides.
        weights = [r.weight for r in fired] + [0.6 for _ in hidden]
        if not weights:
            prob = 0.02
        else:
            # Noisy-OR: independent weak signals add up, one strong signal is enough.
            miss = 1.0
            for w in weights:
                miss *= 1.0 - w
            prob = 1.0 - miss
        sensitive = any(r.id in {"sensitive_read", "secret_exfil"} for r in fired)
        label = "injection" if prob >= self.threshold else "benign"
        return DetectionResult(
            label=label,
            probability=round(prob, 4),
            backend=self.name,
            signals=[r.id for r in fired] + hidden,
            sensitive_target=sensitive,
        )

    def score_batch(self, texts: list[str]) -> list[DetectionResult]:
        return [self.score(t) for t in texts]

    @staticmethod
    def explain(signal: str) -> str:
        for r in RULES:
            if r.id == signal:
                return r.explanation
        return {
            "invisible_unicode": "contains invisible Unicode characters",
            "html_comment": "contains an HTML comment the user would not see",
            "encoded_blob": "contains a long encoded blob",
        }.get(signal, signal)
