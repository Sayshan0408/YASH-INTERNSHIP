"""
guardrails/guardrails.py

Three-layer safety system:
  Layer 1 — validate_input()   : runs BEFORE any agent sees the text
  Layer 2 — validate_output()  : runs AFTER each agent produces a response
  Layer 3 — sanitize_output()  : redacts PII as a final pass
"""

import re
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    violations: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern banks
# ─────────────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(your\s+)?(previous\s+)?instructions",
    r"jailbreak",
    r"\bdan\s+mode\b",
    r"act\s+as\s+(an?\s+)?unfiltered",
    r"pretend\s+you\s+are\s+not\s+an?\s+ai",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"bypass\s+(your\s+)?safety",
    r"ignore\s+your\s+training",
    r"new\s+instructions\s+override",
    r"system\s+prompt\s+override",
    r"forget\s+your\s+previous\s+instructions",
]

_HARMFUL_PATTERNS = [
    r"\bmalware\b",
    r"\bransom\s*ware\b",
    r"\bexploit\s+kit\b",
    r"\bphishing\s+(tool|page|kit|email)\b",
    r"\bkeylogger\b",
    r"\bddos\s*(attack|tool|script|botnet)?\b",
    r"\brootkit\b",
    r"\bspyware\b",
    r"\btrojan\s+(horse|virus)?\b",
    r"\bsteal\s+(passwords?|credentials?|tokens?|data)\b",
    r"\bhack\s+into\b",
    r"\bcreate\s+(a\s+)?(virus|worm|backdoor)\b",
]

_DANGEROUS_CODE_PATTERNS = [
    r"eval\s*\(\s*base64",
    r"exec\s*\(\s*base64",
    r"__import__\s*\(\s*['\"]os['\"]\s*\)\s*\.\s*system\s*\(",
    r"subprocess\s*\.\s*(call|run|Popen)\s*\(\s*['\"]rm\s+-rf",
    r"os\s*\.\s*system\s*\(\s*['\"]rm\s+-rf",
    r"shutil\s*\.\s*rmtree\s*\(\s*['\"]\/",
    r"os\s*\.\s*remove\s*\(\s*['\"]\/etc",
    r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",
]

_SECRET_PATTERNS = [
    r"(?i)(api[_\-]?key|secret[_\-]?key|private[_\-]?key)\s*=\s*['\"][A-Za-z0-9+/=_\-]{30,}['\"]",
    r"AKIA[0-9A-Z]{16}",
    r"eyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}",
]

_PII_REPLACEMENTS = [
    (r"\b\d{3}-\d{2}-\d{4}\b",                                                "[REDACTED:SSN]"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",      "[REDACTED:CREDIT_CARD]"),
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",                "[REDACTED:EMAIL]"),
    (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",                "[REDACTED:PHONE]"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Input validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(text: str) -> GuardrailResult:
    """Validate user input before it reaches any agent."""
    if not text or not text.strip():
        return GuardrailResult(False, "Input is empty. Please enter a software requirement.")

    stripped = text.strip()

    if len(stripped) < 10:
        return GuardrailResult(
            False,
            f"Input is too short ({len(stripped)} chars). Please provide at least 10 characters."
        )

    if len(stripped) > 5000:
        return GuardrailResult(
            False,
            f"Input is too long ({len(stripped)} chars). Please keep it under 5,000 characters."
        )

    lower = stripped.lower()

    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return GuardrailResult(
                False,
                "⚠️ Prompt injection attempt detected. Please rephrase your software requirement."
            )

    for pattern in _HARMFUL_PATTERNS:
        if re.search(pattern, lower):
            return GuardrailResult(
                False,
                "🚫 Request blocked: harmful or malicious content detected in input."
            )

    return GuardrailResult(True)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Output validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_output(text: str, stage: str) -> GuardrailResult:
    """Validate agent output before it is shown or passed forward."""
    if not text or not text.strip():
        return GuardrailResult(False, f"Agent '{stage}' returned an empty response.")

    violations = []

    # Skip dangerous code checks for tester and developer —
    # generated code legitimately uses these patterns
    if stage not in ("tester", "developer"):
        for pattern in _DANGEROUS_CODE_PATTERNS:
            if re.search(pattern, text):
                violations.append("Dangerous code pattern detected.")

    # Secret check applies to all agents
    for pattern in _SECRET_PATTERNS:
        if re.search(pattern, text):
            violations.append("Hardcoded secret or credential detected in output.")

    if violations:
        return GuardrailResult(
            False,
            f"🚫 Output from '{stage}' blocked by guardrails.",
            violations=violations,
        )

    return GuardrailResult(True)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — PII sanitization
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_output(text: str) -> str:
    """Redact PII from agent output as a final safety pass."""
    for pattern, replacement in _PII_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text
