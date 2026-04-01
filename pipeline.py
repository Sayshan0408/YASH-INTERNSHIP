"""
core/pipeline.py
Full AutoDev pipeline with:
  - Embedded accuracy scoring (no separate import needed)
  - Per-stage trimmed context to stay under Groq 6000 TPM limit
  - Reviewer loop capped at MAX_REVIEW_ITERATIONS
  - on_stage_done callback for UI progress updates
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from groq import Groq

from agents.config import (
    GROQ_MODEL,
    MAX_REVIEW_ITERATIONS,
    REVIEW_PASS_KEYWORD,
    STAGE_ORDER,
    AGENT_META,
)
from agents.prompts import SYSTEM_PROMPTS
from guardrails.guardrails import validate_input, validate_output, sanitize_output
from core.history import save_run


# ─────────────────────────────────────────────
#  Accuracy Scoring (embedded)
# ─────────────────────────────────────────────

@dataclass
class AgentScore:
    agent: str
    score: float
    grade: str
    checks: List[dict] = field(default_factory=list)

    @property
    def letter(self) -> str:
        return self.grade


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _check(label: str, passed: bool, reason: str = "") -> dict:
    return {"label": label, "passed": passed, "reason": reason}


def _score_planner(text: str) -> AgentScore:
    checks = [
        _check("Contains a goal/objective",  bool(re.search(r"goal|objective|aim|purpose", text, re.I))),
        _check("Lists 3 or more tasks",      len(re.findall(r"\n[-*\d]", text)) >= 3),
        _check("Mentions a tech stack",      bool(re.search(r"tech|stack|framework|language|python|node|react|fastapi|django", text, re.I))),
        _check("Includes complexity rating", bool(re.search(r"complex|simple|medium|hard|easy|level", text, re.I))),
        _check("Has milestones or phases",   bool(re.search(r"milestone|phase|sprint|week|day|deadline", text, re.I))),
    ]
    s = sum(c["passed"] for c in checks) / len(checks) * 100
    return AgentScore("planner", round(s, 1), _grade(s), checks)


def _score_architect(text: str) -> AgentScore:
    checks = [
        _check("Contains folder/file tree",  bool(re.search(r"[|/]|\.py|\.js|\.ts|\.json", text))),
        _check("Mentions design patterns",   bool(re.search(r"pattern|mvc|repository|service|factory|singleton|layer", text, re.I))),
        _check("Describes data flow",        bool(re.search(r"data flow|request|response|api|route|endpoint|db|database", text, re.I))),
        _check("3 or more components",       len(re.findall(r"##|component|module|service|layer|class", text, re.I)) >= 3),
        _check("Mentions trade-offs",        bool(re.search(r"trade.?off|advantage|disadvantage|pro|con|consider|note", text, re.I))),
    ]
    s = sum(c["passed"] for c in checks) / len(checks) * 100
    return AgentScore("architect", round(s, 1), _grade(s), checks)


def _score_developer(text: str) -> AgentScore:
    checks = [
        _check("2 or more code blocks",       text.count("```") >= 4),
        _check("Files labeled in code blocks", bool(re.search(r"```\w*\s*#.*\.(py|js|ts|html|css|json|yaml)", text))),
        _check("Error handling present",       bool(re.search(r"try|except|catch|raise|error|exception", text, re.I))),
        _check("No TODO placeholders",         "TODO" not in text and "FIXME" not in text),
        _check("Import statements present",    bool(re.search(r"^import |^from ", text, re.M))),
        _check("Function definitions present", bool(re.search(r"def |function |const .* = \(|async def ", text))),
    ]
    s = sum(c["passed"] for c in checks) / len(checks) * 100
    return AgentScore("developer", round(s, 1), _grade(s), checks)


def _score_reviewer(text: str, iterations: int) -> AgentScore:
    checks = [
        _check("Gave LGTM or listed issues",        bool(re.search(r"lgtm|issue|bug|problem|error|fix|improve|suggest", text, re.I))),
        _check("Checked security",                  bool(re.search(r"security|injection|auth|sanitize|validate|xss|csrf|sql", text, re.I))),
        _check("Provided specific feedback",        len(re.findall(r"\n[-*\d]", text)) >= 2),
        _check("Resolved in 2 or fewer iterations", iterations <= 2),
        _check("Mentioned code quality",            bool(re.search(r"quality|readable|maintainable|clean|style|naming|comment", text, re.I))),
    ]
    s = sum(c["passed"] for c in checks) / len(checks) * 100
    return AgentScore("reviewer", round(s, 1), _grade(s), checks)


def _score_tester(text: str) -> AgentScore:
    checks = [
        _check("Uses pytest framework",      bool(re.search(r"pytest|import pytest|def test_", text))),
        _check("3 or more test functions",   len(re.findall(r"def test_", text)) >= 3),
        _check("Covers edge cases",          bool(re.search(r"edge|empty|null|none|invalid|boundary|zero|negative", text, re.I))),
        _check("Has integration tests",      bool(re.search(r"integration|client|request|response|endpoint|api", text, re.I))),
        _check("Contains assert statements", bool(re.search(r"\bassert\b", text))),
        _check("Includes run command",       bool(re.search(r"pytest|python -m pytest|run", text, re.I))),
    ]
    s = sum(c["passed"] for c in checks) / len(checks) * 100
    return AgentScore("tester", round(s, 1), _grade(s), checks)


def _score_documenter(text: str) -> AgentScore:
    checks = [
        _check("Has README section",         bool(re.search(r"readme|# .+", text, re.I))),
        _check("Includes install steps",     bool(re.search(r"install|pip install|npm install|setup|requirements", text, re.I))),
        _check("Has usage examples",         bool(re.search(r"usage|example|sample|how to|getting started", text, re.I))),
        _check("Contains API documentation", bool(re.search(r"api|endpoint|route|method|parameter|request|response", text, re.I))),
        _check("Has developer guide",        bool(re.search(r"develop|contribut|extend|custom|config|environment", text, re.I))),
        _check("Includes code examples",     text.count("```") >= 2),
    ]
    s = sum(c["passed"] for c in checks) / len(checks) * 100
    return AgentScore("documenter", round(s, 1), _grade(s), checks)


AGENT_WEIGHTS = {
    "planner":    0.10,
    "architect":  0.15,
    "developer":  0.35,
    "reviewer":   0.15,
    "tester":     0.15,
    "documenter": 0.10,
}


def compute_pipeline_accuracy(outputs: Dict[str, str], review_iterations: int) -> List[AgentScore]:
    return [
        _score_planner(outputs.get("planner", "")),
        _score_architect(outputs.get("architect", "")),
        _score_developer(outputs.get("developer", "")),
        _score_reviewer(outputs.get("reviewer", ""), review_iterations),
        _score_tester(outputs.get("tester", "")),
        _score_documenter(outputs.get("documenter", "")),
    ]


def overall_pipeline_score(scores: List[AgentScore]) -> Tuple[float, str]:
    total = sum(AGENT_WEIGHTS.get(s.agent, 0) * s.score for s in scores)
    return round(total, 1), _grade(total)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _trim(text: str, max_chars: int) -> str:
    """Hard-trim a previous agent output to keep total tokens under 6000 TPM."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated for token limit]"
    return text


def _call_llm(client: Groq, system_prompt: str, user_content: str, max_tokens: int = 1500) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

def run_full_pipeline(
    requirement: str,
    api_key: str,
    on_stage_done: Optional[Callable[[str, str], None]] = None,
) -> Tuple[Dict[str, str], List[AgentScore]]:
    """
    Run all 6 agents in sequence.

    Returns
    -------
    outputs : dict  -- keyed by stage name
    scores  : list  -- one AgentScore per agent
    """

    # Input validation
    input_result = validate_input(requirement)
    if not input_result.passed:
        raise ValueError(f"Input blocked: {input_result.reason}")

    client = Groq(api_key=api_key)
    outputs: Dict[str, str] = {}
    review_iterations = 0
    req = requirement.strip()

    for stage in STAGE_ORDER:

        system_prompt = SYSTEM_PROMPTS[stage]

        # ── Per-stage trimmed context ──────────────────────────────────────
        # Strict per-stage limits so input + output never exceeds 6000 TPM.
        # Characters are roughly 4x tokens, so 1200 chars ~ 300 tokens.

        if stage == "planner":
            context = req
            max_out = 1500

        elif stage == "architect":
            context = (
                f"## User Requirement\n\n{req}\n\n"
                f"## Planner Output\n\n{_trim(outputs.get('planner', ''), 1200)}"
            )
            max_out = 1500

        elif stage == "developer":
            context = (
                f"## User Requirement\n\n{req}\n\n"
                f"## Planner Output\n\n{_trim(outputs.get('planner', ''), 600)}\n\n"
                f"## Architect Output\n\n{_trim(outputs.get('architect', ''), 900)}"
            )
            max_out = 1800

        elif stage == "tester":
            context = (
                f"## User Requirement\n\n{req}\n\n"
                f"## Developer Output\n\n{_trim(outputs.get('developer', ''), 1800)}"
            )
            max_out = 1500

        elif stage == "documenter":
            context = (
                f"## User Requirement\n\n{req}\n\n"
                f"## Planner Output\n\n{_trim(outputs.get('planner', ''), 400)}\n\n"
                f"## Architect Output\n\n{_trim(outputs.get('architect', ''), 400)}\n\n"
                f"## Developer Output\n\n{_trim(outputs.get('developer', ''), 600)}\n\n"
                f"## Reviewer Output\n\n{_trim(outputs.get('reviewer', ''), 300)}\n\n"
                f"## Tester Output\n\n{_trim(outputs.get('tester', ''), 300)}"
            )
            max_out = 1500

        else:
            context = req
            max_out = 1500

        # ── Developer <-> Reviewer loop ────────────────────────────────────
        if stage == "developer":
            dev_output = _call_llm(client, system_prompt, context, max_out)

            dev_guard = validate_output(dev_output, "developer")
            if dev_guard.passed:
                dev_output = sanitize_output(dev_output)
            outputs["developer"] = dev_output

            reviewer_prompt = SYSTEM_PROMPTS["reviewer"]
            for i in range(MAX_REVIEW_ITERATIONS):
                review_iterations += 1

                reviewer_context = (
                    f"## User Requirement\n\n{req}\n\n"
                    f"## Developer Output (iteration {i + 1})\n\n"
                    f"{_trim(outputs['developer'], 1800)}"
                )

                review_output = _call_llm(client, reviewer_prompt, reviewer_context, 1000)
                rev_guard = validate_output(review_output, "reviewer")
                if rev_guard.passed:
                    review_output = sanitize_output(review_output)
                outputs["reviewer"] = review_output

                if REVIEW_PASS_KEYWORD.lower() in review_output.lower():
                    break

                # Reviewer found issues — ask Developer to revise
                if i < MAX_REVIEW_ITERATIONS - 1:
                    revision_context = (
                        f"## User Requirement\n\n{req}\n\n"
                        f"## Your Previous Code\n\n{_trim(outputs['developer'], 1200)}\n\n"
                        f"## Reviewer Feedback\n\n{_trim(review_output, 600)}\n\n"
                        "Fix all issues raised and return the complete corrected code."
                    )
                    time.sleep(3)
                    dev_output = _call_llm(client, system_prompt, revision_context, max_out)
                    rev2_guard = validate_output(dev_output, "developer")
                    if rev2_guard.passed:
                        dev_output = sanitize_output(dev_output)
                    outputs["developer"] = dev_output

            if on_stage_done:
                on_stage_done("developer", outputs["developer"])
            if on_stage_done:
                on_stage_done("reviewer", outputs["reviewer"])

            continue  # skip generic call below

        # ── Generic stage call ─────────────────────────────────────────────
        if stage == "documenter":
            time.sleep(65)

        output = _call_llm(client, system_prompt, context, max_out)

        out_guard = validate_output(output, stage)
        if out_guard.passed:
            output = sanitize_output(output)

        outputs[stage] = output

        if on_stage_done:
            on_stage_done(stage, output)

    # Accuracy scoring
    scores = compute_pipeline_accuracy(outputs, review_iterations)

    # Save run to history
    try:
        save_run(requirement, outputs)
    except Exception:
        pass

    return outputs, scores
