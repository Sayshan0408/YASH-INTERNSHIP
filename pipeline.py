"""
core/pipeline.py

Central pipeline engine — uses the Groq SDK directly.
No LangChain. No agent loops. No iteration limit issues.

Flow:
  1. validate_input()
  2. Planner → Architect → [Developer <-> Reviewer loop] → Tester → Documenter
  3. Each stage: validate_output() → sanitize_output() → append to context
  4. save_run() to history
"""

import time
from typing import Callable, Dict, Optional

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


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM call — one clean function, no loops
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(api_key: str, system_prompt: str, user_content: str,
               max_tokens: int = 4096) -> str:
    """
    Make a single direct call to Groq.
    Returns the response text string.
    Raises groq.APIError subclasses on failure (caught by pipeline).
    """
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# Single stage runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage(stage: str, context: str, api_key: str,
               max_tokens: int = 4096) -> str:
    """Call one agent stage and return its raw output string."""
    return _call_groq(api_key, SYSTEM_PROMPTS[stage], context,
                      max_tokens=max_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    requirement: str,
    api_key: str,
    on_stage_start: Optional[Callable[[str], None]] = None,
    on_stage_done: Optional[Callable[[str, str], None]] = None,
    on_guardrail_block: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, str]:
    """
    Run all 6 agents in sequence.

    Callbacks (all optional):
      on_stage_start(stage)          — called just before an agent runs
      on_stage_done(stage, output)   — called after an agent finishes
      on_guardrail_block(stage, msg) — called when a guardrail blocks

    Returns dict of { stage_name: output_text }.
    Returns empty dict if input guardrail blocks.
    """

    # ── Layer 1: Input guardrail ───────────────────────────────────────────
    input_check = validate_input(requirement)
    if not input_check.passed:
        if on_guardrail_block:
            on_guardrail_block("input", input_check.reason)
        return {}

    outputs: Dict[str, str] = {}
    context = f"## User Requirement\n\n{requirement.strip()}\n"

    for stage in STAGE_ORDER:

        # ── Developer <-> Reviewer iterative loop ──────────────────────────
        if stage == "developer":
            dev_context = context
            final_dev_output = ""
            final_rev_output = ""

            for iteration in range(1, MAX_REVIEW_ITERATIONS + 1):

                # --- Developer turn ---
                if on_stage_start:
                    on_stage_start("developer")
                try:
                    dev_output = _run_stage("developer", dev_context, api_key)
                except Exception as exc:
                    dev_output = f"[ERROR]: Developer agent failed — {exc}"

                dev_check = validate_output(dev_output, "developer")
                if not dev_check.passed:
                    if on_guardrail_block:
                        on_guardrail_block("developer", dev_check.reason)
                    dev_output = f"[BLOCKED]: {dev_check.reason}"
                    outputs["developer"] = dev_output
                    final_dev_output = dev_output
                    if on_stage_done:
                        on_stage_done("developer", dev_output)
                    break

                dev_output = sanitize_output(dev_output)
                final_dev_output = dev_output
                if on_stage_done:
                    on_stage_done("developer", dev_output)

                # --- Reviewer turn ---
                if on_stage_start:
                    on_stage_start("reviewer")

                rev_context = (
                    dev_context
                    + f"\n\n## Developer Output (Iteration {iteration})\n\n"
                    + dev_output
                )
                try:
                    rev_output = _run_stage("reviewer", rev_context, api_key)
                except Exception as exc:
                    rev_output = f"[ERROR]: Reviewer agent failed — {exc}"

                rev_check = validate_output(rev_output, "reviewer")
                if not rev_check.passed:
                    if on_guardrail_block:
                        on_guardrail_block("reviewer", rev_check.reason)
                    rev_output = f"[BLOCKED]: {rev_check.reason}"

                rev_output = sanitize_output(rev_output)
                final_rev_output = rev_output
                if on_stage_done:
                    on_stage_done("reviewer", rev_output)

                # Exit loop early if reviewer approves
                if REVIEW_PASS_KEYWORD in rev_output:
                    break

                # Feed reviewer feedback back into developer context
                dev_context = (
                    dev_context
                    + f"\n\n## Developer Output (Iteration {iteration})\n\n"
                    + dev_output
                    + f"\n\n## Reviewer Feedback (Iteration {iteration})\n\n"
                    + rev_output
                    + "\n\nPlease address ALL issues raised by the reviewer "
                    "and produce a fully revised version of the code."
                )

            outputs["developer"] = final_dev_output
            outputs["reviewer"] = final_rev_output
            context += (
                f"\n\n## Developer Code (Final)\n\n{final_dev_output}"
                f"\n\n## Code Review (Final)\n\n{final_rev_output}"
            )
            continue

        if stage == "reviewer":
            continue  # handled inside the developer loop above

# ── Trim context + delay before Documenter to avoid 413/429 ───────
        if stage == "documenter":
            time.sleep(65)  # wait 65s to reset Groq's TPM window
            # Only pass requirement + final developer code to Documenter
            # Sending full context exceeds Groq's 6000 TPM limit
            context = (
                f"## User Requirement\n\n{requirement.strip()}\n\n"
                f"## Developer Code (Final)\n\n{outputs.get('developer', '')}\n\n"
                f"## Code Review (Final)\n\n{outputs.get('reviewer', '')}"
            )

        # ── Generic stages: planner, architect, tester, documenter ────────
        if on_stage_start:
            on_stage_start(stage)

        # Use smaller max_tokens for documenter to stay within TPM limits
        stage_max_tokens = 2048 if stage == "documenter" else 4096

        try:
            output = _run_stage(stage, context, api_key,
                                max_tokens=stage_max_tokens)
        except Exception as exc:
            output = f"[ERROR]: {stage.capitalize()} agent failed — {exc}"

        out_check = validate_output(output, stage)
        if not out_check.passed:
            if on_guardrail_block:
                on_guardrail_block(stage, out_check.reason)
            output = f"[BLOCKED]: {out_check.reason}"
        else:
            output = sanitize_output(output)

        outputs[stage] = output
        if on_stage_done:
            on_stage_done(stage, output)

        label = AGENT_META[stage]["label"]
        context += f"\n\n## {label} Output\n\n{output}"

    # ── Persist to history ─────────────────────────────────────────────────
    status = "success" if len(outputs) == len(STAGE_ORDER) else "partial"
    save_run(requirement, outputs, status=status)

    return outputs