"""
core/accuracy.py

Per-agent accuracy scoring system.
Each agent is evaluated on criteria specific to its role.
Returns a score 0–100 and a breakdown of what passed/failed.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class AgentScore:
    agent: str
    score: float                          # 0.0 – 100.0
    grade: str                            # A / B / C / D / F
    checks: List[Tuple[str, bool, str]]   # (check_name, passed, detail)
    summary: str                          # one-line verdict


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _score_from_checks(checks: List[Tuple[str, bool, str]],
                        weights: List[float]) -> float:
    """Weighted average: each check has a weight in [0,1]."""
    total_weight = sum(weights)
    earned = sum(w for (_, passed, _), w in zip(checks, weights) if passed)
    return round((earned / total_weight) * 100, 1) if total_weight else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER
# Checks: has goal, has numbered tasks, mentions tech stack, has complexity
# ─────────────────────────────────────────────────────────────────────────────

def score_planner(output: str) -> AgentScore:
    checks = []

    has_goal = bool(re.search(r"(goal|objective|purpose|aim)", output, re.I))
    checks.append(("Contains project goal", has_goal,
                   "Found goal/objective statement" if has_goal else "No goal statement found"))

    task_count = len(re.findall(r"^\s*\d+[\.\)]\s+\S", output, re.MULTILINE))
    has_tasks = task_count >= 3
    checks.append(("Has ≥3 numbered tasks", has_tasks,
                   f"Found {task_count} numbered tasks" if has_tasks else f"Only {task_count} tasks — needs at least 3"))

    has_stack = bool(re.search(
        r"(python|fastapi|flask|django|node|react|vue|angular|sql|mongodb|redis|docker)",
        output, re.I))
    checks.append(("Recommends tech stack", has_stack,
                   "Tech stack mentioned" if has_stack else "No tech stack found"))

    has_complexity = bool(re.search(r"\b(low|medium|high)\b.{0,60}complex", output, re.I)
                          or re.search(r"complex.{0,60}\b(low|medium|high)\b", output, re.I))
    checks.append(("Includes complexity rating", has_complexity,
                   "Complexity rating present" if has_complexity else "Missing complexity rating"))

    has_milestones = bool(re.search(r"(milestone|phase|sprint|step|stage)", output, re.I))
    checks.append(("Mentions milestones/phases", has_milestones,
                   "Milestones present" if has_milestones else "No milestones mentioned"))

    weights = [2.0, 2.5, 2.0, 1.5, 1.0]
    score = _score_from_checks(checks, weights)
    passed = sum(1 for _, p, _ in checks if p)
    summary = f"{passed}/{len(checks)} checks passed — plan is {'well-structured' if score >= 75 else 'incomplete'}"
    return AgentScore("planner", score, _grade(score), checks, summary)


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECT
# Checks: folder tree, design patterns, data flow, components
# ─────────────────────────────────────────────────────────────────────────────

def score_architect(output: str) -> AgentScore:
    checks = []

    has_tree = bool(re.search(r"(├|└|│|--\s+\w+\.\w+|\w+/\n)", output))
    checks.append(("Contains folder/file tree", has_tree,
                   "File tree structure found" if has_tree else "No folder tree detected"))

    has_patterns = bool(re.search(
        r"\b(mvc|mvvm|repository|factory|singleton|observer|decorator|facade|service layer|rest)",
        output, re.I))
    checks.append(("Mentions design patterns", has_patterns,
                   "Design patterns referenced" if has_patterns else "No design patterns mentioned"))

    has_dataflow = bool(re.search(r"(data flow|request|response|api|endpoint|database|db)", output, re.I))
    checks.append(("Describes data flow", has_dataflow,
                   "Data flow described" if has_dataflow else "No data flow description"))

    component_count = len(re.findall(
        r"\b(service|controller|model|router|middleware|handler|manager|repository|utils|config)\b",
        output, re.I))
    has_components = component_count >= 3
    checks.append(("Defines ≥3 components", has_components,
                   f"{component_count} components found" if has_components else f"Only {component_count} components"))

    has_tradeoffs = bool(re.search(r"(trade.?off|pros?|cons?|consider|alternative|drawback|benefit)", output, re.I))
    checks.append(("Discusses trade-offs", has_tradeoffs,
                   "Trade-offs discussed" if has_tradeoffs else "No trade-off analysis"))

    weights = [2.5, 2.0, 2.0, 2.0, 1.5]
    score = _score_from_checks(checks, weights)
    passed = sum(1 for _, p, _ in checks if p)
    summary = f"{passed}/{len(checks)} checks passed — architecture is {'complete' if score >= 75 else 'needs more detail'}"
    return AgentScore("architect", score, _grade(score), checks, summary)


# ─────────────────────────────────────────────────────────────────────────────
# DEVELOPER
# Checks: code blocks, filenames, error handling, no TODOs, imports
# ─────────────────────────────────────────────────────────────────────────────

def score_developer(output: str) -> AgentScore:
    checks = []

    code_blocks = len(re.findall(r"```[\w]*\n", output))
    has_code = code_blocks >= 2
    checks.append(("Has ≥2 code blocks", has_code,
                   f"{code_blocks} code blocks found" if has_code else f"Only {code_blocks} code block(s)"))

    has_filenames = bool(re.search(r"(#\s*filename:|# \w+\.\w+|## \w+\.\w+)", output, re.I))
    checks.append(("Files are labeled", has_filenames,
                   "File labels present" if has_filenames else "No file labels found — hard to navigate"))

    has_error_handling = bool(re.search(r"\b(try|except|raise|error|exception|HTTPException|ValueError)\b", output))
    checks.append(("Includes error handling", has_error_handling,
                   "Error handling found" if has_error_handling else "No error handling detected"))

    todo_count = len(re.findall(r"#\s*TODO", output, re.I))
    no_todos = todo_count == 0
    checks.append(("No TODO placeholders", no_todos,
                   "No TODOs — code is complete" if no_todos else f"{todo_count} TODO(s) found — incomplete code"))

    has_imports = bool(re.search(r"^(import |from )", output, re.MULTILINE))
    checks.append(("Has import statements", has_imports,
                   "Imports present" if has_imports else "No imports — may be incomplete"))

    has_functions = bool(re.search(r"\b(def |class |async def )", output))
    checks.append(("Defines functions/classes", has_functions,
                   "Functions/classes defined" if has_functions else "No function or class definitions"))

    weights = [2.5, 1.5, 2.5, 2.0, 1.5, 2.0]
    score = _score_from_checks(checks, weights)
    passed = sum(1 for _, p, _ in checks if p)
    summary = f"{passed}/{len(checks)} checks passed — code is {'production-ready' if score >= 75 else 'needs improvement'}"
    return AgentScore("developer", score, _grade(score), checks, summary)


# ─────────────────────────────────────────────────────────────────────────────
# REVIEWER
# Checks: found issues OR gave LGTM, specific feedback, security mention
# ─────────────────────────────────────────────────────────────────────────────

def score_reviewer(output: str, review_iterations: int = 1) -> AgentScore:
    checks = []

    gave_lgtm = "LGTM" in output
    found_issues = bool(re.search(r"\b(bug|issue|error|fix|problem|vulnerability|concern|missing|incorrect)\b", output, re.I))
    is_decisive = gave_lgtm or found_issues
    checks.append(("Gave clear verdict (LGTM or issues)", is_decisive,
                   "LGTM given" if gave_lgtm else ("Issues listed" if found_issues else "Vague — no clear verdict")))

    has_security = bool(re.search(r"(security|injection|auth|sanitiz|valida|xss|csrf|sql\s*inject)", output, re.I))
    checks.append(("Checked security", has_security,
                   "Security review present" if has_security else "No security check"))

    has_specific = bool(re.search(r"(line \d+|function \w+|class \w+|variable \w+|missing \w+)", output, re.I))
    checks.append(("Gave specific feedback", has_specific,
                   "Specific feedback given" if has_specific else "Feedback is too generic"))

    efficient_loop = review_iterations <= 2
    checks.append(("Loop resolved in ≤2 iterations", efficient_loop,
                   f"Resolved in {review_iterations} iteration(s)" if efficient_loop
                   else f"Took {review_iterations} iterations — slow convergence"))

    has_quality = bool(re.search(r"(naming|duplication|structure|readability|best practice|clean)", output, re.I))
    checks.append(("Checked code quality", has_quality,
                   "Code quality checked" if has_quality else "No code quality feedback"))

    weights = [3.0, 2.5, 2.0, 1.5, 1.0]
    score = _score_from_checks(checks, weights)
    passed = sum(1 for _, p, _ in checks if p)
    summary = f"{passed}/{len(checks)} checks passed — review is {'thorough' if score >= 75 else 'superficial'}"
    return AgentScore("reviewer", score, _grade(score), checks, summary)


# ─────────────────────────────────────────────────────────────────────────────
# TESTER
# Checks: pytest usage, test count, edge cases, integration tests, run command
# ─────────────────────────────────────────────────────────────────────────────

def score_tester(output: str) -> AgentScore:
    checks = []

    has_pytest = bool(re.search(r"\b(pytest|unittest|def test_|@pytest\.fixture)\b", output))
    checks.append(("Uses pytest framework", has_pytest,
                   "pytest detected" if has_pytest else "No pytest usage found"))

    test_count = len(re.findall(r"def test_\w+", output))
    has_tests = test_count >= 3
    checks.append(("Has ≥3 test functions", has_tests,
                   f"{test_count} test functions found" if has_tests else f"Only {test_count} test(s) — not enough coverage"))

    has_edge = bool(re.search(r"(edge case|boundary|empty|null|none|zero|negative|invalid|exception|raises)", output, re.I))
    checks.append(("Covers edge cases", has_edge,
                   "Edge cases mentioned" if has_edge else "No edge case coverage"))

    has_integration = bool(re.search(r"(integration|end.to.end|e2e|scenario|flow|api test)", output, re.I))
    checks.append(("Includes integration tests", has_integration,
                   "Integration tests present" if has_integration else "Unit tests only — no integration coverage"))

    has_run_cmd = bool(re.search(r"pytest\s+[\w/]", output) or re.search(r"```bash.*pytest", output, re.DOTALL))
    checks.append(("Includes run command", has_run_cmd,
                   "Run command provided" if has_run_cmd else "No pytest run command"))

    has_assertions = bool(re.search(r"\bassert\b", output))
    checks.append(("Uses assertions", has_assertions,
                   "Assertions present" if has_assertions else "No assert statements found"))

    weights = [2.0, 2.5, 2.0, 1.5, 1.0, 2.0]
    score = _score_from_checks(checks, weights)
    passed = sum(1 for _, p, _ in checks if p)
    summary = f"{passed}/{len(checks)} checks passed — test suite is {'comprehensive' if score >= 75 else 'thin'}"
    return AgentScore("tester", score, _grade(score), checks, summary)


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTER
# Checks: README sections, API docs, installation steps, usage examples
# ─────────────────────────────────────────────────────────────────────────────

def score_documenter(output: str) -> AgentScore:
    checks = []

    has_readme = bool(re.search(r"(# .+|## .+README|## .+Overview|## .+Description)", output, re.I))
    checks.append(("Has README/title section", has_readme,
                   "README heading found" if has_readme else "No README structure"))

    has_install = bool(re.search(r"(install|pip install|npm install|setup|requirements)", output, re.I))
    checks.append(("Includes installation steps", has_install,
                   "Installation steps present" if has_install else "No installation instructions"))

    has_usage = bool(re.search(r"(usage|example|how to|getting started|quickstart)", output, re.I))
    checks.append(("Has usage examples", has_usage,
                   "Usage examples present" if has_usage else "No usage examples"))

    has_api_docs = bool(re.search(r"(endpoint|route|GET|POST|PUT|DELETE|parameter|request|response)", output, re.I))
    checks.append(("Documents API endpoints", has_api_docs,
                   "API docs present" if has_api_docs else "No API documentation"))

    has_dev_guide = bool(re.search(r"(extend|contribut|develop|architecture|how it works|structure)", output, re.I))
    checks.append(("Has developer guide", has_dev_guide,
                   "Developer guide present" if has_dev_guide else "No developer guide"))

    has_code_examples = bool(re.search(r"```", output))
    checks.append(("Includes code examples", has_code_examples,
                   "Code examples present" if has_code_examples else "No code examples in docs"))

    weights = [2.0, 2.5, 2.0, 2.0, 1.5, 1.0]
    score = _score_from_checks(checks, weights)
    passed = sum(1 for _, p, _ in checks if p)
    summary = f"{passed}/{len(checks)} checks passed — docs are {'complete' if score >= 75 else 'incomplete'}"
    return AgentScore("documenter", score, _grade(score), checks, summary)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-level aggregator
# ─────────────────────────────────────────────────────────────────────────────

def compute_pipeline_accuracy(
    outputs: Dict[str, str],
    review_iterations: int = 1,
) -> Dict[str, AgentScore]:
    """
    Score all available agent outputs.
    Returns dict of { stage_name: AgentScore }.
    """
    scorers = {
        "planner":    lambda o: score_planner(o),
        "architect":  lambda o: score_architect(o),
        "developer":  lambda o: score_developer(o),
        "reviewer":   lambda o: score_reviewer(o, review_iterations),
        "tester":     lambda o: score_tester(o),
        "documenter": lambda o: score_documenter(o),
    }
    results = {}
    for stage, output in outputs.items():
        if stage in scorers and output and not output.startswith("[BLOCKED]") and not output.startswith("[ERROR]"):
            results[stage] = scorers[stage](output)
    return results


def overall_pipeline_score(scores: Dict[str, AgentScore]) -> float:
    """Weighted average across all agent scores."""
    weights = {
        "planner": 1.0, "architect": 1.0, "developer": 2.0,
        "reviewer": 1.5, "tester": 1.5, "documenter": 1.0,
    }
    total_w = sum(weights[s] for s in scores if s in weights)
    earned  = sum(scores[s].score * weights[s] for s in scores if s in weights)
    return round(earned / total_w, 1) if total_w else 0.0