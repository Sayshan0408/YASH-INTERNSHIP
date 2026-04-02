"""
utils/report.py
Generates downloadable plain-text compliance report.
"""

from datetime import datetime


def generate_report_text(scores: dict, safety: float, status: str,
                         flags: dict, preview: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 62,
        "   AVIATION INCIDENT REPORT COMPLIANCE CHECKER",
        "   Regulatory Summary Verification Report",
        "=" * 62,
        f"Generated        : {now}",
        f"Compliance Score : {safety}/100",
        f"Status           : {status}",
        "",
        "-" * 62,
        "METRIC BREAKDOWN",
        "-" * 62,
        f"Exact Match  (weight 35%) : {scores['exact']:.1f}/100",
        f"BERTScore    (weight 25%) : {scores['bert']:.1f}/100",
        f"ROUGE-L      (weight 20%) : {scores['rouge']:.1f}/100",
        f"METEOR       (weight 10%) : {scores['meteor']:.1f}/100",
        f"BLEU         (weight 10%) : {scores['bleu']:.1f}/100",
        "",
        "-" * 62,
        "CRITICAL FIELD CHECKLIST",
        "-" * 62,
    ]

    checklist = flags.get("checklist", {})
    for field, present in checklist.items():
        mark = "PRESENT" if present else "MISSING"
        lines.append(f"  [{mark:7s}] {field}")

    lines.append("")

    if flags["critical"]:
        lines += ["-" * 62, "CRITICAL ISSUES (must fix before submission)", "-" * 62]
        for f in flags["critical"]:
            lines.append(f"  [CRITICAL] {f}")
        lines.append("")

    if flags["warnings"]:
        lines += ["-" * 62, "WARNINGS", "-" * 62]
        for f in flags["warnings"]:
            lines.append(f"  [WARNING] {f}")
        lines.append("")

    if not flags["critical"] and not flags["warnings"]:
        lines += ["-" * 62, "No critical issues found.", ""]

    lines += [
        "-" * 62,
        "SUMMARY PREVIEW (first 200 chars)",
        "-" * 62,
        preview,
        "",
        "=" * 62,
        "END OF REPORT",
        "=" * 62,
    ]
    return "\n".join(lines)
