"""
utils/scorer.py
Aviation-specific compliance scoring and critical field checking.
"""

import re

WEIGHTS = {
    "exact":  0.35,
    "bert":   0.25,
    "rouge":  0.20,
    "meteor": 0.10,
    "bleu":   0.10,
}

# Critical aviation fields that MUST appear in the summary
CRITICAL_FIELDS = {
    "Flight number":      [r"flight\s*\d+", r"[A-Z]{2,3}\d{2,4}"],
    "Aircraft type":      [r"boeing|airbus|cessna|embraer|bombardier|aircraft\s*type|[A-Z]\d{3}"],
    "Date of incident":   [r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", r"january|february|march|april|may|june|july|august|september|october|november|december"],
    "Time of incident":   [r"\d{1,2}:\d{2}", r"\d{4}\s*hours?", r"UTC|GMT|local\s*time"],
    "Location/Airport":   [r"airport|runway|[A-Z]{3,4}\s*airport|ICAO|IATA"],
    "Altitude":           [r"\d+\s*ft|feet|\d+\s*m\s*(AGL|MSL|altitude)|flight\s*level|FL\d+"],
    "Weather conditions": [r"weather|visibility|cloud|wind|turbulence|fog|rain|snow|VMC|IMC"],
    "Crew actions":       [r"pilot|captain|crew|co-pilot|declared|executed|initiated|performed"],
    "Injuries reported":  [r"injur|casualt|fatality|fatalities|no\s*injur|minor|serious|fatal"],
    "Damage assessment":  [r"damage|undamaged|minor\s*damage|substantial|destroyed|hull\s*loss"],
}


def _field_present(text: str, patterns: list) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def aviation_safety_score(scores: dict) -> tuple:
    safety = round(sum(WEIGHTS[k] * scores[k] for k in WEIGHTS), 1)
    if safety >= 85:
        return safety, "COMPLIANT — Safe to submit to regulators", "cleared"
    elif safety >= 60:
        return safety, "REVIEW REQUIRED — Human verification needed", "review"
    else:
        return safety, "NON-COMPLIANT — Do not submit", "grounded"


def get_aviation_flags(candidate: str, reference: str, scores: dict) -> dict:
    critical = []
    warnings = []

    # Critical field checklist
    checklist = {}
    for field, patterns in CRITICAL_FIELDS.items():
        in_ref = _field_present(reference, patterns)
        in_can = _field_present(candidate, patterns)
        checklist[field] = in_can
        if in_ref and not in_can:
            critical.append(
                f"'{field}' is present in pilot report but MISSING from AI summary"
            )

    # Number mismatch check
    ref_nums = set(re.findall(r'\b\d+\.?\d*\b', reference))
    can_nums = set(re.findall(r'\b\d+\.?\d*\b', candidate))
    missing  = ref_nums - can_nums
    extra    = can_nums - ref_nums

    if missing:
        critical.append(
            f"Critical values missing from summary: {', '.join(sorted(missing)[:6])}"
        )
    if extra:
        warnings.append(
            f"Unverified values in summary (not in pilot report): "
            f"{', '.join(sorted(extra)[:6])}"
        )

    # Metric-based flags
    if scores["exact"] < 50:
        critical.append(
            f"Exact Match very low ({scores['exact']}%) — "
            "flight numbers, altitudes, and times may be wrong"
        )
    elif scores["exact"] < 75:
        warnings.append(
            f"Exact Match moderate ({scores['exact']}%) — "
            "verify all numerical values manually"
        )

    if scores["bert"] < 50:
        critical.append(
            f"BERTScore very low ({scores['bert']}%) — "
            "summary meaning significantly differs from original report"
        )
    elif scores["bert"] < 70:
        warnings.append(
            f"BERTScore moderate ({scores['bert']}%) — "
            "some sections may have changed in meaning"
        )

    if scores["rouge"] < 50:
        critical.append(
            f"ROUGE score low ({scores['rouge']}%) — "
            "summary is missing major sections from the incident report"
        )
    elif scores["rouge"] < 70:
        warnings.append(
            f"ROUGE score moderate ({scores['rouge']}%) — "
            "some incident details may have been omitted"
        )

    if scores["bleu"] < 30:
        warnings.append(
            f"BLEU score low ({scores['bleu']}%) — "
            "phrasing differs significantly from official report language"
        )

    # Length check
    ref_words = len(reference.split())
    can_words = len(candidate.split())
    if can_words < ref_words * 0.15:
        critical.append(
            f"Summary is extremely short ({can_words} words) vs "
            f"original report ({ref_words} words) — critical details likely omitted"
        )
    elif can_words < ref_words * 0.30:
        warnings.append(
            f"Summary may be too brief ({can_words} words vs {ref_words} words in report)"
        )

    return {"critical": critical, "warnings": warnings, "checklist": checklist}
