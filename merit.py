"""
merit.py
--------
Merit Scholarship for Batch Toppers (handbook 4.4) and the semester
review of scholarship holders (4.11). Rules A1-A9 and R1-R2 (Saad's rule base).
"""

from membership import GPA, POSITION, MERIT_AWARD, fuzzify
from engine import fuzzy_and, fuzzy_not, run_inference


# ---------------------------------------------------------------------------
# Crisp gates (yes/no conditions checked before the fuzzy step)
# ---------------------------------------------------------------------------

def merit_gates(inputs):
    """Return the gate that blocks the merit award, or None if all pass.
    A7, A8, A9 are yes/no rules, so they act as gates (award = 0%)."""
    if inputs["semester"] == 1:
        return "Not applicable: merit uses previous-semester GPA (4.4.3.2)"
    if not inputs["regular_load"]:
        return "A7: no regular course load (4.4.3.4)"
    if inputs["summer_or_final"]:
        return "A8: no merit award in summer/final semester (4.4.3.7)"
    if inputs["other_benefit"]:
        return "A9: already has another scholarship/concession (4.4.3.5-4.4.3.8)"
    return None


# ---------------------------------------------------------------------------
# GPA_Met: degree to which the GPA meets the programme's cut-off
# ---------------------------------------------------------------------------

def gpa_met(gpa_terms, engineering):
    """Engineering: GPA >= 3.60 -> High or VeryHigh (bounded sum).
    Non-engineering: GPA >= 3.75 -> VeryHigh."""
    if engineering:
        return min(1.0, gpa_terms["High"] + gpa_terms["VeryHigh"])
    return gpa_terms["VeryHigh"]


# ---------------------------------------------------------------------------
# Rule base
# ---------------------------------------------------------------------------

def merit_rules(pos, met, is_ug):
    """Evaluate A1-A6. Returns a list of (rule_id, strength, output_term).
    `is_ug` is 1 for undergraduate, 0 for postgraduate."""
    is_pg = 1.0 - is_ug
    rules = [
        ("A1", fuzzy_and(pos["First"], met), "Full70"),
        ("A2", fuzzy_and(pos["Second"], met), "Award40"),
        ("A3", fuzzy_and(pos["Third"], is_ug, met), "Award30"),
        ("A4", fuzzy_and(pos["Third"], is_pg), "None"),     # PG: top two only
        ("A5", pos["Beyond"], "None"),                       # position 4 or lower
        ("A6", fuzzy_not(met), "None"),                      # GPA cut-off not met
    ]
    return rules


def evaluate_merit(inputs):
    """Run the whole merit system. Returns a result dictionary for the GUI."""
    gate = merit_gates(inputs)
    if gate is not None:
        return {"crisp": 0.0, "label": "None", "fired_rules": [], "gate": gate,
                "review": None, "memberships": {}}

    gpa_terms = fuzzify(inputs["gpa"], GPA)
    pos_terms = fuzzify(inputs["position"], POSITION)
    met = gpa_met(gpa_terms, inputs["engineering"])
    is_ug = 1.0 if inputs["level"] == "UG" else 0.0

    rules = merit_rules(pos_terms, met, is_ug)
    fired = [r for r in rules if r[1] > 0]
    crisp, label = run_inference(fired, MERIT_AWARD)

    # Borderline: GPA_Met is between 0 and 1 (inside the 0.05 band below the cut-off)
    if 0 < met < 1:
        label = label + " (Borderline GPA)"

    return {
        "crisp": max(0.0, crisp) if crisp is not None else 0.0,
        "label": label,
        "fired_rules": [(rid, round(s, 3), term) for rid, s, term in fired],
        "gate": None,
        "review": evaluate_continuation(met),
        "memberships": {"GPA": gpa_terms, "Position": pos_terms, "GPA_Met": round(met, 3)},
    }


# ---------------------------------------------------------------------------
# Review of existing holders (4.11): R1 Continue, R2 Suspend
# ---------------------------------------------------------------------------

def evaluate_continuation(met):
    """R1: IF GPA_Met THEN Continue (strength = met).
    R2: IF NOT GPA_Met THEN Suspend (strength = 1 - met).
    The stronger rule decides."""
    r1 = met
    r2 = fuzzy_not(met)
    decision = "Continue" if r1 >= r2 else "Suspend"
    return {"decision": decision, "R1": round(r1, 3), "R2": round(r2, 3)}
