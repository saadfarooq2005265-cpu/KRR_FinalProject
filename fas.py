"""
fas.py
------
Financial Assistance Scholarship (handbook 4.5, 4.6, 4.12).
Rules F1-F6 (Salman's rule base).
"""

from membership import GPA, INCOME, FAS_AWARD, fuzzify
from engine import fuzzy_and, run_inference


def fas_gpa_terms(gpa, semester):
    """GPA terms used by FAS: Low and Eligible (= 1 - Low).
    Semester 1: no previous GPA, so the GPA check is skipped (FAS-05)."""
    if semester == 1:
        return {"Low": 0.0, "Eligible": 1.0}
    low = fuzzify(gpa, GPA)["Low"]
    return {"Low": low, "Eligible": round(1.0 - low, 4)}


def fas_rules(gpa_terms, income_terms, preferred):
    """Evaluate F1-F6. Returns a list of (rule_id, strength, output_term).
    `preferred` is 1 if at least one preference criterion is met, else 0."""
    eligible = gpa_terms["Eligible"]
    not_preferred = 1.0 - preferred

    rules = [
        # F1: GPA below 2.50 -> withheld
        ("F1", gpa_terms["Low"], "None"),
        # F2: no financial constraint
        ("F2", fuzzy_and(eligible, income_terms["High"]), "None"),
        # F3: medium income, no preference
        ("F3", fuzzy_and(eligible, income_terms["Medium"], not_preferred), "Partial"),
        # F4: medium income, preferred
        ("F4", fuzzy_and(eligible, income_terms["Medium"], preferred), "Large"),
        # F5: low income, no preference
        ("F5", fuzzy_and(eligible, income_terms["Low"], not_preferred), "Large"),
        # F6: low income, preferred
        ("F6", fuzzy_and(eligible, income_terms["Low"], preferred), "Full"),
    ]
    return rules


def evaluate_fas(inputs):
    """Run the whole FAS system. Returns a result dictionary for the GUI."""
    gpa_terms = fas_gpa_terms(inputs["gpa"], inputs["semester"])
    income_terms = fuzzify(inputs["income"], INCOME)
    preferred = 1.0 if inputs["preference_count"] >= 1 else 0.0

    rules = fas_rules(gpa_terms, income_terms, preferred)
    fired = [r for r in rules if r[1] > 0]

    crisp, label = run_inference(fired, FAS_AWARD)

    # Borderline: GPA sits inside the 2.45-2.50 band (partly Low, partly Eligible)
    if 0 < gpa_terms["Low"] < 1:
        label = label + " (Borderline GPA)"

    return {
        "crisp": crisp,
        "label": label,
        "fired_rules": [(rid, round(s, 3), term) for rid, s, term in fired],
        "memberships": {"GPA": gpa_terms, "Income": income_terms},
        "note": "Recommended % of tuition; final % decided by the scholarship committee (4.6.2).",
    }
