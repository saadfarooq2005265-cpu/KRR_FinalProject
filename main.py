"""
main.py
-------
Entry point. `evaluate(inputs)` runs the full system and returns one
result dictionary (this is the function the GUI and the tests call).

Run:
    python main.py          -> asks for inputs in the terminal
    python main.py --demo   -> runs a few sample students
"""

import sys

from validate import validate_inputs
from merit import evaluate_merit
from fas import evaluate_fas
from engine import combine


def evaluate(raw_inputs):
    """Validate -> Merit system -> FAS system -> combine (C1-C4)."""
    inputs, errors, warnings = validate_inputs(raw_inputs)
    if errors:
        return {"errors": errors, "warnings": warnings}

    merit = evaluate_merit(inputs)
    fas = evaluate_fas(inputs)

    # An output whose label is 'None' means "not awarded", even though the
    # centroid of the None set is a small number (e.g. 5.4%). Use 0 for those.
    merit_award = 0.0 if merit["label"].startswith("None") else merit["crisp"]
    fas_award = 0.0 if fas["label"].startswith(("None", "No rule")) else fas["crisp"]

    decision, reason, flags = combine(merit_award, fas_award,
                                      inputs["other_benefit"], inputs["international"])
    return {
        "errors": [],
        "warnings": warnings,
        "merit": merit,
        "fas": fas,
        "final": {"decision": decision, "reason": reason, "flags": flags},
    }


# ---------------------------------------------------------------------------
# Printing the result in the terminal
# ---------------------------------------------------------------------------

def print_result(result):
    """Show crisp output, label and fired rules in a readable way."""
    if result["errors"]:
        print("\nINVALID INPUT:")
        for e in result["errors"]:
            print("  -", e)
        return

    for w in result["warnings"]:
        print("  Warning:", w)

    m = result["merit"]
    print("\nMERIT SCHOLARSHIP")
    if m["gate"]:
        print("  Blocked by gate:", m["gate"])
    print(f"  Award: {m['crisp']}%  ({m['label']})")
    for rid, strength, term in m["fired_rules"]:
        print(f"    {rid} fired at {strength} -> {term}")
    if m["review"]:
        print(f"  If already a merit holder, review (4.11): {m['review']['decision']}")

    f = result["fas"]
    print("\nFINANCIAL ASSISTANCE SCHOLARSHIP")
    print(f"  Award: {f['crisp']}%  ({f['label']})")
    for rid, strength, term in f["fired_rules"]:
        print(f"    {rid} fired at {strength} -> {term}")
    print("  Note:", f["note"])

    fin = result["final"]
    print("\nFINAL DECISION")
    print("  ", fin["decision"])
    print("  Reason:", fin["reason"])
    for flag in fin["flags"]:
        print("  Flag:", flag)


# ---------------------------------------------------------------------------
# Terminal input and demo cases
# ---------------------------------------------------------------------------

def ask_inputs():
    """Read every input from the keyboard."""
    print("Enter student details (yes/no answers accepted as y/n):")
    return {
        "semester": input("Current semester number: "),
        "gpa": input("Previous semester GPA (0-4, blank if semester 1): "),
        "income": input("Monthly household income (PKR thousand): "),
        "preference_count": input("Preference criteria met (0, 1 or 2): "),
        "position": input("Batch position (1, 2, 3, ...): "),
        "level": input("Level (UG or PG): "),
        "engineering": input("Engineering programme? "),
        "regular_load": input("Regular course load? "),
        "summer_or_final": input("Summer or final semester? "),
        "other_benefit": input("Already has another scholarship/concession? "),
        "international": input("International student? "),
    }


DEMO_STUDENTS = [
    ("1st position, non-engineering, GPA 3.80, high income",
     dict(semester=4, gpa=3.80, income=150, preference_count=0, position=1, level="UG",
          engineering=False, regular_load=True, summer_or_final=False,
          other_benefit=False, international=False)),
    ("Borderline: 1st position, non-engineering, GPA 3.73",
     dict(semester=4, gpa=3.73, income=150, preference_count=0, position=1, level="UG",
          engineering=False, regular_load=True, summer_or_final=False,
          other_benefit=False, international=False)),
    ("FAS: GPA 3.00, income 55k, no preference",
     dict(semester=3, gpa=3.00, income=55, preference_count=0, position=20, level="UG",
          engineering=True, regular_load=True, summer_or_final=False,
          other_benefit=False, international=False)),
    ("Invalid input: GPA 4.5",
     dict(semester=3, gpa=4.5, income=55, preference_count=0, position=20, level="UG",
          engineering=True, regular_load=True, summer_or_final=False,
          other_benefit=False, international=False)),
]


if __name__ == "__main__":
    if "--demo" in sys.argv:
        for title, student in DEMO_STUDENTS:
            print("\n" + "=" * 60)
            print(title)
            print("=" * 60)
            print_result(evaluate(student))
    else:
        print_result(evaluate(ask_inputs()))
