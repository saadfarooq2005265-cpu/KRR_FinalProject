"""
validate.py
-----------
Checks the user's inputs before any fuzzy work is done.
Returns clean values plus a list of errors and warnings, so invalid
inputs fail with a clear message instead of crashing the program.
"""


def to_number(value, name, errors):
    """Convert text or number to float; record an error if it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number (got '{value}').")
        return None


def to_yes_no(value, name, errors):
    """Accept True/False, 1/0, 'yes'/'no' and return a Python bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "yes", "y", "true"):
        return True
    if text in ("0", "no", "n", "false"):
        return False
    errors.append(f"{name} must be yes or no (got '{value}').")
    return None


def validate_inputs(raw):
    """Check every input. Returns (clean_inputs, errors, warnings)."""
    errors = []
    warnings = []
    clean = {}

    # --- Semester (whole number, 1 or more) ---
    sem = to_number(raw.get("semester"), "Semester", errors)
    if sem is not None:
        if sem < 1 or sem > 8 or sem != int(sem):
            errors.append("Semester must be a whole number between 1 and 8.")
        else:
            clean["semester"] = int(sem)

    # --- GPA (0 to 4). Not needed in semester 1 (no previous semester). ---
    if clean.get("semester") == 1 and raw.get("gpa") in (None, ""):
        clean["gpa"] = None
    else:
        gpa = to_number(raw.get("gpa"), "GPA", errors)
        if gpa is not None:
            if gpa < 0 or gpa > 4:
                errors.append("GPA must be between 0.00 and 4.00.")
            else:
                clean["gpa"] = gpa

    # --- Monthly household income in PKR thousand ---
    income = to_number(raw.get("income"), "Income", errors)
    if income is not None:
        if income < 0:
            errors.append("Income cannot be negative.")
        elif income > 300:
            # Above the universe: treat as 300 (fully 'High') and warn.
            warnings.append("Income above 300k treated as 300k (fully High).")
            clean["income"] = 300.0
        else:
            clean["income"] = income

    # --- Preference criteria met: 0, 1 or 2 (disability, less developed area) ---
    pref = to_number(raw.get("preference_count", 0), "Preference count", errors)
    if pref is not None:
        if pref not in (0, 1, 2):
            errors.append("Preference count must be 0, 1 or 2.")
        else:
            clean["preference_count"] = int(pref)

    # --- Batch position (whole number, 1 or more) ---
    pos = to_number(raw.get("position"), "Position", errors)
    if pos is not None:
        if pos < 1 or pos != int(pos):
            errors.append("Position must be a whole number of 1 or more.")
        else:
            clean["position"] = int(pos)

    # --- Level: UG or PG ---
    level = str(raw.get("level", "")).strip().upper()
    if level not in ("UG", "PG"):
        errors.append("Level must be UG or PG.")
    else:
        clean["level"] = level

    # --- Yes/no flags ---
    for key, label in [("engineering", "Engineering programme"),
                       ("regular_load", "Regular course load"),
                       ("summer_or_final", "Summer or final semester"),
                       ("other_benefit", "Already has another scholarship/concession"),
                       ("international", "International student")]:
        flag = to_yes_no(raw.get(key), label, errors)
        if flag is not None:
            clean[key] = flag

    return clean, errors, warnings
