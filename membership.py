"""
membership.py
-------------
Membership function shapes and the parameters of every fuzzy variable.

All numbers that define the fuzzy sets live in this one file, so a change
here updates the whole system.
"""


# ---------------------------------------------------------------------------
# 1. Shape functions
# ---------------------------------------------------------------------------

def trimf(x, a, b, c):
    """Triangle: 0 at a, rises to 1 at b, falls back to 0 at c."""
    if x <= a or x >= c:
        # Special case: a peak sitting on the edge (a == b or b == c)
        if x == b:
            return 1.0
        return 0.0
    if x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapmf(x, a, b, c, d):
    """Trapezoid: 0 at a, rises to 1 at b, flat to c, falls to 0 at d.
    a == b gives a left 'shoulder' (already 1 at the left edge),
    c == d gives a right shoulder."""
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:                      # rising edge
        if b == a:
            return 1.0
        return (x - a) / (b - a)
    if d == c:                     # falling edge
        return 1.0
    return (d - x) / (d - c)


def membership(x, shape, params):
    """Call the right shape function for one fuzzy term."""
    if shape == "tri":
        return trimf(x, *params)
    return trapmf(x, *params)


# ---------------------------------------------------------------------------
# 2. Variable definitions
#    Each variable: universe (min, max) and its terms -> (shape, parameters)
# ---------------------------------------------------------------------------

# Width of the fuzzy band just below every GPA cut-off (2.50, 3.60, 3.75)
NEAR_MISS = 0.05

GPA = {
    "universe": (0.0, 4.0),
    "terms": {
        "Low":      ("trap", (0.00, 0.00, 2.45, 2.50)),   # below FAS cut-off 2.50
        "Moderate": ("trap", (2.45, 2.50, 3.55, 3.60)),   # 2.50 to below 3.60
        "High":     ("trap", (3.55, 3.60, 3.70, 3.75)),   # meets engineering cut-off 3.60
        "VeryHigh": ("trap", (3.70, 3.75, 4.00, 4.00)),   # meets non-engineering cut-off 3.75
    },
}

# Monthly household income in PKR thousand (team's own assumption, not handbook values)
INCOME = {
    "universe": (0.0, 300.0),
    "terms": {
        "Low":    ("trap", (0, 0, 40, 70)),
        "Medium": ("trap", (40, 70, 100, 130)),
        "High":   ("trap", (100, 130, 300, 300)),
    },
}

# Batch position (rank 1, 2, 3, ...)
POSITION = {
    "universe": (1.0, 100.0),
    "terms": {
        "First":  ("tri",  (0, 1, 2)),
        "Second": ("tri",  (1, 2, 3)),
        "Third":  ("tri",  (2, 3, 4)),
        "Beyond": ("trap", (3, 4, 100, 100)),
    },
}

# Output: FAS award as % of tuition fee
FAS_AWARD = {
    "universe": (0.0, 100.0),
    "terms": {
        "None":    ("trap", (0, 0, 5, 15)),
        "Partial": ("tri",  (10, 30, 50)),
        "Large":   ("tri",  (40, 65, 90)),
        "Full":    ("trap", (80, 95, 100, 100)),
    },
}

# Output: merit award %. Narrow symmetric triangles, universe extended to
# -10..80 so a single fired rule gives exactly 0, 30, 40 or 70.
MERIT_AWARD = {
    "universe": (-10.0, 80.0),
    "terms": {
        "None":    ("tri", (-5, 0, 5)),
        "Award30": ("tri", (25, 30, 35)),
        "Award40": ("tri", (35, 40, 45)),
        "Full70":  ("tri", (65, 70, 75)),
    },
}


# ---------------------------------------------------------------------------
# 3. Fuzzification
# ---------------------------------------------------------------------------

def fuzzify(value, variable):
    """Return the membership degree of `value` in every term of `variable`,
    e.g. {'Low': 0.5, 'Medium': 0.5, 'High': 0.0}."""
    result = {}
    for term_name, (shape, params) in variable["terms"].items():
        result[term_name] = round(membership(value, shape, params), 4)
    return result
