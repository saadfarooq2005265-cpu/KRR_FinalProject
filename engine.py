"""
engine.py
---------
The shared Mamdani steps used by both the merit and the FAS systems:
  implication (clip) -> aggregation (max) -> defuzzification (centroid)
plus the linguistic label and the combination rules C1-C4.
"""

from membership import membership

GRID_POINTS = 2001   # number of sample points on each output universe


# ---------------------------------------------------------------------------
# Fuzzy operators
# ---------------------------------------------------------------------------

def fuzzy_and(*values):
    """AND = minimum."""
    return min(values)


def fuzzy_or(*values):
    """OR = maximum."""
    return max(values)


def fuzzy_not(value):
    """NOT = 1 - membership."""
    return 1.0 - value


# ---------------------------------------------------------------------------
# Output grid
# ---------------------------------------------------------------------------

def make_grid(universe):
    """List of evenly spaced x values covering the output universe."""
    low, high = universe
    step = (high - low) / (GRID_POINTS - 1)
    return [low + i * step for i in range(GRID_POINTS)]


# ---------------------------------------------------------------------------
# Mamdani steps
# ---------------------------------------------------------------------------

def mamdani(fired_rules, output_variable):
    """fired_rules: list of (rule_id, strength, output_term).
    Clips each rule's output set at its strength (implication = min),
    joins all clipped sets (aggregation = max) and returns
    (grid, aggregated_shape)."""
    grid = make_grid(output_variable["universe"])
    aggregated = [0.0] * len(grid)

    for rule_id, strength, term in fired_rules:
        if strength <= 0:
            continue
        shape, params = output_variable["terms"][term]
        for i, x in enumerate(grid):
            clipped = min(strength, membership(x, shape, params))   # implication
            if clipped > aggregated[i]:                             # aggregation
                aggregated[i] = clipped
    return grid, aggregated


def centroid(grid, shape):
    """Centre of gravity: sum(x * mu) / sum(mu).
    Returns None if no rule fired (empty shape), to avoid dividing by zero."""
    top = 0.0
    bottom = 0.0
    for x, mu in zip(grid, shape):
        top += x * mu
        bottom += mu
    if bottom == 0:
        return None
    return top / bottom


def linguistic_label(value, output_variable):
    """Name of the output term with the highest membership at `value`.
    If every term is 0 there (between narrow sets), use the closest peak."""
    best_term, best_mu = None, 0.0
    for term, (shape, params) in output_variable["terms"].items():
        mu = membership(value, shape, params)
        if mu > best_mu:
            best_term, best_mu = term, mu
    if best_term is not None:
        return best_term

    # Fallback: closest peak (middle parameter of the triangle)
    closest, distance = None, None
    for term, (shape, params) in output_variable["terms"].items():
        peak = params[1]
        if distance is None or abs(value - peak) < distance:
            closest, distance = term, abs(value - peak)
    return closest


def run_inference(fired_rules, output_variable):
    """Full Mamdani run for one output. Returns (crisp_value, label)."""
    grid, shape = mamdani(fired_rules, output_variable)
    crisp = centroid(grid, shape)
    if crisp is None:
        return None, "No rule fired"
    return round(crisp, 1), linguistic_label(crisp, output_variable)


# ---------------------------------------------------------------------------
# Combination layer: "only one scholarship" (handbook 4.12 c, e)
# ---------------------------------------------------------------------------

def combine(merit_award, fas_award, other_benefit, international):
    """Rules C1-C4. Awards are crisp percentages (0 if not granted)."""
    flags = []

    # C1: already has another scholarship or concession -> nothing new
    if other_benefit:
        return "No scholarship", "C1: already receives another scholarship/concession (4.12 c)", flags

    merit_ok = merit_award is not None and merit_award >= 1
    fas_ok = fas_award is not None and fas_award >= 1

    # C2: qualifies for both -> recommend the larger one, student chooses
    if merit_ok and fas_ok:
        if merit_award >= fas_award:
            decision = f"Merit Scholarship {merit_award:.0f}%"
        else:
            decision = f"FAS {fas_award:.0f}%"
        reason = "C2: eligible for both; only one allowed, larger recommended (student confirms)"
    # C3: qualifies for exactly one -> that one
    elif merit_ok:
        decision, reason = f"Merit Scholarship {merit_award:.0f}%", "C3: eligible for Merit only"
    elif fas_ok:
        decision, reason = f"FAS {fas_award:.0f}%", "C3: eligible for FAS only"
    else:
        decision, reason = "No scholarship", "Not eligible for Merit or FAS"

    # C4: international student getting FAS needs Rector approval
    if international and decision.startswith("FAS"):
        flags.append("C4: special approval from the Honorable Rector needed (4.12 e)")

    return decision, reason, flags
