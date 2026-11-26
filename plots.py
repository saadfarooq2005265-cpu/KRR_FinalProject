"""
plots.py
--------
Optional: draws the membership functions and the FAS output surface.
Needs matplotlib (pip install matplotlib).

Run:  python plots.py
Saves membership_functions.png and fas_surface.png in this folder.
"""

import matplotlib.pyplot as plt

from membership import GPA, INCOME, POSITION, FAS_AWARD, MERIT_AWARD, membership
from engine import make_grid
from fas import evaluate_fas


def plot_variable(ax, variable, title, x_range=None):
    """Draw every term of one variable on one set of axes."""
    xs = make_grid(x_range or variable["universe"])
    for term, (shape, params) in variable["terms"].items():
        ys = [membership(x, shape, params) for x in xs]
        ax.plot(xs, ys, label=term, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("membership")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_all_membership_functions():
    """One figure with all five variables."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    plot_variable(axes[0][0], GPA, "GPA (shared)")
    plot_variable(axes[0][1], INCOME, "Monthly household income (PKR thousand)")
    plot_variable(axes[1][0], POSITION, "Batch position (zoom 0-8)", (0, 8))
    plot_variable(axes[1][1], FAS_AWARD, "FAS award (% of tuition)")
    plot_variable(axes[2][0], MERIT_AWARD, "Merit award (%)")
    plot_variable(axes[2][1], GPA, "GPA zoom 3.40-4.00", (3.4, 4.0))
    plt.tight_layout()
    plt.savefig("membership_functions.png", dpi=120)
    print("Saved membership_functions.png")


def plot_fas_surface():
    """FAS award for every (GPA, income) pair, no preference, semester 3."""
    gpas = [2.0 + i * 0.02 for i in range(51)]        # 2.00 to 3.00
    incomes = [i * 3 for i in range(61)]                # 0 to 180
    z = []
    for g in gpas:
        row = []
        for inc in incomes:
            result = evaluate_fas({"gpa": g, "income": inc,
                                   "preference_count": 0, "semester": 3})
            row.append(result["crisp"] or 0)
        z.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    picture = ax.contourf(incomes, gpas, z, levels=20, cmap="viridis")
    fig.colorbar(picture, label="FAS award (%)")
    ax.set_xlabel("Monthly household income (PKR thousand)")
    ax.set_ylabel("GPA")
    ax.set_title("FAS output surface (no preference)")
    plt.savefig("fas_surface.png", dpi=120)
    print("Saved fas_surface.png")


if __name__ == "__main__":
    plot_all_membership_functions()
    plot_fas_surface()
