"""
gui.py
------
Tkinter (ttk) front end for the Merit / FAS fuzzy inference system.

The GUI only collects inputs and shows results. All validation and fuzzy
work is done by main.evaluate() (validate.py, merit.py, fas.py, engine.py).

Run:
    python gui.py
"""

import math
import os
import time
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import font as tkfont

import numpy as np

# Embedded matplotlib (no pyplot windows)
import matplotlib
from matplotlib import colormaps, font_manager
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)

from main import evaluate
from membership import GPA, INCOME, POSITION, FAS_AWARD, MERIT_AWARD, NEAR_MISS, membership
from engine import make_grid, mamdani
from fas import evaluate_fas
from merit import evaluate_merit

# Position universe ends at 100 (see POSITION in membership.py); a larger rank
# would have zero membership in every term, so the GUI rejects it up front.
POSITION_MAX = int(POSITION["universe"][1])   # 100


# ---------------------------------------------------------------------------
# Colours and fonts
# ---------------------------------------------------------------------------

FONT = "Segoe UI"

# Use the same font in the plots when it is installed (Windows); otherwise
# matplotlib keeps its default, so no "font not found" warnings elsewhere.
if FONT in {f.name for f in font_manager.fontManager.ttflist}:
    matplotlib.rcParams["font.family"] = [FONT, "DejaVu Sans"]

CARD_BG = "white"
CARD_BORDER = "#c8c6c0"
TEXT = "#222222"
MUTED = "#6b6b6b"

GREEN = "#1e7d32"     # an award was given
GREY = "#7a7a7a"      # None / not awarded
AMBER = "#b26a00"     # borderline GPA
ORANGE = "#d35400"    # warnings and flags (e.g. C4)
RED = "#c62828"       # errors, gate messages, Suspend

ERROR_BG = "#fdecea"  # background of the error box and of invalid fields
AMBER_BG = "#fff4e0"  # borderline-GPA banner background
AMBER_TEXT = "#7a4a00"  # dark amber text on that banner (readable contrast)
TOP_RULE_BG = "#e3f2e5"   # highlight of the strongest rule in a table

# --- Plot colours ---
# Term colours in a fixed order (1st term blue, 2nd orange, 3rd aqua, 4th yellow).
# Colour-blind-checked categorical palette; red is kept for the student marker.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STUDENT_RED = "#d62728"   # dashed line at the student's input value
AGG_FILL = "#52514e"      # neutral grey for the aggregated output shape
CUTOFF_GREY = "#8a8984"   # handbook cut-off lines and borderline bands

# Handbook GPA cut-offs = left edge of the Moderate / High / VeryHigh plateaus
# in membership.py (2.50 FAS, 3.60 engineering, 3.75 non-engineering).
CUT_FAS, CUT_ENG, CUT_NON_ENG = (GPA["terms"][t][1][1] for t in ("Moderate", "High", "VeryHigh"))

# Single-hue sequential colormap for award % (light = low, dark = high).
# The palest 12% of 'Blues' is cut so 0% is still visible on the white figure.
AWARD_CMAP = ListedColormap(colormaps["Blues"](np.linspace(0.12, 1.0, 256)))

# --- Output surface tab: plotted ranges ---
SURF_INCOME = (0.0, 200.0)      # PKR thousand
SURF_GPA = (2.0, 4.0)
SURF_POINTS = 33                # evenly spaced points per axis (+ breakpoints below)
HEAT_GPA = (3.40, 4.00)
HEAT_POSITIONS = [1, 2, 3, 4, 5]
# Default 3D view: looking from the low-GPA / low-income corner, so the GPA
# 2.50 cliff faces the viewer and the income steps descend to the right.
SURF_VIEW = dict(elev=25, azim=-60)

# Maps the first word(s) of an error message to the input key,
# so the matching field can be highlighted in red.
ERROR_PREFIX_TO_FIELD = [
    ("Semester", "semester"),
    ("GPA", "gpa"),
    ("Income", "income"),
    ("Preference count", "preference_count"),
    ("Position", "position"),
    ("Level", "level"),
]

# Numeric text fields checked for nan/inf before calling evaluate()
NUMERIC_FIELDS = [
    ("semester", "Semester"),
    ("gpa", "GPA"),
    ("income", "Income"),
    ("position", "Position"),
]

# ---------------------------------------------------------------------------
# Sample cases for the "Load sample case" dropdown
# ---------------------------------------------------------------------------

def _case(name, slug, **fields):
    """One sample case: (display name, file-name slug, form values).
    Unspecified fields: regular load yes, other boxes no, UG, preference 0."""
    form = dict(semester="", gpa="", income="", position="", preference_count="0",
                level="UG", engineering=False, regular_load=True, summer_or_final=False,
                other_benefit=False, international=False)
    form.update(fields)
    return name, slug, form


SAMPLE_CASES = [
    _case("Merit topper", "merit_topper",
          semester="4", gpa="3.85", income="150", position="1"),
    _case("Borderline merit", "borderline_merit",
          semester="4", gpa="3.73", income="150", position="1"),
    _case("Engineering at cut-off", "engineering_cutoff",
          semester="4", gpa="3.60", income="150", position="2", engineering=True),
    _case("PG third position", "pg_third_position",
          semester="3", gpa="3.90", income="150", position="3", level="PG"),
    _case("FAS low income + preference", "fas_low_income_pref",
          semester="3", gpa="3.00", income="30", preference_count="1", position="20"),
    _case("FAS medium income", "fas_medium_income",
          semester="5", gpa="3.20", income="85", position="15"),
    _case("FAS borderline GPA", "fas_borderline_gpa",
          semester="4", gpa="2.48", income="30", position="25"),
    _case("Eligible for both (C2)", "both_c2",
          semester="4", gpa="3.80", income="30", position="1"),
    _case("International FAS (C4)", "international_c4",
          semester="3", gpa="3.10", income="35", position="12", international=True),
    _case("Already has concession (C1)", "concession_c1",
          semester="4", gpa="3.90", income="30", position="1", other_benefit=True),
    _case("Semester 1 student", "semester_1",
          semester="1", gpa="", income="40", position="1"),
    _case("Summer/final semester (A8)", "summer_final_a8",
          semester="8", gpa="3.90", income="150", position="1", summer_or_final=True),
    _case("Invalid input", "invalid_input",
          semester="4", gpa="4.5", income="50", position="0"),
]
CASE_PLACEHOLDER = "Choose a sample case…"

# Plain-English text of every rule (merit.py A1-A6, fas.py F1-F6)
RULE_TEXT = {
    "A1": "IF Position is First AND GPA meets cut-off THEN Full70 (70%)",
    "A2": "IF Position is Second AND GPA meets cut-off THEN Award40 (40%)",
    "A3": "IF Position is Third AND Level is UG AND GPA meets cut-off THEN Award30 (30%)",
    "A4": "IF Position is Third AND Level is PG THEN None (PG: top two only)",
    "A5": "IF Position is Beyond 3rd THEN None",
    "A6": "IF GPA does NOT meet cut-off THEN None",
    "F1": "IF GPA is Low (below 2.50) THEN None",
    "F2": "IF GPA Eligible AND Income High THEN None",
    "F3": "IF GPA Eligible AND Income Medium AND not preferred THEN Partial",
    "F4": "IF GPA Eligible AND Income Medium AND preferred THEN Large",
    "F5": "IF GPA Eligible AND Income Low AND not preferred THEN Large",
    "F6": "IF GPA Eligible AND Income Low AND preferred THEN Full",
}


# ---------------------------------------------------------------------------
# Small display helpers (formatting only, no fuzzy logic)
# ---------------------------------------------------------------------------

def is_not_awarded(crisp, label):
    """A 'None' label (or no fired rule) means no award, even though the
    centroid of the None set is a small number such as 5.4%."""
    return crisp is None or label.startswith(("None", "No rule fired"))


def award_text(crisp, label):
    """Text for the big percentage in a card."""
    if is_not_awarded(crisp, label):
        return "0% – Not awarded"
    return f"{crisp:.1f}%"


def label_colour(label):
    """Amber for borderline, grey for None / no rule, green for an award."""
    if "Borderline" in label:
        return AMBER
    if label.startswith(("None", "No rule fired")):
        return GREY
    return GREEN


def strength_bar(strength, width=10):
    """Text bar for a firing strength in [0, 1]: `width` blocks = 1.0, with a
    half block for the remainder (e.g. 0.6 -> ▓▓▓▓▓▓, 0.25 -> ▓▓▒)."""
    halves = round(strength * width * 2)
    full, half = divmod(halves, 2)
    return "▓" * full + ("▒" if half else "")


def award_value(crisp, label):
    """Award % as reported to the student: 0 when not awarded (see is_not_awarded)."""
    return 0.0 if is_not_awarded(crisp, label) else crisp


def axis_with_breakpoints(lo, hi, n, breakpoints):
    """n evenly spaced values in [lo, hi] plus the given breakpoints, so sharp
    edges (e.g. the 0.05-wide GPA band) are not skipped between grid points."""
    values = set(np.round(np.linspace(lo, hi, n), 4))
    values.update(b for b in breakpoints if lo <= b <= hi)
    return sorted(values)


# The two grid builders below are generators: they yield after every grid
# point so the GUI can run them in short slices between Tk events (the full
# FAS grid takes ~5 s). The finished grid is the generator's return value.

def compute_fas_surface(preferred):
    """FAS award % on a (GPA x income) grid, using fas.evaluate_fas()."""
    income_breaks = [p for _, params in INCOME["terms"].values() for p in params]
    incomes = axis_with_breakpoints(*SURF_INCOME, SURF_POINTS, income_breaks)
    gpas = axis_with_breakpoints(*SURF_GPA, SURF_POINTS, [CUT_FAS - NEAR_MISS, CUT_FAS])
    z = []
    for g in gpas:
        row = []
        for inc in incomes:
            r = evaluate_fas({"gpa": g, "income": inc,
                              "preference_count": 1 if preferred else 0, "semester": 3})
            row.append(award_value(r["crisp"], r["label"]))
            yield
        z.append(row)
    return {"incomes": np.array(incomes), "gpas": np.array(gpas), "z": np.array(z)}


def compute_merit_heatmap(engineering, level):
    """Merit award % on a (position x GPA) grid, using merit.evaluate_merit()."""
    gpas = np.round(np.arange(HEAT_GPA[0], HEAT_GPA[1] + 1e-9, 0.01), 2)
    z = []
    for pos in HEAT_POSITIONS:
        row = []
        for g in gpas:
            r = evaluate_merit({"gpa": float(g), "position": pos, "semester": 4,
                                "regular_load": True, "summer_or_final": False,
                                "other_benefit": False, "engineering": engineering,
                                "level": level})
            row.append(award_value(r["crisp"], r["label"]))
            yield
        z.append(row)
    return {"gpas": gpas, "positions": np.array(HEAT_POSITIONS), "z": np.array(z)}


def field_for_error(message):
    """Input key named at the start of an error message, or None."""
    for prefix, key in ERROR_PREFIX_TO_FIELD:
        if message.startswith(prefix):
            return key
    return None


# ---------------------------------------------------------------------------
# Scrollable container (Canvas + Scrollbar)
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    """A frame whose content (`self.inner`) scrolls vertically.
    The scrollbar only appears when the content is taller than the view.
    fit_width=True stretches the content to the canvas width; otherwise the
    canvas takes the content's natural width (used for the input form)."""

    def __init__(self, parent, bg, fit_width=False):
        super().__init__(parent)
        self.fit_width = fit_width

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, borderwidth=0,
                                yscrollincrement=20)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, _event):
        """Content changed size: update the scroll region (and width)."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if not self.fit_width:
            self.canvas.configure(width=self.inner.winfo_reqwidth())
        self._update_scrollbar()

    def _on_canvas_configure(self, event):
        """Window resized: stretch content to the new width if requested."""
        if self.fit_width:
            self.canvas.itemconfigure(self._window, width=event.width)
        self._update_scrollbar()

    def _needs_scroll(self):
        return self.inner.winfo_reqheight() > self.canvas.winfo_height()

    def _update_scrollbar(self):
        """Show the scrollbar only when the content does not fit."""
        if self._needs_scroll():
            self.vbar.grid()
        else:
            self.vbar.grid_remove()
            self.canvas.yview_moveto(0)

    def scroll(self, steps):
        """Scroll by `steps` units (negative = up). Used by the mouse wheel."""
        if self._needs_scroll():
            self.canvas.yview_scroll(steps, "units")

    def scroll_to_top(self):
        self.canvas.yview_moveto(0)


class FuzzyGUI:
    """Main window: input form on the left, result notebook on the right."""

    def __init__(self, root):
        self.root = root
        self.root.title("Fuzzy Scholarship Advisor – Bahria University")
        self.root.geometry("1300x800")          # size when un-maximized
        self.root.minsize(900, 500)
        self.root.resizable(True, True)
        self._maximize()

        self._setup_styles()
        self._create_variables()

        # Two columns: fixed-width form (col 0) and expanding notebook (col 1);
        # row 1 is the status bar across the whole window.
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_input_panel()
        self._build_output_panel()
        self._build_status_bar()
        self._bind_mousewheel()
        self._bind_shortcuts()
        self._show_placeholders()
        self.set_status("Ready")

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def _maximize(self):
        """Start maximized ('zoomed' works on Windows and macOS)."""
        try:
            self.root.state("zoomed")
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)    # most Linux window managers
            except tk.TclError:
                pass

    def _setup_styles(self):
        """ttk styles. The 'clam' theme is used because, unlike the native
        Windows theme, it lets us colour entry fields red for errors."""
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.page_bg = style.lookup("TFrame", "background")
        self.root.configure(bg=self.page_bg)

        # One font family everywhere: Tk's named fonts (used by ttk widgets,
        # entries, the combobox lists and the matplotlib toolbar).
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=FONT, size=9)
        style.configure(".", font=(FONT, 9))

        style.configure("Hint.TLabel", foreground="grey45", font=(FONT, 8))
        style.configure("Status.TLabel", foreground=TEXT, padding=(12, 3))
        style.configure("Field.TLabel", font=(FONT, 9, "bold"))

        # Readonly comboboxes look like normal white fields
        style.map("TCombobox", fieldbackground=[("readonly", "white")])

        # Error styles: light red background and red border
        style.configure("Error.TEntry", fieldbackground=ERROR_BG,
                        bordercolor=RED, lightcolor=RED, darkcolor=RED)
        style.configure("Error.TCombobox", bordercolor=RED, lightcolor=RED, darkcolor=RED)
        style.map("Error.TCombobox", fieldbackground=[("readonly", ERROR_BG)])

        # Fired-rules tables
        style.configure("Rules.Treeview", rowheight=26, font=(FONT, 10))
        style.configure("Rules.Treeview.Heading", font=(FONT, 10, "bold"))

    def _create_variables(self):
        """Tk variables behind every input widget (all raw text / booleans)."""
        # Numeric fields stay as raw text; validate.py does the checking.
        self.semester_var = tk.StringVar()
        self.gpa_var = tk.StringVar()
        self.income_var = tk.StringVar()
        self.position_var = tk.StringVar()

        # Comboboxes
        self.preference_var = tk.StringVar(value="0")
        self.level_var = tk.StringVar(value="UG")

        # Yes/no flags (key in the input dictionary -> BooleanVar)
        self.flag_vars = {
            "engineering": tk.BooleanVar(value=False),
            "regular_load": tk.BooleanVar(value=True),
            "summer_or_final": tk.BooleanVar(value=False),
            "other_benefit": tk.BooleanVar(value=False),
            "international": tk.BooleanVar(value=False),
        }

        # Output surface tab
        self.surface_pref_var = tk.IntVar(value=0)   # FAS surface: 0 no preference, 1 preferred
        self.last_result = None     # result of the last successful evaluation
        self.last_raw = None        # the inputs it was computed from
        self.surface_cache = {}     # ("fas", pref) / ("merit", eng, level) -> grid data
        self._surface_jobs = set()  # cache keys whose grids are being computed
        self._surface_refresh_pending = False

        # Membership functions tab: what to draw next time it is visible
        self._mf_state = (None, None)
        self._mf_dirty = True

        # Sample cases: dropdown value and the slug used in saved file names
        self.case_var = tk.StringVar(value=CASE_PLACEHOLDER)
        self.current_case_slug = None

    def _build_status_bar(self):
        """One-line status bar along the bottom of the window."""
        bar = ttk.Frame(self.root)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Separator(bar).pack(side="top", fill="x")
        self.status_var = tk.StringVar()
        ttk.Label(bar, textvariable=self.status_var, style="Status.TLabel",
                  anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Label(bar, text="Enter = Evaluate    Esc = Clear", style="Hint.TLabel",
                  padding=(12, 3)).pack(side="right")

    def set_status(self, text):
        self.status_var.set(text)

    def _bind_shortcuts(self):
        """Enter evaluates and Esc clears, from anywhere in the main window.
        (Open combobox lists are separate windows, so Enter/Esc still work
        normally there.)"""
        self.root.bind("<Return>", lambda e: self.on_evaluate())
        self.root.bind("<KP_Enter>", lambda e: self.on_evaluate())
        self.root.bind("<Escape>", lambda e: self.clear_inputs())

    def _bind_mousewheel(self):
        """One global wheel handler that scrolls whichever ScrollableFrame
        is under the mouse pointer (works over child widgets too)."""
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)            # Windows / macOS
        self.root.bind_all("<Button-4>", lambda e: self._scroll_under(e, -1))   # Linux up
        self.root.bind_all("<Button-5>", lambda e: self._scroll_under(e, 1))    # Linux down

    def _on_mousewheel(self, event):
        self._scroll_under(event, -1 if event.delta > 0 else 1)

    def _scroll_under(self, event, steps):
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):      # pointer over a popup (e.g. combobox list)
            return
        while widget is not None:
            if isinstance(widget, ScrollableFrame):
                widget.scroll(steps)
                return
            widget = widget.master

    # -----------------------------------------------------------------------
    # Left panel: student details (inside a scrollable container)
    # -----------------------------------------------------------------------

    def _build_input_panel(self):
        """Form with entries, comboboxes, checkbuttons and the two buttons."""
        self.input_scroll = ScrollableFrame(self.root, self.page_bg)
        self.input_scroll.grid(row=0, column=0, sticky="ns", padx=(12, 6), pady=8)

        panel = ttk.LabelFrame(self.input_scroll.inner, text="Student details",
                               padding=(12, 8))
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        self.input_panel = panel

        row = 0   # running grid row inside the panel

        # --- Sample cases (fills the form, does not evaluate) ---------------
        ttk.Label(panel, text="Load sample case", style="Field.TLabel").grid(
            row=row, column=0, sticky="w")
        names = [f"{i}. {name}" for i, (name, _, _) in enumerate(SAMPLE_CASES, start=1)]
        self.case_box = ttk.Combobox(panel, textvariable=self.case_var, values=names,
                                     state="readonly", width=30, height=len(names))
        self.case_box.grid(row=row + 1, column=0, sticky="ew", pady=(2, 0))
        self.case_box.bind("<<ComboboxSelected>>",
                           lambda e: self.load_case(self.case_box.current()))
        self.case_box.bind("<MouseWheel>", lambda e: (self._on_mousewheel(e), "break")[1])
        self._add_hint(panel, row + 2, "fills the form; press Evaluate (or Enter)")
        row += 3
        ttk.Separator(panel).grid(row=row, column=0, sticky="ew", pady=(2, 8))
        row += 1

        # --- Text entries -------------------------------------------------
        self.semester_entry = self._add_entry(
            panel, row, "Current semester", self.semester_var, "1–8")
        row += 3

        self.gpa_entry = self._add_entry(
            panel, row, "Previous semester GPA", self.gpa_var,
            "0.00–4.00 (blank if semester 1)")
        row += 3

        self.income_entry = self._add_entry(
            panel, row, "Monthly household income (PKR thousand)", self.income_var,
            "0 or more; above 300 treated as 300")
        row += 3

        # --- Preference criteria (readonly combobox) ------------------------
        self.preference_box = self._add_combobox(
            panel, row, "Preference criteria met", self.preference_var,
            ("0", "1", "2"), "disability, less-developed area")
        row += 3

        self.position_entry = self._add_entry(
            panel, row, "Batch position", self.position_var, "1–100")
        row += 3

        # --- Level (readonly combobox, no hint) -----------------------------
        self.level_box = self._add_combobox(
            panel, row, "Level", self.level_var, ("UG", "PG"), hint=None)
        row += 3

        # --- Yes/no checkbuttons --------------------------------------------
        ttk.Separator(panel).grid(row=row, column=0, sticky="ew", pady=(2, 6))
        row += 1

        checkbox_labels = [
            ("engineering", "Engineering programme"),
            ("regular_load", "Regular course load"),
            ("summer_or_final", "Summer or final semester"),
            ("other_benefit", "Already has another scholarship/concession"),
            ("international", "International student"),
        ]
        for key, text in checkbox_labels:
            ttk.Checkbutton(panel, text=text, variable=self.flag_vars[key]).grid(
                row=row, column=0, sticky="w", pady=1)
            row += 1

        # --- Buttons ----------------------------------------------------------
        buttons = ttk.Frame(panel)
        buttons.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        buttons.columnconfigure((0, 1), weight=1)

        self.evaluate_button = ttk.Button(buttons, text="Evaluate", command=self.on_evaluate)
        self.evaluate_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.clear_button = ttk.Button(buttons, text="Clear", command=self.clear_inputs)
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.save_button = ttk.Button(buttons, text="Save plots…", command=self.save_plots)
        self.save_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # Input key -> (widget, normal style), used to highlight invalid fields
        self.field_widgets = {
            "semester": (self.semester_entry, "TEntry"),
            "gpa": (self.gpa_entry, "TEntry"),
            "income": (self.income_entry, "TEntry"),
            "preference_count": (self.preference_box, "TCombobox"),
            "position": (self.position_entry, "TEntry"),
            "level": (self.level_box, "TCombobox"),
        }

        # Remove a field's red highlight as soon as the user changes it
        field_vars = {
            "semester": self.semester_var, "gpa": self.gpa_var,
            "income": self.income_var, "preference_count": self.preference_var,
            "position": self.position_var, "level": self.level_var,
        }
        for key, var in field_vars.items():
            var.trace_add("write", lambda *_, k=key: self._unmark_field(k))

    def _add_entry(self, parent, row, label, variable, hint):
        """Label, Entry and grey hint stacked in three grid rows."""
        ttk.Label(parent, text=label, style="Field.TLabel").grid(
            row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=variable, width=32)
        entry.grid(row=row + 1, column=0, sticky="ew", pady=(2, 0))
        self._add_hint(parent, row + 2, hint)
        return entry

    def _add_combobox(self, parent, row, label, variable, values, hint):
        """Label, readonly Combobox and optional grey hint in three grid rows."""
        ttk.Label(parent, text=label, style="Field.TLabel").grid(
            row=row, column=0, sticky="w")
        box = ttk.Combobox(parent, textvariable=variable, values=values,
                           state="readonly", width=30)
        box.grid(row=row + 1, column=0, sticky="ew", pady=(2, 0))
        # The wheel should scroll the panel, not silently change the value
        box.bind("<MouseWheel>", lambda e: (self._on_mousewheel(e), "break")[1])
        self._add_hint(parent, row + 2, hint)
        return box

    def _add_hint(self, parent, row, hint):
        """Small grey hint under a field (or just a small gap if there is none)."""
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").grid(
                row=row, column=0, sticky="w", pady=(0, 4))
        else:
            ttk.Frame(parent, height=6).grid(row=row, column=0)

    # -----------------------------------------------------------------------
    # Right panel: result notebook
    # -----------------------------------------------------------------------

    def _build_output_panel(self):
        """Notebook with four tabs ('Result' and 'Fired rules' are filled in)."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=8)

        self.result_tab = ttk.Frame(self.notebook, padding=10)
        self.rules_tab = ttk.Frame(self.notebook, padding=(10, 10, 0, 10))
        self.membership_tab = ttk.Frame(self.notebook, padding=10)
        self.surface_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.result_tab, text="Result")
        self.notebook.add(self.rules_tab, text="Fired rules")
        self.notebook.add(self.membership_tab, text="Membership functions")
        self.notebook.add(self.surface_tab, text="Output surface")

        # Result tab: everything is drawn inside this frame, emptied each time
        self.result_area = tk.Frame(self.result_tab, bg=self.page_bg)
        self.result_area.pack(fill="both", expand=True)

        # Fired rules tab: scrollable, content stretches to the tab width
        self.rules_scroll = ScrollableFrame(self.rules_tab, self.page_bg, fit_width=True)
        self.rules_scroll.pack(fill="both", expand=True)
        self.rules_area = self.rules_scroll.inner

        self._build_membership_tab()

    def _build_membership_tab(self):
        """Embedded matplotlib figure (2 x 3 subplots) with the zoom/pan toolbar."""
        self.mf_fig = Figure(figsize=(12, 7), dpi=100, facecolor=CARD_BG)
        # 'tight' engine = fig.tight_layout() re-applied on every draw/resize
        self.mf_fig.set_layout_engine("tight", pad=1.2, h_pad=2.0, w_pad=1.5)
        axes = self.mf_fig.subplots(2, 3)
        self.mf_axes = {
            "gpa": axes[0][0], "gpa_zoom": axes[0][1], "income": axes[0][2],
            "position": axes[1][0], "merit": axes[1][1], "fas": axes[1][2],
        }

        self.mf_canvas = FigureCanvasTkAgg(self.mf_fig, master=self.membership_tab)
        toolbar = NavigationToolbar2Tk(self.mf_canvas, self.membership_tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.mf_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self._build_surface_tab()

    def _build_surface_tab(self):
        """Controls row + embedded figure (3D FAS surface, merit heatmap).
        Nothing is computed until the tab is first opened."""
        controls = ttk.Frame(self.surface_tab)
        controls.pack(side="top", fill="x", pady=(0, 6))

        ttk.Label(controls, text="FAS surface:", style="Field.TLabel").pack(side="left")
        for text, value in (("No preference", 0), ("Preferred", 1)):
            ttk.Radiobutton(controls, text=text, value=value,
                            variable=self.surface_pref_var).pack(side="left", padx=(8, 0))

        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", padx=14)
        self.surface_info = ttk.Label(controls, text="")
        self.surface_info.pack(side="left")
        self.surface_status = ttk.Label(controls, text="", foreground=AMBER)
        self.surface_status.pack(side="right")

        self.surf_fig = Figure(figsize=(12, 6), dpi=100, facecolor=CARD_BG)
        self.surf_canvas = FigureCanvasTkAgg(self.surf_fig, master=self.surface_tab)
        toolbar = NavigationToolbar2Tk(self.surf_canvas, self.surface_tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.surf_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Redraw when the tab is opened or when an input that the plots
        # depend on changes (radio button, Engineering, Level).
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        for var in (self.surface_pref_var, self.flag_vars["engineering"], self.level_var):
            var.trace_add("write", lambda *_: self._schedule_surface_refresh())

    # -----------------------------------------------------------------------
    # Input handling
    # -----------------------------------------------------------------------

    def get_raw_inputs(self):
        """Collect the form into the dictionary main.evaluate() expects.
        Numeric fields are passed as raw text (validate.py converts them)."""
        raw = {
            "semester": self.semester_var.get().strip(),
            "gpa": self.gpa_var.get().strip(),
            "income": self.income_var.get().strip(),
            "preference_count": self.preference_var.get(),
            "position": self.position_var.get().strip(),
            "level": self.level_var.get(),
        }
        for key, var in self.flag_vars.items():
            raw[key] = var.get()
        return raw

    def gui_side_errors(self, raw):
        """Checks that validate.py does not do. Returns (errors, bad_keys).
        1. nan / inf in a numeric field (validate.py would crash or accept them).
        2. Position above the POSITION universe (1-100).
        Non-numeric or out-of-range text is left to validate.py."""
        errors = []
        bad_keys = set()      # fields that must not be passed to evaluate()

        for key, name in NUMERIC_FIELDS:
            try:
                value = float(raw[key])
            except (TypeError, ValueError):
                continue                     # validate.py reports "must be a number"
            if not math.isfinite(value):
                errors.append(f"{name} must be a real number (nan/inf not allowed).")
                bad_keys.add(key)

        if "position" not in bad_keys:
            try:
                if float(raw["position"]) > POSITION_MAX:
                    errors.append(f"Position must be between 1 and {POSITION_MAX}.")
            except (TypeError, ValueError):
                pass
        return errors, bad_keys

    def on_evaluate(self):
        """Evaluate button: collect inputs, run main.evaluate(), show result."""
        self._unmark_all_fields()
        raw = self.get_raw_inputs()
        gui_errors, bad_keys = self.gui_side_errors(raw)

        # nan/inf fields are blanked so validate.py never sees them; its own
        # messages about those fields are dropped (ours are clearer).
        safe_raw = dict(raw)
        for key in bad_keys:
            safe_raw[key] = ""

        try:
            result = evaluate(safe_raw)
        except Exception as exc:   # safety net: never let a crash kill the GUI
            self.show_errors([f"Unexpected error while evaluating: {exc}"])
            return

        own_errors = [e for e in result["errors"] if field_for_error(e) not in bad_keys]
        errors = own_errors + gui_errors

        # Stay on the current tab (every tab updates); errors are shown on Result.
        if errors:
            self.notebook.select(self.result_tab)
            self.show_errors(errors)
        else:
            self.show_result(result, safe_raw)
            self.show_fired_rules(result)
            self.draw_membership_plots(result, safe_raw)
            self.last_result, self.last_raw = result, safe_raw
            # Show the FAS surface that matches the student's preference
            # (setting the variable also schedules a surface redraw).
            self.surface_pref_var.set(1 if int(float(safe_raw["preference_count"])) >= 1 else 0)
            self.set_status("Evaluated: " + result["final"]["decision"])

    def clear_inputs(self):
        """Reset every field to its default value and clear the results."""
        for var in (self.semester_var, self.gpa_var, self.income_var, self.position_var):
            var.set("")
        self.preference_var.set("0")
        self.level_var.set("UG")
        for key, var in self.flag_vars.items():
            var.set(key == "regular_load")    # only regular load is checked by default
        self.case_var.set(CASE_PLACEHOLDER)
        self.current_case_slug = None
        self._unmark_all_fields()
        self._show_placeholders()
        self.semester_entry.focus_set()
        self.set_status("Ready")

    def load_case(self, index):
        """Fill the form with sample case `index` (0-based). Does not evaluate."""
        if index < 0:
            return
        name, slug, form = SAMPLE_CASES[index]
        self.semester_var.set(form["semester"])
        self.gpa_var.set(form["gpa"])
        self.income_var.set(form["income"])
        self.position_var.set(form["position"])
        self.preference_var.set(form["preference_count"])
        self.level_var.set(form["level"])
        for key, var in self.flag_vars.items():
            var.set(form[key])
        self.current_case_slug = slug
        self._unmark_all_fields()
        self.evaluate_button.focus_set()     # so Enter evaluates straight away
        self.set_status(f"Loaded sample case {index + 1}: {name} – press Evaluate")

    # -----------------------------------------------------------------------
    # Saving the plots
    # -----------------------------------------------------------------------

    def save_plots(self):
        """Ask for two PNG file names and save the Membership functions and
        Output surface figures. The surface is computed first if needed."""
        case = self.current_case_slug or "custom"
        mf_path = filedialog.asksaveasfilename(
            parent=self.root, title="Save Membership functions plot",
            initialfile=f"membership_{case}.png", defaultextension=".png",
            filetypes=[("PNG image", "*.png")])
        folder = os.path.dirname(mf_path) if mf_path else None
        surf_path = filedialog.asksaveasfilename(
            parent=self.root, title="Save Output surface plot",
            initialfile=f"surface_{case}.png", initialdir=folder,
            defaultextension=".png", filetypes=[("PNG image", "*.png")])

        saved = []
        if mf_path:
            if self._mf_dirty:                   # tab not drawn since last change
                self._render_membership_plots()
            self.mf_fig.savefig(mf_path, dpi=150, facecolor=CARD_BG)
            saved.append(os.path.basename(mf_path))
        if surf_path:
            self._save_surface_when_ready(surf_path, saved)
        elif saved:
            self.set_status("Saved: " + ", ".join(saved))

    def _save_surface_when_ready(self, path, saved):
        """Compute any missing grid (in slices, as on the tab), then draw the
        surface figure for the current settings and save it."""
        preferred, engineering, level = self._surface_settings()
        keys = [("fas", preferred), ("merit", engineering, level)]
        missing = [k for k in keys if k not in self.surface_cache]
        if self.surface_status.cget("text").startswith("Computation failed"):
            self.set_status("Could not save the surface plot: " + self.surface_status.cget("text"))
            return
        if missing:
            for key in missing:
                self._start_surface_job(key)
            self.set_status("Computing the output surface before saving …")
            self.root.after(200, self._save_surface_when_ready, path, saved)
            return
        self._draw_surface_figure(self.surface_cache[keys[0]], self.surface_cache[keys[1]],
                                  preferred, engineering, level)
        self.surf_fig.savefig(path, dpi=150, facecolor=CARD_BG)
        saved.append(os.path.basename(path))
        self.set_status("Saved: " + ", ".join(saved))

    # -----------------------------------------------------------------------
    # Field highlighting
    # -----------------------------------------------------------------------

    def _mark_fields_for_errors(self, errors):
        """Give every field named in an error message the red error style."""
        for message in errors:
            key = field_for_error(message)
            if key is not None:
                widget, normal_style = self.field_widgets[key]
                widget.configure(style="Error." + normal_style)

    def _unmark_field(self, key):
        """Return one field to its normal style."""
        widget, normal_style = self.field_widgets[key]
        widget.configure(style=normal_style)

    def _unmark_all_fields(self):
        for key in self.field_widgets:
            self._unmark_field(key)

    # -----------------------------------------------------------------------
    # Shared drawing helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _clear(area):
        for child in area.winfo_children():
            child.destroy()

    def _placeholder(self, area, text):
        """Grey centred message in an empty tab."""
        self._clear(area)
        tk.Label(area, text=text, font=(FONT, 11), fg=MUTED,
                 bg=self.page_bg).pack(pady=40)

    def _show_placeholders(self):
        """State before the first evaluation and after Clear."""
        message = "Enter the student's details and press Evaluate."
        self._placeholder(self.result_area, message)
        self._placeholder(self.rules_area, message)
        self.rules_scroll.scroll_to_top()
        self.draw_membership_plots(None)
        self.last_result = self.last_raw = None
        self._schedule_surface_refresh()

    def _make_card(self, parent, title, border=CARD_BORDER, bg=CARD_BG, title_fg=MUTED,
                   pady=14):
        """White box with a small upper-case title. Labels added with
        _card_text() re-wrap automatically when the card is resized."""
        card = tk.Frame(parent, bg=bg, highlightbackground=border,
                        highlightcolor=border, highlightthickness=1, padx=18, pady=pady)
        card.wrap_labels = []
        card.bind("<Configure>", lambda e, c=card: self._rewrap(c, e.width))
        tk.Label(card, text=title.upper(), font=(FONT, 9, "bold"),
                 fg=title_fg, bg=bg).pack(anchor="w")
        return card

    def _card_text(self, card, text, size=10, weight="normal", fg=TEXT, pady=(2, 0)):
        """Left-aligned, wrapping text line inside a card."""
        label = tk.Label(card, text=text, font=(FONT, size, weight), fg=fg,
                         bg=card["bg"], justify="left", anchor="w")
        label.pack(anchor="w", fill="x", pady=pady)
        card.wrap_labels.append(label)
        return label

    @staticmethod
    def _rewrap(card, width):
        for label in card.wrap_labels:
            label.configure(wraplength=max(width - 40, 100))

    def _section_title(self, parent, text, pady=(0, 6)):
        tk.Label(parent, text=text, font=(FONT, 12, "bold"), fg=TEXT,
                 bg=self.page_bg, anchor="w").pack(fill="x", pady=pady)

    # -----------------------------------------------------------------------
    # Result tab
    # -----------------------------------------------------------------------

    def show_errors(self, errors):
        """Red 'Invalid input' box listing every error, and nothing else."""
        self._clear(self.result_area)
        self._mark_fields_for_errors(errors)
        self.set_status(f"Invalid input ({len(errors)} error{'s' if len(errors) != 1 else ''})")

        box = self._make_card(self.result_area, "Invalid input",
                              border=RED, bg=ERROR_BG, title_fg=RED)
        box.pack(fill="x")
        for message in errors:
            self._card_text(box, "•  " + message, size=10, fg=RED, pady=(4, 0))

        self._placeholder(self.rules_area,
                          "No result – fix the input errors shown in the Result tab.")
        self.draw_membership_plots(None)
        self.last_result = self.last_raw = None
        self._schedule_surface_refresh()

    def show_result(self, result, raw):
        """One screen that tells the whole story, top to bottom:
        inputs -> final decision (+ borderline banner, warnings) ->
        Merit and FAS cards -> compact list of fired rules."""
        area = self.result_area
        self._clear(area)
        final, merit, fas = result["final"], result["merit"], result["fas"]

        # --- Inputs summary ----------------------------------------------------
        self._wrapping_line(area, "Inputs:  " + self._inputs_summary(raw),
                            size=9, fg=MUTED, pady=(0, 6))

        # --- FINAL DECISION card --------------------------------------------
        awarded = final["decision"] != "No scholarship"
        card = self._make_card(area, "Final decision",
                               border=GREEN if awarded else CARD_BORDER, pady=10)
        card.pack(fill="x")
        self._card_text(card, final["decision"], size=20, weight="bold",
                        fg=GREEN if awarded else GREY, pady=(4, 2))
        self._card_text(card, final["reason"], size=10, fg=TEXT)
        for flag in final["flags"]:
            self._card_text(card, "⚠  " + flag, size=10, weight="bold",
                            fg=ORANGE, pady=(4, 0))

        # --- Borderline GPA banner (interpretation only; the decision above
        #     comes unchanged from engine.combine()) ----------------------------
        banner = self._borderline_text(result, raw)
        if banner:
            box = tk.Frame(area, bg=AMBER_BG, highlightbackground=AMBER,
                           highlightcolor=AMBER, highlightthickness=1, padx=14, pady=8)
            box.pack(fill="x", pady=(6, 0))
            box.wrap_labels = []
            box.bind("<Configure>", lambda e, c=box: self._rewrap(c, e.width))
            self._card_text(box, banner, size=10, fg=AMBER_TEXT, pady=(0, 0))

        # --- Warnings (if any) --------------------------------------------------
        for warning in result["warnings"]:
            self._wrapping_line(area, "⚠  Warning: " + warning, size=10, fg=ORANGE,
                                pady=(6, 0))

        # --- Merit and FAS cards side by side ------------------------------------
        row = tk.Frame(area, bg=self.page_bg)
        row.pack(fill="x", pady=(10, 0))
        row.columnconfigure((0, 1), weight=1, uniform="cards")

        merit_card = self._make_card(row, "Merit Scholarship", pady=10)
        merit_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._fill_merit_card(merit_card, merit)

        fas_card = self._make_card(row, "Financial Assistance (FAS)", pady=10)
        fas_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._fill_fas_card(fas_card, fas, overridden_by_c1=final["reason"].startswith("C1"))

        # --- Compact fired-rules summary -------------------------------------------
        if merit["gate"]:
            merit_line = "Merit rules: not evaluated – " + merit["gate"]
        else:
            merit_line = "Merit rules fired: " + self._rules_summary(merit["fired_rules"])
        self._wrapping_line(area, merit_line, size=10, fg=TEXT, pady=(10, 0))
        self._wrapping_line(area, "FAS rules fired: " + self._rules_summary(fas["fired_rules"]),
                            size=10, fg=TEXT, pady=(2, 0))

    def _wrapping_line(self, parent, text, size, fg, pady):
        """Plain text line on the page background that wraps to the tab width."""
        label = tk.Label(parent, text=text, font=(FONT, size), fg=fg, bg=self.page_bg,
                         anchor="w", justify="left")
        label.pack(fill="x", pady=pady)
        label.bind("<Configure>", lambda e: e.widget.configure(wraplength=max(e.width - 4, 100)))
        return label

    @staticmethod
    def _rules_summary(fired_rules):
        """'A1 (0.60 → Full70), A6 (0.40 → None)' – strongest first."""
        if not fired_rules:
            return "none"
        rows = sorted(fired_rules, key=lambda r: r[1], reverse=True)
        return ", ".join(f"{rid} ({strength:.2f} → {term})" for rid, strength, term in rows)

    def _inputs_summary(self, raw):
        """One line with every input as the user entered it."""
        flag_names = {"engineering": "Engineering", "regular_load": "Regular load",
                      "summer_or_final": "Summer/final semester",
                      "other_benefit": "Other scholarship/concession",
                      "international": "International"}
        ticked = [name for key, name in flag_names.items() if raw[key]]
        gpa = raw["gpa"] if raw["gpa"] != "" else "– (semester 1)"
        parts = [f"Semester {raw['semester']}", f"GPA {gpa}",
                 f"Income {raw['income']}k PKR/month",
                 f"Preference {raw['preference_count']}",
                 f"Position {raw['position']}", raw["level"],
                 "Ticked: " + (", ".join(ticked) if ticked else "none")]
        return "   ·   ".join(parts)

    @staticmethod
    def _borderline_text(result, raw):
        """Amber banner text when Merit and/or FAS reports a borderline GPA,
        naming the system, the student's GPA and the cut-off it is just below."""
        cases = []
        if "Borderline" in result["merit"]["label"]:
            engineering = bool(raw["engineering"])
            cut = CUT_ENG if engineering else CUT_NON_ENG
            prog = "engineering" if engineering else "non-engineering"
            cases.append(("Merit", cut, f"{prog} merit cut-off"))
        if "Borderline" in result["fas"]["label"]:
            cases.append(("FAS", CUT_FAS, "FAS cut-off"))
        if not cases:
            return None
        gpa = float(raw["gpa"])
        which = " and ".join(name for name, _, _ in cases)
        # Name each system in the sentence only when both are borderline
        prefix = (lambda name: f"{name}: ") if len(cases) > 1 else (lambda name: "")
        below = "; ".join(f"{prefix(name)}GPA {gpa:.2f} is just below the {cut:.2f} {desc}"
                          for name, cut, desc in cases)
        return (f"⚠  Borderline GPA ({which}): {below}. The fuzzy system gives a reduced "
                "recommendation, but under the strict handbook rule this student does not "
                "meet the cut-off. Refer to the scholarship committee.")

    def _fill_award_header(self, card, crisp, label):
        """Big percentage and coloured linguistic label (shared by both cards)."""
        colour = label_colour(label)
        big_fg = GREY if is_not_awarded(crisp, label) else colour
        self._card_text(card, award_text(crisp, label), size=22, weight="bold",
                        fg=big_fg, pady=(4, 0))
        self._card_text(card, "Label: " + label, size=11, weight="bold", fg=colour)

    def _fill_merit_card(self, card, merit):
        self._fill_award_header(card, merit["crisp"], merit["label"])
        if "Borderline" in merit["label"]:
            self._card_text(card, "Blended value between the award and None — "
                                  "not a separate award tier.", size=9, fg=MUTED)

        if merit["gate"]:
            self._card_text(card, "Blocked: " + merit["gate"], fg=RED, pady=(8, 0))

        review = merit["review"]
        if review:
            colour = GREEN if review["decision"] == "Continue" else RED
            self._card_text(card, "Review of existing holder (4.11): " + review["decision"],
                            weight="bold", fg=colour, pady=(8, 0))
            self._card_text(card, f"R1 Continue = {review['R1']}    "
                                  f"R2 Suspend = {review['R2']}", fg=MUTED)
        else:
            self._card_text(card, "Review (4.11): not evaluated (merit blocked)",
                            fg=MUTED, pady=(8, 0))

    def _fill_fas_card(self, card, fas, overridden_by_c1=False):
        self._fill_award_header(card, fas["crisp"], fas["label"])
        if overridden_by_c1:
            self._card_text(card, "Overridden by C1 in the final decision.",
                            size=9, fg=MUTED)
        self._card_text(card, fas["note"], size=9, fg=MUTED, pady=(8, 0))

    # -----------------------------------------------------------------------
    # Fired rules tab
    # -----------------------------------------------------------------------

    def show_fired_rules(self, result):
        """Merit and FAS rule tables, then the fuzzified inputs."""
        area = self.rules_area
        self._clear(area)
        merit, fas = result["merit"], result["fas"]

        # --- Merit table (or the gate that blocked it) -----------------------
        self._section_title(area, "Merit rules (A1–A6)")
        if merit["gate"]:
            box = self._make_card(area, "Merit rules not evaluated",
                                  border=RED, bg=ERROR_BG, title_fg=RED)
            box.pack(fill="x", padx=(0, 10))
            self._card_text(box, merit["gate"], size=10, fg=RED, pady=(4, 0))
        else:
            self._rules_table(area, merit["fired_rules"])

        # --- FAS table ----------------------------------------------------------
        self._section_title(area, "FAS rules (F1–F6)", pady=(18, 6))
        self._rules_table(area, fas["fired_rules"])

        tk.Label(area, text="Only rules with firing strength > 0 are listed, strongest "
                            "first. The strongest rule is highlighted.",
                 font=(FONT, 9), fg=MUTED, bg=self.page_bg, anchor="w").pack(
            fill="x", pady=(6, 0))

        # --- Fuzzified inputs ---------------------------------------------------
        self._section_title(area, "Fuzzified inputs", pady=(18, 6))
        self._fuzzified_inputs(area, merit, fas)
        self.rules_scroll.scroll_to_top()

    def _rules_table(self, parent, fired_rules):
        """Treeview of (rule_id, strength, term) sorted by strength."""
        if not fired_rules:
            tk.Label(parent, text="No rule fired.", font=(FONT, 10), fg=MUTED,
                     bg=self.page_bg, anchor="w").pack(fill="x")
            return

        rows = sorted(fired_rules, key=lambda r: r[1], reverse=True)
        strongest = rows[0][1]

        columns = ("rule", "strength", "term", "meaning")
        tree = ttk.Treeview(parent, columns=columns, show="headings",
                            height=len(rows), style="Rules.Treeview", selectmode="none")
        headings = {"rule": "Rule", "strength": "Firing strength",
                    "term": "Output term", "meaning": "Rule meaning"}
        widths = {"rule": 60, "strength": 200, "term": 110, "meaning": 520}
        for col in columns:
            tree.heading(col, text=headings[col], anchor="w")
            tree.column(col, width=widths[col], minwidth=widths[col] // 2,
                        anchor="center" if col == "rule" else "w",
                        stretch=(col == "meaning"))

        for rule_id, strength, term in rows:
            tags = ("strongest",) if strength == strongest else ()
            tree.insert("", "end", tags=tags, values=(
                rule_id,
                f"{strength:.3f}   {strength_bar(strength)}",
                term,
                RULE_TEXT.get(rule_id, "—"),
            ))
        tree.tag_configure("strongest", background=TOP_RULE_BG, font=(FONT, 10, "bold"))
        tree.pack(fill="x", padx=(0, 10))

    def _fuzzified_inputs(self, parent, merit, fas):
        """Two cards (stacked, so long term rows are never clipped) with the
        membership degree of every input term."""
        # Merit system inputs
        merit_card = self._make_card(parent, "Merit system")
        merit_card.pack(fill="x", padx=(0, 10))
        memberships = merit["memberships"]
        if not memberships:
            self._card_text(merit_card, "Not computed – merit blocked by gate.",
                            fg=MUTED, pady=(6, 0))
        else:
            grid = tk.Frame(merit_card, bg=CARD_BG)
            grid.pack(anchor="w", pady=(6, 0))
            self._membership_row(grid, 0, "GPA", memberships["GPA"])
            self._membership_row(grid, 1, "Position", memberships["Position"])
            self._membership_row(grid, 2, "GPA_Met", {"degree": memberships["GPA_Met"]},
                                 show_names=False)

        # FAS system inputs
        fas_card = self._make_card(parent, "FAS system")
        fas_card.pack(fill="x", padx=(0, 10), pady=(10, 0))
        grid = tk.Frame(fas_card, bg=CARD_BG)
        grid.pack(anchor="w", pady=(6, 0))
        self._membership_row(grid, 0, "GPA", fas["memberships"]["GPA"])
        self._membership_row(grid, 1, "Income", fas["memberships"]["Income"])

    def _membership_row(self, grid, row, variable, terms, show_names=True):
        """One line: variable name, then 'Term 0.60' for every term.
        Non-zero degrees are bold and dark, zero degrees grey."""
        tk.Label(grid, text=variable, font=(FONT, 10, "bold"), fg=TEXT,
                 bg=CARD_BG, anchor="w", width=9).grid(row=row, column=0, sticky="w", pady=2)
        for col, (term, degree) in enumerate(terms.items(), start=1):
            active = degree > 0
            text = f"{term} {degree:.2f}" if show_names else f"{degree:.2f}"
            tk.Label(grid, text=text, bg=CARD_BG, anchor="w",
                     font=(FONT, 10, "bold" if active else "normal"),
                     fg=TEXT if active else GREY).grid(
                row=row, column=col, sticky="w", padx=(0, 16), pady=2)

    # -----------------------------------------------------------------------
    # Membership functions tab (matplotlib)
    # -----------------------------------------------------------------------

    def draw_membership_plots(self, result=None, raw=None):
        """Remember what to show and draw it now only if the tab is visible.
        Drawing this figure takes ~1.5 s, so a hidden tab is drawn lazily
        when it is opened (see _on_tab_changed) instead of on every Evaluate."""
        self._mf_state = (result, raw)
        self._mf_dirty = True
        if self.notebook.select() == str(self.membership_tab):
            self._render_membership_plots()

    def _render_membership_plots(self):
        """Redraw all six subplots. Without a result only the curves are shown;
        with a result the student's inputs and the inferred outputs are added."""
        result, raw = self._mf_state
        self._mf_dirty = False
        ax = self.mf_axes
        for a in ax.values():
            a.clear()

        gpa = income = position = None
        if result is not None:
            # Inputs already passed validate.py, so the text converts cleanly.
            gpa = float(raw["gpa"]) if raw["gpa"] != "" else None    # blank in semester 1
            income = min(float(raw["income"]), INCOME["universe"][1])  # validate.py caps at 300
            position = float(raw["position"])

        # Legend positions are fixed to areas the curves and student labels
        # never reach (the flat middle of each plot), so nothing is hidden.
        self._plot_input(ax["gpa"], GPA, "GPA (0–4)", "GPA (grade points)", gpa,
                         cutoffs=[(CUT_FAS, "2.50", "center"),
                                  (CUT_ENG, "3.60", "right"),
                                  (CUT_NON_ENG, "3.75", "left")],
                         legend_loc="center left")
        self._plot_input(ax["gpa_zoom"], GPA, "GPA zoom 3.40–4.00", "GPA (grade points)", gpa,
                         x_range=(3.40, 4.00),
                         cutoffs=[(CUT_ENG, "3.60 Eng. ", "right"),
                                  (CUT_NON_ENG, " 3.75 Non-eng.", "left")],
                         legend_loc="center left")
        self._plot_input(ax["income"], INCOME, "Monthly household income",
                         "Income (PKR thousand / month)", income, legend_loc="center right")
        self._plot_input(ax["position"], POSITION, "Batch position (zoom 0–8)",
                         "Position in batch (rank)", position, x_range=(0, 8),
                         legend_loc="center right")

        merit = result["merit"] if result else None
        fas = result["fas"] if result else None
        self._plot_output(ax["merit"], MERIT_AWARD, "Merit output", "Merit award (% of tuition)", merit)
        self._plot_output(ax["fas"], FAS_AWARD, "FAS output", "FAS award (% of tuition)", fas)

        self.mf_canvas.draw_idle()

    @staticmethod
    def _style_axes(ax, title, xlabel, x_range, title_pad=6, y_max=1.05):
        """Common look: fonts, limits, recessive grid."""
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT, pad=title_pad)
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        ax.set_ylabel("membership", fontsize=8, color=MUTED)
        ax.set_xlim(*x_range)
        ax.set_ylim(0, y_max)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        ax.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _plot_input(self, ax, variable, title, xlabel, value, x_range=None, cutoffs=(),
                    legend_loc="best"):
        """Term curves of one input variable, optional handbook cut-offs and,
        after an evaluation, the student's value with its membership degrees."""
        lo, hi = x_range or variable["universe"]
        xs = make_grid((lo, hi))
        terms = list(variable["terms"].items())

        # 1. Term curves (reused from membership.py, nothing redefined)
        for i, (term, (shape, params)) in enumerate(terms):
            ys = [membership(x, shape, params) for x in xs]
            ax.plot(xs, ys, color=SERIES[i], linewidth=1.8, label=term)

        self._style_axes(ax, title, xlabel, (lo, hi), title_pad=16 if cutoffs else 6)

        # 2. Handbook cut-offs: dotted line, label above the axes, and the
        #    NEAR_MISS-wide borderline band just below the cut-off
        band_labelled = False
        for cut, text, ha in cutoffs:
            if not lo <= cut <= hi:
                continue
            ax.axvspan(cut - NEAR_MISS, cut, color=CUTOFF_GREY, alpha=0.15, linewidth=0,
                       label=None if band_labelled else f"borderline band ({NEAR_MISS})")
            band_labelled = True
            ax.axvline(cut, color=CUTOFF_GREY, linestyle=":", linewidth=1)
            ax.text(cut, 1.0, text, transform=ax.get_xaxis_transform(), ha=ha, va="bottom",
                    fontsize=7, color=MUTED)

        # 3. Student marker
        if value is not None:
            degrees = [(term, membership(value, shape, params), SERIES[i])
                       for i, (term, (shape, params)) in enumerate(terms)]
            degrees = [d for d in degrees if d[1] > 0]
            if lo <= value <= hi:
                self._mark_student(ax, value, degrees, lo, hi)
            else:
                # Value outside a zoomed axis: say so and list the degrees
                arrow = "→" if value > hi else "←"
                lines = [f"student = {value:g} {arrow} (outside zoom)"]
                lines += [f"{term} {mu:.2f}" for term, mu, _ in degrees]
                # y = 0.78: below the 1.0 plateau, above the centred legend
                ax.text(0.97 if value > hi else 0.03, 0.78, "\n".join(lines),
                        transform=ax.transAxes, ha="right" if value > hi else "left",
                        va="center", fontsize=7.5, color=TEXT,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=STUDENT_RED, lw=0.8))

        ax.legend(fontsize=7, loc=legend_loc, framealpha=0.9)

    @staticmethod
    def _mark_student(ax, value, degrees, lo, hi):
        """Dashed red line at the student's value; a dot and 'Term 0.40' label
        on every curve it crosses with non-zero membership."""
        ax.axvline(value, color=STUDENT_RED, linestyle="--", linewidth=1.3)
        right_side = (value - lo) / (hi - lo) > 0.7     # put labels left of the line
        ha, dx = ("right", -7) if right_side else ("left", 7)
        ax.text(value, 0.01, f" student {value:g} ", transform=ax.get_xaxis_transform(),
                ha=ha, va="bottom", fontsize=7, color=TEXT)

        # Stack labels whose degrees are close (e.g. Low 0.50 and Medium 0.50)
        degrees = sorted(degrees, key=lambda d: d[1])
        previous, stack = None, 0
        for term, mu, colour in degrees:
            stack = stack + 1 if previous is not None and mu - previous < 0.1 else 0
            previous = mu
            dy = -11 * stack if mu > 0.85 else 11 * stack     # near the top: stack downwards
            ax.plot(value, mu, "o", markersize=6, color=colour,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5)
            ax.annotate(f"{term} {mu:.2f}", (value, mu), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=7.5,
                        color=TEXT, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    def _plot_output(self, ax, variable, name, xlabel, system):
        """Output term outlines; after an evaluation the aggregated (clipped)
        shape from engine.mamdani() and the crisp centroid."""
        lo, hi = variable["universe"]
        grid = make_grid((lo, hi))
        for i, (term, (shape, params)) in enumerate(variable["terms"].items()):
            ys = [membership(x, shape, params) for x in grid]
            ax.plot(grid, ys, color=SERIES[i], linewidth=1.2, alpha=0.55)
        self._label_output_peaks(ax, variable)

        title = name
        if system is not None:
            title = f"{name} – {system['label']}"
            fired = system["fired_rules"]
            if system.get("gate") or not fired:
                if system.get("gate"):
                    title = f"{name} – blocked by gate"
                ax.text(0.5, 0.5, "No rules fired / blocked by gate", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color=MUTED,
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=CARD_BORDER))
            else:
                agg_x, agg = mamdani(fired, variable)          # reused from engine.py
                ax.fill_between(agg_x, agg, color=AGG_FILL, alpha=0.30, linewidth=0)
                ax.plot(agg_x, agg, color=AGG_FILL, linewidth=1.2)
                xlabel += "\nshaded = aggregated (clipped) rule outputs"

                crisp = system["crisp"]
                ax.axvline(crisp, color=TEXT, linewidth=1.5)
                text = f"centroid = {crisp:.1f}%"
                if system["label"].startswith("None"):
                    text += "\n(None → reported as 0%)"
                right_side = (crisp - lo) / (hi - lo) > 0.6
                ax.annotate(text, (crisp, 0.82), xytext=(-6 if right_side else 6, 0),
                            textcoords="offset points", ha="right" if right_side else "left",
                            va="center", fontsize=8, fontweight="bold", color=TEXT, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

        # Extra headroom above 1.0 for the term names written over each peak
        self._style_axes(ax, title, xlabel, (lo, hi), y_max=1.18)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    @staticmethod
    def _label_output_peaks(ax, variable):
        """Write each output term's name just above its peak (direct labels
        instead of a legend, so no legend can cover the result). Neighbouring
        peaks that are close (Award30 / Award40) are pushed apart."""
        lo, hi = variable["universe"]
        peaks = []
        for term, (shape, params) in variable["terms"].items():
            centre = params[1] if shape == "tri" else (params[1] + params[2]) / 2
            margin = 0.04 * (hi - lo)
            peaks.append([term, min(max(centre, lo + margin), hi - margin), "center", 0])
        peaks.sort(key=lambda p: p[1])
        for left, right in zip(peaks, peaks[1:]):
            if right[1] - left[1] < 0.15 * (hi - lo):
                left[2], left[3] = "right", -2
                right[2], right[3] = "left", 2
        for term, x, ha, dx in peaks:
            ax.annotate(term, (x, 1.0), xytext=(dx, 2), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=7, color=TEXT)

    # -----------------------------------------------------------------------
    # Output surface tab (3D FAS surface + merit heatmap)
    # -----------------------------------------------------------------------

    def _on_tab_changed(self, _event=None):
        """The two plot tabs are drawn lazily, when they are opened."""
        if self.notebook.select() == str(self.membership_tab) and self._mf_dirty:
            self._render_membership_plots()
        self._schedule_surface_refresh()

    def _surface_tab_visible(self):
        return self.notebook.select() == str(self.surface_tab)

    def _schedule_surface_refresh(self):
        """Redraw the surface tab once, after the current event, if it is
        visible. Several triggers in a row (e.g. Clear) give one redraw."""
        if not self._surface_tab_visible() or self._surface_refresh_pending:
            return
        self._surface_refresh_pending = True
        self.root.after_idle(self._refresh_surface_tab)

    def _surface_settings(self):
        """(preferred, engineering, level) currently selected for the plots."""
        return (self.surface_pref_var.get() == 1,
                self.flag_vars["engineering"].get(),
                self.level_var.get())

    def _refresh_surface_tab(self):
        """Draw from the cache; start background jobs for anything missing."""
        self._surface_refresh_pending = False
        preferred, engineering, level = self._surface_settings()
        fas_key = ("fas", preferred)
        merit_key = ("merit", engineering, level)
        for key in (fas_key, merit_key):
            if key not in self.surface_cache:
                self._start_surface_job(key)

        self.surface_info.configure(
            text=f"Merit heatmap uses the form: {'Engineering' if engineering else 'Non-engineering'}, "
                 f"{level}")
        self._draw_surface_figure(self.surface_cache.get(fas_key),
                                  self.surface_cache.get(merit_key),
                                  preferred, engineering, level)

    # --- incremental computation ------------------------------------------------
    # A worker thread does not help here: the fuzzy code is pure Python and
    # holds the GIL, which starved Tk and froze the window just the same.
    # Instead each grid is computed in ~40 ms slices between Tk events.

    SLICE_SECONDS = 0.04

    def _start_surface_job(self, key):
        """Begin computing one grid (no-op if it is already running)."""
        if key in self._surface_jobs:
            return
        self._surface_jobs.add(key)
        if key[0] == "fas":
            steps = compute_fas_surface(key[1])
        else:
            steps = compute_merit_heatmap(key[1], key[2])
        self._update_surface_status()
        self.root.after(1, self._run_surface_slice, key, steps)

    def _run_surface_slice(self, key, steps):
        """Advance a grid generator for one time slice, then yield to Tk."""
        deadline = time.perf_counter() + self.SLICE_SECONDS
        try:
            while time.perf_counter() < deadline:
                next(steps)
        except StopIteration as done:            # grid finished
            self.surface_cache[key] = done.value
            self._surface_jobs.discard(key)
            self._update_surface_status()
            self._schedule_surface_refresh()
            return
        except Exception as exc:                 # report instead of crashing
            self._surface_jobs.discard(key)
            self.surface_status.configure(text=f"Computation failed: {exc}")
            return
        self.root.after(1, self._run_surface_slice, key, steps)

    def _update_surface_status(self):
        if self._surface_jobs:
            names = sorted("FAS surface" if k[0] == "fas" else "merit heatmap"
                           for k in self._surface_jobs)
            self.surface_status.configure(text="Computing " + " and ".join(names) + " …")
        elif self.surface_status.cget("text").startswith("Computing"):
            self.surface_status.configure(text="")

    # --- drawing ----------------------------------------------------------------

    def _draw_surface_figure(self, fas_data, merit_data, preferred, engineering, level):
        """Rebuild the whole figure (so colorbars do not pile up)."""
        fig = self.surf_fig
        fig.clear()
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.18,
                                left=0.03, right=0.95, top=0.88, bottom=0.12)
        ax3d = fig.add_subplot(grid[0], projection="3d", computed_zorder=False)
        ax2d = fig.add_subplot(grid[1])

        pref_text = "Preferred" if preferred else "No preference"
        ax3d.set_title(f"FAS output surface – {pref_text}",
                       fontsize=10, fontweight="bold", color=TEXT)
        if fas_data is None:
            ax3d.set_axis_off()
            ax3d.text2D(0.5, 0.5, "Computing FAS surface …", transform=ax3d.transAxes,
                        ha="center", fontsize=10, color=MUTED)
        else:
            self._draw_fas_surface(ax3d, fas_data, preferred)

        prog = "Engineering" if engineering else "Non-engineering"
        ax2d.set_title(f"Merit award – {prog}, {level}",
                       fontsize=10, fontweight="bold", color=TEXT, pad=16)
        if merit_data is None:
            ax2d.set_axis_off()
            ax2d.text(0.5, 0.5, "Computing merit heatmap …", transform=ax2d.transAxes,
                      ha="center", fontsize=10, color=MUTED)
        else:
            self._draw_merit_heatmap(ax2d, merit_data, engineering, level)

        self.surf_canvas.draw_idle()

    def _draw_fas_surface(self, ax, data, preferred):
        X, Y = np.meshgrid(data["incomes"], data["gpas"])
        surf = ax.plot_surface(X, Y, data["z"], cmap=AWARD_CMAP, vmin=0, vmax=100,
                               rstride=1, cstride=1, linewidth=0, antialiased=True,
                               zorder=1)
        ax.set_xlabel("Income (PKR thousand)", fontsize=8, color=MUTED)
        ax.set_ylabel("GPA", fontsize=8, color=MUTED)
        ax.set_zlabel("FAS award (%)", fontsize=8, color=MUTED)
        ax.set_xlim(*SURF_INCOME)
        ax.set_ylim(*SURF_GPA)
        ax.set_zlim(0, 100)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.view_init(**SURF_VIEW)
        self._colorbar(surf, ax, "FAS award (%)", shrink=0.6, pad=0.13)

        # Current student
        note = None
        if self.last_result is not None:
            raw, fas = self.last_raw, self.last_result["fas"]
            student_pref = int(float(raw["preference_count"])) >= 1
            if raw["gpa"] == "":
                note = "Student: semester 1 (no GPA) – not plotted"
            elif student_pref != preferred:
                note = (f"Student is on the '{'Preferred' if student_pref else 'No preference'}'"
                        " surface – switch above")
            else:
                gpa = float(raw["gpa"])
                income = min(float(raw["income"]), INCOME["universe"][1])
                if SURF_INCOME[0] <= income <= SURF_INCOME[1] and SURF_GPA[0] <= gpa <= SURF_GPA[1]:
                    z = award_value(fas["crisp"], fas["label"])
                    ax.plot([income, income], [gpa, gpa], [0, z], color=STUDENT_RED,
                            linestyle="--", linewidth=1, zorder=10)
                    ax.scatter([income], [gpa], [z], color=STUDENT_RED, s=40,
                               edgecolors="white", linewidths=1, zorder=11, depthshade=False)
                    ax.text(income, gpa, z + 7, f"student {z:.1f}%", fontsize=8,
                            color=TEXT, zorder=12, ha="center",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
                else:
                    note = (f"Student outside plotted range "
                            f"(GPA {gpa:g}, income {income:g})")
        if note:
            ax.text2D(0.02, 0.97, note, transform=ax.transAxes, fontsize=8, color=TEXT,
                      va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                          ec=STUDENT_RED, lw=0.8))

    def _colorbar(self, mappable, ax, label, **kwargs):
        """Colorbar with the same small, muted text as the axes."""
        bar = self.surf_fig.colorbar(mappable, ax=ax, **kwargs)
        bar.set_label(label, fontsize=8, color=MUTED)
        bar.ax.tick_params(labelsize=7.5, colors=MUTED)
        return bar

    def _draw_merit_heatmap(self, ax, data, engineering, level):
        mesh = ax.pcolormesh(data["gpas"], data["positions"], data["z"], shading="nearest",
                             cmap=AWARD_CMAP, vmin=0, vmax=70)
        ax.set_xlim(*HEAT_GPA)
        ax.set_yticks(HEAT_POSITIONS)
        ax.set_yticklabels(["1st", "2nd", "3rd", "4th", "5th"])
        ax.invert_yaxis()                       # 1st position at the top
        ax.set_xlabel("GPA (grade points)", fontsize=8, color=MUTED)
        ax.set_ylabel("Batch position", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        self._colorbar(mesh, ax, "Merit award (%)", pad=0.03)

        cut = CUT_ENG if engineering else CUT_NON_ENG
        ax.axvline(cut, color=TEXT, linestyle=":", linewidth=1.4)
        ax.text(cut, 1.0, f"{cut:.2f} cut-off", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=MUTED)

        # Current student
        note = None
        if self.last_result is not None:
            raw, merit = self.last_raw, self.last_result["merit"]
            student_eng = bool(raw["engineering"])
            student_level = str(raw["level"]).upper()
            if merit["gate"]:
                note = "Student: merit blocked by gate – not plotted"
            elif (student_eng, student_level) != (engineering, level):
                prog = "Engineering" if student_eng else "Non-engineering"
                note = f"Student was evaluated as {prog}, {student_level}\n(form now differs) – not plotted"
            else:
                gpa, pos = float(raw["gpa"]), float(raw["position"])
                if HEAT_GPA[0] <= gpa <= HEAT_GPA[1] and pos in HEAT_POSITIONS:
                    award = award_value(merit["crisp"], merit["label"])
                    ax.plot(gpa, pos, "o", color=STUDENT_RED, markersize=8,
                            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
                    left = gpa > 3.85
                    ax.annotate(f"student {award:.1f}%", (gpa, pos),
                                xytext=(-9 if left else 9, 0), textcoords="offset points",
                                ha="right" if left else "left", va="center", fontsize=8,
                                color=TEXT, zorder=6,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                                          alpha=0.9))
                else:
                    note = f"Student outside plotted range (GPA {gpa:g}, position {pos:g})"
        if note:
            ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8, color=TEXT,
                    va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                           ec=STUDENT_RED, lw=0.8))


if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzyGUI(root)
    root.mainloop()
