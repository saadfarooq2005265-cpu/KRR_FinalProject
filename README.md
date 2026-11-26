# Fuzzy Inference System: Merit Scholarship and Financial Assistance Scholarship

KRR Assignment 2, Bahria University Student Handbook §4.4, §4.5, §4.6, §4.11 and §4.12.
It is a Mamdani fuzzy system written in plain Python.

## Requirements
- Python 3.8 or newer
- `matplotlib`, which is only needed for the plots:
  ```
  pip install matplotlib
  ```
  The fuzzy system itself uses only standard Python.

## Files
| File | What it does |
|---|---|
| `membership.py` | Triangle and trapezoid functions, all variable definitions (GPA, income, position, outputs), fuzzification |
| `validate.py` | Checks inputs (range, type, yes/no) and returns clear error messages |
| `engine.py` | Mamdani steps (clip with min, aggregate with max, centroid), linguistic label, combination rules C1–C4 |
| `fas.py` | Financial Assistance rules F1–F6 |
| `merit.py` | Merit gates (A7–A9), rules A1–A6, review rules R1–R2 |
| `main.py` | `evaluate(inputs)`: runs everything; terminal input and demo mode |
| `plots.py` | Saves membership function plots and the FAS output surface |
| `gui.py` | Tkinter GUI: input form with validation, result, fired rules, membership plots, output surface, sample cases |

## How to run
Open a terminal in this folder, then run one of these:

**1. Enter a student's details yourself**
```
python main.py
```
Answer the questions. Yes/no questions accept `y` or `n`.

**2. Run the built-in sample students**
```
python main.py --demo
```

**3. Draw the plots** (needs matplotlib)
```
python plots.py
```
This creates `membership_functions.png` and `fas_surface.png`.

## GUI
```
python gui.py
```
Needs matplotlib (`pip install matplotlib`); Tkinter comes with Python. The window opens maximized. Enter the student's details on the left and press **Evaluate**. Enter evaluates and Esc clears the form.

**Load sample case** (top of the left panel) fills the form with one of 13 prepared students, such as a merit topper, borderline GPA, FAS with preference, C1/C2/C4, semester 1, a summer/final semester, or invalid input. It doesn't evaluate the case automatically, so you can look at or change the inputs first.

The four tabs show:
| Tab | What it shows |
|---|---|
| Result | The final decision with its reason and any flags (e.g. C4 Rector approval). Then one card each for Merit and FAS with the award %, the label and notes. Invalid input shows a red list of errors and highlights the fields. |
| Fired rules | Tables of the merit rules (A1–A6) and FAS rules (F1–F6) that fired, with firing strength and meaning, strongest first. Below them is the degree of every input term (fuzzified inputs). |
| Membership functions | The membership functions of every input and output. After an evaluation it also shows the student's values (red line with degrees), the aggregated output shapes and the centroids. |
| Output surface | A 3D FAS surface (income × GPA) and a merit heatmap (GPA × position), with the student as a red dot. They are computed the first time the tab is opened, which takes a few seconds, and then reused. |

**Save plots…** saves the Membership functions and Output surface figures as PNG files. The status bar at the bottom shows the last action.

## Using it from the GUI or test code
```python
from main import evaluate

result = evaluate({
    "semester": 4, "gpa": 3.73, "income": 55, "preference_count": 0,
    "position": 1, "level": "UG", "engineering": False,
    "regular_load": True, "summer_or_final": False,
    "other_benefit": False, "international": False,
})
```
`result` contains:
- `errors`: a list of messages. If it isn't empty, the input was invalid and nothing else is returned.
- `merit`: `crisp`, `label`, `fired_rules` as (rule ID, strength, output term), `gate`, `review`, `memberships`
- `fas`: `crisp`, `label`, `fired_rules`, `memberships`, `note`
- `final`: `decision`, `reason`, `flags`

## Inputs
| Input | Allowed values |
|---|---|
| semester | whole number of 1 or more |
| gpa | 0.00–4.00; may be left blank in semester 1 |
| income | monthly household income in PKR thousand, 0 or more (values above 300 are treated as 300) |
| preference_count | 0, 1 or 2 (disability, less-developed area) |
| position | batch rank, whole number of 1 or more |
| level | UG or PG |
| engineering, regular_load, summer_or_final, other_benefit, international | yes or no |

## Assumptions
- The income breakpoints are the team's own assumption; the handbook gives no income figures.
- Merit is not evaluated in semester 1, because it is based on the previous semester's GPA (§4.4.3.2).
- A FAS award labelled "None" is reported as 0% in the final decision. On its own, the centroid of the None set is about 5%.
