"""Build the chemistry notebook from scratch.

12 sections covering KS3 + GCSE chemistry, in the same exercise-driven format
as the mathematics notebook.

Run from project root:
    python fundamentals/build_chemistry.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat

sys.path.insert(0, str(Path(__file__).parent))
from _build import assemble_notebook  # noqa: E402

NB_PATH = Path(__file__).parent / "chemistry.ipynb"


TITLE = """
# Chemistry for Ages 11–16 — Learn by Coding

A code-along revision notebook for KS3 and GCSE chemistry. We use Python to do the
arithmetic (moles, masses, pH, rates) and the periodic-table bookkeeping, so you
can focus on the chemistry instead of the calculator.

**How it works:**
1. Each section starts with a **concept explanation**.
2. Then a **worked example** with code you can run.
3. Then **exercises** — write code to solve each one. A `check()` will tell you if you're right.
4. If you're stuck, click the *"Click to reveal solution"* toggle to see a worked answer.

**Topics:**
1. Atoms, isotopes and relative atomic mass
2. Electron configuration and ions
3. Compounds, formulae and balancing equations
4. The mole and Avogadro's number
5. Reacting masses and limiting reagents
6. Concentrations and dilutions
7. Acids, bases and pH
8. Energy changes
9. Rates of reaction
10. Electrolysis and redox
11. Chemical bonding (ionic, covalent, metallic)
12. Organic chemistry — alkanes, alkenes, alcohols
"""

SETUP = """
import math
from collections import Counter

# Common atomic masses (rounded as on a GCSE periodic table).
A_R = {
    'H': 1, 'He': 4, 'Li': 7, 'Be': 9, 'B': 11, 'C': 12, 'N': 14, 'O': 16,
    'F': 19, 'Ne': 20, 'Na': 23, 'Mg': 24, 'Al': 27, 'Si': 28, 'P': 31,
    'S': 32, 'Cl': 35.5, 'Ar': 40, 'K': 39, 'Ca': 40, 'Fe': 56, 'Cu': 63.5,
    'Zn': 65, 'Br': 80, 'Ag': 108, 'I': 127, 'Au': 197, 'Pb': 207,
}

AVOGADRO = 6.022e23

def check(got, expected, tol=1e-3):
    if isinstance(expected, float) or isinstance(got, float):
        ok = abs(got - expected) < tol
    else:
        ok = got == expected
    print('✅ Correct!' if ok else f'❌ Got {got!r}, expected {expected!r}')

print('Ready to go!')
"""

FOOTER = """
---
## 🎉 You've finished GCSE chemistry essentials

You can now:
- Compute relative formula masses, moles, reacting masses, concentrations and pH.
- Balance an equation by atom-counting and identify limiting reagents.
- Calculate rates from concentration-vs-time data and energy changes from bond enthalpies.
- Identify ionic vs covalent vs metallic bonding from a formula.
- Recognise the basic homologous series of organic chemistry.

The next step (A-level chemistry) layers in equilibrium constants, kinetics rate
laws, electrochemistry and a much richer organic mechanism vocabulary — but the
quantitative scaffolding is exactly what you've practised here.
"""


# ----------------------------------------------------------------------
# Section 1 — Atoms, isotopes, relative atomic mass
# ----------------------------------------------------------------------
SEC_1 = {
    "title": "## 1. Atoms, isotopes and relative atomic mass",
    "intro": (
        "Every atom has:\n"
        "- a **proton number** $Z$ (also called atomic number) — the number of protons. "
        "Determines which element it is.\n"
        "- a **mass number** $A$ — the number of protons + neutrons.\n"
        "- so the **number of neutrons** is $A - Z$.\n\n"
        "**Isotopes** are atoms of the same element with different mass numbers. The "
        "**relative atomic mass** ($A_r$) is the *weighted mean* of the isotope masses, "
        "weighted by their natural abundance.\n\n"
        "$A_r = \\dfrac{\\sum (\\text{mass} \\times \\text{abundance \\%})}{100}$"
    ),
    "worked_intro": "Chlorine has two stable isotopes: $^{35}$Cl (75%) and $^{37}$Cl (25%). Find $A_r$.",
    "worked_code": """
# Weighted mean of the isotope masses.
A_r = (35 * 75 + 37 * 25) / 100
print(f"A_r(Cl) = {A_r}")   # 35.5 — matches the periodic table
""",
    "exercises": [
        {
            "id": "1.1",
            "prompt": "Write `neutrons(A, Z)` returning the number of neutrons in an atom with mass number A and proton number Z.",
            "student": """
def neutrons(A, Z):
    # TODO
    pass

check(neutrons(23, 11), 12)   # sodium-23
check(neutrons(35, 17), 18)   # chlorine-35
""",
            "solution": """
def neutrons(A, Z):
    return A - Z

check(neutrons(23, 11), 12)
check(neutrons(35, 17), 18)
""",
            "explanation": (
                "Mass number = protons + neutrons, so neutrons = A − Z. Sodium-23 "
                "has 11 protons and 12 neutrons; chlorine-35 has 17 protons and 18 "
                "neutrons."
            ),
        },
        {
            "id": "1.2",
            "prompt": "Write `relative_atomic_mass(masses, abundances)` taking two lists (same length) and returning the weighted mean.",
            "student": """
def relative_atomic_mass(masses, abundances):
    # TODO: sum(m * a for m, a in zip(...)) / sum(abundances)
    pass

check(relative_atomic_mass([35, 37], [75, 25]), 35.5)
check(relative_atomic_mass([24, 25, 26], [79, 10, 11]), 24.32)   # magnesium
""",
            "solution": """
def relative_atomic_mass(masses, abundances):
    total = sum(m * a for m, a in zip(masses, abundances))
    return total / sum(abundances)

check(relative_atomic_mass([35, 37], [75, 25]), 35.5)
check(relative_atomic_mass([24, 25, 26], [79, 10, 11]), 24.32)
""",
            "explanation": (
                "Generalised weighted mean. Dividing by `sum(abundances)` rather than "
                "100 makes the function robust to abundances given as fractions or "
                "percentages."
            ),
        },
        {
            "id": "1.3",
            "prompt": "Element X has two isotopes: mass 63 (69.2%) and mass 65 (30.8%). Write `copper_a_r()` returning $A_r$ rounded to 1 decimal place.",
            "student": """
def copper_a_r():
    # TODO
    pass

check(copper_a_r(), 63.6)
""",
            "solution": """
def copper_a_r():
    return round((63 * 69.2 + 65 * 30.8) / 100, 1)

check(copper_a_r(), 63.6)
""",
            "explanation": (
                "Plug into the weighted-mean formula: (63×69.2 + 65×30.8)/100 = "
                "63.616. Round to 63.6 — matches the GCSE periodic-table value."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 2 — Electron configuration and ions
# ----------------------------------------------------------------------
SEC_2 = {
    "title": "## 2. Electron configuration and ions",
    "intro": (
        "Electrons sit in shells (energy levels). The simplest model fills shells in "
        "order: 2, 8, 8, 2 — i.e. shell 1 holds 2, shells 2 and 3 hold 8 each, then "
        "shell 4 starts with 2 (for K and Ca).\n\n"
        "**Atoms become ions** by losing or gaining electrons to reach a full outer "
        "shell:\n"
        "- Group 1 metals lose 1 electron → **+1** ion (e.g. Na⁺).\n"
        "- Group 2 metals lose 2 electrons → **+2** ion (e.g. Mg²⁺).\n"
        "- Group 7 non-metals gain 1 electron → **−1** ion (e.g. Cl⁻).\n"
        "- Group 6 non-metals gain 2 electrons → **−2** ion (e.g. O²⁻)."
    ),
    "worked_intro": "Distribute 19 electrons of potassium across shells.",
    "worked_code": """
def shells(n_electrons):
    capacities = [2, 8, 8, 2]   # simple GCSE model
    out = []
    remaining = n_electrons
    for cap in capacities:
        if remaining <= 0:
            break
        out.append(min(cap, remaining))
        remaining -= cap
    return out

print("K (Z=19):", shells(19))   # [2, 8, 8, 1]
print("Ca (Z=20):", shells(20))  # [2, 8, 8, 2]
print("Cl (Z=17):", shells(17))  # [2, 8, 7]
""",
    "exercises": [
        {
            "id": "2.1",
            "prompt": "Implement `shells(n_electrons)` for the simple GCSE model (capacities 2, 8, 8, 2). Return a list of the electrons in each occupied shell.",
            "student": """
def shells(n_electrons):
    # TODO
    pass

check(shells(11), [2, 8, 1])      # Na
check(shells(17), [2, 8, 7])      # Cl
check(shells(20), [2, 8, 8, 2])   # Ca
""",
            "solution": """
def shells(n_electrons):
    out = []
    for cap in [2, 8, 8, 2]:
        if n_electrons <= 0:
            break
        out.append(min(cap, n_electrons))
        n_electrons -= cap
    return out

check(shells(11), [2, 8, 1])
check(shells(17), [2, 8, 7])
check(shells(20), [2, 8, 8, 2])
""",
            "explanation": (
                "Walk the capacity list, adding `min(cap, remaining)` to the output "
                "and decrementing the remainder. Stop early when there are no more "
                "electrons to place."
            ),
        },
        {
            "id": "2.2",
            "prompt": "Write `ion_charge(group, period_metal_or_nonmetal)` taking a periodic-table group number (1, 2, 6 or 7) and a string `'metal'` or `'nonmetal'`. Return the charge of the typical ion (e.g. group 1 metal → +1; group 7 nonmetal → −1).",
            "student": """
def ion_charge(group, kind):
    # TODO: groups 1/2 (metal) lose electrons; groups 6/7 (nonmetal) gain electrons
    pass

check(ion_charge(1, 'metal'), +1)
check(ion_charge(2, 'metal'), +2)
check(ion_charge(7, 'nonmetal'), -1)
check(ion_charge(6, 'nonmetal'), -2)
""",
            "solution": """
def ion_charge(group, kind):
    if kind == 'metal':
        return +group
    return -(8 - group)

check(ion_charge(1, 'metal'), +1)
check(ion_charge(2, 'metal'), +2)
check(ion_charge(7, 'nonmetal'), -1)
check(ion_charge(6, 'nonmetal'), -2)
""",
            "explanation": (
                "Metals lose all their outer-shell electrons (group number). "
                "Non-metals gain enough to reach 8 in the outer shell, so the "
                "charge magnitude is `8 - group`."
            ),
        },
        {
            "id": "2.3",
            "prompt": "Write `outer_electrons(n)` returning the number of electrons in the outermost shell for an atom with `n` electrons (n ≤ 20).",
            "student": """
def outer_electrons(n):
    # TODO: build shells, return last entry
    pass

check(outer_electrons(11), 1)    # Na — 1 outer
check(outer_electrons(17), 7)    # Cl — 7 outer
check(outer_electrons(10), 8)    # Ne — full
""",
            "solution": """
def outer_electrons(n):
    out = []
    for cap in [2, 8, 8, 2]:
        if n <= 0:
            break
        out.append(min(cap, n))
        n -= cap
    return out[-1]

check(outer_electrons(11), 1)
check(outer_electrons(17), 7)
check(outer_electrons(10), 8)
""",
            "explanation": (
                "Reuse the `shells` logic, then take the last element. The number of "
                "outer electrons equals the group number, which is why group 1 metals "
                "and group 7 non-metals are reactive — they're one electron away "
                "from a full shell."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 3 — Compounds, formulae and balancing equations
# ----------------------------------------------------------------------
SEC_3 = {
    "title": "## 3. Compounds, formulae and balancing equations",
    "intro": (
        "A chemical formula tells you which atoms are in a compound and in what ratio. "
        "We can parse a formula like `C6H12O6` into a `{'C': 6, 'H': 12, 'O': 6}` count "
        "of atoms.\n\n"
        "**Balancing** an equation means making the count of each element equal on both "
        "sides — atoms are conserved in chemistry. You only ever add coefficients (the "
        "numbers in front), never change the formulae themselves.\n\n"
        "$\\mathrm{CH_4 + 2O_2 \\rightarrow CO_2 + 2H_2O}$ — count both sides: 1 C, 4 H, "
        "4 O on each."
    ),
    "worked_intro": "Parse `H2SO4` into a dict of atom counts and check the formula mass.",
    "worked_code": """
import re

def parse_formula(formula):
    # Match an uppercase letter, optional lowercase letter, optional digits.
    pattern = re.compile(r'([A-Z][a-z]?)(\\d*)')
    return {sym: int(count) if count else 1 for sym, count in pattern.findall(formula) if sym}

print(parse_formula('H2SO4'))     # {'H': 2, 'S': 1, 'O': 4}
print(parse_formula('CO2'))        # {'C': 1, 'O': 2}
print(parse_formula('C6H12O6'))    # {'C': 6, 'H': 12, 'O': 6}

# Relative formula mass M_r = sum of (count * A_r) for each element.
counts = parse_formula('H2SO4')
M_r = sum(n * A_R[el] for el, n in counts.items())
print(f"M_r(H2SO4) = {M_r}")   # 2*1 + 32 + 4*16 = 98
""",
    "exercises": [
        {
            "id": "3.1",
            "prompt": "Use the `parse_formula` helper from the worked example. Write `formula_mass(formula)` returning the relative formula mass using the global `A_R` dict.",
            "student": """
import re

def parse_formula(formula):
    pat = re.compile(r'([A-Z][a-z]?)(\\d*)')
    return {sym: int(c) if c else 1 for sym, c in pat.findall(formula) if sym}

def formula_mass(formula):
    # TODO: parse, then sum n * A_R[element]
    pass

check(formula_mass('H2O'), 18)
check(formula_mass('CO2'), 44)
check(formula_mass('NaCl'), 58.5)
""",
            "solution": """
import re

def parse_formula(formula):
    pat = re.compile(r'([A-Z][a-z]?)(\\d*)')
    return {sym: int(c) if c else 1 for sym, c in pat.findall(formula) if sym}

def formula_mass(formula):
    counts = parse_formula(formula)
    return sum(n * A_R[el] for el, n in counts.items())

check(formula_mass('H2O'), 18)
check(formula_mass('CO2'), 44)
check(formula_mass('NaCl'), 58.5)
""",
            "explanation": (
                "Parse, then sum the per-element contributions. Sodium chloride "
                "has Cl with $A_r=35.5$ — a non-integer formula mass is normal."
            ),
        },
        {
            "id": "3.2",
            "prompt": "Write `is_balanced(left_counts, right_counts)` taking two dicts (atom counts on each side of an equation) and returning `True` if every element has the same count on both sides.",
            "student": """
def is_balanced(left, right):
    # TODO
    pass

check(is_balanced({'C': 1, 'H': 4, 'O': 4}, {'C': 1, 'O': 4, 'H': 4}), True)
check(is_balanced({'H': 2, 'O': 1}, {'H': 1, 'O': 1}), False)
""",
            "solution": """
def is_balanced(left, right):
    return left == right

check(is_balanced({'C': 1, 'H': 4, 'O': 4}, {'C': 1, 'O': 4, 'H': 4}), True)
check(is_balanced({'H': 2, 'O': 1}, {'H': 1, 'O': 1}), False)
""",
            "explanation": (
                "Two dicts are equal iff they have the same keys with the same values. "
                "Order doesn't matter, so the test passes for the first case "
                "(combustion of methane: counts match)."
            ),
        },
        {
            "id": "3.3",
            "prompt": "Combustion of ethane: $\\mathrm{2C_2H_6 + 7O_2 \\rightarrow 4CO_2 + 6H_2O}$. Write `ethane_balanced()` returning `True` if the equation as written is balanced (count C, H, O on both sides).",
            "student": """
def ethane_balanced():
    # Left:  2 * C2H6 + 7 * O2  =>  C: 2*2=4,  H: 2*6=12,  O: 7*2=14
    # Right: 4 * CO2 + 6 * H2O  =>  C: 4,      H: 6*2=12,  O: 4*2 + 6*1 = 14
    # TODO: build the two dicts and compare
    pass

check(ethane_balanced(), True)
""",
            "solution": """
def ethane_balanced():
    left  = {'C': 2*2,    'H': 2*6, 'O': 7*2}
    right = {'C': 4,      'H': 6*2, 'O': 4*2 + 6*1}
    return left == right

check(ethane_balanced(), True)
""",
            "explanation": (
                "Multiply each formula by its coefficient and sum element-by-element. "
                "Both sides give 4 C, 12 H, 14 O — equation is balanced."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 4 — The mole and Avogadro's number
# ----------------------------------------------------------------------
SEC_4 = {
    "title": "## 4. The mole and Avogadro's number",
    "intro": (
        "The **mole** is just a counting unit: 1 mol of anything is $6.022 \\times 10^{23}$ "
        "particles — Avogadro's number, $N_A$.\n\n"
        "Two formulae you'll use constantly:\n\n"
        "$$n = \\dfrac{m}{M_r}\\quad\\text{(moles from mass)}\\qquad N = n \\cdot N_A\\quad\\text{(particles from moles)}$$\n\n"
        "where $n$ = moles, $m$ = mass in grams, $M_r$ = relative formula mass."
    ),
    "worked_intro": "How many moles in 18 g of water? How many molecules?",
    "worked_code": """
M_r_water = 18                       # H2O: 2*1 + 16
m = 18                                # grams
n = m / M_r_water                     # moles
N = n * AVOGADRO                      # number of molecules

print(f"n = {n} mol")
print(f"N = {N:.3e} molecules")
""",
    "exercises": [
        {
            "id": "4.1",
            "prompt": "Write `moles(mass_g, M_r)` returning the number of moles.",
            "student": """
def moles(mass_g, M_r):
    # TODO
    pass

check(moles(18, 18), 1.0)         # 1 mol H2O
check(moles(44, 44), 1.0)         # 1 mol CO2
check(moles(8.8, 44), 0.2)        # 8.8 g CO2 → 0.2 mol
""",
            "solution": """
def moles(mass_g, M_r):
    return mass_g / M_r

check(moles(18, 18), 1.0)
check(moles(44, 44), 1.0)
check(moles(8.8, 44), 0.2)
""",
            "explanation": (
                "Definition of the mole. The third test shows the typical exam "
                "computation: 8.8 g of CO₂ ÷ 44 g/mol = 0.2 mol."
            ),
        },
        {
            "id": "4.2",
            "prompt": "Write `particles(n_mol)` returning the number of particles in `n_mol` moles.",
            "student": """
def particles(n_mol):
    # TODO: use the global AVOGADRO
    pass

check(particles(1.0), 6.022e23)
check(particles(0.5), 3.011e23)
""",
            "solution": """
def particles(n_mol):
    return n_mol * AVOGADRO

check(particles(1.0), 6.022e23)
check(particles(0.5), 3.011e23)
""",
            "explanation": (
                "Avogadro's number is the link between counts of particles and the "
                "moles you can actually weigh on a balance."
            ),
        },
        {
            "id": "4.3",
            "prompt": "Write `mass_from_moles(n, M_r)` returning the mass in grams.",
            "student": """
def mass_from_moles(n, M_r):
    # TODO: rearrange n = m/M_r for m
    pass

check(mass_from_moles(2, 40), 80)    # 2 mol of NaOH (M_r=40) is 80 g
check(mass_from_moles(0.25, 18), 4.5)
""",
            "solution": """
def mass_from_moles(n, M_r):
    return n * M_r

check(mass_from_moles(2, 40), 80)
check(mass_from_moles(0.25, 18), 4.5)
""",
            "explanation": (
                "The mass formula is the inverse of the mole formula: m = n × M_r. "
                "GCSE often gives you any two of the three and asks for the third."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 5 — Reacting masses and limiting reagents
# ----------------------------------------------------------------------
SEC_5 = {
    "title": "## 5. Reacting masses and limiting reagents",
    "intro": (
        "Once an equation is balanced, the coefficients give the **mole ratio** of the "
        "reactants and products. To predict masses:\n\n"
        "1. Convert masses to moles using $n = m/M_r$.\n"
        "2. Use the mole ratio to find moles of the desired species.\n"
        "3. Convert back to mass using $m = n \\cdot M_r$.\n\n"
        "When you have two reactants, the **limiting reagent** is the one that runs "
        "out first — the smaller of (mol available) ÷ (mol needed per equation)."
    ),
    "worked_intro": "How many grams of CO₂ form when 16 g of CH₄ burn? CH₄ + 2O₂ → CO₂ + 2H₂O.",
    "worked_code": """
m_CH4 = 16
M_r_CH4 = 16          # 12 + 4
M_r_CO2 = 44          # 12 + 32

n_CH4 = m_CH4 / M_r_CH4         # = 1.0 mol
n_CO2 = n_CH4 * (1 / 1)          # 1:1 mole ratio CH4 → CO2
m_CO2 = n_CO2 * M_r_CO2

print(f"n(CH4)  = {n_CH4} mol")
print(f"n(CO2)  = {n_CO2} mol")
print(f"m(CO2)  = {m_CO2} g")
""",
    "exercises": [
        {
            "id": "5.1",
            "prompt": "$\\mathrm{2H_2 + O_2 \\rightarrow 2H_2O}$. Write `mass_water(g_H2)` returning the mass of water produced when `g_H2` grams of hydrogen react completely. ($M_r$: H₂=2, H₂O=18.)",
            "student": """
def mass_water(g_H2):
    # n(H2) → mole ratio 2:2 (= 1:1) → m(H2O)
    pass

check(mass_water(2), 18)        # 1 mol H2 → 1 mol H2O
check(mass_water(4), 36)
""",
            "solution": """
def mass_water(g_H2):
    n_H2 = g_H2 / 2
    n_H2O = n_H2 * (2 / 2)
    return n_H2O * 18

check(mass_water(2), 18)
check(mass_water(4), 36)
""",
            "explanation": (
                "The 2:2 coefficient gives a 1:1 mole ratio between H₂ and H₂O. "
                "2 g of H₂ is one mole, which produces one mole of water = 18 g."
            ),
        },
        {
            "id": "5.2",
            "prompt": "$\\mathrm{C + O_2 \\rightarrow CO_2}$. Write `mass_co2_from_c(g_C)` returning grams of CO₂ from `g_C` grams of carbon. ($A_r$: C=12, CO₂=44.)",
            "student": """
def mass_co2_from_c(g_C):
    # TODO
    pass

check(mass_co2_from_c(12), 44)     # 1 mol C → 1 mol CO2
check(mass_co2_from_c(6),  22)
""",
            "solution": """
def mass_co2_from_c(g_C):
    n_C = g_C / 12
    return n_C * 44

check(mass_co2_from_c(12), 44)
check(mass_co2_from_c(6),  22)
""",
            "explanation": (
                "1:1 mole ratio. The factor 44/12 ≈ 3.67 is the mass of CO₂ "
                "produced per gram of carbon — handy for environmental "
                "calculations."
            ),
        },
        {
            "id": "5.3",
            "prompt": "$\\mathrm{2H_2 + O_2 \\rightarrow 2H_2O}$. Write `limiting(g_H2, g_O2)` returning `'H2'` or `'O2'` — whichever runs out first. Use $M_r$: H₂=2, O₂=32.",
            "student": """
def limiting(g_H2, g_O2):
    # n(H2)/coef(H2)  vs  n(O2)/coef(O2)  — whichever is smaller runs out first.
    # TODO
    pass

check(limiting(4, 16), 'O2')    # 2 mol H2 needs 1 mol O2, but only 0.5 mol O2 → O2 limits
check(limiting(2, 32), 'H2')    # 1 mol H2 (units 0.5) vs 1 mol O2 (units 1.0) → H2 limits
""",
            "solution": """
def limiting(g_H2, g_O2):
    units_H2 = (g_H2 / 2) / 2       # n(H2)/coef(H2)
    units_O2 = (g_O2 / 32) / 1      # n(O2)/coef(O2)
    return 'H2' if units_H2 < units_O2 else 'O2'

check(limiting(4, 16), 'O2')      # 2 mol H2 (units 1.0) vs 0.5 mol O2 (units 0.5) → O2 limits
check(limiting(2, 32), 'H2')      # 1 mol H2 (units 0.5) vs 1 mol O2 (units 1.0) → H2 limits
""",
            "explanation": (
                "Divide each reactant's mole count by its coefficient in the balanced "
                "equation. The smaller \"reaction-units\" value runs out first. "
                "The first test: 2 mol H₂ would need 1 mol O₂, but you only have "
                "0.5 mol O₂ — oxygen limits."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 6 — Concentrations and dilutions
# ----------------------------------------------------------------------
SEC_6 = {
    "title": "## 6. Concentrations and dilutions",
    "intro": (
        "Concentration is moles (or mass) per unit volume:\n\n"
        "$\\text{conc (mol/dm}^3\\text{)} = \\dfrac{n}{V}$ where V is in **dm³** (= litres).\n"
        "$\\text{conc (g/dm}^3\\text{)} = \\dfrac{m}{V}$\n\n"
        "**Dilution rule** $C_1 V_1 = C_2 V_2$ — the moles before and after dilution "
        "are conserved. Watch units: V can be in cm³, dm³, or m³ as long as both "
        "sides match.\n\n"
        "Conversion: 1 dm³ = 1000 cm³ = 1 litre."
    ),
    "worked_intro": "What concentration of NaOH (M_r=40) results from dissolving 8 g in 250 cm³ of water?",
    "worked_code": """
m = 8                # g
M_r = 40             # NaOH
V_cm3 = 250
V_dm3 = V_cm3 / 1000

n = m / M_r           # moles
conc = n / V_dm3      # mol/dm^3

print(f"n        = {n} mol")
print(f"conc     = {conc} mol/dm^3")
""",
    "exercises": [
        {
            "id": "6.1",
            "prompt": "Write `concentration(n_mol, V_cm3)` returning concentration in mol/dm³ when given moles and volume in cm³.",
            "student": """
def concentration(n_mol, V_cm3):
    # TODO: V_dm3 = V_cm3 / 1000; conc = n / V_dm3
    pass

check(concentration(0.2, 250), 0.8)
check(concentration(1.0, 1000), 1.0)
""",
            "solution": """
def concentration(n_mol, V_cm3):
    return n_mol / (V_cm3 / 1000)

check(concentration(0.2, 250), 0.8)
check(concentration(1.0, 1000), 1.0)
""",
            "explanation": (
                "Concentration in mol/dm³ uses volume in dm³, but lab measurements "
                "are usually in cm³ — so divide V by 1000 first."
            ),
        },
        {
            "id": "6.2",
            "prompt": "Write `dilution_volume(C1, V1, C2)` returning the new volume `V2` (same units as `V1`) when a solution is diluted from `C1` to `C2` using $C_1 V_1 = C_2 V_2$.",
            "student": """
def dilution_volume(C1, V1, C2):
    # TODO
    pass

check(dilution_volume(2.0, 50, 0.5), 200.0)   # 50 cm^3 of 2 M → 200 cm^3 of 0.5 M
check(dilution_volume(1.0, 100, 0.1), 1000.0)
""",
            "solution": """
def dilution_volume(C1, V1, C2):
    return C1 * V1 / C2

check(dilution_volume(2.0, 50, 0.5), 200.0)
check(dilution_volume(1.0, 100, 0.1), 1000.0)
""",
            "explanation": (
                "Rearrange the dilution equation: $V_2 = C_1 V_1 / C_2$. The number "
                "of moles is conserved — diluting 4-fold drops the concentration "
                "4-fold."
            ),
        },
        {
            "id": "6.3",
            "prompt": "Write `mass_from_conc(C_mol_dm3, V_cm3, M_r)` returning the mass of solute in grams. Combine $n = CV$ then $m = nM_r$.",
            "student": """
def mass_from_conc(C, V_cm3, M_r):
    # TODO
    pass

check(mass_from_conc(0.5, 200, 40), 4.0)    # 0.5 mol/dm3 × 0.2 dm3 × 40 = 4 g NaOH
check(mass_from_conc(1.0, 1000, 58.5), 58.5)
""",
            "solution": """
def mass_from_conc(C, V_cm3, M_r):
    n = C * (V_cm3 / 1000)
    return n * M_r

check(mass_from_conc(0.5, 200, 40), 4.0)
check(mass_from_conc(1.0, 1000, 58.5), 58.5)
""",
            "explanation": (
                "Two steps in one. The second test computes how much NaCl (M_r=58.5) "
                "you'd need to make a 1 mol/dm³ solution of one litre — exactly the "
                "formula mass."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 7 — Acids, bases, pH
# ----------------------------------------------------------------------
SEC_7 = {
    "title": "## 7. Acids, bases and pH",
    "intro": (
        "**pH** measures hydrogen-ion concentration on a log scale:\n\n"
        "$\\text{pH} = -\\log_{10}[\\mathrm{H}^+]\\qquad [\\mathrm{H}^+] = 10^{-\\text{pH}}$\n\n"
        "- pH < 7 → acidic\n"
        "- pH = 7 → neutral (at 25 °C)\n"
        "- pH > 7 → alkaline\n\n"
        "A **strong acid** (e.g. HCl) fully dissociates — [H⁺] equals the acid concentration. "
        "A **weak acid** (e.g. ethanoic acid) only partially dissociates, so [H⁺] is less.\n\n"
        "**Neutralisation**: acid + base → salt + water. With a strong monoprotic acid "
        "and monoprotic base, $n_\\text{acid} = n_\\text{base}$ at the end-point."
    ),
    "worked_intro": "Find the pH of 0.01 mol/dm³ HCl, and the [H⁺] of pH 4.2.",
    "worked_code": """
import math

H_plus = 0.01                       # HCl is a strong acid: [H+] = conc
pH = -math.log10(H_plus)
print(f"pH of 0.01 M HCl = {pH:.2f}")    # 2.00

H_plus_2 = 10 ** (-4.2)
print(f"[H+] at pH 4.2 = {H_plus_2:.3e} mol/dm^3")
""",
    "exercises": [
        {
            "id": "7.1",
            "prompt": "Write `pH(H_plus)` returning $-\\log_{10}[H^+]$.",
            "student": """
import math

def pH(H_plus):
    # TODO
    pass

check(pH(0.01), 2.0)
check(pH(1e-7), 7.0)
""",
            "solution": """
import math

def pH(H_plus):
    return -math.log10(H_plus)

check(pH(0.01), 2.0)
check(pH(1e-7), 7.0)
""",
            "explanation": (
                "Direct definition. Multiplying [H⁺] by 10 lowers pH by 1 — that's "
                "the whole reason pH is a log scale."
            ),
        },
        {
            "id": "7.2",
            "prompt": "Write `H_plus_from_pH(pH_value)` returning $[H^+] = 10^{-\\text{pH}}$.",
            "student": """
def H_plus_from_pH(pH_value):
    # TODO
    pass

check(H_plus_from_pH(3),   0.001)
check(H_plus_from_pH(7),   1e-7)
""",
            "solution": """
def H_plus_from_pH(pH_value):
    return 10 ** (-pH_value)

check(H_plus_from_pH(3),   0.001)
check(H_plus_from_pH(7),   1e-7)
""",
            "explanation": (
                "Inverse of pH. A change of 1 pH unit = a 10× change in [H⁺]; a "
                "change of 2 = 100× change. That's why pH 3 is 10 000× more acidic "
                "than pH 7."
            ),
        },
        {
            "id": "7.3",
            "prompt": "Neutralisation: how many cm³ of 0.1 mol/dm³ NaOH neutralise 25 cm³ of 0.05 mol/dm³ HCl? Write `vol_naoh()` returning the answer in cm³.",
            "student": """
def vol_naoh():
    # n(HCl) = n(NaOH).  n = C × V (consistent units).
    # TODO
    pass

check(vol_naoh(), 12.5)
""",
            "solution": """
def vol_naoh():
    n_HCl = 0.05 * (25 / 1000)              # 0.00125 mol
    V_NaOH_dm3 = n_HCl / 0.1                 # 0.0125 dm^3
    return V_NaOH_dm3 * 1000                 # → cm^3

check(vol_naoh(), 12.5)
""",
            "explanation": (
                "1:1 reaction (HCl + NaOH → NaCl + H₂O). Moles of acid = "
                "0.05 × 0.025 = 0.00125. Same moles of NaOH at 0.1 mol/dm³ "
                "needs 0.0125 dm³ = 12.5 cm³."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 8 — Energy changes
# ----------------------------------------------------------------------
SEC_8 = {
    "title": "## 8. Energy changes",
    "intro": (
        "**Exothermic** reactions release energy (ΔH < 0); **endothermic** reactions "
        "absorb it (ΔH > 0).\n\n"
        "**Bond enthalpy method**: every bond stores energy. To estimate the enthalpy "
        "change of a reaction:\n\n"
        "$\\Delta H \\approx \\sum E(\\text{bonds broken}) - \\sum E(\\text{bonds formed})$\n\n"
        "If more energy is needed to break bonds than is released forming new ones, "
        "ΔH is positive (endothermic) — and vice versa."
    ),
    "worked_intro": "Estimate ΔH for $\\mathrm{H_2 + Cl_2 \\rightarrow 2HCl}$ given E(H–H)=436, E(Cl–Cl)=243, E(H–Cl)=432 kJ/mol.",
    "worked_code": """
broken = 436 + 243          # H-H + Cl-Cl
formed = 2 * 432            # 2 × H-Cl
delta_H = broken - formed   # kJ/mol

print(f"Bonds broken: {broken} kJ")
print(f"Bonds formed: {formed} kJ")
print(f"ΔH ≈ {delta_H} kJ/mol")
print("Negative → exothermic" if delta_H < 0 else "Positive → endothermic")
""",
    "exercises": [
        {
            "id": "8.1",
            "prompt": "Write `delta_H(broken, formed)` taking total bond-energy values for bonds broken and bonds formed, returning the enthalpy change.",
            "student": """
def delta_H(broken, formed):
    # TODO
    pass

check(delta_H(679, 864), -185)        # H2 + Cl2 → 2HCl
check(delta_H(2640, 1856), 784)       # endothermic example
""",
            "solution": """
def delta_H(broken, formed):
    return broken - formed

check(delta_H(679, 864), -185)
check(delta_H(2640, 1856), 784)
""",
            "explanation": (
                "Definition of bond-enthalpy ΔH. Negative is exothermic — the system "
                "loses energy to the surroundings. The chlorination of hydrogen is a "
                "classic exothermic example."
            ),
        },
        {
            "id": "8.2",
            "prompt": "Write `is_exothermic(broken, formed)` returning `True` if ΔH < 0.",
            "student": """
def is_exothermic(broken, formed):
    # TODO
    pass

check(is_exothermic(679, 864), True)
check(is_exothermic(800, 600), False)
""",
            "solution": """
def is_exothermic(broken, formed):
    return (broken - formed) < 0

check(is_exothermic(679, 864), True)
check(is_exothermic(800, 600), False)
""",
            "explanation": (
                "Exothermic when more energy is released forming new bonds than is "
                "needed to break the old ones. Combustion is always strongly "
                "exothermic (otherwise fires wouldn't sustain)."
            ),
        },
        {
            "id": "8.3",
            "prompt": "Methane combustion: $\\mathrm{CH_4 + 2O_2 \\rightarrow CO_2 + 2H_2O}$. With bond energies (kJ/mol) C–H 412, O=O 498, C=O 803, O–H 463, write `combustion_dh()` returning ΔH.",
            "student": """
def combustion_dh():
    # bonds broken: 4 × C-H + 2 × O=O
    # bonds formed: 2 × C=O + 4 × O-H
    # TODO
    pass

check(combustion_dh(), -814)
""",
            "solution": """
def combustion_dh():
    broken = 4 * 412 + 2 * 498
    formed = 2 * 803 + 4 * 463
    return broken - formed

check(combustion_dh(), -814)
""",
            "explanation": (
                "Count CH₄ (4 C-H bonds) and 2 O₂ (each O=O) on the left; "
                "CO₂ (2 C=O) and 2 H₂O (each 2 O-H) on the right. "
                "broken = 1648+996 = 2644; formed = 1606+1852 = 3458. "
                "ΔH = 2644 − 3458 = −814 kJ/mol — close to the textbook value of "
                "−890 with experimental bond energies."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 9 — Rates of reaction
# ----------------------------------------------------------------------
SEC_9 = {
    "title": "## 9. Rates of reaction",
    "intro": (
        "**Rate of reaction** = how fast a reactant is consumed (or product formed) per unit time.\n\n"
        "$\\text{rate} = \\dfrac{\\Delta[\\text{reactant}]}{\\Delta t}$\n\n"
        "Factors that increase rate:\n"
        "- Higher concentration → more particles per unit volume → more collisions.\n"
        "- Higher temperature → more particles have ≥ activation energy.\n"
        "- Smaller particle size → more surface area exposed.\n"
        "- Catalyst → lowers activation energy without being consumed.\n\n"
        "**Collision theory** (qualitative): a successful reaction needs particles to "
        "collide *with enough energy* and in the *right orientation*."
    ),
    "worked_intro": "Marble + acid: 100 cm³ CO₂ collected after 50 s. Compute the average rate.",
    "worked_code": """
volume_cm3 = 100
time_s = 50
rate = volume_cm3 / time_s
print(f"average rate = {rate} cm^3/s")
""",
    "exercises": [
        {
            "id": "9.1",
            "prompt": "Write `mean_rate(amount, time_s)` returning amount per second.",
            "student": """
def mean_rate(amount, time_s):
    # TODO
    pass

check(mean_rate(100, 50), 2.0)
check(mean_rate(0.05, 25), 0.002)
""",
            "solution": """
def mean_rate(amount, time_s):
    return amount / time_s

check(mean_rate(100, 50), 2.0)
check(mean_rate(0.05, 25), 0.002)
""",
            "explanation": (
                "Total change ÷ total time. Useful for a single 'how fast did "
                "this reaction go on average' number, but doesn't capture rate "
                "changing over time."
            ),
        },
        {
            "id": "9.2",
            "prompt": "Write `rate_at_interval(volumes, times, i)` returning the gradient of (volume vs time) between sample i and sample i-1: $(V_i - V_{i-1}) / (t_i - t_{i-1})$.",
            "student": """
def rate_at_interval(volumes, times, i):
    # TODO
    pass

check(rate_at_interval([0, 20, 35, 45, 50], [0, 10, 20, 30, 40], 1), 2.0)
check(rate_at_interval([0, 20, 35, 45, 50], [0, 10, 20, 30, 40], 4), 0.5)
""",
            "solution": """
def rate_at_interval(volumes, times, i):
    return (volumes[i] - volumes[i-1]) / (times[i] - times[i-1])

check(rate_at_interval([0, 20, 35, 45, 50], [0, 10, 20, 30, 40], 1), 2.0)
check(rate_at_interval([0, 20, 35, 45, 50], [0, 10, 20, 30, 40], 4), 0.5)
""",
            "explanation": (
                "Gradient of the secant line — the rate is fastest at the start and "
                "slows as reactants are consumed. This is exactly what you'd compute "
                "by hand from a graph."
            ),
        },
        {
            "id": "9.3",
            "prompt": "From the same data ($\\mathrm{V}=$ [0, 20, 35, 45, 50] cm³ at $t=$ [0, 10, 20, 30, 40] s), write `total_rate()` returning the average rate over the full reaction.",
            "student": """
def total_rate():
    # TODO: total volume change / total time
    pass

check(total_rate(), 1.25)
""",
            "solution": """
def total_rate():
    return (50 - 0) / (40 - 0)

check(total_rate(), 1.25)
""",
            "explanation": (
                "Average rate is just total change over total time: 50 cm³ in 40 s = "
                "1.25 cm³/s. The instantaneous rate at t=0 is much higher than this "
                "average — that's why a sketch of volume vs time is curved."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 10 — Electrolysis and redox
# ----------------------------------------------------------------------
SEC_10 = {
    "title": "## 10. Electrolysis and redox",
    "intro": (
        "**Redox**: a reaction where electrons move. **OIL RIG** = "
        "**O**xidation **I**s **L**oss (of electrons), **R**eduction **I**s **G**ain.\n\n"
        "**Electrolysis** uses electricity to drive a non-spontaneous redox reaction:\n"
        "- **Cathode** (negative electrode) — positive ions arrive, gain electrons → "
        "**reduction**.\n"
        "- **Anode** (positive electrode) — negative ions arrive, lose electrons → "
        "**oxidation**.\n\n"
        "For a metal salt solution: the metal goes to the cathode unless it's more "
        "reactive than hydrogen, in which case H₂ is produced instead."
    ),
    "worked_intro": "Electrolysis of molten NaCl: write the half-equations.",
    "worked_code": """
# Cathode (reduction):  Na+ + e-  →  Na
# Anode  (oxidation): 2Cl-       →  Cl2 + 2e-
#
# Net cell:           2Na+ + 2Cl- → 2Na + Cl2

print("Cathode: Na+ + e- → Na           (reduction, gain of electrons)")
print("Anode:   2Cl- → Cl2 + 2e-        (oxidation, loss of electrons)")
""",
    "exercises": [
        {
            "id": "10.1",
            "prompt": "Write `is_oxidation(electrons_lost, electrons_gained)` returning `True` if more electrons are lost than gained (oxidation).",
            "student": """
def is_oxidation(electrons_lost, electrons_gained):
    # TODO
    pass

check(is_oxidation(2, 0), True)    # Mg → Mg2+ + 2e-
check(is_oxidation(0, 1), False)   # Cl + e- → Cl-
""",
            "solution": """
def is_oxidation(electrons_lost, electrons_gained):
    return electrons_lost > electrons_gained

check(is_oxidation(2, 0), True)
check(is_oxidation(0, 1), False)
""",
            "explanation": (
                "Definition of oxidation: net loss of electrons. Mg → Mg²⁺ + 2e⁻ "
                "is the textbook example."
            ),
        },
        {
            "id": "10.2",
            "prompt": "Electrolysis of molten lead(II) bromide. Write `pbbr2_products()` returning a tuple of strings naming the cathode and anode product.",
            "student": """
def pbbr2_products():
    # Cathode: Pb2+ + 2e- → Pb
    # Anode:   2Br- → Br2 + 2e-
    # TODO: return ('Pb', 'Br2')
    pass

check(pbbr2_products(), ('Pb', 'Br2'))
""",
            "solution": """
def pbbr2_products():
    return ('Pb', 'Br2')

check(pbbr2_products(), ('Pb', 'Br2'))
""",
            "explanation": (
                "Molten salt electrolysis (no water competition): metal at cathode, "
                "non-metal at anode. The same logic applies to molten Al₂O₃ in "
                "aluminium production: Al at cathode, O₂ at anode."
            ),
        },
        {
            "id": "10.3",
            "prompt": "Charge passed in electrolysis: $Q = It$ (coulombs = amps × seconds). Write `charge(I, t)` returning Q.",
            "student": """
def charge(I, t):
    # TODO
    pass

check(charge(2.0, 600), 1200)
check(charge(0.5, 1800), 900)
""",
            "solution": """
def charge(I, t):
    return I * t

check(charge(2.0, 600), 1200)
check(charge(0.5, 1800), 900)
""",
            "explanation": (
                "Q = It is the foundation for calculating the moles of electrons "
                "delivered (= Q / 96485, the Faraday constant). At GCSE level you "
                "usually stop at Q itself."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 11 — Bonding (ionic, covalent, metallic)
# ----------------------------------------------------------------------
SEC_11 = {
    "title": "## 11. Chemical bonding",
    "intro": (
        "Three main bond types:\n\n"
        "- **Ionic** — between a metal and a non-metal. Electrons are *transferred*. "
        "Forms a giant lattice. High melting points; conducts when molten or dissolved.\n"
        "- **Covalent** — between two non-metals. Electrons are *shared*. Forms simple "
        "molecules (low melting points) or giant covalent structures (e.g. diamond, "
        "graphite, SiO₂ — very high melting points).\n"
        "- **Metallic** — between metal atoms. A *sea of delocalised electrons* held by "
        "positive metal cations. Conducts; malleable.\n\n"
        "A simple test: is at least one element a metal?"
    ),
    "worked_intro": "Classify NaCl, H₂O, CH₄, MgO, Cu, Fe.",
    "worked_code": """
METALS = {'Li','Na','K','Be','Mg','Ca','Al','Fe','Cu','Zn','Ag','Au','Pb'}

def bond_type(els):
    has_metal = any(e in METALS for e in els)
    has_nonmetal = any(e not in METALS for e in els)
    if has_metal and has_nonmetal: return 'ionic'
    if has_metal: return 'metallic'
    return 'covalent'

print(bond_type(['Na', 'Cl']))   # ionic
print(bond_type(['H', 'O']))      # covalent (H2O)
print(bond_type(['C', 'H']))      # covalent
print(bond_type(['Mg', 'O']))     # ionic
print(bond_type(['Cu']))           # metallic
""",
    "exercises": [
        {
            "id": "11.1",
            "prompt": "Implement `bond_type(elements)` from the worked example using the global `METALS` set defined here.",
            "student": """
METALS = {'Li','Na','K','Be','Mg','Ca','Al','Fe','Cu','Zn','Ag','Au','Pb'}

def bond_type(elements):
    # TODO
    pass

check(bond_type(['Na', 'Cl']), 'ionic')
check(bond_type(['H', 'O']),    'covalent')
check(bond_type(['Fe']),        'metallic')
""",
            "solution": """
METALS = {'Li','Na','K','Be','Mg','Ca','Al','Fe','Cu','Zn','Ag','Au','Pb'}

def bond_type(elements):
    has_metal = any(e in METALS for e in elements)
    has_nonmetal = any(e not in METALS for e in elements)
    if has_metal and has_nonmetal:
        return 'ionic'
    if has_metal:
        return 'metallic'
    return 'covalent'

check(bond_type(['Na', 'Cl']), 'ionic')
check(bond_type(['H', 'O']),    'covalent')
check(bond_type(['Fe']),        'metallic')
""",
            "explanation": (
                "Two booleans — has_metal and has_nonmetal — fully decide the type. "
                "A pure metallic element gives 'metallic'; a pure non-metal compound "
                "gives 'covalent'; a mix gives 'ionic'."
            ),
        },
        {
            "id": "11.2",
            "prompt": "Write `conducts_when_solid(bond)` returning `True` only for metallic bonding (ionic solids don't conduct because the ions are locked in place).",
            "student": """
def conducts_when_solid(bond):
    # TODO
    pass

check(conducts_when_solid('metallic'), True)
check(conducts_when_solid('ionic'),    False)
check(conducts_when_solid('covalent'), False)
""",
            "solution": """
def conducts_when_solid(bond):
    return bond == 'metallic'

check(conducts_when_solid('metallic'), True)
check(conducts_when_solid('ionic'),    False)
check(conducts_when_solid('covalent'), False)
""",
            "explanation": (
                "Only metallic structures have free-moving electrons in the solid "
                "state. Ionic compounds need to be molten or dissolved before "
                "their ions can carry current."
            ),
        },
        {
            "id": "11.3",
            "prompt": "How many shared pairs of electrons are in a single, double and triple covalent bond? Write `shared_pairs(order)` taking 1, 2 or 3.",
            "student": """
def shared_pairs(order):
    # TODO
    pass

check(shared_pairs(1), 1)
check(shared_pairs(2), 2)
check(shared_pairs(3), 3)
""",
            "solution": """
def shared_pairs(order):
    return order

check(shared_pairs(1), 1)
check(shared_pairs(2), 2)
check(shared_pairs(3), 3)
""",
            "explanation": (
                "By definition: bond order = number of shared electron pairs. A "
                "single bond shares 2 electrons (1 pair), double 4 (2 pairs), "
                "triple 6 (3 pairs). N₂ has a triple bond — exceptionally strong, "
                "which is why nitrogen gas is so unreactive."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 12 — Organic chemistry basics
# ----------------------------------------------------------------------
SEC_12 = {
    "title": "## 12. Organic chemistry — alkanes, alkenes, alcohols",
    "intro": (
        "**Homologous series** are families of organic molecules sharing a general "
        "formula and similar properties:\n\n"
        "| Series | General formula | Suffix | First member |\n"
        "|---|---|---|---|\n"
        "| Alkanes | $C_nH_{2n+2}$ | -ane | $\\text{CH}_4$ (methane) |\n"
        "| Alkenes | $C_nH_{2n}$ | -ene | $C_2H_4$ (ethene) |\n"
        "| Alcohols | $C_nH_{2n+1}\\text{OH}$ | -ol | $\\text{CH}_3\\text{OH}$ (methanol) |\n\n"
        "Alkenes have a C=C double bond — the source of most of their reactivity. "
        "Alcohols carry an -OH group; their boiling points are much higher than "
        "the corresponding alkane's because of hydrogen bonding."
    ),
    "worked_intro": "Generate the molecular formula for the n-th alkane and alkene.",
    "worked_code": """
def alkane_formula(n):
    return f"C{n}H{2*n + 2}"

def alkene_formula(n):
    return f"C{n}H{2*n}"

print(alkane_formula(1))   # CH4 → 'C1H4' (we'll fix the '1' subscript visually)
print(alkene_formula(2))   # C2H4
print(alkane_formula(4))   # C4H10 (butane)
""",
    "exercises": [
        {
            "id": "12.1",
            "prompt": "Write `n_hydrogens_alkane(n)` returning the number of H atoms in the n-th alkane ($C_nH_{2n+2}$).",
            "student": """
def n_hydrogens_alkane(n):
    # TODO
    pass

check(n_hydrogens_alkane(1), 4)    # CH4
check(n_hydrogens_alkane(4), 10)   # butane C4H10
check(n_hydrogens_alkane(8), 18)   # octane C8H18
""",
            "solution": """
def n_hydrogens_alkane(n):
    return 2*n + 2

check(n_hydrogens_alkane(1), 4)
check(n_hydrogens_alkane(4), 10)
check(n_hydrogens_alkane(8), 18)
""",
            "explanation": (
                "Each carbon needs 4 bonds. In a straight chain, two end carbons "
                "have 3 H + 1 C-C, and each interior carbon has 2 H + 2 C-C — "
                "totals 2n + 2 hydrogens."
            ),
        },
        {
            "id": "12.2",
            "prompt": "Write `is_alkene(formula_dict)` returning `True` if a parsed formula `{'C': c, 'H': h}` matches the alkene general formula $C_nH_{2n}$ (n ≥ 2).",
            "student": """
def is_alkene(d):
    # TODO
    pass

check(is_alkene({'C': 2, 'H': 4}), True)    # ethene
check(is_alkene({'C': 3, 'H': 6}), True)    # propene
check(is_alkene({'C': 2, 'H': 6}), False)   # ethane
check(is_alkene({'C': 1, 'H': 2}), False)   # n must be >= 2
""",
            "solution": """
def is_alkene(d):
    c = d.get('C', 0); h = d.get('H', 0)
    return c >= 2 and h == 2 * c

check(is_alkene({'C': 2, 'H': 4}), True)
check(is_alkene({'C': 3, 'H': 6}), True)
check(is_alkene({'C': 2, 'H': 6}), False)
check(is_alkene({'C': 1, 'H': 2}), False)
""",
            "explanation": (
                "Alkenes need at least 2 carbons (you need a C=C double bond). "
                "After that, $H = 2C$. Same molecular formula as a cycloalkane — "
                "GCSE skips that subtlety."
            ),
        },
        {
            "id": "12.3",
            "prompt": "Write `alcohol_formula(n)` returning a string for the n-th alcohol's formula in the form `'CnH(2n+1)OH'`. For example, `alcohol_formula(2)` → `'C2H5OH'`.",
            "student": """
def alcohol_formula(n):
    # TODO
    pass

check(alcohol_formula(1), 'C1H3OH')      # methanol
check(alcohol_formula(2), 'C2H5OH')      # ethanol
check(alcohol_formula(4), 'C4H9OH')      # butan-1-ol
""",
            "solution": """
def alcohol_formula(n):
    return f"C{n}H{2*n + 1}OH"

check(alcohol_formula(1), 'C1H3OH')
check(alcohol_formula(2), 'C2H5OH')
check(alcohol_formula(4), 'C4H9OH')
""",
            "explanation": (
                "Alcohol = alkyl group ($C_nH_{2n+1}$) + -OH. Methanol is "
                "CH₃OH (n=1: H = 3), ethanol C₂H₅OH (n=2: H = 5), and so on."
            ),
        },
    ],
}


SECTIONS = [SEC_1, SEC_2, SEC_3, SEC_4, SEC_5, SEC_6, SEC_7, SEC_8, SEC_9, SEC_10, SEC_11, SEC_12]


def main():
    nb = assemble_notebook(TITLE, SETUP, SECTIONS, FOOTER)
    nbformat.validate(nb)
    nbformat.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} — {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
