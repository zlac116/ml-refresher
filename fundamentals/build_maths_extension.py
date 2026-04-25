"""Extend the existing mathematics notebook with 6 new GCSE-level sections (13–18).

Run from project root:
    python fundamentals/build_maths_extension.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat

sys.path.insert(0, str(Path(__file__).parent))
from _build import build_section, md  # noqa: E402


NB_PATH = Path(__file__).parent / "mathematics.ipynb"


# ----------------------------------------------------------------------
# Section 13 — Quadratics
# ----------------------------------------------------------------------
SEC_13 = {
    "title": "## 13. Quadratics",
    "intro": (
        "A **quadratic** is anything of the form $ax^2 + bx + c$. The graph is a parabola.\n\n"
        "Three ways to solve $ax^2 + bx + c = 0$:\n"
        "1. **Factorising**: rewrite as $(x - p)(x - q) = 0$ — solutions are $p, q$.\n"
        "2. **Completing the square**: rewrite as $(x + p)^2 + q$ — useful for finding the vertex.\n"
        "3. **Quadratic formula**: $x = \\dfrac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$ — always works.\n\n"
        "The **discriminant** $\\Delta = b^2 - 4ac$ tells you how many real solutions exist:\n"
        "$\\Delta > 0$ ⇒ two solutions, $\\Delta = 0$ ⇒ one solution, $\\Delta < 0$ ⇒ no real solutions."
    ),
    "worked_intro": "Solve $x^2 - 5x + 6 = 0$ three ways.",
    "worked_code": """
import math

# 1) Factorising: x^2 - 5x + 6 = (x - 2)(x - 3) = 0  →  x = 2 or x = 3
print("Factored:", [2, 3])

# 2) Completing the square: x^2 - 5x + 6 = (x - 2.5)^2 - 0.25
#    (x - 2.5)^2 = 0.25  →  x - 2.5 = ±0.5  →  x = 2 or 3
p = -5 / 2
q = 6 - p**2
print(f"Completed:  (x + {p})^2 + {q}  →  x = {-p + 0.5} or {-p - 0.5}")

# 3) Quadratic formula.
a, b, c = 1, -5, 6
disc = b**2 - 4*a*c
x1 = (-b + math.sqrt(disc)) / (2*a)
x2 = (-b - math.sqrt(disc)) / (2*a)
print(f"Formula:   discriminant={disc}, x = {x1}, {x2}")
""",
    "exercises": [
        {
            "id": "13.1",
            "prompt": "Write `quadratic_formula(a, b, c)` that returns a tuple `(x1, x2)` of the two real roots, with `x1 >= x2`. Assume the discriminant is non-negative.",
            "student": """
import math

def quadratic_formula(a, b, c):
    # TODO: use the quadratic formula and return (larger_root, smaller_root)
    pass

check(quadratic_formula(1, -5, 6), (3.0, 2.0))
check(quadratic_formula(1, 2, -3), (1.0, -3.0))
""",
            "solution": """
import math

def quadratic_formula(a, b, c):
    disc = b**2 - 4*a*c
    x1 = (-b + math.sqrt(disc)) / (2*a)
    x2 = (-b - math.sqrt(disc)) / (2*a)
    return (max(x1, x2), min(x1, x2))

check(quadratic_formula(1, -5, 6), (3.0, 2.0))
check(quadratic_formula(1, 2, -3), (1.0, -3.0))
""",
            "explanation": (
                "Compute the discriminant first; both roots use it. Sorting the pair "
                "by `max`/`min` makes the output order predictable, so the test can "
                "compare directly."
            ),
        },
        {
            "id": "13.2",
            "prompt": "Write `n_real_roots(a, b, c)` returning 0, 1, or 2 — the number of distinct real roots, decided by the discriminant.",
            "student": """
def n_real_roots(a, b, c):
    # TODO: compute discriminant; return 0, 1 or 2
    pass

check(n_real_roots(1, -5, 6), 2)   # disc = 1
check(n_real_roots(1, -4, 4), 1)   # disc = 0 (perfect square)
check(n_real_roots(1, 1, 1), 0)    # disc = -3
""",
            "solution": """
def n_real_roots(a, b, c):
    d = b**2 - 4*a*c
    if d > 0: return 2
    if d == 0: return 1
    return 0

check(n_real_roots(1, -5, 6), 2)
check(n_real_roots(1, -4, 4), 1)
check(n_real_roots(1, 1, 1), 0)
""",
            "explanation": (
                "Discriminant $b^2 - 4ac$ is the entire story: positive ⇒ two real roots, "
                "zero ⇒ one (a 'repeated' root), negative ⇒ none."
            ),
        },
        {
            "id": "13.3",
            "prompt": "Write `vertex(a, b, c)` returning `(h, k)` where the parabola $ax^2+bx+c$ has its turning point at $(h, k)$. Use $h = -b/(2a)$ and substitute.",
            "student": """
def vertex(a, b, c):
    # TODO: return (h, k) where h is the x-coord of the vertex
    pass

check(vertex(1, -4, 5), (2.0, 1.0))    # x^2 - 4x + 5 = (x-2)^2 + 1
check(vertex(2, 8, 3), (-2.0, -5.0))   # 2(x+2)^2 - 5
""",
            "solution": """
def vertex(a, b, c):
    h = -b / (2*a)
    k = a*h*h + b*h + c
    return (h, k)

check(vertex(1, -4, 5), (2.0, 1.0))
check(vertex(2, 8, 3), (-2.0, -5.0))
""",
            "explanation": (
                "Differentiating gives $2ax + b = 0$, so the turning point is at "
                "$x = -b/(2a)$. Plug back in to get $y$. This is exactly what "
                "completing the square reveals."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 14 — Simultaneous equations & inequalities
# ----------------------------------------------------------------------
SEC_14 = {
    "title": "## 14. Simultaneous equations and inequalities",
    "intro": (
        "**Simultaneous equations** — two equations, two unknowns. Two methods:\n"
        "- **Elimination**: add or subtract the equations to cancel one variable.\n"
        "- **Substitution**: rearrange one equation, plug into the other.\n\n"
        "**Linear inequalities** behave like equations, but flip the sign whenever you "
        "multiply or divide by a negative.\n\n"
        "**Compound inequalities** (e.g. $-3 \\le x < 5$) describe a range. The set of "
        "integers inside is finite — Python can list them with `range`."
    ),
    "worked_intro": "Solve $\\begin{cases} 2x + 3y = 12 \\\\ x - y = 1 \\end{cases}$ by elimination.",
    "worked_code": """
# Eliminate x: multiply the second equation by 2 to match x's coefficient,
# then subtract from the first.
#
#   2x + 3y = 12
# - 2x - 2y =  2     (= 2 × (x - y = 1))
#   ---------
#         5y = 10  →  y = 2
#
# Substitute back: x - 2 = 1  →  x = 3.

x, y = 3, 2
print(f"x = {x}, y = {y}")

# Check both original equations.
print("eq1 holds:", 2*x + 3*y == 12)
print("eq2 holds:",   x -   y ==  1)
""",
    "exercises": [
        {
            "id": "14.1",
            "prompt": "Write `solve_2x2(a, b, c, d, e, f)` that solves the linear system $ax + by = c$, $dx + ey = f$. Use Cramer's rule: $x = (ce - bf)/(ae - bd)$, $y = (af - cd)/(ae - bd)$. Return `(x, y)`.",
            "student": """
def solve_2x2(a, b, c, d, e, f):
    # TODO: apply Cramer's rule
    pass

check(solve_2x2(2, 3, 12, 1, -1, 1), (3.0, 2.0))
check(solve_2x2(1, 1, 5, 2, -1, 4), (3.0, 2.0))
""",
            "solution": """
def solve_2x2(a, b, c, d, e, f):
    det = a*e - b*d
    x = (c*e - b*f) / det
    y = (a*f - c*d) / det
    return (x, y)

check(solve_2x2(2, 3, 12, 1, -1, 1), (3.0, 2.0))
check(solve_2x2(1, 1, 5, 2, -1, 4), (3.0, 2.0))
""",
            "explanation": (
                "Cramer's rule: the determinant of the 2x2 coefficient matrix is "
                "$ae - bd$. The numerator for x replaces the x-column with the "
                "constants; same idea for y. If $det = 0$ the system has no unique "
                "solution (parallel lines)."
            ),
        },
        {
            "id": "14.2",
            "prompt": "Write `integers_in_range(lo, hi)` that returns a list of integers $n$ with $\\text{lo} \\le n < \\text{hi}$. Use `range`.",
            "student": """
def integers_in_range(lo, hi):
    # TODO
    pass

check(integers_in_range(-3, 5), [-3, -2, -1, 0, 1, 2, 3, 4])
check(integers_in_range(0, 1), [0])
""",
            "solution": """
def integers_in_range(lo, hi):
    return list(range(lo, hi))

check(integers_in_range(-3, 5), [-3, -2, -1, 0, 1, 2, 3, 4])
check(integers_in_range(0, 1), [0])
""",
            "explanation": (
                "`range(lo, hi)` is half-open: it includes `lo` but stops before `hi`. "
                "That matches the inequality `lo <= n < hi` exactly."
            ),
        },
        {
            "id": "14.3",
            "prompt": "Write `solve_inequality(a, b)` returning the boundary value of $ax + b > 0$ — i.e. the value of $x$ where the expression equals zero. Doesn't matter which side is satisfied.",
            "student": """
def solve_inequality(a, b):
    # TODO: solve a*x + b = 0 for x
    pass

check(solve_inequality(2, -8), 4.0)    # 2x - 8 > 0  →  x > 4
check(solve_inequality(-3, 6), 2.0)    # -3x + 6 > 0  →  x < 2 (sign flip!)
""",
            "solution": """
def solve_inequality(a, b):
    return -b / a

check(solve_inequality(2, -8), 4.0)
check(solve_inequality(-3, 6), 2.0)
""",
            "explanation": (
                "The boundary is just $x = -b/a$. The interesting part of inequalities "
                "is which side is the solution: divide-by-negative flips the inequality "
                "sign, so $-3x + 6 > 0$ becomes $x < 2$."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 15 — Surds & advanced indices
# ----------------------------------------------------------------------
SEC_15 = {
    "title": "## 15. Surds and advanced indices",
    "intro": (
        "**Surds** are irrational roots written exactly: $\\sqrt{2}$, $\\sqrt{3}$, $\\sqrt{12}$ etc.\n\n"
        "Useful identities:\n"
        "- $\\sqrt{ab} = \\sqrt{a}\\sqrt{b}$  → simplify $\\sqrt{12} = \\sqrt{4 \\cdot 3} = 2\\sqrt{3}$.\n"
        "- Rationalising the denominator: multiply numerator and denominator by the surd.\n\n"
        "**Index laws** you should know cold:\n"
        "- $a^m \\cdot a^n = a^{m+n}$,  $a^m / a^n = a^{m-n}$,  $(a^m)^n = a^{mn}$\n"
        "- $a^0 = 1$,  $a^{-n} = 1/a^n$,  $a^{1/n} = \\sqrt[n]{a}$,  $a^{m/n} = (\\sqrt[n]{a})^m$"
    ),
    "worked_intro": "Simplify $\\sqrt{50}$ and rationalise $\\dfrac{1}{\\sqrt{3}}$.",
    "worked_code": """
import math

# sqrt(50) = sqrt(25 * 2) = 5*sqrt(2).
print(f"sqrt(50) ≈ {math.sqrt(50):.6f}")
print(f"5*sqrt(2) ≈ {5 * math.sqrt(2):.6f}   (should match)")

# 1 / sqrt(3) — rationalise by multiplying top and bottom by sqrt(3):
#   1/sqrt(3)  =  sqrt(3) / 3
print(f"1/sqrt(3)  ≈ {1 / math.sqrt(3):.6f}")
print(f"sqrt(3)/3 ≈ {math.sqrt(3) / 3:.6f}   (should match)")

# Fractional indices: 8 ** (1/3) is the cube root.
print(f"8^(1/3) = {8 ** (1/3):.6f}   (cube root of 8 is 2)")

# Negative indices: 2 ** -3 = 1/8.
print(f"2^-3 = {2 ** -3}   (= 1/8)")
""",
    "exercises": [
        {
            "id": "15.1",
            "prompt": "Write `simplify_surd(n)` that returns `(a, b)` where $\\sqrt{n} = a\\sqrt{b}$ and $b$ is square-free. For example, $\\sqrt{12} = 2\\sqrt{3}$ → `(2, 3)`.",
            "student": """
def simplify_surd(n):
    # TODO: pull out the largest square factor of n.
    pass

check(simplify_surd(12), (2, 3))
check(simplify_surd(50), (5, 2))
check(simplify_surd(7),  (1, 7))   # already square-free
""",
            "solution": """
def simplify_surd(n):
    a = 1
    k = 2
    while k * k <= n:
        while n % (k*k) == 0:
            n //= k*k
            a *= k
        k += 1
    return (a, n)

check(simplify_surd(12), (2, 3))
check(simplify_surd(50), (5, 2))
check(simplify_surd(7),  (1, 7))
""",
            "explanation": (
                "Loop $k$ from 2; while $k^2$ divides $n$, divide it out and multiply the "
                "outer factor by $k$. When the loop ends, $n$ has no square factors left."
            ),
        },
        {
            "id": "15.2",
            "prompt": "Write `power(a, m, n)` returning $a^{m/n}$ as a float. Handle negative `m` (so `power(8, -2, 3)` = $8^{-2/3} = 1/4$).",
            "student": """
def power(a, m, n):
    # TODO: compute a**(m/n) — Python's ** handles fractional exponents directly.
    pass

check(power(8, 1, 3), 2.0)    # cube root of 8
check(power(16, 3, 4), 8.0)   # 16^(3/4) = (16^(1/4))^3 = 2^3
check(power(8, -2, 3), 0.25)  # 1 / 8^(2/3) = 1 / 4
""",
            "solution": """
def power(a, m, n):
    return a ** (m / n)

check(power(8, 1, 3), 2.0)
check(power(16, 3, 4), 8.0)
check(power(8, -2, 3), 0.25)
""",
            "explanation": (
                "`a ** (m/n)` covers all the index laws automatically. The only thing "
                "to remember when working by hand is the *order* — taking the nth root "
                "first (`a^(1/n)`) keeps numbers small."
            ),
        },
        {
            "id": "15.3",
            "prompt": "Write `rationalise(num, surd_denom)` that returns the rationalised numerator. For $\\dfrac{\\text{num}}{\\sqrt{d}}$, multiplying by $\\sqrt{d}/\\sqrt{d}$ gives $\\dfrac{\\text{num} \\cdot \\sqrt{d}}{d}$ — return the numerator coefficient as a float (i.e. just `num`, since the surd part stays $\\sqrt{d}$ symbolically). Specifically, return `num / d` (the coefficient of $\\sqrt{d}$ in the rationalised form).",
            "student": """
def rationalise(num, d):
    # 1 / sqrt(3)  →  sqrt(3) / 3  →  return 1/3 (coefficient of sqrt(3))
    # 5 / sqrt(2)  →  5*sqrt(2)/2  →  return 5/2
    # TODO
    pass

check(rationalise(1, 3), 1/3)
check(rationalise(5, 2), 2.5)
check(rationalise(6, 3), 2.0)
""",
            "solution": """
def rationalise(num, d):
    return num / d

check(rationalise(1, 3), 1/3)
check(rationalise(5, 2), 2.5)
check(rationalise(6, 3), 2.0)
""",
            "explanation": (
                "Multiplying top and bottom by $\\sqrt{d}$ turns the denominator from "
                "$\\sqrt{d}$ into $d$, so $\\dfrac{n}{\\sqrt{d}} = \\dfrac{n\\sqrt{d}}{d}$. "
                "The coefficient of $\\sqrt{d}$ in the result is simply $n/d$."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 16 — Functions and rearranging formulae
# ----------------------------------------------------------------------
SEC_16 = {
    "title": "## 16. Functions and rearranging formulae",
    "intro": (
        "**Function notation**: $f(x) = 3x + 2$ means \"the function f sends x to 3x+2\".\n"
        "- $f(4) = 14$ — substitute the number for x.\n"
        "- **Composite**: $fg(x) = f(g(x))$ — apply g first, then f.\n"
        "- **Inverse** $f^{-1}$ undoes f. To find it, swap x and y, then solve for y.\n\n"
        "**Rearranging formulae** (changing the subject) uses the same rules as solving "
        "equations: do the same to both sides until the variable you want is alone."
    ),
    "worked_intro": "If $f(x) = 3x + 2$ and $g(x) = x^2$, find $f(4)$, $fg(2)$, and $f^{-1}(11)$.",
    "worked_code": """
def f(x): return 3*x + 2
def g(x): return x**2

print(f"f(4)    = {f(4)}")           # 3*4 + 2 = 14
print(f"f(g(2)) = {f(g(2))}")        # g(2)=4, f(4)=14
print(f"g(f(2)) = {g(f(2))}")        # f(2)=8,  g(8)=64

# Inverse of f(x) = 3x + 2:
#   y = 3x + 2  →  x = (y - 2) / 3  →  f^-1(y) = (y - 2) / 3
def f_inv(y): return (y - 2) / 3
print(f"f^-1(11) = {f_inv(11)}")     # = 3
""",
    "exercises": [
        {
            "id": "16.1",
            "prompt": "Define `compose(f, g, x)` that returns $f(g(x))$.",
            "student": """
def compose(f, g, x):
    # TODO
    pass

check(compose(lambda a: a + 1, lambda a: a * 2, 5), 11)   # 2*5 + 1 = 11
check(compose(lambda a: a**2, lambda a: a - 3, 5), 4)     # (5-3)^2 = 4
""",
            "solution": """
def compose(f, g, x):
    return f(g(x))

check(compose(lambda a: a + 1, lambda a: a * 2, 5), 11)
check(compose(lambda a: a**2, lambda a: a - 3, 5), 4)
""",
            "explanation": (
                "Composition is just nested function calls: apply g first, then feed "
                "its output into f. Reading $fg(x)$ left-to-right but evaluating "
                "right-to-left is a common stumbling block."
            ),
        },
        {
            "id": "16.2",
            "prompt": "The formula for the period of a pendulum is $T = 2\\pi\\sqrt{L/g}$. Make $L$ the subject and write `length_from_period(T, g)` returning $L$. (Square both sides, then rearrange.)",
            "student": """
import math

def length_from_period(T, g):
    # T = 2*pi*sqrt(L/g)  →  T^2 = 4*pi^2 * L/g  →  L = g * T^2 / (4*pi^2)
    # TODO
    pass

check(length_from_period(2.0, 9.81), 0.9939608115313336)
check(length_from_period(1.0, 9.81), 0.2484902028828334)
""",
            "solution": """
import math

def length_from_period(T, g):
    return g * T**2 / (4 * math.pi**2)

check(length_from_period(2.0, 9.81), 0.9939608115313336)
check(length_from_period(1.0, 9.81), 0.2484902028828334)
""",
            "explanation": (
                "Square both sides to remove the root, then multiply both sides by g and "
                "divide by $4\\pi^2$. A 1-second pendulum is roughly 25 cm long — "
                "matches the second test."
            ),
        },
        {
            "id": "16.3",
            "prompt": "Inverse of $f(x) = (x - 4)/5$ is $f^{-1}(x) = 5x + 4$. Write `inverse_linear(m, c, y)` returning $f^{-1}(y)$ for $f(x) = mx + c$ — i.e. solve $y = mx + c$ for $x$.",
            "student": """
def inverse_linear(m, c, y):
    # TODO
    pass

check(inverse_linear(5, -4, 21), 5.0)    # 5*5 - 4 = 21
check(inverse_linear(0.5, 2, 4), 4.0)    # 0.5*4 + 2 = 4
""",
            "solution": """
def inverse_linear(m, c, y):
    return (y - c) / m

check(inverse_linear(5, -4, 21), 5.0)
check(inverse_linear(0.5, 2, 4), 4.0)
""",
            "explanation": (
                "Subtract c, divide by m. The inverse of a linear function is always "
                "linear, with reciprocal gradient and a flipped intercept."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 17 — Advanced trigonometry
# ----------------------------------------------------------------------
SEC_17 = {
    "title": "## 17. Advanced trigonometry",
    "intro": (
        "Right-angled trig (SOH-CAH-TOA) only works when there's a right angle. "
        "For *any* triangle, you have:\n\n"
        "- **Sine rule**: $\\dfrac{a}{\\sin A} = \\dfrac{b}{\\sin B} = \\dfrac{c}{\\sin C}$\n"
        "- **Cosine rule**: $a^2 = b^2 + c^2 - 2bc\\cos A$\n"
        "- **Area**: $\\text{Area} = \\dfrac{1}{2}ab \\sin C$\n\n"
        "Use sine rule when you have a *side and its opposite angle*. "
        "Use cosine rule when you have *all three sides* (to find an angle), "
        "or *two sides and the angle between them* (to find the third side). "
        "Python's `math.sin/cos` expect **radians** — convert with `math.radians`."
    ),
    "worked_intro": "Triangle: a=7, b=10, angle C=42°. Find side c (cosine rule) and area.",
    "worked_code": """
import math

a, b, C_deg = 7, 10, 42
C = math.radians(C_deg)

# Cosine rule for the third side: c^2 = a^2 + b^2 - 2ab*cos(C).
c = math.sqrt(a**2 + b**2 - 2*a*b*math.cos(C))
print(f"c = {c:.4f}")

# Area = 1/2 * a * b * sin(C).
area = 0.5 * a * b * math.sin(C)
print(f"area = {area:.4f}")
""",
    "exercises": [
        {
            "id": "17.1",
            "prompt": "Write `triangle_area(a, b, C_deg)` returning the area of a triangle with two sides of length `a`, `b` and angle `C` (in degrees) between them.",
            "student": """
import math

def triangle_area(a, b, C_deg):
    # Area = 0.5 * a * b * sin(C). Convert C to radians first!
    pass

check(triangle_area(7, 10, 42), 23.419571222560037)
check(triangle_area(5, 5, 90), 12.5)         # right-isoceles
""",
            "solution": """
import math

def triangle_area(a, b, C_deg):
    return 0.5 * a * b * math.sin(math.radians(C_deg))

check(triangle_area(7, 10, 42), 23.419571222560037)
check(triangle_area(5, 5, 90), 12.5)
""",
            "explanation": (
                "Direct formula. The conversion to radians via `math.radians` is the "
                "step most often forgotten — Python's trig functions don't take degrees."
            ),
        },
        {
            "id": "17.2",
            "prompt": "Write `cosine_rule_side(b, c, A_deg)` returning side `a` opposite angle `A`, given the other two sides and the included angle. Use $a^2 = b^2 + c^2 - 2bc\\cos A$.",
            "student": """
import math

def cosine_rule_side(b, c, A_deg):
    # TODO
    pass

check(cosine_rule_side(10, 7, 42), 6.705201296990628)
check(cosine_rule_side(3, 4, 90), 5.0)   # right triangle: 3-4-5
""",
            "solution": """
import math

def cosine_rule_side(b, c, A_deg):
    A = math.radians(A_deg)
    return math.sqrt(b**2 + c**2 - 2*b*c*math.cos(A))

check(cosine_rule_side(10, 7, 42), 6.705201296990628)
check(cosine_rule_side(3, 4, 90), 5.0)
""",
            "explanation": (
                "When $A = 90°$, $\\cos A = 0$ and the cosine rule reduces to "
                "Pythagoras: $a^2 = b^2 + c^2$. The 3-4-5 test confirms it."
            ),
        },
        {
            "id": "17.3",
            "prompt": "Sine rule: write `sine_rule_angle(a, A_deg, b)` returning angle B (in degrees) opposite side b, given side a and its angle A. Use $\\sin B = b \\sin A / a$.",
            "student": """
import math

def sine_rule_angle(a, A_deg, b):
    # TODO
    pass

check(sine_rule_angle(8, 30, 6), 22.024312837)
check(sine_rule_angle(10, 60, 5), 25.65890622)
""",
            "solution": """
import math

def sine_rule_angle(a, A_deg, b):
    A = math.radians(A_deg)
    sinB = b * math.sin(A) / a
    return math.degrees(math.asin(sinB))

check(sine_rule_angle(8, 30, 6), 22.024312837)
check(sine_rule_angle(10, 60, 5), 25.65890622)
""",
            "explanation": (
                "Rearrange the sine rule to $\\sin B = b\\sin A/a$, then take "
                "$\\arcsin$ and convert back to degrees. Watch for the ambiguous case: "
                "if $\\sin B$ is between 0 and 1, B could be acute or obtuse — context "
                "decides which is right."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 18 — Vectors and GCSE statistics
# ----------------------------------------------------------------------
SEC_18 = {
    "title": "## 18. Vectors and GCSE statistics",
    "intro": (
        "**Vectors** in 2D have a direction and a magnitude. Written as a column "
        "$\\binom{a}{b}$ or as $a\\mathbf{i} + b\\mathbf{j}$.\n"
        "- Adding vectors: add the components.\n"
        "- Scalar multiplication: multiply each component.\n"
        "- Magnitude: $|\\mathbf{v}| = \\sqrt{a^2 + b^2}$ (Pythagoras).\n\n"
        "**Cumulative frequency**: running total of the frequencies. The median sits at "
        "the n/2-th data point; the lower and upper quartiles at n/4 and 3n/4.\n\n"
        "**Box plots** show min, lower quartile, median, upper quartile, max — the "
        "**five-number summary**. The **interquartile range** (IQR) is upper − lower."
    ),
    "worked_intro": "Vector arithmetic and a five-number summary on a small dataset.",
    "worked_code": """
import math
import numpy as np

# Vector addition + magnitude.
v = (3, 4)
w = (1, 2)
v_plus_w = (v[0] + w[0], v[1] + w[1])
mag = math.sqrt(v[0]**2 + v[1]**2)
print(f"v + w = {v_plus_w}")
print(f"|v|   = {mag}")

# Five-number summary on a tiny dataset.
data = [5, 7, 8, 12, 13, 14, 18, 21, 22, 30]
q1, med, q3 = np.percentile(data, [25, 50, 75])
print(f"min={min(data)}  Q1={q1}  median={med}  Q3={q3}  max={max(data)}")
print(f"IQR = {q3 - q1}")
""",
    "exercises": [
        {
            "id": "18.1",
            "prompt": "Write `vector_magnitude(a, b)` returning $\\sqrt{a^2 + b^2}$.",
            "student": """
import math

def vector_magnitude(a, b):
    # TODO
    pass

check(vector_magnitude(3, 4), 5.0)
check(vector_magnitude(5, 12), 13.0)
""",
            "solution": """
import math

def vector_magnitude(a, b):
    return math.sqrt(a**2 + b**2)

check(vector_magnitude(3, 4), 5.0)
check(vector_magnitude(5, 12), 13.0)
""",
            "explanation": (
                "Pure Pythagoras applied to the two components. Both test cases use "
                "well-known Pythagorean triples (3-4-5 and 5-12-13)."
            ),
        },
        {
            "id": "18.2",
            "prompt": "Write `add_vectors(v, w)` taking two `(a, b)` tuples and returning their sum as a tuple.",
            "student": """
def add_vectors(v, w):
    # TODO
    pass

check(add_vectors((3, 4), (1, 2)), (4, 6))
check(add_vectors((-1, 5), (1, -5)), (0, 0))
""",
            "solution": """
def add_vectors(v, w):
    return (v[0] + w[0], v[1] + w[1])

check(add_vectors((3, 4), (1, 2)), (4, 6))
check(add_vectors((-1, 5), (1, -5)), (0, 0))
""",
            "explanation": (
                "Component-wise addition. Two equal-and-opposite vectors sum to zero — "
                "the second test sanity-checks that."
            ),
        },
        {
            "id": "18.3",
            "prompt": "Write `iqr(data)` taking a list of numbers and returning the interquartile range (upper quartile minus lower quartile). Use `np.percentile`.",
            "student": """
import numpy as np

def iqr(data):
    # TODO
    pass

check(iqr([5, 7, 8, 12, 13, 14, 18, 21, 22, 30]), 11.25)
check(iqr([1, 2, 3, 4, 5, 6, 7, 8]), 3.5)
""",
            "solution": """
import numpy as np

def iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    return q3 - q1

check(iqr([5, 7, 8, 12, 13, 14, 18, 21, 22, 30]), 11.25)
check(iqr([1, 2, 3, 4, 5, 6, 7, 8]), 3.5)
""",
            "explanation": (
                "IQR is robust to outliers — that's why a box plot's box uses Q1 and Q3 "
                "rather than min/max. NumPy's `percentile` uses linear interpolation "
                "for sub-integer ranks, which matches GCSE expectations."
            ),
        },
    ],
}


SECTIONS = [SEC_13, SEC_14, SEC_15, SEC_16, SEC_17, SEC_18]


def main():
    nb = nbformat.read(NB_PATH, as_version=4)

    # 1) Update the title cell from "Ages 11-15" to "Ages 11-16".
    title = nb.cells[0]
    title.source = (
        title.source
        .replace("11–15", "11–16")
        .replace("Key Stage 3 and early KS4", "Key Stage 3 and KS4 / GCSE")
    )

    # 2) Drop any pre-existing "Well done!" / footer cell — we'll re-add at the end.
    while nb.cells and nb.cells[-1].cell_type == "markdown" and (
        "Well done" in nb.cells[-1].source or "🎉" in nb.cells[-1].source
    ):
        nb.cells.pop()

    # 3) Append the six new sections.
    for sec in SECTIONS:
        nb.cells.extend(build_section(sec))

    # 4) Add the topic list update + new footer.
    nb.cells.append(md(
        "---\n"
        "## 🎉 You've finished the full age 11–16 syllabus\n\n"
        "You've now covered:\n\n"
        "**KS3 (ages 11–14):** integers & BIDMAS, fractions/decimals/percentages, ratio, "
        "powers & standard form, basic algebra, sequences, straight-line graphs, angles, "
        "area & volume, Pythagoras & basic trig, statistics, probability.\n\n"
        "**KS4 / GCSE (ages 14–16):** quadratics, simultaneous equations & inequalities, "
        "surds & advanced indices, functions & formula manipulation, sine/cosine rules, "
        "vectors and GCSE statistics.\n\n"
        "Re-run any cell, tweak the numbers, and watch how the answer changes — that's "
        "the fastest way to make these patterns stick."
    ))

    # Update the title-cell topic list too.
    title.source = title.source.replace(
        "12. Probability (with simulation)",
        "12. Probability (with simulation)\n13. Quadratics\n14. Simultaneous equations and inequalities\n"
        "15. Surds and advanced indices\n16. Functions and rearranging formulae\n"
        "17. Advanced trigonometry\n18. Vectors and GCSE statistics",
    )

    nbformat.validate(nb)
    nbformat.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} — {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
