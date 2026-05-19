"""
visual_math.py — five fun maths experiments with pictures!

Run me with:
    python visual_math.py

You'll see:
    1. A magic U-shaped curve  (quadratic equations)
    2. Nature's favourite numbers  (Fibonacci)
    3. Why losing 10% then gaining 10% doesn't break even  (percentages)
    4. Folding paper to the moon  (exponentials)
    5. The "backwards question"  (logarithms)

Each one has a little 🌟 TRY THIS at the bottom. Have a go!
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log2, log10


# ===========================================================================
# 1. THE MAGIC U-CURVE  (quadratic equations)
# ===========================================================================
#
# When you throw a ball, its height makes a U-shape (upside down).
# This curve has a name: a "parabola". Maths writes it like:
#       y = a*x*x + b*x + c
#
# To find where the curve crosses zero (the "roots"), we use the famous
# QUADRATIC FORMULA:
#
#       x = ( -b ± √(b² - 4ac) ) / (2a)
#
# The bit under the square root, b² - 4ac, tells us a LOT:
#       positive  →  two crossings
#       zero      →  the curve just touches zero (one crossing)
#       negative  →  no crossings (the curve floats above or below)

def draw_curve(a, b, c):
    # Step 1: use the QUADRATIC FORMULA to find roots
    inside_sqrt = b * b - 4 * a * c

    if inside_sqrt > 0:
        root1 = (-b + sqrt(inside_sqrt)) / (2 * a)
        root2 = (-b - sqrt(inside_sqrt)) / (2 * a)
        roots = [root1, root2]
        message = f"Two crossings:  x = {root1:.2f}  and  x = {root2:.2f}"
    elif inside_sqrt == 0:
        root = -b / (2 * a)
        roots = [root]
        message = f"Just touches zero at x = {root:.2f}"
    else:
        roots = []
        message = "No crossings — the curve floats above or below the line!"

    print(f"  b² - 4ac = {inside_sqrt}")
    print(f"  {message}")

    # Step 2: draw the curve and mark the roots
    x = np.linspace(-6, 6, 200)
    y = a * x * x + b * x + c

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=3, color="purple",
             label=f"y = {a}·x² + {b}·x + {c}")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)

    for r in roots:
        plt.plot(r, 0, "o", color="red", markersize=15)
        plt.text(r, 1.5, f"x = {r:.2f}", ha="center", color="red", fontsize=11)

    plt.title(f"y = {a}·x² + {b}·x + {c}    ({message})", fontsize=11)
    plt.legend(); plt.grid(alpha=0.3)
    plt.show()


# 🌟 TRY THIS for quadratics:
#   - draw_curve(1, -5, 6)     The classic textbook one — predict the roots first!
#   - draw_curve(1, 0, -4)     Two roots, nice and symmetric
#   - draw_curve(1, -2, 1)     The discriminant is exactly zero — what does that look like?
#   - draw_curve(1, 0, 4)      Negative discriminant — does it touch zero?


# ===========================================================================
# 2. NATURE'S FAVOURITE NUMBERS  (Fibonacci)
# ===========================================================================
#
# Start with: 1, 1
# Add the last two to get the next: 1+1=2, so we have 1, 1, 2
# Keep going:  1+2=3,  2+3=5,  3+5=8,  5+8=13 ...
#
# You can spot these numbers in flower petals, sunflower spirals, pinecones,
# and even how rabbits multiply (the story Fibonacci first told!).

def fibonacci_numbers(how_many):
    numbers = [1, 1]
    while len(numbers) < how_many:
        numbers.append(numbers[-1] + numbers[-2])
    return numbers


def draw_fibonacci():
    nums = fibonacci_numbers(12)
    print("  First 12 Fibonacci numbers:", nums)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(range(1, 13), nums, color="goldenrod")
    for bar, num in zip(bars, nums):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(num), ha="center", va="bottom", fontsize=11)
    plt.title("Each Fibonacci number = sum of the two before it!", fontsize=12)
    plt.xlabel("position"); plt.ylabel("value")
    plt.grid(alpha=0.3)
    plt.show()


# 🌟 TRY THIS for Fibonacci:
#   1. Without using the computer: what are the next 3 numbers after 89?
#      (Answer at the bottom of the file — no peeking!)
#   2. Pick ANY two starting numbers (like 4 and 7) and follow the same rule.
#      What kind of sequence do you get?


# ===========================================================================
# 3. POCKET MONEY MAGIC  (percentage returns)
# ===========================================================================
#
# Imagine you have £100 in your money box.
# Week 1: it grows by 10%   →  £110  (gained £10)
# Week 2: it shrinks by 10%  →   £99  (lost £11!)
# Surprise: you end up with LESS than you started.

def pocket_money_demo(weekly_changes=None):
    if weekly_changes is None:
        weekly_changes = [0.10, -0.10, 0.20, -0.05, 0.15, -0.20, 0.10]

    start = 100.0
    money = start
    history = [start]
    for change in weekly_changes:
        money = money * (1 + change)
        history.append(money)

    weeks = list(range(len(history)))

    plt.figure(figsize=(9, 5))
    plt.plot(weeks, history, "o-", linewidth=3, markersize=10, color="seagreen")
    plt.axhline(start, color="grey", linestyle="--",
                label=f"start: £{start:.0f}")
    for w, m in zip(weeks, history):
        plt.text(w, m + 2, f"£{m:.2f}", ha="center", fontsize=9)
    plt.title("How your £100 changes week by week", fontsize=13)
    plt.xlabel("week"); plt.ylabel("money (£)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.show()

    print(f"  Started with:  £{start:.2f}")
    print(f"  Ended with:    £{money:.2f}")
    print(f"  Change:        £{money - start:+.2f}")


# 🌟 TRY THIS for pocket money:
#   pocket_money_demo([0.50, -0.50])    # gain 50% then lose 50% — surprise!
#   pocket_money_demo([0.10, 0.10, 0.10])  # three good weeks in a row


# ===========================================================================
# 4. FOLDING PAPER TO THE MOON  (exponentials)
# ===========================================================================
#
# A sheet of paper is about 0.1 mm thick.
# Fold it once: 0.2 mm. Fold again: 0.4 mm. Again: 0.8 mm.
# Each fold DOUBLES the thickness.
#
# Crazy question: how many folds before the paper reaches the MOON
# (which is 384,400 km away)?
#
# Spoiler: just 42 folds. That's exponential growth in action!

def paper_to_moon():
    start_mm = 0.1
    moon_mm = 384_400 * 1_000_000   # 384,400 km → millimetres

    folds = list(range(45))
    thickness = [start_mm * (2 ** n) for n in folds]

    # find the first fold that reaches the moon
    moon_fold = next(n for n, t in zip(folds, thickness) if t >= moon_mm)
    print(f"  After {moon_fold} folds the paper is {thickness[moon_fold]:,.0f} mm thick")
    print(f"  — that's farther than the moon!")

    plt.figure(figsize=(9, 5))
    plt.plot(folds, thickness, "o-", linewidth=2, color="purple")
    plt.axhline(moon_mm, color="red", linestyle="--",
                label=f"moon: {moon_mm:.0e} mm away")
    plt.plot(moon_fold, thickness[moon_fold], "r*", markersize=20,
             label=f"reaches moon at fold #{moon_fold}")
    plt.yscale("log")   # numbers get HUGE, so we squash them with a log scale
    plt.title("Thickness of folded paper (each fold = ×2)", fontsize=12)
    plt.xlabel("number of folds"); plt.ylabel("thickness in mm (log scale)")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.show()


# 🌟 TRY THIS for exponentials:
#   Imagine you offer your parent two pocket-money options for 30 days:
#     Option A: £100 every day
#     Option B: 1p on day 1, 2p on day 2, 4p on day 3, 8p on day 4, ...
#               (doubles each day)
#   Which one do you think pays more by day 30?
#   Calculate it: total_B = sum of 0.01 * (2 ** day) for day from 0 to 29
#   (Hint: the answer might surprise you...)


# ===========================================================================
# 5. THE BACKWARDS QUESTION  (logarithms)
# ===========================================================================
#
# Exponentials answer:  "Start at 1. How big after N doublings?"
# Logarithms answer:    "How many doublings does it take to reach X?"
#
# It's the SAME question, asked backwards. That's all a logarithm is —
# the opposite of an exponential.
#
# Examples:
#   2¹⁰ = 1024     so    log₂(1024) = 10
#   10³ = 1000     so    log₁₀(1000) = 3

def logs_demo():
    # how many doublings to reach each of these numbers?
    targets = [10, 100, 1000, 1_000_000, 1_000_000_000]
    doublings = [log2(t) for t in targets]

    plt.figure(figsize=(9, 5))
    bars = plt.bar([str(t) for t in targets], doublings, color="teal")
    for bar, d in zip(bars, doublings):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{d:.1f}", ha="center", va="bottom", fontsize=11)
    plt.title("How many doublings to reach this number?", fontsize=13)
    plt.xlabel("target number"); plt.ylabel("number of doublings (log₂)")
    plt.grid(alpha=0.3, axis="y")
    plt.show()

    print("  Notice: going from 1 to ONE BILLION only takes about 30 doublings!")
    print("  Logs squash huge numbers into small ones.")


# 🌟 TRY THIS for logarithms:
#   1. Without a calculator: 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 = 256.
#      So log₂(256) = ? (count the 2s)
#   2. Earthquakes use a LOG scale (Richter). A magnitude 7 quake is not
#      "a bit bigger" than magnitude 5 — it's 10 × 10 = 100 TIMES stronger.
#      Why? Because each step on the scale means ×10.


# ===========================================================================
# Run all five
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60); print("1. The Magic U-Curve (quadratic formula)"); print("=" * 60)
    draw_curve(1, -3, -4)

    # print("\n" + "=" * 60); print("2. Nature's Favourite Numbers"); print("=" * 60)
    # draw_fibonacci()

    # print("\n" + "=" * 60); print("3. Pocket Money Magic"); print("=" * 60)
    # pocket_money_demo()

    # print("\n" + "=" * 60); print("4. Folding Paper to the Moon"); print("=" * 60)
    # paper_to_moon()

    # print("\n" + "=" * 60); print("5. The Backwards Question (logs)"); print("=" * 60)
    # logs_demo()

    # print("\nDone! Scroll up and look for 🌟 TRY THIS bits to play with.")


# ---------------------------------------------------------------------------
# Fibonacci answer (no peeking!):  after 89 comes 144, 233, 377
# Exponential answer:  Option B's total ≈ £10,737,418 — over £10 million!
# Logs answer:  log₂(256) = 8  (since 2⁸ = 256)
# ---------------------------------------------------------------------------
