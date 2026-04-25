"""Build the physics notebook from scratch.

12 sections covering KS3 + GCSE physics in the same exercise-driven format
as the mathematics and chemistry notebooks.

Run from project root:
    python fundamentals/build_physics.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat

sys.path.insert(0, str(Path(__file__).parent))
from _build import assemble_notebook  # noqa: E402

NB_PATH = Path(__file__).parent / "physics.ipynb"


TITLE = """
# Physics for Ages 11–16 — Learn by Coding

A code-along revision notebook for KS3 and GCSE physics. We use Python to do the
arithmetic and to *simulate* the systems where it helps — projectiles, decay
curves, circuit calculations.

**How it works:**
1. Each section starts with a **concept explanation**.
2. Then a **worked example** with code you can run.
3. Then **exercises** — write code to solve each one. A `check()` will tell you if you're right.
4. If you're stuck, click the *"Click to reveal solution"* toggle to see a worked answer.

**Topics:**
1. Speed, velocity and acceleration (SUVAT)
2. Newton's laws and forces
3. Momentum and impulse
4. Work and energy (KE, GPE)
5. Power and efficiency
6. Density and pressure
7. Waves
8. Light: reflection and refraction
9. Electric circuits
10. Magnetism and electromagnetism
11. Atomic structure and radioactivity
12. Heat: specific heat capacity and changes of state
"""

SETUP = """
import math
import matplotlib.pyplot as plt
import numpy as np

# Useful constants for GCSE physics.
g_EARTH = 9.81   # m/s^2 (acceleration due to gravity, GCSE often uses 9.8 or 10)
SPEED_OF_LIGHT = 3.0e8       # m/s
SPEED_OF_SOUND_AIR = 343      # m/s at room temperature

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
## 🎉 You've finished GCSE physics essentials

You can now:
- Solve kinematics problems with the SUVAT equations.
- Apply Newton's laws to find forces, masses and accelerations.
- Use momentum conservation in collisions.
- Compute kinetic and gravitational potential energy and apply conservation of energy.
- Calculate power, efficiency, density, pressure (including hydrostatic), wave speed, refraction angles, circuit currents, half-lives, and heat-capacity transfers.

The next step (A-level physics) layers in vectors as a first-class object,
calculus-based mechanics, fields (electric, magnetic, gravitational), wave-particle
duality and a much more rigorous treatment of thermodynamics — but the
quantitative scaffolding is exactly what you've practised here.
"""


# ----------------------------------------------------------------------
# Section 1 — Speed, velocity, acceleration (SUVAT)
# ----------------------------------------------------------------------
SEC_1 = {
    "title": "## 1. Speed, velocity and acceleration",
    "intro": (
        "**Speed** = distance / time. **Velocity** is speed *with a direction* "
        "(positive for one way, negative for the other).\n\n"
        "**Acceleration** = change of velocity / time = $\\dfrac{v - u}{t}$.\n\n"
        "For uniform acceleration, the **SUVAT** equations link the five quantities:\n"
        "- $v = u + at$\n"
        "- $s = ut + \\tfrac{1}{2}at^2$\n"
        "- $v^2 = u^2 + 2as$\n\n"
        "Use whichever equation matches the variables you have. SI units throughout: "
        "m, m/s, m/s², s."
    ),
    "worked_intro": "A car accelerates from 5 m/s to 25 m/s in 8 s. Find its acceleration and the distance covered.",
    "worked_code": """
u = 5.0   # initial velocity (m/s)
v = 25.0  # final velocity   (m/s)
t = 8.0   # time             (s)

a = (v - u) / t                  # m/s^2
s = 0.5 * (u + v) * t            # area of v-t trapezium

print(f"acceleration a = {a} m/s^2")
print(f"distance     s = {s} m")
""",
    "exercises": [
        {
            "id": "1.1",
            "prompt": "Write `acceleration(u, v, t)` returning $\\dfrac{v - u}{t}$.",
            "student": """
def acceleration(u, v, t):
    # TODO
    pass

check(acceleration(0, 30, 6), 5.0)     # 0 to 30 m/s in 6 s
check(acceleration(20, 8, 4), -3.0)    # decelerating
""",
            "solution": """
def acceleration(u, v, t):
    return (v - u) / t

check(acceleration(0, 30, 6), 5.0)
check(acceleration(20, 8, 4), -3.0)
""",
            "explanation": (
                "Acceleration is the rate of change of velocity. A negative "
                "answer means decelerating (slowing down) when moving in the "
                "positive direction."
            ),
        },
        {
            "id": "1.2",
            "prompt": "Use $v = u + at$. Write `final_velocity(u, a, t)`.",
            "student": """
def final_velocity(u, a, t):
    # TODO
    pass

check(final_velocity(0, 9.81, 2), 19.62)    # free-fall from rest for 2 s
check(final_velocity(10, -2, 3), 4.0)       # decelerating
""",
            "solution": """
def final_velocity(u, a, t):
    return u + a * t

check(final_velocity(0, 9.81, 2), 19.62)
check(final_velocity(10, -2, 3), 4.0)
""",
            "explanation": (
                "Direct rearrangement of acceleration's definition. After 2 s of "
                "free-fall (a ≈ g = 9.81), an object reaches ~19.6 m/s — that's "
                "≈ 70 km/h."
            ),
        },
        {
            "id": "1.3",
            "prompt": "Use $v^2 = u^2 + 2as$ to write `final_velocity_dist(u, a, s)` returning $v$ (positive root).",
            "student": """
import math

def final_velocity_dist(u, a, s):
    # TODO: square root of (u^2 + 2*a*s).
    pass

check(final_velocity_dist(0, 9.81, 5), 9.9045)   # dropped from 5 m
check(final_velocity_dist(10, -2, 25), 0.0)      # decelerates to rest
""",
            "solution": """
import math

def final_velocity_dist(u, a, s):
    return math.sqrt(u*u + 2*a*s)

check(final_velocity_dist(0, 9.81, 5), 9.9045)
check(final_velocity_dist(10, -2, 25), 0.0)
""",
            "explanation": (
                "This SUVAT equation doesn't involve t — useful when the time "
                "isn't given. Dropping from 5 m gives $v = \\sqrt{2gs} \\approx 9.9$ m/s."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 2 — Newton's laws and forces
# ----------------------------------------------------------------------
SEC_2 = {
    "title": "## 2. Newton's laws and forces",
    "intro": (
        "**Newton's laws:**\n"
        "1. An object stays at rest, or moves in a straight line at constant velocity, "
        "unless a *resultant* force acts on it.\n"
        "2. $F = ma$ — resultant force = mass × acceleration. SI units: N = kg·m/s².\n"
        "3. Every action has an equal and opposite reaction.\n\n"
        "**Weight** is just gravity acting on mass: $W = mg$ (g ≈ 9.81 N/kg on Earth, "
        "GCSE often uses 10 N/kg).\n\n"
        "If multiple forces act on a body, the **resultant** is their vector sum. "
        "If they balance (zero resultant), the body keeps moving at constant velocity."
    ),
    "worked_intro": "A 60 kg passenger experiences a 240 N resultant force during braking. Find the deceleration.",
    "worked_code": """
F = 240   # N
m = 60    # kg

a = F / m
print(f"a = {a} m/s^2")     # 4.0 m/s^2

# Negative if you want to make the deceleration explicit:
print("This is a deceleration (force points backwards).")
""",
    "exercises": [
        {
            "id": "2.1",
            "prompt": "Write `force(m, a)` returning $F = ma$.",
            "student": """
def force(m, a):
    # TODO
    pass

check(force(10, 2),  20)
check(force(0.5, 9.81), 4.905)   # weight of a 500 g object
""",
            "solution": """
def force(m, a):
    return m * a

check(force(10, 2),  20)
check(force(0.5, 9.81), 4.905)
""",
            "explanation": (
                "Newton's second law. Notice that weight is just F = ma with "
                "a = g — a 500 g apple weighs about 4.9 N."
            ),
        },
        {
            "id": "2.2",
            "prompt": "Write `weight(m)` using `g_EARTH` (already defined globally as 9.81 N/kg).",
            "student": """
def weight(m):
    # TODO
    pass

check(weight(10), 98.1)     # 10 kg → 98.1 N
check(weight(0.2), 1.962)   # 200 g → ~2 N
""",
            "solution": """
def weight(m):
    return m * g_EARTH

check(weight(10), 98.1)
check(weight(0.2), 1.962)
""",
            "explanation": (
                "Weight = mg. On Earth a 1 kg mass weighs ~9.81 N; on the Moon "
                "(g ≈ 1.6 N/kg) the same mass would weigh ~1.6 N — your mass "
                "doesn't change but your weight does."
            ),
        },
        {
            "id": "2.3",
            "prompt": "Write `acceleration_from_forces(m, F_drive, F_friction)` returning the acceleration when a drive force pushes one way and friction opposes it. Use $a = (F_\\text{drive} - F_\\text{friction})/m$.",
            "student": """
def acceleration_from_forces(m, F_drive, F_friction):
    # TODO
    pass

check(acceleration_from_forces(1500, 4500, 1500), 2.0)
check(acceleration_from_forces(800, 2000, 2000), 0.0)   # balanced → constant velocity
""",
            "solution": """
def acceleration_from_forces(m, F_drive, F_friction):
    return (F_drive - F_friction) / m

check(acceleration_from_forces(1500, 4500, 1500), 2.0)
check(acceleration_from_forces(800, 2000, 2000), 0.0)
""",
            "explanation": (
                "Resultant force = drive − friction. When they balance, the resultant "
                "is zero so the acceleration is zero — the car moves at "
                "constant velocity (Newton's first law)."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 3 — Momentum and impulse
# ----------------------------------------------------------------------
SEC_3 = {
    "title": "## 3. Momentum and impulse",
    "intro": (
        "**Momentum** $p = mv$ — kg·m/s. Always *carries a sign* in 1-D problems.\n\n"
        "**Conservation of momentum**: in any closed system, the total momentum "
        "before = total momentum after. This is what lets you solve collision "
        "problems algebraically.\n\n"
        "**Impulse** $\\Delta p = F \\cdot \\Delta t$ — the change of momentum equals "
        "force × time. Doubling the time over which a force acts doubles the "
        "momentum change. (That's why airbags work: same Δp, longer Δt → smaller F.)"
    ),
    "worked_intro": "A 0.5 kg ball moving at 8 m/s hits a wall and rebounds at 6 m/s. Find the change in momentum.",
    "worked_code": """
m = 0.5
u = +8       # before
v = -6       # after (opposite direction → negative)

p_before = m * u
p_after  = m * v
delta_p  = p_after - p_before

print(f"p before: {p_before} kg·m/s")
print(f"p after:  {p_after} kg·m/s")
print(f"Δp     :  {delta_p} kg·m/s")    # = -7
""",
    "exercises": [
        {
            "id": "3.1",
            "prompt": "Write `momentum(m, v)` returning $p = mv$.",
            "student": """
def momentum(m, v):
    # TODO
    pass

check(momentum(2, 5), 10)
check(momentum(0.1, -3), -0.3)
""",
            "solution": """
def momentum(m, v):
    return m * v

check(momentum(2, 5), 10)
check(momentum(0.1, -3), -0.3)
""",
            "explanation": (
                "Definition. Sign matters — if you ignore direction, "
                "conservation calculations break."
            ),
        },
        {
            "id": "3.2",
            "prompt": "Two trolleys collide and stick. $m_1 = 2$ kg at $u_1 = 3$ m/s; $m_2 = 1$ kg at rest. Use conservation of momentum to write `final_velocity_after_collision()` returning their common velocity after.",
            "student": """
def final_velocity_after_collision():
    # m1*u1 + m2*u2 = (m1 + m2) * v
    # TODO
    pass

check(final_velocity_after_collision(), 2.0)
""",
            "solution": """
def final_velocity_after_collision():
    m1, u1 = 2, 3
    m2, u2 = 1, 0
    return (m1 * u1 + m2 * u2) / (m1 + m2)

check(final_velocity_after_collision(), 2.0)
""",
            "explanation": (
                "Total momentum before is $m_1u_1 + m_2u_2 = 6$ kg·m/s. After they "
                "stick, the combined 3 kg mass carries the same 6 kg·m/s, so the "
                "shared velocity is 2 m/s."
            ),
        },
        {
            "id": "3.3",
            "prompt": "Impulse: write `impulse_force(delta_p, delta_t)` returning the average force during the impact.",
            "student": """
def impulse_force(delta_p, delta_t):
    # TODO: F = Δp / Δt
    pass

check(impulse_force(7, 0.05), 140)        # short stop → big force
check(impulse_force(7, 0.5), 14)          # 10× longer → 10× smaller force (airbag!)
""",
            "solution": """
def impulse_force(delta_p, delta_t):
    return delta_p / delta_t

check(impulse_force(7, 0.05), 140)
check(impulse_force(7, 0.5), 14)
""",
            "explanation": (
                "Impulse = Δp = FΔt → F = Δp/Δt. Lengthening the contact time by "
                "10× cuts the peak force by 10×. Crumple zones, airbags and "
                "trampolines all exploit this."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 4 — Work and energy
# ----------------------------------------------------------------------
SEC_4 = {
    "title": "## 4. Work and energy (KE and GPE)",
    "intro": (
        "**Work done** by a force: $W = F \\cdot d$ — N·m = J.\n\n"
        "Two key forms of mechanical energy:\n"
        "- **Kinetic energy**: $E_K = \\tfrac{1}{2}mv^2$\n"
        "- **Gravitational potential energy**: $E_P = mgh$\n\n"
        "**Conservation of energy**: in the absence of friction, $E_K + E_P$ is "
        "constant. A ball dropped from height h converts GPE into KE: $\\tfrac{1}{2}mv^2 "
        "= mgh \\Rightarrow v = \\sqrt{2gh}$."
    ),
    "worked_intro": "A 2 kg ball is dropped from 5 m. What's its speed at the bottom (ignoring air resistance)?",
    "worked_code": """
import math

m = 2
h = 5
g = 9.81

# Energy method: GPE at top = KE at bottom.
GPE = m * g * h          # joules
v = math.sqrt(2 * g * h)  # rearranging 0.5 m v^2 = m g h

print(f"GPE at top  = {GPE:.2f} J")
print(f"speed at bottom = {v:.3f} m/s")
""",
    "exercises": [
        {
            "id": "4.1",
            "prompt": "Write `work_done(F, d)` returning the work done by force F over distance d.",
            "student": """
def work_done(F, d):
    # TODO
    pass

check(work_done(50, 4), 200)
check(work_done(0.5, 10), 5.0)
""",
            "solution": """
def work_done(F, d):
    return F * d

check(work_done(50, 4), 200)
check(work_done(0.5, 10), 5.0)
""",
            "explanation": (
                "Work in joules = force in newtons × distance moved in metres. "
                "Strictly W = Fd cos θ, but at GCSE the force is along the motion "
                "so cos θ = 1."
            ),
        },
        {
            "id": "4.2",
            "prompt": "Write `kinetic_energy(m, v)` and `gpe(m, h)` (use `g_EARTH`).",
            "student": """
def kinetic_energy(m, v):
    # TODO
    pass

def gpe(m, h):
    # TODO
    pass

check(kinetic_energy(2, 10), 100.0)
check(gpe(50, 4), 1962.0)            # 50 kg × 9.81 × 4
""",
            "solution": """
def kinetic_energy(m, v):
    return 0.5 * m * v**2

def gpe(m, h):
    return m * g_EARTH * h

check(kinetic_energy(2, 10), 100.0)
check(gpe(50, 4), 1962.0)
""",
            "explanation": (
                "KE has the ½ and the v² — quadrupling the speed gives 16× the KE. "
                "GPE depends only on the *height* lifted, not the path taken."
            ),
        },
        {
            "id": "4.3",
            "prompt": "From conservation of energy ($mgh = \\tfrac{1}{2}mv^2$) write `speed_after_drop(h)` returning $v = \\sqrt{2gh}$.",
            "student": """
import math

def speed_after_drop(h):
    # TODO
    pass

check(speed_after_drop(5), 9.9045)
check(speed_after_drop(20), 19.8091)
""",
            "solution": """
import math

def speed_after_drop(h):
    return math.sqrt(2 * g_EARTH * h)

check(speed_after_drop(5), 9.9045)
check(speed_after_drop(20), 19.8091)
""",
            "explanation": (
                "Mass cancels — every object hits the ground at the same speed when "
                "air resistance is neglected. Galileo first deduced this; Apollo 15 "
                "famously demonstrated it on the Moon with a hammer and a feather."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 5 — Power and efficiency
# ----------------------------------------------------------------------
SEC_5 = {
    "title": "## 5. Power and efficiency",
    "intro": (
        "**Power** = rate of energy transfer: $P = \\dfrac{E}{t}$ (watts = J/s). "
        "When force is applied at velocity v: $P = Fv$.\n\n"
        "**Efficiency** is what fraction of input energy ends up where you want it:\n\n"
        "$\\eta = \\dfrac{\\text{useful energy out}}{\\text{total energy in}}$\n\n"
        "Always between 0 and 1 (or 0–100%). The lost energy is usually heat from "
        "friction or electrical resistance."
    ),
    "worked_intro": "A motor uses 1500 J to lift a load and dissipates 500 J as heat. Find efficiency.",
    "worked_code": """
useful = 1500
total = 1500 + 500

eff = useful / total
print(f"efficiency = {eff:.2%}")    # 75.00%
""",
    "exercises": [
        {
            "id": "5.1",
            "prompt": "Write `power(E, t)` returning energy/time.",
            "student": """
def power(E, t):
    # TODO
    pass

check(power(600, 30), 20.0)
check(power(1500, 60), 25.0)
""",
            "solution": """
def power(E, t):
    return E / t

check(power(600, 30), 20.0)
check(power(1500, 60), 25.0)
""",
            "explanation": (
                "1 W = 1 J/s. A 100 W bulb left on for 1 hour uses "
                "100 × 3600 = 360 000 J = 0.1 kWh."
            ),
        },
        {
            "id": "5.2",
            "prompt": "Write `efficiency(useful, total)` returning the ratio (0–1, not a percentage).",
            "student": """
def efficiency(useful, total):
    # TODO
    pass

check(efficiency(800, 1000), 0.8)
check(efficiency(45, 100), 0.45)
""",
            "solution": """
def efficiency(useful, total):
    return useful / total

check(efficiency(800, 1000), 0.8)
check(efficiency(45, 100), 0.45)
""",
            "explanation": (
                "Useful energy out divided by total energy in. To express as a "
                "percentage, multiply by 100."
            ),
        },
        {
            "id": "5.3",
            "prompt": "Write `power_from_force(F, v)` returning $P = Fv$.",
            "student": """
def power_from_force(F, v):
    # TODO
    pass

check(power_from_force(500, 20), 10000)   # 500 N at 20 m/s = 10 kW
check(power_from_force(40, 1.5), 60)
""",
            "solution": """
def power_from_force(F, v):
    return F * v

check(power_from_force(500, 20), 10000)
check(power_from_force(40, 1.5), 60)
""",
            "explanation": (
                "P = Fv when force and velocity are in the same direction — handy "
                "for cars at constant speed where the engine force exactly balances "
                "drag."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 6 — Density and pressure
# ----------------------------------------------------------------------
SEC_6 = {
    "title": "## 6. Density and pressure",
    "intro": (
        "**Density**: $\\rho = \\dfrac{m}{V}$ — kg/m³. Water is 1000 kg/m³ (or 1 g/cm³).\n\n"
        "**Pressure**: $P = \\dfrac{F}{A}$ — Pa = N/m². Same force on a smaller area = "
        "much higher pressure (knife edges, drawing pins).\n\n"
        "**Hydrostatic pressure** in a liquid increases with depth:\n\n"
        "$P = \\rho g h$\n\n"
        "(often added to atmospheric pressure for total pressure)."
    ),
    "worked_intro": "A 200 g block measures 10 cm × 5 cm × 4 cm. Find its density. Then find the pressure at 50 m depth in seawater (ρ ≈ 1025 kg/m³).",
    "worked_code": """
# Density.
mass_kg = 0.200
V_m3 = (0.10 * 0.05 * 0.04)         # 0.0002 m^3
rho = mass_kg / V_m3
print(f"density = {rho} kg/m^3")

# Hydrostatic pressure at 50 m depth.
P = 1025 * 9.81 * 50
print(f"pressure at 50 m depth = {P:.0f} Pa  (≈ 5 atm)")
""",
    "exercises": [
        {
            "id": "6.1",
            "prompt": "Write `density(mass_kg, volume_m3)` returning kg/m³.",
            "student": """
def density(mass_kg, volume_m3):
    # TODO
    pass

check(density(2, 0.001), 2000.0)         # 2 kg in 1 litre
check(density(0.001, 1e-6), 1000.0)      # 1 g in 1 cm^3 → water
""",
            "solution": """
def density(mass_kg, volume_m3):
    return mass_kg / volume_m3

check(density(2, 0.001), 2000.0)
check(density(0.001, 1e-6), 1000.0)
""",
            "explanation": (
                "ρ = m/V. Sanity check: pure water has 1 g per cm³, which is 1000 "
                "kg per m³ — the second test case."
            ),
        },
        {
            "id": "6.2",
            "prompt": "Write `pressure(F, A)` returning $P = F/A$ in pascals.",
            "student": """
def pressure(F, A):
    # TODO
    pass

check(pressure(500, 0.01), 50000)        # foot stamp on 100 cm^2
check(pressure(50, 0.0001), 500000)      # same force on a much smaller area
""",
            "solution": """
def pressure(F, A):
    return F / A

check(pressure(500, 0.01), 50000)
check(pressure(50, 0.0001), 500000)
""",
            "explanation": (
                "Smaller area → larger pressure. That's why a sharp knife (tiny "
                "contact area) cuts more easily than a blunt one (large area)."
            ),
        },
        {
            "id": "6.3",
            "prompt": "Write `hydrostatic_pressure(rho, h)` returning $\\rho g h$ using `g_EARTH`.",
            "student": """
def hydrostatic_pressure(rho, h):
    # TODO
    pass

check(hydrostatic_pressure(1000, 10), 98100)      # 10 m of water
check(hydrostatic_pressure(1025, 50), 502762.5)   # 50 m seawater
""",
            "solution": """
def hydrostatic_pressure(rho, h):
    return rho * g_EARTH * h

check(hydrostatic_pressure(1000, 10), 98100)
check(hydrostatic_pressure(1025, 50), 502762.5)
""",
            "explanation": (
                "P = ρgh — pressure rises linearly with depth. 10 m of water adds "
                "roughly 1 atm to the pressure above (atmospheric is 101 325 Pa). "
                "Submarines have to handle this."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 7 — Waves
# ----------------------------------------------------------------------
SEC_7 = {
    "title": "## 7. Waves",
    "intro": (
        "Waves transfer energy without transferring matter. Key quantities:\n"
        "- **Wavelength** λ — distance between successive crests, in metres.\n"
        "- **Frequency** f — number of crests per second, in hertz (Hz).\n"
        "- **Period** T — time for one cycle, in seconds: $T = 1/f$.\n"
        "- **Wave speed** v — how fast crests move, in m/s.\n\n"
        "The single equation tying them together:\n\n"
        "$v = f \\lambda$\n\n"
        "**Transverse** waves (light, water surface, EM): oscillate perpendicular to "
        "travel direction. **Longitudinal** waves (sound, P-waves): oscillate along "
        "the travel direction."
    ),
    "worked_intro": "A radio transmits at 100 MHz. What's its wavelength?",
    "worked_code": """
f = 100e6              # 100 MHz
v = SPEED_OF_LIGHT     # all EM waves travel at c in vacuum

wavelength = v / f
print(f"wavelength = {wavelength} m")    # 3 m
""",
    "exercises": [
        {
            "id": "7.1",
            "prompt": "Write `wave_speed(f, wavelength)` returning $v = f\\lambda$.",
            "student": """
def wave_speed(f, wavelength):
    # TODO
    pass

check(wave_speed(440, 0.78), 343.2)    # middle-A in air
check(wave_speed(2e9, 0.15), 3e8)      # 2 GHz radio
""",
            "solution": """
def wave_speed(f, wavelength):
    return f * wavelength

check(wave_speed(440, 0.78), 343.2)
check(wave_speed(2e9, 0.15), 3e8)
""",
            "explanation": (
                "v = fλ. The first test confirms the speed of sound (~343 m/s) "
                "from middle A's frequency and wavelength."
            ),
        },
        {
            "id": "7.2",
            "prompt": "Write `period(f)` returning $T = 1/f$.",
            "student": """
def period(f):
    # TODO
    pass

check(period(50), 0.02)    # mains AC: 50 Hz → 20 ms
check(period(1000), 0.001) # 1 kHz → 1 ms
""",
            "solution": """
def period(f):
    return 1 / f

check(period(50), 0.02)
check(period(1000), 0.001)
""",
            "explanation": (
                "Period is the inverse of frequency. 50 Hz mains has a period of "
                "20 ms — that's why old fluorescent lights flicker at 100 Hz "
                "(twice per cycle)."
            ),
        },
        {
            "id": "7.3",
            "prompt": "Sound from a person 700 m away takes about 2 seconds. Use this to confirm the speed of sound: write `sound_speed(d, t)`.",
            "student": """
def sound_speed(d, t):
    # TODO
    pass

check(sound_speed(700, 2), 350.0)
check(sound_speed(1715, 5), 343.0)
""",
            "solution": """
def sound_speed(d, t):
    return d / t

check(sound_speed(700, 2), 350.0)
check(sound_speed(1715, 5), 343.0)
""",
            "explanation": (
                "Same as average speed: distance ÷ time. Sound in air at room "
                "temperature is ~343 m/s. Light covers the same distance in "
                "microseconds — that's why you see lightning before hearing thunder."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 8 — Light: reflection and refraction
# ----------------------------------------------------------------------
SEC_8 = {
    "title": "## 8. Light: reflection and refraction",
    "intro": (
        "**Reflection law**: angle of incidence = angle of reflection (both measured "
        "from the normal — the line perpendicular to the surface).\n\n"
        "**Refraction**: light bends when entering a different medium because its "
        "speed changes. Snell's law:\n\n"
        "$n_1 \\sin\\theta_1 = n_2 \\sin\\theta_2$\n\n"
        "where $n$ is the **refractive index** (n_air ≈ 1, n_water ≈ 1.33, "
        "n_glass ≈ 1.5).\n\n"
        "When going from a denser to a less dense medium (e.g. water to air), at "
        "angles past the **critical angle** the light totally internally reflects "
        "instead of escaping."
    ),
    "worked_intro": "Light enters water (n=1.33) from air at 30°. Find the refraction angle.",
    "worked_code": """
import math

theta_1 = math.radians(30)
n1, n2 = 1.0, 1.33

theta_2 = math.asin(n1 * math.sin(theta_1) / n2)
print(f"refracted angle = {math.degrees(theta_2):.2f}°")  # ~22.08°
""",
    "exercises": [
        {
            "id": "8.1",
            "prompt": "Write `snell_angle_2(n1, theta_1_deg, n2)` returning the refracted angle in degrees.",
            "student": """
import math

def snell_angle_2(n1, theta_1_deg, n2):
    # TODO
    pass

check(snell_angle_2(1.0, 30, 1.33), 22.082405)
check(snell_angle_2(1.33, 22.082405, 1.0), 30.0)   # reversibility
""",
            "solution": """
import math

def snell_angle_2(n1, theta_1_deg, n2):
    theta_1 = math.radians(theta_1_deg)
    return math.degrees(math.asin(n1 * math.sin(theta_1) / n2))

check(snell_angle_2(1.0, 30, 1.33), 22.082405)
check(snell_angle_2(1.33, 22.082405, 1.0), 30.0)
""",
            "explanation": (
                "$\\theta_2 = \\arcsin(n_1 \\sin\\theta_1 / n_2)$. The second test "
                "reverses the path — light is reversible, so the same equation "
                "takes you back."
            ),
        },
        {
            "id": "8.2",
            "prompt": "Write `critical_angle(n1, n2)` returning the critical angle (in degrees) for light passing from medium 1 (denser) to medium 2 (less dense). Use $\\sin\\theta_c = n_2 / n_1$.",
            "student": """
import math

def critical_angle(n1, n2):
    # TODO
    pass

check(critical_angle(1.5, 1.0), 41.81031)   # glass to air
check(critical_angle(1.33, 1.0), 48.75348)  # water to air
""",
            "solution": """
import math

def critical_angle(n1, n2):
    return math.degrees(math.asin(n2 / n1))

check(critical_angle(1.5, 1.0), 41.81031)
check(critical_angle(1.33, 1.0), 48.75348)
""",
            "explanation": (
                "Total internal reflection occurs above the critical angle. Glass "
                "fibres (~42°) and the swimmer's view of the underwater world both "
                "depend on this."
            ),
        },
        {
            "id": "8.3",
            "prompt": "Write `refractive_index_from_speeds(c_vacuum, c_medium)` returning $n = c_\\text{vacuum} / c_\\text{medium}$.",
            "student": """
def refractive_index_from_speeds(c_vacuum, c_medium):
    # TODO
    pass

check(refractive_index_from_speeds(3e8, 2.25e8), 1.3333333)   # water
check(refractive_index_from_speeds(3e8, 2.0e8), 1.5)           # glass
""",
            "solution": """
def refractive_index_from_speeds(c_vacuum, c_medium):
    return c_vacuum / c_medium

check(refractive_index_from_speeds(3e8, 2.25e8), 1.3333333)
check(refractive_index_from_speeds(3e8, 2.0e8), 1.5)
""",
            "explanation": (
                "The refractive index *is* the ratio of light speeds. Water slows "
                "light by 1.33×, glass by 1.5×. That slowing is what causes the "
                "bend."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 9 — Electric circuits
# ----------------------------------------------------------------------
SEC_9 = {
    "title": "## 9. Electric circuits",
    "intro": (
        "Three quantities, three relationships:\n"
        "- **Voltage** V (volts) — energy per unit charge.\n"
        "- **Current** I (amps) — rate of charge flow, $I = Q/t$.\n"
        "- **Resistance** R (ohms) — opposition to current.\n\n"
        "**Ohm's law**: $V = IR$.\n\n"
        "**Power**: $P = IV = I^2 R = V^2/R$.\n\n"
        "**Series** circuits: same current through every component, voltages add: "
        "$R_\\text{total} = R_1 + R_2$.\n\n"
        "**Parallel** circuits: same voltage across each branch, currents add: "
        "$\\dfrac{1}{R_\\text{total}} = \\dfrac{1}{R_1} + \\dfrac{1}{R_2}$."
    ),
    "worked_intro": "12 V battery, 6 Ω resistor. Find the current and power.",
    "worked_code": """
V = 12
R = 6

I = V / R
P = I * V

print(f"I = {I} A")
print(f"P = {P} W")    # 24 W
""",
    "exercises": [
        {
            "id": "9.1",
            "prompt": "Write `current(V, R)` and `power(V, I)`.",
            "student": """
def current(V, R):
    # TODO
    pass

def power(V, I):
    # TODO
    pass

check(current(12, 6), 2.0)
check(power(12, 2), 24)
""",
            "solution": """
def current(V, R):
    return V / R

def power(V, I):
    return V * I

check(current(12, 6), 2.0)
check(power(12, 2), 24)
""",
            "explanation": (
                "Ohm's law and the power equation, the two most-used formulae of "
                "GCSE electricity."
            ),
        },
        {
            "id": "9.2",
            "prompt": "Series total resistance: write `series_R(resistors)` taking a list of values and returning their sum.",
            "student": """
def series_R(resistors):
    # TODO
    pass

check(series_R([10, 20, 30]), 60)
check(series_R([5, 5, 5, 5]), 20)
""",
            "solution": """
def series_R(resistors):
    return sum(resistors)

check(series_R([10, 20, 30]), 60)
check(series_R([5, 5, 5, 5]), 20)
""",
            "explanation": (
                "Series resistors stack up — the same current has to push through "
                "all of them, so the total opposition is just the sum."
            ),
        },
        {
            "id": "9.3",
            "prompt": "Parallel total resistance: write `parallel_R(resistors)` returning $\\bigl(\\sum 1/R_i\\bigr)^{-1}$.",
            "student": """
def parallel_R(resistors):
    # TODO
    pass

check(parallel_R([10, 10]), 5.0)         # two equal in parallel halve
check(parallel_R([2, 4, 4]), 1.0)        # 1/2 + 1/4 + 1/4 = 1 → R = 1
""",
            "solution": """
def parallel_R(resistors):
    return 1 / sum(1/r for r in resistors)

check(parallel_R([10, 10]), 5.0)
check(parallel_R([2, 4, 4]), 1.0)
""",
            "explanation": (
                "In parallel, current has multiple routes — total resistance is "
                "always *less* than the smallest individual one. Two equal "
                "resistors in parallel = half their value."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 10 — Magnetism and electromagnetism
# ----------------------------------------------------------------------
SEC_10 = {
    "title": "## 10. Magnetism and electromagnetism",
    "intro": (
        "A current-carrying wire creates a magnetic field around it. Coiling it into "
        "a **solenoid** strengthens the field; adding an iron core makes an electromagnet.\n\n"
        "**Force on a current-carrying conductor in a magnetic field**:\n\n"
        "$F = BIL$  (force = flux density × current × length)\n\n"
        "where B is in **tesla** (T). The force direction follows Fleming's left-hand "
        "rule (thumb = force, first finger = field, second finger = current).\n\n"
        "**Electromagnetic induction**: moving a magnet near a coil (or vice-versa) "
        "generates a voltage. This is how generators, microphones and induction "
        "stoves all work."
    ),
    "worked_intro": "A 0.5 m wire carrying 4 A sits in a 0.2 T field. Find the force on it.",
    "worked_code": """
B = 0.2     # tesla
I = 4       # amps
L = 0.5     # metres

F = B * I * L
print(f"force = {F} N")   # 0.4 N
""",
    "exercises": [
        {
            "id": "10.1",
            "prompt": "Write `motor_force(B, I, L)` returning $F = BIL$.",
            "student": """
def motor_force(B, I, L):
    # TODO
    pass

check(motor_force(0.2, 4, 0.5), 0.4)
check(motor_force(1.5, 10, 0.25), 3.75)
""",
            "solution": """
def motor_force(B, I, L):
    return B * I * L

check(motor_force(0.2, 4, 0.5), 0.4)
check(motor_force(1.5, 10, 0.25), 3.75)
""",
            "explanation": (
                "F = BIL is the motor equation. Increase any of B, I or L to make a "
                "stronger motor. The force direction is given by Fleming's left-hand "
                "rule."
            ),
        },
        {
            "id": "10.2",
            "prompt": "Transformer voltage: $V_s/V_p = N_s/N_p$. Write `secondary_voltage(Vp, Np, Ns)` returning the secondary voltage.",
            "student": """
def secondary_voltage(Vp, Np, Ns):
    # TODO
    pass

check(secondary_voltage(230, 1000, 100), 23.0)     # step-down 10×
check(secondary_voltage(12, 200, 1000), 60.0)       # step-up 5×
""",
            "solution": """
def secondary_voltage(Vp, Np, Ns):
    return Vp * Ns / Np

check(secondary_voltage(230, 1000, 100), 23.0)
check(secondary_voltage(12, 200, 1000), 60.0)
""",
            "explanation": (
                "Voltages are in the same ratio as the turns. Step-down (Ns < Np) "
                "drops voltage and raises current proportionally — that's how the "
                "national grid distributes power efficiently."
            ),
        },
        {
            "id": "10.3",
            "prompt": "Write `is_electromagnet_stronger(turns_a, current_a, turns_b, current_b)` returning `True` if magnet A is stronger (using NI as a proxy for field strength).",
            "student": """
def is_electromagnet_stronger(NA, IA, NB, IB):
    # TODO
    pass

check(is_electromagnet_stronger(100, 2, 50, 3), True)     # 200 vs 150
check(is_electromagnet_stronger(50, 2,  200, 1), False)   # 100 vs 200
""",
            "solution": """
def is_electromagnet_stronger(NA, IA, NB, IB):
    return NA * IA > NB * IB

check(is_electromagnet_stronger(100, 2, 50, 3), True)
check(is_electromagnet_stronger(50, 2,  200, 1), False)
""",
            "explanation": (
                "The product N × I (ampere-turns) determines field strength for "
                "an air-cored solenoid. Doubling either turns or current doubles "
                "the field."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 11 — Atomic structure and radioactivity
# ----------------------------------------------------------------------
SEC_11 = {
    "title": "## 11. Atomic structure and radioactivity",
    "intro": (
        "Atoms have a tiny **nucleus** (protons + neutrons) surrounded by electrons. "
        "Radioactive nuclei decay into more stable ones, emitting one of three "
        "kinds of radiation:\n\n"
        "- **Alpha** (α): helium-4 nucleus. Heavy, stops in paper.\n"
        "- **Beta** (β): high-speed electron. Stops in a few mm of aluminium.\n"
        "- **Gamma** (γ): high-energy EM photon. Needs lead/concrete to attenuate.\n\n"
        "**Half-life** $t_{1/2}$ is the time for half the radioactive nuclei in a "
        "sample to decay. After $n$ half-lives:\n\n"
        "$N = N_0 \\cdot \\left(\\dfrac{1}{2}\\right)^n$"
    ),
    "worked_intro": "A sample has 1000 nuclei and a half-life of 5 years. How many remain after 15 years?",
    "worked_code": """
N0 = 1000
half_life = 5
t = 15

n_halves = t / half_life
N = N0 * (0.5 ** n_halves)
print(f"after {t} years ({n_halves} half-lives): {N:.0f} nuclei")
""",
    "exercises": [
        {
            "id": "11.1",
            "prompt": "Write `remaining_after(N0, t, half_life)` returning the number of nuclei remaining after time t.",
            "student": """
def remaining_after(N0, t, half_life):
    # TODO
    pass

check(remaining_after(1000, 5, 5), 500.0)       # one half-life
check(remaining_after(1000, 15, 5), 125.0)      # three half-lives
check(remaining_after(8000, 0, 5), 8000.0)      # zero time
""",
            "solution": """
def remaining_after(N0, t, half_life):
    return N0 * 0.5 ** (t / half_life)

check(remaining_after(1000, 5, 5), 500.0)
check(remaining_after(1000, 15, 5), 125.0)
check(remaining_after(8000, 0, 5), 8000.0)
""",
            "explanation": (
                "Decay is exponential: every half-life, half remains. After "
                "n half-lives the fraction left is $(1/2)^n$. Notice this is "
                "*not* the same as 'after n half-lives, none remain' — there's "
                "always some fraction left."
            ),
        },
        {
            "id": "11.2",
            "prompt": "Write `n_half_lives(N0, N_now, half_life)` returning how many half-lives have passed (use $\\log_{1/2}$).",
            "student": """
import math

def n_half_lives(N0, N_now, half_life):
    # n = log_{0.5}(N_now / N0)  =  log(N_now/N0) / log(0.5)
    pass

check(n_half_lives(1000, 250, 5), 2.0)
check(n_half_lives(8000, 1000, 5), 3.0)
""",
            "solution": """
import math

def n_half_lives(N0, N_now, half_life):
    return math.log(N_now / N0) / math.log(0.5)

check(n_half_lives(1000, 250, 5), 2.0)
check(n_half_lives(8000, 1000, 5), 3.0)
""",
            "explanation": (
                "Inverting the decay formula: take the log of the ratio, divide by "
                "log(½). Multiply the result by half-life to get the elapsed time."
            ),
        },
        {
            "id": "11.3",
            "prompt": "Carbon-14 has a half-life of 5730 years. A bone has 25% of the C-14 of a fresh bone. How old is it? Write `bone_age()` returning the age in years.",
            "student": """
import math

def bone_age():
    # 25% remaining → n = log(0.25)/log(0.5) = 2 half-lives → age = 2 × 5730
    # TODO
    pass

check(bone_age(), 11460.0)
""",
            "solution": """
import math

def bone_age():
    n = math.log(0.25) / math.log(0.5)
    return n * 5730

check(bone_age(), 11460.0)
""",
            "explanation": (
                "Carbon dating exploits the fact that ¹⁴C in living tissue stays at "
                "atmospheric levels until death; afterwards it decays. 25% means "
                "exactly two half-lives — about 11 460 years."
            ),
        },
    ],
}


# ----------------------------------------------------------------------
# Section 12 — Heat, specific heat capacity and changes of state
# ----------------------------------------------------------------------
SEC_12 = {
    "title": "## 12. Heat: specific heat capacity and changes of state",
    "intro": (
        "**Specific heat capacity** $c$ — the energy needed to raise 1 kg of a "
        "substance by 1 °C. Water's c ≈ 4180 J/(kg·°C).\n\n"
        "$E = m c \\Delta\\theta$\n\n"
        "**Specific latent heat** $L$ — the energy needed to *change state* "
        "(melt or evaporate) without changing temperature. For water: "
        "$L_f$ (fusion) ≈ 334 000 J/kg, $L_v$ (vaporisation) ≈ 2.26 × 10⁶ J/kg.\n\n"
        "$E = m L$\n\n"
        "These two equations cover almost every GCSE thermal-energy problem."
    ),
    "worked_intro": "Heat 0.5 kg of water from 20 °C to 100 °C. How much energy is needed?",
    "worked_code": """
m = 0.5
c_water = 4180     # J/(kg·°C)
delta_theta = 100 - 20

E = m * c_water * delta_theta
print(f"energy = {E:.0f} J  (= {E/1000:.1f} kJ)")    # 167 200 J = 167.2 kJ
""",
    "exercises": [
        {
            "id": "12.1",
            "prompt": "Write `heat_required(m, c, delta_theta)` returning $E = mc\\Delta\\theta$.",
            "student": """
def heat_required(m, c, delta_theta):
    # TODO
    pass

check(heat_required(0.5, 4180, 80), 167200.0)
check(heat_required(2.0, 900, 30), 54000.0)         # aluminium
""",
            "solution": """
def heat_required(m, c, delta_theta):
    return m * c * delta_theta

check(heat_required(0.5, 4180, 80), 167200.0)
check(heat_required(2.0, 900, 30), 54000.0)
""",
            "explanation": (
                "Linear in mass and in temperature change. Water has an unusually "
                "high specific heat capacity (4180) — that's why coastal climates "
                "are mild and a hot-water bottle stays warm for ages."
            ),
        },
        {
            "id": "12.2",
            "prompt": "Write `latent_energy(m, L)` returning $E = mL$ (energy needed to melt or boil mass m).",
            "student": """
def latent_energy(m, L):
    # TODO
    pass

check(latent_energy(0.1, 334000),  33400.0)        # melt 100 g of ice
check(latent_energy(0.1, 2260000), 226000.0)       # boil 100 g of water
""",
            "solution": """
def latent_energy(m, L):
    return m * L

check(latent_energy(0.1, 334000),  33400.0)
check(latent_energy(0.1, 2260000), 226000.0)
""",
            "explanation": (
                "During a phase change, energy goes into breaking molecular bonds, "
                "not raising temperature — that's why the temperature stays at the "
                "boiling point while water is being turned to steam."
            ),
        },
        {
            "id": "12.3",
            "prompt": "Total energy to take 0.1 kg of ice from −10 °C to 100 °C steam: heat ice (c=2100), melt it (L=334000), heat water (c=4180), boil it (L=2260000). Write `ice_to_steam_energy()` returning the total in joules.",
            "student": """
def ice_to_steam_energy():
    # ice -10 → 0:   m c_ice ΔT     = 0.1 × 2100 × 10
    # melt:           m L_f           = 0.1 × 334000
    # water 0 → 100:  m c_water ΔT   = 0.1 × 4180 × 100
    # boil:           m L_v           = 0.1 × 2260000
    # TODO
    pass

check(ice_to_steam_energy(), 303300.0)
""",
            "solution": """
def ice_to_steam_energy():
    return (0.1 * 2100 * 10) + (0.1 * 334000) + (0.1 * 4180 * 100) + (0.1 * 2260000)

check(ice_to_steam_energy(), 303300.0)
""",
            "explanation": (
                "Add the four stages: 2100 + 33 400 + 41 800 + 226 000 = 303 300 J. "
                "Note vaporising is by far the biggest chunk — phase changes "
                "dominate the total energy."
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
