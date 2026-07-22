# TRS Security Leg — Discount Factor Methodology

Context: TRS funding leg matches BBG (correctly discounted on SONIA). Security
leg does not — implied discount factors are lower than pure SONIA by >0.05.
Reference security: **GB00BPJJKP77 / TR43**, UK gilt, 4.75% coupon, maturing
22-Oct-2043. Root cause: the security leg needs a gilt-specific discount curve
(SONIA + spread), not pure SONIA — the spread-solving step is currently
missing.

## Steps

1. **Pull the gilt's own cashflows and market price.**
   TR43's coupon dates/amounts + redemption, and its current observed dirty
   price (e.g. BBG `YAS`).

2. **Reuse the existing SONIA zero curve.**
   Same curve already used for the funding leg — no separate curve build.

3. **Bootstrap the flat Z-spread `s`.**
   Solve for the single spread such that discounting TR43's own cashflows on
   `z_SONIA(t) + s` reproduces its observed dirty price:
   ```
   DirtyPrice = Σ CF_i × exp[−(z_SONIA(t_i) + s) × t_i]
   ```
   Root-find `s`. **This step is currently missing from the engine.**

4. **Build the gilt-specific zero curve.**
   `z_gilt(t) = z_SONIA(t) + s`, on the **same compounding basis** as the
   SONIA zeros — convert first if the spread was quoted on a different basis.

5. **Convert to discount factors at the exact TRS cashflow dates.**
   Use ACT/ACT (gilt convention) for the year fraction:
   ```
   DF_gilt(t) = exp[−z_gilt(t) × t]
   ```

6. **Apply `DF_gilt(t)` to the security-leg cashflows** — not `DF_SONIA(t)`.

## Checks before accepting the result

- **Sign/magnitude sanity check.** Gilts often trade with a small, sometimes
  negative, spread to SONIA (collateral scarcity premium) — a large positive
  `s` implied by a 0.05 DF gap is a flag to re-verify, not a given, for a gilt
  specifically (unlike a corporate/bank credit).
- **Tenor consistency.** Re-run the DF-gap-to-spread estimate
  (`Δz ≈ ΔDF / (t·DF)`) at the TRS security leg's actual cashflow tenor
  (~17y+ for TR43), not an assumed shorter tenor — the same DF gap implies a
  much smaller spread at longer tenors.
- **Date/duration mismatch.** Confirm cashflows are discounted to their own
  dates, not to the TRS maturity — a long-dated gilt referenced in a
  shorter-dated TRS is a common source of date-driven error distinct from any
  spread issue.
