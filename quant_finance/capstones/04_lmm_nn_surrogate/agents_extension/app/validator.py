"""Validator node — pure-function retry controller.

Reads the most recent calibration result and decides:
  - ACCEPT (rmse below threshold) → routes back to supervisor; pricing proceeds.
  - RETRY (rmse too high, retries left) → picks one of three strategies by
    failure mode, writes its inputs to state, routes back to supervisor.
  - GIVE UP (max retries hit OR all strategies exhausted) → restores best-so-
    far calibration into state["calibration"], routes back to supervisor.

This is a plain Python node (not an LLM agent) — pure function of state,
deterministic, unit-testable without any LLM stub.

Failure-mode picker (per Q1 design):
  - outlier_detected   → strategy "drop_worst"     (one quote much worse than others)
  - optimizer_failed   → strategy "random_restart" (convergence failed)
  - otherwise          → strategy "perturb"        (near-optimal but stuck)
  - strategies_tried gates so the same strategy never fires twice.

Refs:
  - State updates / Command:   https://docs.langchain.com/oss/python/langgraph/use-graph-api
  - Anthropic prompt patterns: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
"""
import numpy as np
from langchain.messages import HumanMessage
from langgraph.types import Command

from app.config import get_settings
from app.state import WorkflowState

# Mirror of the LMM surrogate's training-region bounds. Strategy outputs MUST
# clip to these — the API rejects x0 outside this box via its Pydantic validator.
LMM_NAMES = ("sig_a", "sig_c", "sabr_alpha", "rho_inf")
LMM_LO    = np.array([0.10, 0.30, 0.005, 0.10])
LMM_HI    = np.array([0.25, 0.50, 0.025, 0.50])


# ============================================================================
# Strategy implementations — each PURE function returns the next attempt's
# input (a new x0 list, or a new quotes list). They do NOT call the API.
# ============================================================================
def _perturb_x0(theta_star: dict[str, float]) -> list[float]:
    """Strategy "perturb": Gaussian jitter around best-so-far theta_star.
    Used when calibration converged but rmse is mediocre — we're near a
    local optimum and a small kick may escape it.
    """
    current   = np.array([theta_star[k] for k in LMM_NAMES])
    noise     = np.random.normal(0, 0.02 * (LMM_HI - LMM_LO))
    perturbed = np.clip(current + noise, LMM_LO, LMM_HI)
    return perturbed.tolist()


def _random_x0(seed: int | None = None) -> list[float]:
    """Strategy "random_restart": fresh uniform sample in the bounding box.
    Used when the optimizer failed to converge — escape the basin entirely.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(LMM_LO, LMM_HI).tolist()


def _drop_worst_quote(quotes: list[dict], verify_rows: list[dict]) -> list[dict]:
    """Strategy "drop_worst": remove the quote with the largest |calib_bp|.
    Used when one quote is dramatically worse than others — likely an
    outlier dragging the fit.
    """
    worst_idx = max(range(len(verify_rows)), key=lambda i: abs(verify_rows[i]["calib_bp"]))
    return [q for i, q in enumerate(quotes) if i != worst_idx]


# ============================================================================
# Failure-mode picker — picks the strategy that matches the failure type.
# ============================================================================
def _pick_strategy(calibration: dict, strategies_tried: list[str]) -> str | None:
    """Return the strategy name to attempt next, or None if all three are exhausted.

    Diagnostic signals (all derived from the current calibration result):
      - outlier_detected: one quote's |calib_bp| exceeds 2 * median of all
      - optimizer_failed: scipy.optimize reported success=False
    """
    rows      = calibration["verify"]["rows"]
    residuals = [abs(r["calib_bp"]) for r in rows]

    outlier_detected = len(residuals) >= 2 and max(residuals) > 2 * float(np.median(residuals))
    optimizer_failed = not calibration.get("success", True)

    if outlier_detected and "drop_worst" not in strategies_tried:
        return "drop_worst"
    if optimizer_failed and "random_restart" not in strategies_tried:
        return "random_restart"
    if "perturb" not in strategies_tried:
        return "perturb"
    return None  # all three already tried


# ============================================================================
# The validator node
# ============================================================================
def validator_node(state: WorkflowState) -> Command:
    """Evaluate the most recent calibration; accept, retry, or give up."""
    s = get_settings()

    current_calibration = state["calibration"]
    current_rmse        = current_calibration["verify"]["rmse_calib_bp"]
    retries             = state.get("retries", 0)
    strategies_tried    = state.get("strategies_tried", [])

    # Track the BEST calibration across retries. Carrying the whole dict
    # (theta_star + verify report + success flag) means the give-up path
    # can restore the full result, not just theta_star.
    best_calibration = state.get("best_calibration")
    best_rmse        = best_calibration["verify"]["rmse_calib_bp"] if best_calibration else float("inf")
    if current_rmse < best_rmse:
        best_calibration = current_calibration
        best_rmse        = current_rmse

    # ---- ACCEPT path ----
    if current_rmse < s.rmse_accept_bp:
        return Command(
            update={
                "messages":         [HumanMessage(
                    f"Calibration accepted (rmse={current_rmse:.1f} bp < threshold={s.rmse_accept_bp:.1f} bp)."
                )],
                "best_calibration": best_calibration,
            },
        )

    # ---- GIVE-UP path: max retries OR all strategies exhausted ----
    strategy = _pick_strategy(current_calibration, strategies_tried)
    if retries >= s.retry_max or strategy is None:
        return Command(
            update={
                "messages":         [HumanMessage(
                    f"Max retries reached. Restoring best calibration (rmse={best_rmse:.1f} bp) "
                    f"and proceeding to pricing."
                )],
                "calibration":      best_calibration,   # ← RESTORE best into the field pricing reads
                "best_calibration": best_calibration,
            },
        )

    # ---- RETRY path: pick a strategy and emit its inputs ----
    base_update: dict = {
        "retries":          retries + 1,
        "strategies_tried": strategies_tried + [strategy],
        "best_calibration": best_calibration,
    }

    if strategy == "perturb":
        # Perturb around the BEST theta_star so far (not necessarily the most
        # recent attempt — best may be from an earlier retry).
        x0 = _perturb_x0(best_calibration["theta_star"])
        return Command(
            update={
                **base_update,
                "retry_x0": x0,
                "messages": [HumanMessage(
                    f"Rerun calibration: rmse={current_rmse:.1f} bp above {s.rmse_accept_bp:.1f} bp threshold; "
                    f"perturbing best-so-far theta_star."
                )],
            },
        )

    if strategy == "random_restart":
        # No seed → genuinely random across retries. Set a seed from Settings
        # later if you want reproducibility for tests.
        x0 = _random_x0(seed=None)
        return Command(
            update={
                **base_update,
                "retry_x0": x0,
                "messages": [HumanMessage(
                    f"Rerun calibration: optimizer failed to converge; retrying with a random x0."
                )],
            },
        )

    # strategy == "drop_worst"
    rows           = current_calibration["verify"]["rows"]
    residuals      = [abs(r["calib_bp"]) for r in rows]
    worst_idx      = max(range(len(rows)), key=lambda i: abs(rows[i]["calib_bp"]))
    worst_bp       = residuals[worst_idx]
    current_quotes = state.get("market_quotes", [])
    new_quotes     = _drop_worst_quote(current_quotes, rows)
    return Command(
        update={
            **base_update,
            "market_quotes": new_quotes,                 # MUTATE quotes; calibration tool reads from state
            "retry_x0":      None,                       # don't reuse a stale x0 alongside a quote drop
            "messages":      [HumanMessage(
                f"Rerun calibration: outlier detected (quote {worst_idx} has |calib_bp|={worst_bp:.1f}, "
                f"~{worst_bp / float(np.median(residuals)):.1f}× the median); dropping it and re-fitting."
            )],
        },
    )
