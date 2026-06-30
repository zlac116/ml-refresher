"""Pure-function tests for the validator node.

The validator is a plain Python function — no LLM, no httpx, no graph.
Every test here:
  1. Constructs a WorkflowState dict in-line.
  2. Calls validator_node(state).
  3. Asserts on the returned Command's `update` dict.

No mocks needed; no fixtures other than `env_with_api_key` (the validator
reads thresholds from Settings, which needs the env keys to construct).

Refs:
  - LangGraph Command:    https://docs.langchain.com/oss/python/langgraph/use-graph-api
  - pytest assertions:    https://docs.pytest.org/en/stable/how-to/assert.html
"""
import numpy as np
import pytest

from app.validator import (
    LMM_HI,
    LMM_LO,
    LMM_NAMES,
    _drop_worst_quote,
    _perturb_x0,
    _pick_strategy,
    _random_x0,
    validator_node,
)


# ============================================================================
# Helpers — build small state fragments to keep test bodies readable.
# ============================================================================
_DEFAULT_THETA = {"sig_a": 0.18, "sig_c": 0.40, "sabr_alpha": 0.015, "rho_inf": 0.30}


def _calibration(rmse: float, rows: list[dict] | None = None, success: bool = True) -> dict:
    """Build a minimal `state['calibration']` dict for tests."""
    return {
        "theta_star":    dict(_DEFAULT_THETA),
        "cost":          0.001,
        "success":       success,
        "message":       "ok",
        "model_version": 1,
        "verify": {
            "rmse_calib_bp":     rmse,
            "rmse_surrogate_bp": rmse,
            "rows":              rows or [{"calib_bp": -rmse}, {"calib_bp": -rmse}],
        },
    }


# ============================================================================
# Tests on `validator_node` — one per decision branch.
# ============================================================================
class TestValidatorAccepts:
    """rmse below the accept threshold → no retry, no x0 written."""

    def test_accepts_when_rmse_below_threshold(self, env_with_api_key):
        state = {
            "calibration":      _calibration(rmse=10.0),
            "retries":          0,
            "strategies_tried": [],
        }
        cmd = validator_node(state)

        # Should NOT trigger any retry mechanics
        assert "retry_x0" not in cmd.update
        assert "market_quotes" not in cmd.update
        assert "strategies_tried" not in cmd.update

        # SHOULD update best_calibration (current is the best so far)
        assert cmd.update["best_calibration"]["verify"]["rmse_calib_bp"] == 10.0

        # Message confirms acceptance
        msg = cmd.update["messages"][0].content
        assert "accepted" in msg.lower()


class TestValidatorRetriesByFailureMode:
    """The picker chooses strategy based on the failure mode."""

    def test_picks_drop_worst_when_one_quote_is_an_outlier(self, env_with_api_key):
        # One quote's residual is ~10x worse than the others → outlier.
        rows = [{"calib_bp": -120}, {"calib_bp": -5}, {"calib_bp": -3}]
        state = {
            "calibration":      _calibration(rmse=60.0, rows=rows),
            "market_quotes":    [{"T": 1.0}, {"T": 2.0}, {"T": 5.0}],
            "retries":          0,
            "strategies_tried": [],
        }
        cmd = validator_node(state)

        # Drop_worst was picked
        assert "drop_worst" in cmd.update["strategies_tried"]
        # Market quotes shrunk by one (the worst was index 0 since |-120| was largest)
        assert len(cmd.update["market_quotes"]) == 2
        # retry_x0 is None (drop_worst doesn't change x0)
        assert cmd.update["retry_x0"] is None
        # Retry counter incremented
        assert cmd.update["retries"] == 1
        # Message starts with the "Rerun calibration" prefix the supervisor watches for
        assert cmd.update["messages"][0].content.startswith("Rerun calibration")

    def test_picks_random_restart_when_optimizer_failed(self, env_with_api_key):
        state = {
            "calibration":      _calibration(rmse=60.0, success=False),
            "retries":          0,
            "strategies_tried": [],
        }
        cmd = validator_node(state)

        assert "random_restart" in cmd.update["strategies_tried"]
        assert len(cmd.update["retry_x0"]) == 4
        # All x0 values within LMM bounds
        for name, val, lo, hi in zip(LMM_NAMES, cmd.update["retry_x0"], LMM_LO, LMM_HI):
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"
        assert cmd.update["messages"][0].content.startswith("Rerun calibration")

    def test_picks_perturb_when_no_outlier_no_failure(self, env_with_api_key):
        """Healthy convergence but bad rmse → near-optimal but stuck → perturb."""
        # Tight residuals (no outlier), success=True (no convergence failure)
        rows = [{"calib_bp": -50}, {"calib_bp": -45}, {"calib_bp": -55}]
        state = {
            "calibration":      _calibration(rmse=50.0, rows=rows, success=True),
            "retries":          0,
            "strategies_tried": [],
        }
        cmd = validator_node(state)

        assert "perturb" in cmd.update["strategies_tried"]
        assert len(cmd.update["retry_x0"]) == 4
        assert cmd.update["messages"][0].content.startswith("Rerun calibration")

    def test_skips_strategy_already_tried(self, env_with_api_key):
        """If perturb already failed and an outlier is now visible, drop_worst fires.

        Note: the outlier heuristic uses `max > 2 * median`, so we need at least
        3 rows where one is much larger than the rest for it to trigger.
        """
        rows = [{"calib_bp": -120}, {"calib_bp": -5}, {"calib_bp": -3}]
        state = {
            "calibration":      _calibration(rmse=80.0, rows=rows),
            "market_quotes":    [{"T": 1.0}, {"T": 2.0}, {"T": 5.0}],
            "retries":          1,
            "strategies_tried": ["perturb"],
        }
        cmd = validator_node(state)

        assert cmd.update["strategies_tried"] == ["perturb", "drop_worst"]


class TestValidatorGivesUp:
    """When retries exhaust OR no untried strategy remains → restore best."""

    def test_gives_up_after_max_retries_and_restores_best(self, env_with_api_key):
        # Best-so-far has low rmse; current attempt is worse.
        best = _calibration(rmse=22.0)
        current = _calibration(rmse=55.0)

        state = {
            "calibration":      current,
            "best_calibration": best,
            "retries":          3,  # at the cap (s.retry_max default = 3)
            "strategies_tried": ["perturb", "random_restart", "drop_worst"],
        }
        cmd = validator_node(state)

        # The BEST calibration is restored into the field pricing reads
        assert cmd.update["calibration"]["verify"]["rmse_calib_bp"] == 22.0
        assert cmd.update["best_calibration"]["verify"]["rmse_calib_bp"] == 22.0
        # Message starts with "Max retries reached" — matches the supervisor's prompt rule
        assert cmd.update["messages"][0].content.startswith("Max retries reached")

    def test_gives_up_when_no_untried_strategies_remain(self, env_with_api_key):
        """retries below cap but all 3 strategies tried → still give up."""
        best = _calibration(rmse=22.0)
        state = {
            "calibration":      _calibration(rmse=40.0),
            "best_calibration": best,
            "retries":          2,  # below cap
            "strategies_tried": ["perturb", "random_restart", "drop_worst"],
        }
        cmd = validator_node(state)

        # Best is still restored
        assert cmd.update["calibration"] == best


# ============================================================================
# Tests on the strategy helpers — they're pure, test them in isolation.
# ============================================================================
class TestStrategyHelpers:

    def test_perturb_x0_returns_4_values_within_bounds(self):
        result = _perturb_x0(_DEFAULT_THETA)
        assert len(result) == 4
        for name, val, lo, hi in zip(LMM_NAMES, result, LMM_LO, LMM_HI):
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_perturb_x0_clips_when_jitter_would_exceed_bounds(self):
        """Near-bound theta_star + large noise → clipped to bound."""
        # Pin sig_a at upper bound; perturbation can ONLY clip back
        edge_theta = {"sig_a": 0.25, "sig_c": 0.40, "sabr_alpha": 0.015, "rho_inf": 0.30}
        # Set a seed where np.random.normal returns a large positive value for sig_a
        np.random.seed(0)
        result = _perturb_x0(edge_theta)
        # sig_a must still be within bounds (clipped at 0.25)
        assert LMM_LO[0] <= result[0] <= LMM_HI[0]

    def test_random_x0_is_deterministic_under_seed(self):
        a = _random_x0(seed=42)
        b = _random_x0(seed=42)
        assert a == b

    def test_random_x0_returns_4_values_within_bounds(self):
        result = _random_x0(seed=42)
        assert len(result) == 4
        for name, val, lo, hi in zip(LMM_NAMES, result, LMM_LO, LMM_HI):
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_drop_worst_quote_removes_the_largest_residual(self):
        quotes = [{"id": 0}, {"id": 1}, {"id": 2}]
        # Row index 1 has the worst |calib_bp|
        verify_rows = [{"calib_bp": -5}, {"calib_bp": -90}, {"calib_bp": -3}]
        result = _drop_worst_quote(quotes, verify_rows)

        assert len(result) == 2
        assert {q["id"] for q in result} == {0, 2}  # index 1 dropped


# ============================================================================
# Tests on the picker — exercise the diagnostic logic in isolation.
# ============================================================================
class TestPickStrategy:

    def test_outlier_routes_to_drop_worst(self):
        cal = _calibration(rmse=80.0, rows=[{"calib_bp": -120}, {"calib_bp": -5}, {"calib_bp": -3}])
        assert _pick_strategy(cal, strategies_tried=[]) == "drop_worst"

    def test_outlier_already_tried_falls_through(self):
        cal = _calibration(rmse=80.0, rows=[{"calib_bp": -120}, {"calib_bp": -5}, {"calib_bp": -3}])
        # drop_worst already tried → perturb is the only untried with no failure signal
        assert _pick_strategy(cal, strategies_tried=["drop_worst"]) == "perturb"

    def test_optimizer_failure_routes_to_random_restart(self):
        cal = _calibration(rmse=80.0, success=False)
        assert _pick_strategy(cal, strategies_tried=[]) == "random_restart"

    def test_default_routes_to_perturb(self):
        rows = [{"calib_bp": -50}, {"calib_bp": -45}, {"calib_bp": -55}]
        cal = _calibration(rmse=50.0, rows=rows, success=True)
        assert _pick_strategy(cal, strategies_tried=[]) == "perturb"

    def test_all_strategies_tried_returns_none(self):
        cal = _calibration(rmse=80.0)
        result = _pick_strategy(cal, strategies_tried=["perturb", "drop_worst", "random_restart"])
        assert result is None
