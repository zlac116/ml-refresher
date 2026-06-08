"""Inference services — calibrate and price, against a loaded model.

The point of this module is to translate between API-shaped data (pydantic
schemas) and the parent capstone's numpy/PyTorch entry points (`nn_iv`,
`calibrate`, `verify`). Routes call these; tests can call these directly.

We import from the parent capstone via the same sys.path shim the CLI uses.
Importing parent capstone code in a server is normally a code smell — but
the parent is a learning script, the API is the extension, and we don't
own a packaged version of the parent. For real prod you'd factor the
parent's pure functions into a small library importable by both.
"""
import sys
from pathlib import Path

import numpy as np
import torch

# Same shim as train_and_register.py.
PARENT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from surrogate import (  # noqa: E402
    LMM_PARAM_HI,
    LMM_PARAM_LO,
    black76_implied_vol,
    calibrate,
    mock_lmm_price,
    nn_iv,
)

from app.config import LMM_PARAM_NAMES  # noqa: E402
from app.schemas import (  # noqa: E402
    Instrument,
    Params,
    VerifyReport,
    VerifyRow,
)


def _device(model: torch.nn.Module) -> torch.device:
    """Where does the model live? We must feed inputs on the same device."""
    return next(model.parameters()).device


def _instruments_to_tuples(instruments: list[Instrument]) -> list[tuple]:
    """Convert pydantic Instruments to the (T, K, F) tuples nn_iv expects."""
    return [(i.T, i.K, i.F) for i in instruments]


# =============================================================================
# /price
# =============================================================================
def run_pricing(
    model: torch.nn.Module,
    params: Params,
    instruments: list[Instrument],
) -> list[float]:
    """Predict IVs for instruments at one set of params. Wraps parent nn_iv.

    WHY: this is the simplest possible call: one forward pass through the
    surrogate. The route hands us pydantic objects; we convert to numpy /
    tuples, call the parent's pure function, return a plain list.

    PATTERN:
        device = _device(model)
        ivs = nn_iv(
            model=model,
            params=params.as_array(),
            instruments=_instruments_to_tuples(instruments),
            device=device,
        )
        return ivs.tolist()
    """
    # TODO 11 — implement per the docstring.
    raise NotImplementedError("TODO 11: run_pricing")


# =============================================================================
# /calibrate
# =============================================================================
def run_calibration(
    model: torch.nn.Module,
    instruments: list[Instrument],
    market_ivs: list[float],
) -> dict:
    """Calibrate LMM params against market IVs. Returns a dict matching
    CalibrateResponse (minus model_version, which the route adds).

    WHY: this is the WHOLE POINT of the API. The parent capstone's
    `calibrate()` does the heavy lifting; we just wrap input/output and
    add the verify report.

    Bounds come from the parent capstone constants — we trust them rather
    than re-importing the schema constants (which are mirrored from them).

    PATTERN:
        device     = _device(model)
        tuples     = _instruments_to_tuples(instruments)
        ivs_np     = np.asarray(market_ivs, dtype=np.float64)
        x0         = (LMM_PARAM_LO + LMM_PARAM_HI) / 2
        bounds     = (LMM_PARAM_LO, LMM_PARAM_HI)

        res        = calibrate(model, tuples, ivs_np, x0, bounds, device)
        theta_star = res.x

        verify_rep = _build_verify_report(
            model, theta_star, tuples, ivs_np, device
        )

        return {
            "theta_star": dict(zip(LMM_PARAM_NAMES, theta_star.tolist())),
            "cost":       float(res.cost),
            "success":    bool(res.success),
            "message":    str(res.message),
            "verify":     verify_rep,
        }
    """
    # TODO 12 — implement per the docstring.
    raise NotImplementedError("TODO 12: run_calibration")


def _build_verify_report(
    model: torch.nn.Module,
    theta_star: np.ndarray,
    instruments: list[tuple],
    market_ivs: np.ndarray,
    device: torch.device,
) -> VerifyReport:
    """Three-way IV comparison + RMSE summary.

    Same shape as the parent capstone's verify() but returns the typed
    pydantic VerifyReport (the parent returns a DataFrame; we cannot send
    a DataFrame over the wire).

    PATTERN:
        iv_nn = nn_iv(model, theta_star, instruments, device)
        iv_mc = np.array([
            black76_implied_vol(mock_lmm_price(theta_star, T, K, F), F, K, T)
            for (T, K, F) in instruments
        ])
        calib_bp     = (market_ivs - iv_mc) * 1e4
        surrogate_bp = (iv_nn      - iv_mc) * 1e4

        rows = [
            VerifyRow(
                instrument=f"T={T:.2f} K={K:.4f} F={F:.4f}",
                market=float(m), nn=float(n), mc=float(c),
                calib_bp=float(cb), surrogate_bp=float(sb),
            )
            for (T, K, F), m, n, c, cb, sb in zip(
                instruments, market_ivs, iv_nn, iv_mc, calib_bp, surrogate_bp
            )
        ]
        return VerifyReport(
            rows=rows,
            rmse_calib_bp=float(np.sqrt(np.mean(calib_bp**2))),
            rmse_surrogate_bp=float(np.sqrt(np.mean(surrogate_bp**2))),
        )
    """
    # TODO 13 — implement per the docstring.
    raise NotImplementedError("TODO 13: _build_verify_report")
