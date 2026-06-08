"""Capstone (YOUR exercise): LMM NN surrogate for calibration.

The real-world workflow is:
  slow MC LMM pricer  →  generate (params, instrument) → IV rows
                      →  train NN surrogate
                      →  calibrate by running scipy.optimize.least_squares
                         with the NN inside the loop (microseconds per call,
                         instead of seconds with MC)
                      →  verify by repricing with the real MC at the
                         calibrated params and Black-inverting

Here the MC pricer is mocked by a fast toy function (`mock_lmm_iv`) so the
workflow runs in seconds. From the NN's perspective the difference is invisible —
it just learns whatever function maps inputs to outputs.

Read the cheatsheet at ../lmm_nn_surrogate.md and the demo at
../example.py for context. You can A/B against the demo as
you go (intentional: same conventions, same ranges, same target).

Time budget: ~3 hours.

Fill in every function marked `# TODO` (each raises NotImplementedError) and
wire them together in main(). Run from this folder using the existing
ml/neural_networks uv env (or any env with numpy/torch/scipy/joblib):

    <repo>/ml/neural_networks/.venv/bin/python surrogate.py
    <repo>/ml/neural_networks/.venv/bin/python surrogate.py \
        --n-data 20000 --epochs 3000 --hidden 64 64
"""
import argparse
import datetime
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


# =============================================================================
# Sampling ranges + dimensions  (PROVIDED — match the demo's conventions)
# =============================================================================
LMM_PARAM_NAMES = ("sig_a", "sig_c", "sabr_alpha", "rho_inf")
LMM_PARAM_LO = np.array([0.10, 0.30, 0.005, 0.10])
LMM_PARAM_HI = np.array([0.25, 0.50, 0.025, 0.50])

T_LO, T_HI = 0.5, 10.0          # years to expiry
F_LO, F_HI = 0.02, 0.05         # forward rate
LOG_M_LO, LOG_M_HI = -0.3, 0.3  # log-moneyness range  -> K = F * exp(log_m)

N_FEATURES = 7  # [sig_a, sig_c, sabr_alpha, rho_inf, T, log(K/F), F]


# =============================================================================
# 1. Mock LMM pricer + Black-76  (PROVIDED — infrastructure, not the exercise)
# =============================================================================
def mock_lmm_iv(lmm_params, T, K, F) -> float:
    """Mock for the slow MC LMM pricer: deterministic IV as a function of
    (params, instrument). Stand-in only — see the cheatsheet for what the real
    MC computes."""
    sig_a, sig_c, sabr_alpha, rho_inf = lmm_params
    log_m = np.log(K / F)
    return (sig_a + 0.1 * sig_c + 5 * sabr_alpha
            + 0.4 * log_m ** 2 + 0.05 * np.sqrt(T) - 0.1 * rho_inf * log_m)


def mock_lmm_price(lmm_params, T, K, F, is_call: bool = True) -> float:
    """Mock MC price = Black-76 evaluated at the mock IV. Used in verification."""
    iv = mock_lmm_iv(lmm_params, T, K, F)
    return black76_price(F, K, T, iv, is_call)


def black76_price(F, K, T, sigma, is_call: bool = True) -> float:
    """Standard Black-76 formula for caplets / swaptions."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    s = 1.0 if is_call else -1.0
    return s * (F * norm.cdf(s * d1) - K * norm.cdf(s * d2))


def black76_implied_vol(price, F, K, T, is_call: bool = True) -> float:
    """Invert Black-76 via Brent: given a price, find σ that reproduces it."""
    return brentq(
        lambda sig: black76_price(F, K, T, sig, is_call) - price,
        1e-6, 5.0,
    )


# =============================================================================
# 2. Data generation  (YOUR exercise — TODO)
# =============================================================================
def generate_data(n: int, seed: int = 0):
    """Generate `n` (params, instrument) -> IV training rows.

    WHY: this is the NN's textbook. At inference time the calibration loop will
    probe (params, instrument) pairs trying to find LMM params that match the
    market. The NN must be able to answer for ANY (params, instrument) inside
    the calibration bounds, so the training set has to cover that whole region.
    Uniform sampling is the simplest space-filling strategy (Sobol / importance
    sampling are smarter, used in production; uniform is the baseline).

    Sample uniformly from the ranges at the top of this file:
      - n rows of 4 LMM params      (LMM_PARAM_LO/HI, shape (4,))
      - T  uniform in [T_LO, T_HI]
      - F  uniform in [F_LO, F_HI]
      - K = F * exp(uniform(LOG_M_LO, LOG_M_HI))   # log-moneyness in fixed range
    For each row, compute target IV via `mock_lmm_iv(params, T, K, F)`.

    Returns:
        X: (n, 7) float32 — [sig_a, sig_c, sabr_alpha, rho_inf, T, log(K/F), F]
        y: (n,)  float32 — IVs

    HINTS:
      - rng = np.random.default_rng(seed); use rng.uniform(...). Pass size= for n rows.
      - rng.uniform(LMM_PARAM_LO, LMM_PARAM_HI, size=(n, 4)) broadcasts each
        column to its own (low, high). For scalar lo/hi: size=n.
      - Looping `[mock_lmm_iv(p, t, k, f) for p,t,k,f in zip(params, T, K, F)]`
        is fine at n=10k (sub-second). Vectorising is an optional speedup.
      - Use the drawn log_m directly in X (it equals np.log(K/F), no recompute).
      - np.column_stack([(n,4), (n,), (n,), (n,)]) → (n, 7). Cast to float32.
    """
    # TODO: implement per the docstring.
    rng = np.random.default_rng(seed)
    
    params = rng.uniform(LMM_PARAM_LO, LMM_PARAM_HI, size=(n, 4))
    T = rng.uniform(T_LO, T_HI, size=n)
    F = rng.uniform(F_LO, F_HI, size=n)
    K = F * np.exp(rng.uniform(LOG_M_LO, LOG_M_HI, size=n)) # log-moneyness in fixed range
    lmm_iv = np.array([mock_lmm_iv(p, t, k, f) for p, t, k, f in zip(params, T, K, F)], dtype=np.float32)
    
    return np.column_stack((params, T, np.log(K/F), F)).astype(np.float32), lmm_iv


def split_train_val(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    """Random shuffle + split into train / val. Returns (X_tr, y_tr, X_va, y_va).

    WHY: the val set tells you whether the surrogate generalises beyond the rows
    it trained on. Watch val MSE during training -- if train MSE keeps falling
    but val MSE rises, you're overfitting. Cheap insurance on a small model.

    HINTS:
      - rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
      - n_va = int(val_frac * len(X)); slice idx into [n_va:] and [:n_va]
      - return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    """
    # TODO: implement per the docstring.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)) # sampled indexes for shuffling
    n_va = max(1, int(val_frac * len(X))) # number of validation samples
    
    train_idx, val_idx = idx[:-n_va], idx[-n_va:] # split the indexes into training and validation sets
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# =============================================================================
# 3. Surrogate model  (YOUR exercise — TODO)
# =============================================================================
class Surrogate(nn.Module):
    """Small MLP: 7 inputs -> hidden... -> 1 output (Black IV).

    WHY: the surrogate replaces the slow MC pricer inside the calibration loop.
    It needs to be fast (microseconds per call) AND smooth (so the least_squares
    optimiser can compute usable Jacobians by finite differences). A small MLP
    is exactly right: fully-connected, smooth activations, no batchnorm or
    dropout (those would add noise and break the smooth-function assumption the
    optimiser relies on).

    SUGGESTED: 2 hidden layers, 64 units each, SiLU or ReLU activation, no
    activation on the output (IV can be any positive real number; we don't want
    to constrain its range with a sigmoid).

    HINTS:
      - Same `for h in hidden` pattern you used in the previous capstones.
      - Final layer: nn.Linear(prev, 1) with NO activation.
      - .squeeze(-1) on the output collapses (N, 1) -> (N,) so the loss compares
        like-shapes against y of shape (N,).
    """

    def __init__(self, d_in: int = N_FEATURES, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        # TODO: build self.net = nn.Sequential(...) per the docstring.
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return self.net(x).squeeze(-1)
        return self.net(x).squeeze(-1)


def train_surrogate(
    model: nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int, lr: float, device: torch.device,
) -> dict:
    """Train `model` on (X_tr, y_tr); record per-epoch train + val MSE.

    WHY: nothing new vs the previous capstones -- the function the NN is fitting
    is just smoother and lower-dimensional than the Wine/California problems.
    Target loss: train MSE around 1e-4 or below. If you can't get there, the
    network or the data may be too small.

    Full-batch training (no DataLoaders): at 10k rows the whole training set
    fits comfortably in memory, and a single optimiser step per epoch is
    actually faster than mini-batching here.

    PATTERN:
        Xt_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
        yt_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
        # (and similarly for val)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train": [], "val": []}
        for epoch in range(epochs):
            model.train()
            pred = model(Xt_tr)
            loss = ((pred - yt_tr) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = ((model(Xt_va) - yt_va) ** 2).mean()
            history["train"].append(loss.item())
            history["val"].append(val_loss.item())
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"epoch {epoch+1:4d} | train {loss.item():.2e} | val {val_loss.item():.2e}")
        return history
    """
    # TODO: implement per the docstring.
    # x_scaler = StandardScaler().fit(X_tr)
    # X_tr = x_scaler.transform(X_tr)
    # X_va = x_scaler.transform(X_va)
    
    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xt_va = torch.tensor(X_va, dtype=torch.float32).to(device)
    yt_va = torch.tensor(y_va, dtype=torch.float32).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}
    
    for epoch in range(epochs):
        model.train()
        pred = model(Xt_tr)
        loss = loss_fn(pred, yt_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xt_va), yt_va)
        
        history["train"].append(loss.item())
        history["val"].append(val_loss.item())
        
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch+1:4d} | train {loss.item():.2e} | val {val_loss.item():.2e}")
                
    return history
        

# =============================================================================
# 4. NN inference helper  (YOUR exercise — TODO)
# =============================================================================
def nn_iv(model: nn.Module, params: np.ndarray, instruments: list[tuple], device: torch.device) -> np.ndarray:
    """Predict IVs for a list of instruments at ONE set of LMM params.

    WHY: this is the bridge between scipy and PyTorch. scipy's least_squares
    sends in raw numpy arrays of LMM params; you need to predict an IV per
    market instrument and return numpy back. This is called THOUSANDS of times
    inside the optimiser, so it has to be fast: build one feature matrix, do
    one forward pass, return numpy. No Python loops, no per-instrument PyTorch
    calls.

    params:       shape (4,) numpy array — sig_a, sig_c, sabr_alpha, rho_inf
    instruments:  list of (T, K, F) tuples
    Returns:      (len(instruments),) numpy array of predicted IVs

    PATTERN:
        feats = [[*params, T_, np.log(K_/F_), F_] for T_, K_, F_ in instruments]
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            return model(x).cpu().numpy()
    """
    # TODO: implement per the docstring.
    feats = [[*params, T_, np.log(K_/F_), F_] for T_, K_, F_ in instruments]
    X = torch.tensor(feats, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy()


# =============================================================================
# 5. Calibration  (YOUR exercise — TODO)
# =============================================================================
def calibrate(
    model: nn.Module,
    market_instruments: list[tuple],
    market_ivs: np.ndarray,
    x0: np.ndarray,
    bounds: tuple,
    device: torch.device,
):
    """Find LMM params that minimise (NN(params) - market_ivs) ** 2.

    WHY: this is the WHOLE POINT of the capstone. In production this loop runs
    overnight because every residual evaluation kicks off an MC pricing job.
    With the NN surrogate inside the loop, each residual evaluation is
    microseconds, so the same calibration runs in seconds.

    least_squares (Levenberg-Marquardt under the hood) wants you to give it the
    residual VECTOR (one entry per instrument). It squares + sums internally.

        residual(params) = nn_iv(model, params, market_instruments) - market_ivs

    BOUNDS are critical: NNs extrapolate WILDLY outside the region they were
    trained on. If you let the optimiser drift to e.g. sig_a = 0.5 (the
    training set only had sig_a up to 0.25), the surrogate's predictions are
    nonsense and the calibration produces garbage. Pass (LMM_PARAM_LO,
    LMM_PARAM_HI) so the optimiser is constrained to the training region.

    PATTERN:
        from scipy.optimize import least_squares
        res = least_squares(
            fun=lambda p: nn_iv(model, p, market_instruments, device) - market_ivs,
            x0=x0,
            bounds=bounds,
        )
        return res     # res.x is θ*, res.cost is final residual sum
    """
    # TODO: implement per the docstring.
    res = least_squares(
        fun=lambda p: nn_iv(model, p, market_instruments, device) - market_ivs,
        x0=x0,
        bounds=bounds
    )
    
    return res


# =============================================================================
# 6. Verification  (YOUR exercise — TODO)
# =============================================================================
def verify(
    model: nn.Module,
    theta_star: np.ndarray,
    market_instruments: list[tuple],
    market_ivs: np.ndarray,
    device: torch.device,
) -> dict:
    """Run the three-way comparison: IV_market vs IV_NN(θ*) vs IV_MC(θ*).

    WHY: the calibration could be wrong for two independent reasons; you need
    to test both. The "MC at θ*" is the truth-from-the-real-pricer (here mocked
    by mock_lmm_price -> Black inversion); IV_NN(θ*) is what the surrogate
    thought; IV_market is the broker quote.

    Two residuals, two questions, two tolerances:

      calib     = IV_market - IV_MC(θ*)
                  → "Is the LMM well-fitted to market?"
                  → Pass if per-instrument < 25 bp and RMSE < 15 bp.
                    (If this fails the LMM literally can't fit the market
                    quotes; nothing to do with the NN.)

      surrogate = IV_NN(θ*) - IV_MC(θ*)
                  → "Was the NN reliable AT θ*?"
                  → Pass if per-instrument < 10 bp.
                    (If this fails the NN was inaccurate exactly where the
                    optimiser landed; retrain or fall back to MC-in-loop.)

    Both must pass. The cheatsheet's "Verification" section explains the
    valuation-control / model-risk framing of these two checks.

    PATTERN (per instrument):
        iv_nn = nn_iv(model, theta_star, [(T, K, F)], device)[0]
        p_mc  = mock_lmm_price(theta_star, T, K, F)
        iv_mc = black76_implied_vol(p_mc, F, K, T)
        # bp = 1e4 * (a - b)

    Print the four-column table (see README for exact format) and compute the
    two RMSEs as summary numbers.

    Returns: dict with rows (one per instrument) and rmse_calib /
    rmse_surrogate summary keys. main() saves this to JSON.
    
        instrument | market |  NN(θ*)  |  MC(θ*)  |  calib  | surrogate
    ------------------------------------------------------------------------------
    T=1.0 K=0.030 F=0.035 | 0.3591 | 0.3612  | 0.3600  |  -8.7bp | +12.3bp
    T=2.0 K=0.040 F=0.040 | 0.3657 | 0.3654  | 0.3650  |  +6.8bp |  +3.8bp
    """
    # TODO: implement per the docstring.
    iv_nn = nn_iv(model, theta_star, market_instruments, device)
    iv_mc = [
        black76_implied_vol(mock_lmm_price(theta_star, T, K, F), F, K, T)
        for T, K, F in market_instruments
    ]
    
    df = pd.DataFrame(market_instruments, columns=["T", "K", "F"])
    df["market"]       = market_ivs
    df["NN(θ*)"]       = iv_nn
    df["MC(θ*)"]       = iv_mc
    df["calib_bp"]     = (df["market"]  - df["MC(θ*)"]) * 1e4
    df["surrogate_bp"] = (df["NN(θ*)"]  - df["MC(θ*)"]) * 1e4
    
    fmt = {
        "T": "{:.1f}".format,
        "K": "{:.3f}".format,
        "F": "{:.3f}".format,
        "market": "{:.4f}".format,
        "NN(θ*)": "{:.4f}".format,
        "MC(θ*)": "{:.4f}".format,
        "calib_bp": "{:+.1f}".format,
        "surrogate_bp": "{:+.1f}".format,
    }
    
    print("\n" + df.to_string(index=False, formatters=fmt))
    
    rmse_calib     = float(np.sqrt(np.mean(df["calib_bp"] ** 2)))
    rmse_surrogate = float(np.sqrt(np.mean(df["surrogate_bp"] ** 2)))
    print(f"\nRMSE calib: {rmse_calib:5.1f} bp   |   RMSE surrogate: {rmse_surrogate:5.1f} bp")
    
    return df
    
# =============================================================================
# 7. Wiring  (YOUR exercise — TODO)
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="LMM NN-surrogate calibration capstone.")
    p.add_argument("--n-data", type=int, default=10_000, help="number of training rows")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="model_outputs")
    args = p.parse_args()

    # Four market instruments (T, K, F) — same as the demo, includes a smile point.
    market_instruments = [
        (1.0, 0.030, 0.035),
        (2.0, 0.040, 0.040),
        (5.0, 0.045, 0.040),
        (5.0, 0.035, 0.040),  # smile point: same F, T as the line above, different K
    ]
    # "True" params used to generate the market IVs (in production these come from the broker).
    true_params = np.array([0.18, 0.40, 0.015, 0.30])
    market_ivs = np.array([mock_lmm_iv(true_params, T, K, F) for (T, K, F) in market_instruments])

    # TODO: wire the pieces together in this order:
    #   1. torch.manual_seed(args.seed); np.random.seed(args.seed)
    #   2. device = "cuda" if available else "cpu"
    #   3. X, y = generate_data(args.n_data, args.seed)
    #      X_tr, y_tr, X_va, y_va = split_train_val(X, y, args.val_frac, args.seed)
    #   4. model = Surrogate(hidden=tuple(args.hidden)).to(device)
    #   5. history = train_surrogate(model, X_tr, y_tr, X_va, y_va,
    #                                args.epochs, args.lr, device)
    #   6. x0 = midpoint of (LMM_PARAM_LO, LMM_PARAM_HI), or a perturbed point.
    #      bounds = (LMM_PARAM_LO, LMM_PARAM_HI).
    #      res = calibrate(model, market_instruments, market_ivs, x0, bounds, device)
    #      theta_star = res.x
    #   7. report = verify(model, theta_star, market_instruments, market_ivs, device)
    #   8. Save three artifacts:
    #        torch.save(model.state_dict(), out_dir / "surrogate.pt")
    #        # OPTIONAL: if you added an X scaler, joblib.dump it here.
    #        (out_dir / "calibration_result.json").write_text(json.dumps({
    #            "theta_star": theta_star.tolist(),
    #            "true_params": true_params.tolist(),
    #            **report,
    #        }, indent=2))
    
    # Set PyTorch and NumPy seeds for reproducibility
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device (GPU if available, else CPU)
    
    # Generate training data and split into train and validation sets
    X, y = generate_data(args.n_data, args.seed)
    
    # Train/val split
    X_tr, y_tr, X_va, y_va = split_train_val(X, y, args.val_frac, args.seed)
    
    # Build the surrogate model and train it
    model = Surrogate(N_FEATURES, tuple(args.hidden)).to(device)
    
    # Train the surrogate model and record training history
    history = train_surrogate(model, X_tr, y_tr, X_va, y_va, args.epochs, args.lr, device)
    
    # Calibrate: find θ* that minimises (NN(θ) - market_ivs)
    x0 = (LMM_PARAM_LO + LMM_PARAM_HI) / 2
    bounds = (LMM_PARAM_LO, LMM_PARAM_HI)
    res = calibrate(model, market_instruments, market_ivs, x0, bounds, device)
    theta_star = res.x
    
    # Verify the calibration results by comparing NN predictions to market IVs and MC IVs at theta_star
    report = verify(model, theta_star, market_instruments, market_ivs, device)
    
    # Save the surrogate model and calibration results
    run_dir = Path(args.out_dir) / datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "surrogate.pt")
    report.to_csv(run_dir / "verify_report.csv", index=False)
    
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "hyperparams": {
            "n_data":   args.n_data,
            "hidden":   args.hidden,
            "epochs":   args.epochs,
            "lr":       args.lr,
            "val_frac": args.val_frac,
        },
        "calibration": {
            "theta_star":  theta_star.tolist(),
            "x0":          x0.tolist(),
            "bounds_lo":   LMM_PARAM_LO.tolist(),
            "bounds_hi":   LMM_PARAM_HI.tolist(),
            "cost":        float(res.cost),
            "nfev":        int(res.nfev),
            "success":     bool(res.success),
            "message":     res.message,
        },
        "market_inputs": {
            "instruments": [list(inst) for inst in market_instruments],
            "ivs":         market_ivs.tolist(),
        },
        "training_summary": {
            "final_train_loss": float(history["train"][-1]),
            "final_val_loss":   float(history["val"][-1]),
            "best_val_loss":    float(min(history["val"])),
        },
    }
    
    with open(run_dir / "run.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index_label="epoch")
    
    print((f"\nArtifacts saved to {run_dir}"))

if __name__ == "__main__":
    main()
