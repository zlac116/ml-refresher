"""
LMM neural-network calibration surrogate — minimal illustrative example.

End-to-end flow:
  1. Mock the slow LMM Monte-Carlo pricer with a toy vol-surface function.
  2. Generate training data: (LMM params, instrument) -> Black implied vol.
  3. Train a tiny MLP on that data.
  4. "Calibrate": optimiser uses the NN to find LMM params that match market IVs.
  5. Verify: re-run the "full pricer" at the calibrated params, invert Black,
     compare market / NN / MC implied vols.

Run:  python example.py
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

# ---------------------------------------------------------------
# 1. Mock "full LMM Monte Carlo pricer" -- in production this is slow MC.
#    Here it's a deterministic function of (LMM params, instrument).
# ---------------------------------------------------------------
def mock_lmm_iv(lmm_params, T, K, F):
    sig_a, sig_c, sabr_alpha, rho_inf = lmm_params
    log_m = np.log(K / F)
    return (sig_a + 0.1*sig_c + 5*sabr_alpha
            + 0.4*log_m**2 + 0.05*np.sqrt(T) - 0.1*rho_inf*log_m)

def mock_lmm_price(lmm_params, T, K, F, is_call=True):
    iv = mock_lmm_iv(lmm_params, T, K, F)
    return black76_price(F, K, T, iv, is_call)

# ---------------------------------------------------------------
# 2. Black-76 + its inverse (used to convert MC price -> IV at verification)
# ---------------------------------------------------------------
def black76_price(F, K, T, sigma, is_call=True):
    sqrtT = np.sqrt(T)
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    s = 1.0 if is_call else -1.0
    return s * (F*norm.cdf(s*d1) - K*norm.cdf(s*d2))

def black76_implied_vol(price, F, K, T, is_call=True):
    return brentq(lambda sig: black76_price(F, K, T, sig, is_call) - price,
                  1e-6, 5.0)

# ---------------------------------------------------------------
# 3. Generate training data: 5,000 (params, instrument, IV) rows
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
N = 5000
params = np.column_stack([
    rng.uniform(0.10, 0.25, N),    # sig_a
    rng.uniform(0.30, 0.50, N),    # sig_c
    rng.uniform(0.005, 0.025, N),  # sabr_alpha
    rng.uniform(0.10, 0.50, N),    # rho_inf
])
T = rng.uniform(0.5, 10.0, N)
F = rng.uniform(0.02, 0.05, N)
K = F * np.exp(rng.uniform(-0.3, 0.3, N))

X = np.column_stack([params, T, np.log(K/F), F])  # 7 features per row
y = np.array([mock_lmm_iv(p, t, k, f) for p,t,k,f in zip(params, T, K, F)])

# ---------------------------------------------------------------
# 4. Train a tiny MLP surrogate
# ---------------------------------------------------------------
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)

class Surrogate(nn.Module):
    def __init__(self, d_in=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.SiLU(),
            nn.Linear(64, 64),   nn.SiLU(),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

model = Surrogate()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for epoch in range(2000):
    loss = ((model(Xt) - yt) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"final train MSE: {loss.item():.2e}")

def nn_iv(params, instruments):
    feats = [[*params, T_, np.log(K_/F_), F_] for T_,K_,F_ in instruments]
    with torch.no_grad():
        return model(torch.tensor(feats, dtype=torch.float32)).numpy()

# ---------------------------------------------------------------
# 5. "Calibration": optimiser uses the NN inside its inner loop
# ---------------------------------------------------------------
market_instr = [(1.0, 0.030, 0.035),
                (2.0, 0.040, 0.040),
                (5.0, 0.045, 0.040),
                (5.0, 0.035, 0.040)]      # smile point
true_params  = np.array([0.18, 0.40, 0.015, 0.30])
market_iv    = np.array([mock_lmm_iv(true_params, *i) for i in market_instr])

res = least_squares(
    fun=lambda p: nn_iv(p, market_instr) - market_iv,
    x0=[0.20, 0.45, 0.020, 0.40],
    bounds=([0.10, 0.30, 0.005, 0.10],
            [0.25, 0.50, 0.025, 0.50]))
theta_star = res.x
print(f"calibrated params: {theta_star.round(4)}")
print(f"true params:       {true_params}")

# ---------------------------------------------------------------
# 6. Verification: full MC reprice at theta*, invert Black, compare three IVs
# ---------------------------------------------------------------
print(f"\n{'instrument':>20} | market |  NN(θ*)  |  MC(θ*)  |  calib  | surrogate")
print("-" * 78)
for (T_, K_, F_), iv_m in zip(market_instr, market_iv):
    iv_nn = nn_iv(theta_star, [(T_, K_, F_)])[0]
    p_mc  = mock_lmm_price(theta_star, T_, K_, F_)
    iv_mc = black76_implied_vol(p_mc, F_, K_, T_)
    print(f"T={T_:.1f} K={K_:.3f} F={F_:.3f} | "
          f"{iv_m:.4f} | {iv_nn:.4f}  | {iv_mc:.4f}  | "
          f"{(iv_m-iv_mc)*1e4:+5.1f}bp | {(iv_nn-iv_mc)*1e4:+5.1f}bp")
