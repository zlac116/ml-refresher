import numpy as np

fac = np.array(
    [[1,2], [-1,0], [2,-1], [-2,-1]]
)

# covariance matrix

var_xx = np.sum([(f[0] -  fac[:,0].mean())**2 for f in fac]) / (len(fac))
covar_xy = np.sum([(f[0] -  fac[:,0].mean()) * (f[1] -  fac[:,1].mean()) for f in fac]) / (len(fac))
var_yy =  np.sum([(f[1] -  fac[:,1].mean())**2 for f in fac]) / (len(fac))
cov = np.array([[var_xx, covar_xy], [covar_xy, var_yy]])

Xc = fac - fac.mean(axis=0)
cov_m = Xc.T @ Xc / len(fac)

print(f"by hand:\n{cov}")
print(f"by hand (matrix):\n{cov_m}")

print(f"numpy:\n{np.cov(fac, rowvar=False, ddof=0)}")

# correlation coeff
corr_xy = covar_xy / (np.sqrt(var_xx) * np.sqrt(var_yy))
corr_xx = var_xx / (np.sqrt(var_xx) * np.sqrt(var_xx))
corr_yy = var_yy / (np.sqrt(var_yy) * np.sqrt(var_yy))

corr = np.array([[corr_xx, corr_xy], [corr_xy, corr_yy]])

print(f"corr by hand:\n{corr_xy}")
print(f"corr matrix:\n{corr}")
print(f"numpy:\n{np.corrcoef(fac, rowvar=False)}")