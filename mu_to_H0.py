#!/usr/bin/env python3
"""
mu_to_H0.py
Second-to-third rung distance ladder solver.

Given Cepheid distance moduli to calibrator SN host galaxies, this tool
solves for H0 and M_B using the SN apparent magnitudes and their full
covariance matrix (accounting for photometric systematics across surveys).

The model (for the selected SN subset) is:

    Calibrator SN i (host h):   y_i = μ_host[h] + M_B
    Hubble-flow SN j:            y_j = M_B - 5·log10(H0)

where y is the SH0ES magnitude observable (apparent magnitude for
calibrators; m_B - 5·log10(c·z) - 25 for Hubble-flow SNe).

Usage
-----
As a library:
    from mu_to_H0 import solve_H0
    result = solve_H0({'m101': 29.04, 'n1365': 31.48, ...})
    print(result['H0'], result['sigma_H0'])

As a script (demo with SH0ES-implied host distances):
    python mu_to_H0.py

Data files required (default: pre-built SH0ES R22 subset, shipped with the repo):
    data/SH0ES22_partial_y.npy / data/SH0ES22_partial_C.npy / data/SH0ES22_partial_labels.csv

To use your own selection instead, run prep_SN_data.ipynb to produce
data/custom_{y,C,labels}.* and pass those paths to load_truncated_data().
"""

import numpy as np
import pandas as pd
from scipy import linalg

# ---------------------------------------------------------------------------
# Default data file paths (pre-built SH0ES R22 subset shipped with the repo)
# ---------------------------------------------------------------------------
_DEFAULT_Y      = 'data/SH0ES22_partial_y.npy'
_DEFAULT_C      = 'data/SH0ES22_partial_C.npy'
_DEFAULT_LABELS = 'data/SH0ES22_partial_labels.csv'


def load_truncated_data(y_path=_DEFAULT_Y, C_path=_DEFAULT_C,
                        labels_path=_DEFAULT_LABELS):
    """Load the pre-extracted SN data from prep_SH0ES22_partial.py."""
    y      = np.load(y_path)
    C      = np.load(C_path)
    labels = pd.read_csv(labels_path)
    return y, C, labels


def calibrator_hosts(labels):
    """Return the ordered list of unique calibrator host galaxy names."""
    cal = labels[labels['type'] == 'CAL']
    return list(dict.fromkeys(cal['host']))   # preserves insertion order


def solve_H0(mu_host, sigma_mu_host=None, y=None, C=None, labels=None, verbose=True):
    """
    Solve for H0 given Cepheid distance moduli to calibrator host galaxies.

    Parameters
    ----------
    mu_host : dict
        Mapping of host galaxy name → distance modulus (mag).
        All calibrator hosts must be present; run calibrator_hosts()
        to get the required keys.  Setting a host's value to NaN
        effectively drops that host from the fit by inflating the
        covariance diagonal of its calibrator SNe.
    sigma_mu_host : dict, optional
        Mapping of host galaxy name → 1-sigma uncertainty on the distance
        modulus (mag).  When provided, each host's uncertainty is added to
        the covariance matrix as a fully-correlated term across all
        calibrator SNe in that host (i.e. C_eff[i,j] += σ_h² for every
        pair i,j of SNe belonging to host h).  Hosts absent from this dict
        are treated as having zero distance uncertainty.
    y, C, labels : array-like, optional
        Pre-loaded data arrays.  Loaded from default files if not provided.
    verbose : bool
        Print a brief results summary.

    Returns
    -------
    dict with keys:
        H0          float  [km/s/Mpc]
        sigma_H0    float  1-sigma uncertainty on H0
        M_B         float  SN absolute magnitude
        sigma_M_B   float  1-sigma uncertainty on M_B
        chi2        float  chi-squared of the best-fit
        dof         int    degrees of freedom (N - 2)
        cov_q       (2,2)  full covariance of [M_B, 5*log10(H0)]
    """
    if y is None or C is None or labels is None:
        y, C, labels = load_truncated_data()

    N = len(y)
    assert N == len(labels) == C.shape[0] == C.shape[1], 'Shape mismatch'

    # ------------------------------------------------------------------
    # Build effective observation vector: subtract host μ for calibrators
    # ------------------------------------------------------------------
    missing = [h for h in calibrator_hosts(labels) if h not in mu_host]
    if missing:
        raise ValueError(f'Missing distance moduli for hosts: {missing}')

    # Hosts whose μ is NaN are dropped from the fit: any contribution they
    # would make is neutralised by adding a huge variance to the covariance
    # diagonal of their calibrator SNe (see below).
    dropped_hosts = {h for h, mu in mu_host.items() if np.isnan(mu)}

    # labels has a default 0-based integer index; i == positional index into y
    y_eff = y.copy()
    for i, row in labels.iterrows():
        if row['type'] == 'CAL' and row['host'] not in dropped_hosts:
            y_eff[i] -= mu_host[row['host']]

    # ------------------------------------------------------------------
    # Augment covariance with Cepheid distance uncertainties.
    # SNe sharing the same host are subject to the same distance error,
    # so the contribution σ_h² is fully correlated across that host's SNe.
    # For dropped hosts (μ = NaN), add a huge diagonal variance so those
    # rows carry negligible weight in the weighted least-squares solution.
    # ------------------------------------------------------------------
    C_eff = C.copy()
    if sigma_mu_host:
        for h, sigma in sigma_mu_host.items():
            if sigma == 0.0 or h in dropped_hosts:
                continue
            cal_idx = labels.index[
                (labels['type'] == 'CAL') & (labels['host'] == h)
            ].tolist()
            var = sigma ** 2
            for i in cal_idx:
                for j in cal_idx:
                    C_eff[i, j] += var

    HUGE_VAR = 1e10
    for h in dropped_hosts:
        cal_idx = labels.index[
            (labels['type'] == 'CAL') & (labels['host'] == h)
        ].tolist()
        for i in cal_idx:
            y_eff[i] = 0.0                 # value is irrelevant; row is down-weighted
            C_eff[i, i] += HUGE_VAR

    # ------------------------------------------------------------------
    # Design matrix for the 2-parameter system [M_B, 5*log10(H0)]
    #
    #   Calibrator:   y_eff[i] = 1 * M_B + 0 * 5log10(H0)
    #   Hubble-flow:  y_eff[j] = 1 * M_B + (-1) * 5log10(H0)
    # ------------------------------------------------------------------
    L = np.zeros((2, N))
    L[0, :] = 1.0                                             # M_B coefficient
    hf_mask = (labels['type'] == 'HF').values
    L[1, hf_mask] = -1.0                                      # H0 coefficient

    # ------------------------------------------------------------------
    # Weighted least squares  q = (L C^{-1} L^T)^{-1} L C^{-1} y_eff
    # ------------------------------------------------------------------
    C_inv = linalg.inv(C_eff)
    A     = L @ C_inv @ L.T        # (2, 2) Fisher matrix
    b     = L @ C_inv @ y_eff      # (2,)
    q     = linalg.solve(A, b)     # [M_B, 5*log10(H0)]
    cov_q = linalg.inv(A)          # parameter covariance

    M_B       = q[0]
    log5_H0   = q[1]               # 5 * log10(H0)
    H0        = 10 ** (log5_H0 / 5)

    # Uncertainty propagation: σ(H0) = H0 * ln(10)/5 * σ(5log10 H0)
    sigma_M_B  = np.sqrt(cov_q[0, 0])
    sigma_H0   = H0 * np.log(10) / 5 * np.sqrt(cov_q[1, 1])

    # chi-squared
    residuals = y_eff - L.T @ q
    chi2      = float(residuals @ C_inv @ residuals)
    n_dropped = ((labels['type'] == 'CAL') & labels['host'].isin(dropped_hosts)).sum()
    dof       = N - int(n_dropped) - 2

    if verbose:
        print(f'M_B       = {M_B:.4f} ± {sigma_M_B:.4f}  mag')
        print(f'H0        = {H0:.4f} ± {sigma_H0:.4f}  km/s/Mpc')
        print(f'chi2/dof  = {chi2:.1f}/{dof} = {chi2/dof:.3f}')

    return dict(H0=H0, sigma_H0=sigma_H0,
                M_B=M_B, sigma_M_B=sigma_M_B,
                chi2=chi2, dof=dof, cov_q=cov_q)


# ---------------------------------------------------------------------------
# Minimal CLI sanity check — prints dataset summary and required host names.
# For a worked example with real Cepheid distances, see demo_mu_to_H0.ipynb.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    y, C, labels = load_truncated_data()

    hosts = calibrator_hosts(labels)
    print(f'Loaded {len(y)} SN rows ({(labels["type"]=="CAL").sum()} cal, '
          f'{(labels["type"]=="HF").sum()} HF), {len(hosts)} calibrator hosts.')
    print(f'\nCalibrator hosts required as keys in mu_host:')
    for h in hosts:
        print(f'  {h}')
    print('\nSee demo_mu_to_H0.ipynb for a worked example with Cepheid distances.')
