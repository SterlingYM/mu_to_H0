#!/usr/bin/env python3
"""
build_TRGB_partial.py
Build the TRGB-track SN dataset: TRGB_y.npy, TRGB_C.npy, TRGB_labels.csv

This is the TRGB analogue of prep_SH0ES22_partial.py.  It assembles the SN
observation vector and covariance submatrix used by demo_TRGB_to_H0.ipynb.

Calibrators
-----------
All SH0ES R22 calibrator hosts PLUS the three CATS calibrator galaxies that are
not in SH0ES: NGC 1316, NGC 1404, NGC 4526.
Source: Hoyt et al. 2023 (CATS), Table 2.
  https://iopscience.iop.org/article/10.3847/2041-8213/ace978

Hubble-flow selection
---------------------
Drawn from the full Pantheon+ sample (Scolnic et al. 2022) using CATS-style
criteria: redshift 0.023 < z_HD < 0.15, x1 and c within the Pantheon+ natural
range (CATS used -3 < x1 < 3, -0.3 < c < 0.3 — non-restrictive given the actual
Pantheon+ ranges), and IS_CALIBRATOR == 0.  Any SN already assigned as a
calibrator is excluded from the HF sample.

Required inputs (all in this directory)
----------------------------------------
Pantheon+SH0ES.dat          — included in repo
Pantheon+SH0ES_STAT+SYS.cov — NOT included; download from:
  https://github.com/PantheonPlusSH0ES/DataRelease
  (file: Pantheon+SH0ES_STAT+SYS.cov, ~32 MB)
TRGB_extended_y_labels.csv  — calibrator label table (included in repo)

Outputs (saved to this directory)
-----------------------------------
TRGB_y.npy      — y vector (calibrator m_b_corr + HF Hubble intercepts)
TRGB_C.npy      — covariance submatrix for the selected rows
TRGB_labels.csv — row metadata (type, host, label)

These three files are the defaults loaded by demo_TRGB_to_H0.ipynb.
For an interactive walk-through of the same construction, see prep_TRGB_data.ipynb.
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

PPLUS_DAT   = os.path.join(DATA_DIR, 'Pantheon+SH0ES.dat')
PPLUS_COV   = os.path.join(DATA_DIR, 'Pantheon+SH0ES_STAT+SYS.cov')
TRGB_LABELS = os.path.join(DATA_DIR, 'TRGB_extended_y_labels.csv')

OUT_Y      = os.path.join(DATA_DIR, 'TRGB_y.npy')
OUT_C      = os.path.join(DATA_DIR, 'TRGB_C.npy')
OUT_LABELS = os.path.join(DATA_DIR, 'TRGB_labels.csv')

# -- Pre-flight checks --------------------------------------------------------
if not os.path.exists(PPLUS_COV):
    print("ERROR: Pantheon+SH0ES_STAT+SYS.cov not found.")
    print()
    print("Download it from the Pantheon+ data release:")
    print("  https://github.com/PantheonPlusSH0ES/DataRelease")
    print("  File: Pantheon+SH0ES_STAT+SYS.cov  (~32 MB)")
    print(f"Place it in: {DATA_DIR}")
    sys.exit(1)

# -- Load data ----------------------------------------------------------------
print("Loading Pantheon+ data...")
pplus = pd.read_csv(PPLUS_DAT, sep=r'\s+', comment='#')
pplus.index.name = 'pp_idx'

print("Loading covariance matrix (this may take a moment)...")
cov_flat = np.loadtxt(PPLUS_COV, skiprows=1)
cov_full = cov_flat.reshape(len(pplus), len(pplus))
print(f"  Covariance matrix shape: {cov_full.shape}")

print("Loading TRGB extended labels...")
trgb_labels = pd.read_csv(TRGB_LABELS)

# -- Build calibrator set -----------------------------------------------------
# TRGB_extended_y_labels mixes two index systems in 'Unnamed: 0':
#   - SH0ES rows: FITS-file indices (3130+), not valid Pantheon+ row indices
#   - New rows (n1316, n1404, n4526): Pantheon+ row indices (4-55)
# Solution: match ALL calibrators to Pantheon+ by CID + IDSURVEY key, which is
# unambiguous and present in both files.
cal_labels = trgb_labels[trgb_labels['calib_host'].notna()].copy()

pplus['_key'] = pplus['CID'].astype(str) + '_' + pplus['IDSURVEY'].astype(str)
pp_key_to_idx = {k: i for i, k in enumerate(pplus['_key'])}

calib_records = []
skipped_cal = []
for _, row in cal_labels.iterrows():
    host     = str(row['calib_host']).lower()
    cid      = str(row['CID'])
    idsurvey = str(int(float(row['IDSURVEY'])))
    key      = f"{cid}_{idsurvey}"
    if key not in pp_key_to_idx:
        skipped_cal.append((host, cid, idsurvey))
        continue
    pp_idx = pp_key_to_idx[key]
    label  = f"{host}_{cid}_{idsurvey}"
    calib_records.append({'pp_idx': pp_idx, 'label': label, 'type': 'CAL', 'host': host})

if skipped_cal:
    print(f"  Warning: {len(skipped_cal)} calibrator SN(s) not found in Pantheon+ — skipped:")
    for h, c, s in skipped_cal:
        print(f"    {h}: {c}_{s}")

calib_df = pd.DataFrame(calib_records)
calib_pp_idx = calib_df['pp_idx'].values

print(f"\nCalibrators: {len(calib_df)} SN observations across "
      f"{calib_df['host'].nunique()} host galaxies")

# -- Build Hubble-flow set ----------------------------------------------------
# CATS-style selection: z range, IS_CALIBRATOR == 0, exclude our calibrator SNe.
# x1/c cuts are non-restrictive given Pantheon+ actual ranges (see module docstring).
c_light = 299792.458  # km/s

def q0_jerk_correction(z, q0=-0.55, j0=1.0):
    return 0.5*(1 - q0)*z - (1/6)*(1 - q0 - 3*q0**2 + j0)*z**2

calib_cid_set = set(pplus.loc[calib_pp_idx, 'CID'].astype(str).str.lower())

hf_mask = (
    pplus['zHD'].between(0.023, 0.15) &
    (pplus['IS_CALIBRATOR'] == 0) &
    pplus['x1'].between(-3, 3) &
    pplus['c'].between(-0.3, 0.3) &
    ~pplus['CID'].astype(str).str.lower().isin(calib_cid_set)
)

pplus_hf = pplus[hf_mask].copy()
print(f"Hubble-flow SNe: {len(pplus_hf)}")

z_hf  = pplus_hf['zHD'].values
logcz = 5 * np.log10(c_light * z_hf * (1 + q0_jerk_correction(z_hf)))
y_hf  = pplus_hf['m_b_corr'].values - logcz - 25

hf_records = []
for pp_idx, (_, row) in zip(pplus_hf.index.values, pplus_hf.iterrows()):
    label = f"{row['CID']}_{int(row['IDSURVEY'])}"
    hf_records.append({'pp_idx': int(pp_idx), 'label': label, 'type': 'HF', 'host': ''})

hf_df = pd.DataFrame(hf_records)

# -- Assemble y vector and covariance submatrix -------------------------------
y_calib = pplus.loc[calib_pp_idx, 'm_b_corr'].values

y = np.concatenate([y_calib, y_hf])
all_idx = np.concatenate([calib_pp_idx, pplus_hf.index.values]).astype(int)

print(f"\nExtracting {len(all_idx)}x{len(all_idx)} covariance submatrix...")
C = cov_full[np.ix_(all_idx, all_idx)]

labels_out = pd.concat([calib_df, hf_df], ignore_index=True)
labels_out = labels_out.drop(columns='pp_idx')

# -- Save ---------------------------------------------------------------------
np.save(OUT_Y, y)
np.save(OUT_C, C)
labels_out.to_csv(OUT_LABELS, index=False)

print(f"\nSaved:")
print(f"  TRGB_y.npy       shape {y.shape}")
print(f"  TRGB_C.npy       shape {C.shape}")
print(f"  TRGB_labels.csv  {len(labels_out)} rows")
print(f"\nCalibrator hosts ({calib_df['host'].nunique()}):")
for h in sorted(calib_df['host'].unique()):
    n = (calib_df['host'] == h).sum()
    print(f"  {h:<12}  {n} SN obs")
print(f"\nThese files are the defaults loaded by demo_TRGB_to_H0.ipynb.")
