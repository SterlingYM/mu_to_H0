# mu_to_H0

***Demonstration/learning purpose only*** A simplified version of distance ladder solver that allows user to modify input distances to SN Ia host galaxies and study the effect on the Hubble constant. Intended for use in CosmoVerse workshop. Do not use this code for cosmological analysis.

## Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `astropy`
- `jupyter` (to run the notebooks)

Install with:
```bash
pip install numpy pandas scipy matplotlib astropy jupyter
```

## Two demos

There are two parallel demos that share the **same solver** (`mu_to_H0.py`):

| Notebook | First-rung distances | Dataset |
|----------|---------------------|---------|
| [`demo_mu_to_H0.ipynb`](demo_mu_to_H0.ipynb) | **Cepheids** (SH0ES R22) | `data/SH0ES22_partial_*` (37 hosts) |
| [`demo_TRGB_to_H0.ipynb`](demo_TRGB_to_H0.ipynb) | **TRGB** | `data/TRGB_*` (40 hosts) |

## Getting started (Cepheid version)

Open **[`demo_mu_to_H0.ipynb`](demo_mu_to_H0.ipynb)** and run the cells top-to-bottom. The notebook walks through:

1. Loading the pre-built SN dataset (`data/SH0ES22_partial_*`).
2. Providing Cepheid distance moduli to each calibrator host.
3. Calling `solve_H0` to recover H0 and M_B.
4. Exploring how H0 responds to shifts in the Cepheid distance scale.

### Using your own SN data

If you want to regenerate the SN data from Pantheon+ inputs (e.g. to apply different redshift cuts or kinematic corrections), run **[`prep_SN_data.ipynb`](prep_SN_data.ipynb)** first. It writes `data/custom_{y,C,labels}.*`. To use these in the demo, update the load paths in `demo_mu_to_H0.ipynb` from `data/SH0ES22_partial_*` to `data/custom_*`.

Before running `prep_SN_data.ipynb`, download the Pantheon+ statistical+systematic covariance file from the [Pantheon+SH0ES release page](https://github.com/PantheonPlusSH0ES/DataRelease) (file: `Pantheon+SH0ES_STAT+SYS.cov`, ~32 MB) and place it in `data/`. The Pantheon+ data table `data/Pantheon+SH0ES.dat` is already included.

## TRGB version: from TRGB distances to H₀

**[`demo_TRGB_to_H0.ipynb`](demo_TRGB_to_H0.ipynb)** runs the identical solver using **TRGB**
host distances instead of Cepheids. You supply the distances in **`data/TRGB_distances.csv`**
(columns `host, mu, sigma_mu`); any host left blank or omitted is automatically dropped from the
fit. The notebook then computes H₀, draws the Hubble diagram, and shows the sensitivity to a
uniform TRGB zero-point shift.

> The shipped `data/TRGB_distances.csv` is pre-filled with R22 Cepheid distances as
> **placeholders** so the notebook runs immediately. Replace them with your own TRGB
> measurements. (The three CATS-only hosts NGC 1316/1404/4526 are left blank and dropped
> until you provide values.)

The TRGB SN dataset (`data/TRGB_*`, 40 calibrator hosts = the 37 SH0ES R22 hosts plus the
three CATS calibrators from Hoyt et al. 2023, with a CATS-style Hubble-flow selection) is
shipped pre-built. To regenerate or modify the selection, run **[`prep_TRGB_data.ipynb`](prep_TRGB_data.ipynb)**
or the script `data/build_TRGB_partial.py` (both need `data/Pantheon+SH0ES_STAT+SYS.cov`, as above).
