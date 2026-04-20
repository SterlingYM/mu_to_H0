# mu_to_H0

***Demonstration/learning purpose only*** A simplified version of distance ladder solver that allows user to modify input distances to SN Ia host galaxies and study the effect on the Hubble constant. Intended for use in CosmoVerse workshop. Do not use this code for cosmological analysis.

## Getting started

Open **[`demo_mu_to_H0.ipynb`](demo_mu_to_H0.ipynb)** and run the cells top-to-bottom. The notebook walks through:

1. Loading the pre-built SN dataset (`data/SH0ES22_partial_*`).
2. Providing Cepheid distance moduli to each calibrator host.
3. Calling `solve_H0` to recover H0 and M_B.
4. Exploring how H0 responds to shifts in the Cepheid distance scale.

### Using your own SN data

If you want to regenerate the SN data from Pantheon+ inputs (e.g. to apply different redshift cuts or kinematic corrections), run **[`prep_SN_data.ipynb`](prep_SN_data.ipynb)** first. It writes `data/custom_{y,C,labels}.*` which `demo_mu_to_H0.ipynb` loads by default.
