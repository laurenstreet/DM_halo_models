# DM_halo_models
Testing DM models with SPARC

See *insert publication here* for more information.

## Overview

This repository was created to test several galactic dark matter (DM) models against the Spitzer Photometry and Accurate Rotation Curves (SPARC) catalog data (see http://astroweb.case.edu/SPARC/).  It can be used to obtain fit results for each of the DM models analyzed, as well as calculate various properties such as the galactic DM mass, etc.  For more information pertaining to the math and physics behind the calculations, see *insert publication here*.

For more detailed documentation see : https://laurenstreet.github.io/DM_halo_models or DM_halo_models_docs.pdf

## Requirements

See requirements.txt

## Fitting

All results, including figures, in *insert publication here* can be reproduced using the Jupyter notebooks.  The main results of the paper are included in the notebooks : 

- fits_CDM_all.ipynb
- fits_Einasto.ipynb

Comparisons to previous studies can be found in the notebooks :

- checks_CDM_all.ipynb
- checks_DC14_NFW.ipynb
- checks_Einasto_NFW.ipynb

## Results

Some example plots from *insert publication here* are shown.  First, we show the results from the ULDM model analysis assuming the particle mass is free to vary in the fitting procedure.  Below is a plot of the difference in Bayesian Information Criterion (BIC) values between the ULDM and Einasto model vs. ULDM particle mass.

![ULDM_mass_free_BIC_vs_mass](https://github.com/laurenstreet/DM_halo_models/blob/main/example_plots/psi_mfree_BICvsm.jpg?raw=true)

Next, we show the results from the ULDM model analysis assuming the particle mass is fixed and scanned along some range in the fitting procedure.  Below is a plot of the fractional differences in the sum of the reduced chi-square values (summed over galaxies) between the ULDM and Einasto models vs. ULDM particle mass. 

![ULDM_mass_fix_chisqsumfrac_vs_mass](https://github.com/laurenstreet/DM_halo_models/blob/main/example_plots/psi_mfix_chisqvsm.jpg?raw=true)

We also show the results for the best fit particle masses (for the matched models).  Below are the distributions of the BIC differences between the ULDM and Einasto models for the particle masses fit to the best fit values.

![ULDM_mass_fix_BICdist](https://github.com/laurenstreet/DM_halo_models/blob/main/example_plots/psi_mfix_BICvsBIC_ex.jpg?raw=true)

Finally, we show results for the CDM models.  Below are the distributions of BIC differences between each CDM model analyzed.

![CDM_BIC_vs_BIC](https://github.com/laurenstreet/DM_halo_models/blob/main/example_plots/CDM_BICvsBIC.jpg?raw=true)

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy (DOE), Office of Science, Office of
Workforce Development for Teachers and Scientists, Office of Science Graduate Student Research (SCGSR) program.
The SCGSR program is administered by the Oak Ridge Institute for Science and Education (ORISE) for the DOE.
ORISE is managed by ORAU under contract number DE-SC0014664. All opinions expressed in this paper are the
authors’ and do not necessarily reflect the policies and views of DOE, ORAU, or ORISE. 

Thanks to Joshua Eby and Peter Suranyi for valuable discussions and comments from proofreading of *insert publication here*.

Thanks to Mike Sokoloff and Daniel Vieira for setting up computational resources to be used in the next installment of this study.

## Citations

Below is the bibtex formatted citation for *insert publication here*,

*insert bibtex citation here*

Below is the bibtex formatted citation for this repository,

@misc{Street_DM_halo_models,
author = {Street, Lauren and Gnedin, Nickolay Y. and Wijewardhana, L. C. R.},
title = {{DM halo models}},
url = {https://github.com/laurenstreet/DM_halo_models}
}

## Contact

streetlg@mail.uc.edu
