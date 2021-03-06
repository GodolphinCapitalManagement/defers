# README

This repo contains scripts that process the data and estimate the parameters
of a survival model in a mixed-model Bayesian framework and presented in
[Loan Payment Deferments Due to Labor Market Shocks: A Case Study](https://github.com/GodolphinCapitalManagement/corona.git/).
While the files in this repo are all python scripts, they are derived from jupyter notebooks using
the Jupytext extension and can be converted to such using the
**jupytext --set-formats ipynb,py *.py** command.

## Manifest

[common.py](https://github.com/GodolphinCapitalManagement/defers/tree/master/common.py):
 this script contains a set of common functions that get used across the various jupyter notebooks.

### Sub-directories

[pymc3](https://github.com/GodolphinCapitalManagement/defers/tree/master/pymc3):
  contains all the modeling scripts for the paper.

* [covid_pooled.py](https://github.com/GodolphinCapitalManagement/defers/tree/master/pymc3/covid_pooled.py):
  pooled version of model

* [covid_level_I_fraily.py](https://github.com/GodolphinCapitalManagement/defers/tree/master/pymc3/covid_level_I_frailty.py):
  mixed model with frailty terms for originators.

* [deferment.py](https://github.com/GodolphinCapitalManagement/defers/tree/master/pymc3/deferment.py):
  generic script that estimates both models in one go.

* [projections.py](https://github.com/GodolphinCapitalManagement/defers/tree/master/pymc3/projections.py):
   script for model comparisons using (i) PSIS-LOO and
  (ii) the [Integrated Calibration Index](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8570)
  approach of Harrell et. al.
