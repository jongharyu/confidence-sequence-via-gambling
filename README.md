## Universal-Gambling-Based Confidence Sequences for Bounded Random Processes 

This repository contains an official codebase of the two papers 
1. "[On Confidence Sequences for Bounded Random Processes via Universal Gambling](https://arxiv.org/abs/2207.12382)," Jongha (Jon) Ryu and Alankrita Bhatt, Under minor revision for *IEEE Transactions on Information Theory* (2022).
2. "[Gambling-Based Confidence Sequences for Bounded Random Vectors](https://openreview.net/forum?id=mu7Er7f9NQ)," Jongha (Jon) Ryu and Gregory W. Wornell, **ICML 2024 (spotlight)**.

To run the lower bound universal portfolio (LBUP) of `Ryu and Bhatt (2022)` with Cython, first run
```bash
python3 setup_lbup_integrand.py build_ext --inplace
```
This allows ~33% improvement in the runtime.

The experiments in the manuscript can be replicated by running the jupyter notebook `notebooks/exps.ipynb`.

## Install
To run the proposed method, `Cython` should be installed:
```bash
pip install Cython
```
Then, the preamble of the code should include
```python
import pyximport
pyximport.install()
```
This is required to load the `lbup_integrand.pyx` file.


## Acknowledgement
As noted under `methods/precise`, a Python translation of PRECiSE algorithms of [Orabona and Jun (2021)](https://arxiv.org/abs/2110.14099) are included, 
where the original MATLAB implementation can be found at https://github.com/bremen79/precise. 
