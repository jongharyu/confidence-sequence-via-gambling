# On Confidence Sequences for Bounded Random Processes via Universal Gambling

This repository contains an official codebase of the paper ["On Confidence Sequences for Bounded Random Processes via Universal Gambling," Jongha (Jon) Ryu and Alankrita Bhatt, arXiv 2022](https://arxiv.org/abs/2207.12382).
To run the lower bound universal portfolio (LBUP) with Cython, first run
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