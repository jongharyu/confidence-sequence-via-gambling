# On Confidence Sequences for Bounded Random Processes via Universal Gambling

This repository contains an official codebase of the paper ["On Confidence Sequences for Bounded Random Processes via Universal Gambling," Jongha (Jon) Ryu and Alankrita Bhatt, arXiv 2022](https://arxiv.org/abs/2207.12382).
To run the lower bound universal portfolio (LBUP) with Cython, first run
```bash
python3 setup_lbup_integrand.py build_ext --inplace
```
This allows ~33% improvement in the runtime.

The experiments in the manuscript can be replicated by running the jupyter notebook `notebooks/exps.ipynb`. 