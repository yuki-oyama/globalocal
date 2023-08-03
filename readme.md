# Global-local path choice model
Python code for the estimation of a link-based recursive logit model with decomposed global and local utilities

## Paper
For more details, please see the paper

Oyama, Y. (2023) [Global path preference and local response: A reward decomposition approach for network path choice analysis in the presence of locally perceived attributes](https://arxiv.org/abs/2307.08646). *arXiv*.

If you find this code useful, please cite the paper:
```
@article{oyama2023globalocal,
  title = {Global path preference and local response: A reward decomposition approach for network path choice analysis in the presence of locally perceived attributes},
  author = {Oyama, Yuki},
  year = {2023},
  eprint = {2307.08646},
  archivePrefix = {arXiv},
  primaryClass = {physics.soc-ph}
}
```

## Quick Start
**Estimate** a global-local model using probe person data in the Kannai network.

```
python pp_application.py --n_samples 1 --test_ratio 0
```

For **cross-validation**, split the data into estimation and validation samples by setting test ratio greater than zero.

```
python pp_application.py --n_samples 10 --test_ratio 0.2
```

For **bootstrapping**, resample K sets of observations and estimate the model with each of them, by setting n_samples greater than 1 and test ratio to zero. 

```
python pp_application.py --n_samples 200 --test_ratio 0 --isBootstrap True
```
Note: this returns only the estimation results for K observations, so you should analyze the standard error or confidential intervals from the results afterwards.
