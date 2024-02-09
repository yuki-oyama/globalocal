# Global-local path choice model
Python code for the estimation of a link-based route choice model with decomposed utilities (a global-local model)

## Paper
For more details, please see the paper

Oyama, Y. (2023) [Global path preference and local response: A reward decomposition approach for network path choice analysis in the presence of locally perceived attributes](https://arxiv.org/abs/2307.08646). *arXiv*.
(Accepted for publication in Transportation Research Part A! The published version will be available soon!)

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

## Synthetic Data
I have prepared synthetic data generated in the Sioux Falls network.
Two datasets are available in the data folder:
- data_G0.csv: simulated by a global model (i.e., the original recursive logit model)
- data_L0.csv: simulated by a global-local model (i.e., capacity is assumed to have a local impact)
You can specify data to use within the "run_estimation.py" code.

## Quick Start
**Estimate** a global-local model with **both global and local attributes**

```
python run_estimation.py --vars_g "length" --init_beta_g -1 --lb_g None --ub_g 0 --vars_l "caplen" --init_beta_l 0 --lb_l None --ub_l None
```

**Estimate** a global-local model with **only global attributes**

```
python run_estimation.py --vars_g "length" "caplen" --init_beta_g -1 -1 --lb_g None None --ub_g 0 None
```

## Validation and Bootstrapping


For **cross-validation**, split the data into estimation and validation samples by setting test ratio greater than zero.

```
python run_estimation.py --n_samples 10 --test_ratio 0.2
```

For **bootstrapping**, resample K sets of observations and estimate the model with each of them, by setting n_samples greater than 1 and test ratio to zero. 

```
python run_estimation.py --n_samples 200 --test_ratio 0 --isBootstrap True
```
Note: this returns only the estimation results for K observations, so you should analyze the standard error or confidential intervals from the results afterwards.
