# Global-local path choice model
Python code for the estimation of a link-based recursive logit model with decomposed global and local utilities

## Paper
For more details, please see the paper

Oyama, Y. (2023) [Global path preference and local response: A reward decomposition approach for network path choice analysis in the presence of locally perceived attributes](https://arxiv.org/abs/2307.08646). arXiv preprint.

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
Estimate a Prism-RL model using synthetic observations in the Sioux Falls network.

```
python run_estimation.py --rl True --prism True --n_samples 1 --test_ratio 0
```

For cross-validation, split the data into estimation and validation samples by setting test ratio greater than zero.

```
python run_estimation.py --rl True --prism True --n_samples 10 --test_ratio 0.2
```
