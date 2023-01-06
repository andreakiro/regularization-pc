# Generalization of Predictive Coding networks

* Generalization of Predictive Coding networks
* Deep Learning class FW 2022 - ETH Zürich
* Set of useful resources and papers [here](resources.md)

## Get started with our code

We implement two simple models once with BP and once with PC (click for details and all run parameters)
- Basic [regression model](doc/regression.md) to be trained on a noisy sinus time series
- Basic [classification model](doc/classification.md) to be trained on the classic MNIST dataset

The codebase includes `src.layers`, a framework we implement for simple PC blocks in PyTorch.

## Basic usage of the core file

- More details can be found in markdown instructions for both models (linked above)
- Exhaustive list of all parameters can be found using ```python train.py --help``` or in the src.parser.py file

For usage of default values, the simplest command is
```python
python3 train.py --training ${bp, pc} --model ${reg, clf} --dataset ${sine, mnist}
```

## Project requirements

```python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install wheel
python3 -m pip install -r ./env/requirements.txt
```

## Reproducing our results

- All our experiments were performed on CPU
- For reproducing them you can either run `train` file with specific params;
- Or run sepcific [wandb](https://wandb.ai/)  sweeps for dropout and initialization experiments
- Detailled xplanations on how to run sweeps on wandb is available [here](doc/sweep.md)
