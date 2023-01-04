# Generalization of Predictive Coding networks

* Generalization of Predictive Coding networks
* Deep Learning class FW 2022 - ETH Zürich
* Set of useful resources and papers [here](resources.md)

## Get started with our code

We implement two simple models once with BP and once with PC (click for details)
- Basic [regression model](doc/regression.md) to be trained on a noisy sinus time series
- Basic [classification model](doc/classification.md) to be trained on the classic MNIST dataset

The codebase include `src.layers` a framework for simple PC blocks implemented in PyTorch

## Basic usage of the core file

- More details can be found in markdown instructions for both models (linked above)
- Exhaustive list of all parameters is otherwise visible in `src.parser.py` file

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
