# Generalization of Predictive Coding networks

* Generalization of Predictive Coding networks
* Deep Learning class FW 2022 - ETH Zürich
* Set of useful resources and papers [here](resources.md)

## Get started with our code

We implement two simple models once with BP and once with PC (click for details)
- Basic [regression model](doc/regression.md) to be trained on a noisy sinus time series
- Basic [classification model](doc/classification.md) to be trained on the classic MNIST dataset

The codebase includes `src.layers`, a framework we implement for simple PC blocks in PyTorch.

## Basic usage of the core file

- More details can be found in markdown instructions for both models (linked above)
- Exhaustive list of all parameters can be found using ```python train.py --help``` or in the src.parser.py file

For usage of default values, the simplest command is
```python
python3 train.py --training ${bp, pc} --model ${reg, clf} --dataset ${sine, housing, mnist, fashion}
```

## Project requirements

```python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install wheel
python3 -m pip install -r ./env/requirements.txt
```

## Reproducing our results
For reproducing the experiments discussed in our report, use the [Weights and Biases](doc/sweep.md) 
instruction file. Create an account and project on wandb.ai and create sweeps via the 
sweep ".yaml" files found in the "wnb" directory. The ".yaml" files are sorted by experiment type.

### Single Runs
If you want to try out single experiment runs without creating a wandb account, you can use below standard commands:
For training with default hyper-parameters, the run is logged in the `out/` folder, showing each training step updates.

Regression on Sine with default hyperparameters:
```python
python train.py --model reg --training bp --dataset sine --verbose 1 --plot True
```

Classification on MNIST with default hyperparameters:
```python
python train.py --model clf --training bp --dataset mnist --verbose 1
```