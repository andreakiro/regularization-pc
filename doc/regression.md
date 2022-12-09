# Regression model
- Toy model and dataset to compare the effect of regularization techniques on PC and BP network
- One of the dataset consists of points sampled from a generated noisy sine wave [0,4]
- Sampled data and the ground truth function looks [like this](https://github.com/andreakiro/bio-transformers/blob/11-refactor-repo-structure/out/images/reg/20221117111430.png)

![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Csmall%20g(x)%20=%20f(x)%20&plus;%20%5Cepsilon%20=%20%5Csin(1&plus;x%5E2)%20&plus;%20%5Cmathcal%7BN%7D%20(0,%201)) 

## How to train the model

- To train the model, simply use following command at root:

```python
python train.py --model reg --training [bp,pc]
```
### Optional adds. command-line parameters

#### Regression model specific:
- `--batch_size`: batch size used for training and evaluation (default:32)
- `--epochs`: number of epochs in central training loop (default:300)
- `--dropout`: dropout probability applied to every linear layers (default:0)
- `--lr`: learning rate in backpropagation optimization step (default:0.001)
- `--early_stopping`: number of epochs for early stopping (default: 300)
- `--nsamples`: number of generated samlpes for regression sample dataset (default:1000)

#### PC training mode specific:
- `--init`: PC initialization technique in {'zeros', 'normal', 'xavier_normal', 'forward'} (default:forward)
  - 'zeros', hidden values initialized with 0s
  - 'normal', hidden values initialized with a normal distribution with μ=mean and σ=std
  - 'xavier_normal', hidden values initialize with values according to the method described in 
    *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
    (2010), using a normal distribution. 
  - 'forward', hidden values initialized with the forward pass value
- `--clr`: PC convergence learning rate (default:0.2)
- `--iterations`: PC convergence iterations (default:100)

#### IO and logging specific:
- `--verbose`: verbosity level of the training process {0, 1} (default:0)
- `--checkpoint_frequency`: epochs frequency for model checkpoints (`.pt` file) (default:1) 
- `--plot`: boolean indicating whether to plot regression results after training (default:false)

## Reproducing our results
For example, if you want to run the training with the default training hyper-parameters, logging the run in the `out/` folder, showing each training step updates, and plotting the validation result at the end of the training use

```python
python train.py --model reg --training bp --verbose 1 --plot True
```
