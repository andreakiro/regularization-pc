# Regression model
- Toy model and dataset to compare the effect of regularization techniques on PC and BP network
- Two datasets are available:
    - generated noisy sine wave
     `` f(x) = sin(x**2) + c``
      where ``c`` is sampled from a gaussian distribution
    - House Price Regression dataset (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

## How to train the model

- To train the model, simply use the following command at root:

```python
python train.py --model reg --training [bp,pc] --dataset [housing, sine]
```

### Optional adds. command-line parameters

#### Regression model specific:
- `--batch_size`: batch size used for training and evaluation (default:32)
- `--epochs`: number of epochs in central training loop (default:300)
- `--dropout`: dropout probability applied to every linear layers (default:0)
- `--lr`: learning rate in backpropagation optimization step (default:0.001)
- `--early_stopping`: number of epochs for early stopping (default: 300)
- `--nsamples`: number of generated samples for regression sample dataset (default:1000)

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