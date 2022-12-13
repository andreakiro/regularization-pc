# Classification model
- Toy model to compare the effect of regularization techniques on PC and BP network
- One of the dataset consists of the classic digit MNIST samples (available [here](http://yann.lecun.com/exdb/mnist/))

## How to train the model

- To train the model, simply use following command at root:

```python
python train.py --model clf --training [bp,pc]
```

### Optional adds. command-line parameters

#### Regression model specific:
- `--batch_size`: batch size used for training and evaluation (default:32)
- `--epochs`: number of epochs in central training loop (default:300)
- `--dropout`: dropout probability applied to every linear layers (default:0)
- `--lr`: learning rate in backpropagation optimization step (default:0.001)
- `--early_stopping`: number of epochs for early stopping (default: 300)

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

## Reproducing our results
For example, if you want to run the training with the default training hyper-parameters, logging the run in the `out/` folder, showing each training step updates, and plotting the validation result at the end of the training use

```python
python train.py --model clf --training bp --verbose 1 --epochs 30
```
