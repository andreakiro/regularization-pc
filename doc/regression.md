# Regression model
- Toy model and dataset to test regularization techniques for PC
- The dataset are points sampled from a generated noisy sine wave [0,4]
- Sampled data and the ground truth function looks [like this](https://github.com/andreakiro/bio-transformers/blob/11-refactor-repo-structure/out/images/reg/20221117111430.png)

![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Csmall%20g(x)%20=%20f(x)%20&plus;%20%5Cepsilon%20=%20%5Csin(1&plus;x%5E2)%20&plus;%20%5Cmathcal%7BN%7D%20(0,%201)) 

## How to run the model

- To run the model, simply use following command at root:

```python
python train.py --model reg --training [bp,pc]
```

### Optional command-line parameters
- `--num value`: number of samples in the dataset (by default is 1000)
- `--lr value`: learning rate (by default is 1e-3)
- `--epochs value`: number of epochs for the training, no early-stopping is applied (by default is 300)
- `--plot [True,False]`: boolean flag, activate the plotting of validation results compared to the ground truth function at the end of the training (default is False)
- `--verbose [0,1]`: set the verbosity level of the training; if 0, no outputs will be displayed; if 1, the training loop will plot training information for each epoch (default is 0)
- `--output_dir path`: define the relative path (with respect to the root `very_simple_regression/`) of output folder where run information are logged; if not specified, no information is logged (by default no information logged)
- `--init`: PC initialization technique, (default is "forward"); supported values:
    - 'zeros', hidden values initialized with 0s
    - 'normal', hidden values initialized with a normal distribution with μ=mean and σ=std
    - 'xavier_normal', hidden values initialize with values according to the method described in 
      *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
      (2010), using a normal distribution. 
    - 'forward', hidden values initialized with the forward pass value
- `--dropout`: dropout probability, set 0 for no dropout (default is 0)


## Reproduce our results
For example, if you want to run the training with the default training hyper-parameters, logging the run in the `outputs/` folder, showing each training step updates, and plotting the validation result at the end of the training use
```python
python train.py --training bp --verbose 1 --plot True --output_dir outputs/
```
