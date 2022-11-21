## Sinus Regression
Toy model and dataset to test regularization techniques for predictive coding.

### Dataset
The dataset contains observations sampled in the interval [0,4] from a sinusoidal function ground truth function with some random gaussian noise applied to it

![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Csmall%20g(x)%20=%20f(x)%20&plus;%20%5Cepsilon%20=%20%5Csin(1&plus;x%5E2)%20&plus;%20%5Cmathcal%7BN%7D%20(0,%201)) 

The sampled data and the ground truth function look like this
![image](https://github.com/andreakiro/bio-transformers/blob/bp_simple_regressor/very_simple_regression/images/data.png)

### How to run the model
To run the model, use
```python
python train.py --training [bp,pc]
```

Many other (optional) command-line parameters are available in the training script
- `--num value`: number of samples in the dataset (by default is 1000)
- `--lr value`: learning rate (by default is 1e-3)
- `--epochs value`: number of epochs for the training, no early-stopping is applied (by default is 300)
- `--plot [True,False]`: boolean flag, activate the plotting of validation results compared to the ground truth function at the end of the training (default is False)
- `--verbose [0,1]`: set the verbosity level of the training; if 0, no outputs will be displayed; if 1, the training loop will plot training information for each epoch (default is 0)
- `--output_dir path`: define the relative path (with respect to the root `very_simple_regression/`) of output folder where run information are logged; if not specified, no information is logged (by default no information logged)

For example, if you want to run the training with the default training hyper-parameters, logging the run in the `outputs/` folder, showing each training step updates, and plotting the validation result at the end of the training use
```python
python train.py --training bp --verbose 1 --plot True --output_dir outputs/
```
