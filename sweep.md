## Run a wandb sweep

- Define a sweep `yaml` file in `./wnb`
- Should define all hyperparams to optimize
- Change interpreter variable to your system

### Euler setup (skip for local sweep)
```
ssh nethz@euler.ethz.ch
env2lmod && module load gcc/6.3.0 eth_proxy hdf5/1.10.1
```

### Create and run sweep
```
wandb sweep  ./wnb/${SWEEP_FILE}.yaml
wandb agent ${SWEEP_ID} --count ${number}
```
