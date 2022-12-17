## Run a wandb sweep

- Define a sweep `yaml` file in `./wnb`
- Should define all hyperparams to optimize
- Change interpreter with your system ones

### Euler setup (skip for local runs)
```
ssh nethz@euler.ethz.ch
env2lmod && module load gcc/6.3.0 eth_proxy hdf5/1.10.1
```

### Setup dependencies
```
python3 -m pip install venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install wheel
python3 -m pip install -r ./env/requirements.txt
```

### Create sweep and run agent
```
wandb sweep  ./wnb/${SWEEP_FILE}.yaml
wandb agent ${SWEEP_ID} --count ${number}
```

### Run file manually
```
python3 train.py --model ${reg, clf, trf} --training ${bp, pc}
```
