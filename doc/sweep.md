## Run a wandb sweep

- Define a sweep `yaml` file in `./wnb`
- Should define all hyperparams to optimize
- Change interpreter with your system ones

### Euler setup (skip for local runs)
```
ssh nethz@euler.ethz.ch
git clone git@github.com:andreakiro/bio-transformers.git
cd bio-transformers
source setup_cluster.sh
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
