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

### Create sweep and run agent (manual)
```
wandb sweep  ./wnb/${SWEEP_FILE}.yaml
wandb agent ${SWEEP_ID} --count ${number}
```

### Create sweep and run agents (jobs)
```
wandb sweep ./wnb/${SWEEP_FILE}.yaml
sbatch --array=1-${num_agents} --wrap="wandb agent ${SWEEP_ID} --count ${number}"
```

### Create sweep and run agents (for experiments)
```
wandb sweep ./wnb/${SWEEP_FILE}.yaml
sbatch --array=1-20 --wrap="wandb agent ${SWEEP_ID} --count 20" # for bp
sbatch --array=1-20 --wrap="wandb agent ${SWEEP_ID} --count 40" # for pc
```

### Run file manually
```
python3 train.py --model ${reg, clf, trf} --training ${bp, pc} --dataset ${sine, housing, mnist, fashion}
```

## Sweep Records
### sweep-ids record table
| who | sweep | sweep-id |
|---|---|---|
| andrea | mnist_bp_adagrad_seed42.yaml | umt60yvv |
| andrea | mnist_bp_adam_seed42.yaml | fzsm09ta |
| andrea | mnist_bp_rmsprop_seed42.yaml | gp2jvtle |
| andrea | mnist_bp_momentum_seed42.yaml | 5ualz5b8 |

| anne | fashion_bp_adagrad_seed42.yaml | q6j3zu02 |
| anne | fashion_bp_adam_seed42.yaml | 1yutu8s2 |
| anne | fashion_bp_rmsprop_seed42.yaml | uep82qrg |
| anne | fashion_bp_momentum_seed42.yaml | bj7we2l7 |
---
- TBD: test dropout, fix all optimizers
- TBD: test initialization, fix all optimizers