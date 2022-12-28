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
| andrea | mnist_bp_adagrad_seed42.yaml | eq3d63v4 (5310498) |
| andrea | mnist_bp_adam_seed42.yaml | 98w5uup7 (5310547) |
| andrea | mnist_bp_rmsprop_seed42.yaml | 512yd9li (5310619) |
| andrea | mnist_bp_momentum_seed42.yaml | jcvxvaof (5310673) |
| andrea | mnist_pc_adagrad_adam_seed42.yaml | euh6q0tl (5310797) |
| andrea | mnist_pc_adam_adam_seed42.yaml | ycappciw (5310849) |
| andrea | mnist_pc_rmsprop_adam_seed42.yaml | qpcnqeri (5310917) |
| andrea | mnist_pc_momentum_adam_seed42.yaml | 607stvh1 (5310968) |
| andrea | mnist_pc_adagrad_momentum_seed42.yaml | lo6b8318 (5311028) |
| andrea | mnist_pc_adam_momentum_seed42.yaml | g8ffnck4 (5311085) |
| andrea | mnist_pc_rmsprop_momentum_seed42.yaml | qnmpck0o (5311159) |
| andrea | mnist_pc_momentum_momentum_seed42.yaml | h43i61oa (5311211) |
| anne | fashion_bp_adagrad_seed42.yaml | q6j3zu02 |
| anne | fashion_bp_adam_seed42.yaml | 1yutu8s2 |
| anne | fashion_bp_rmsprop_seed42.yaml | uep82qrg |
| anne | fashion_bp_momentum_seed42.yaml | bj7we2l7 |
| anne | fashion_pc_adagrad_seed42.yaml | 97z0xhl6 |
| anne | fashion_pc_adam_seed42.yaml | chb1iz0y |
| anne | fashion_pc_rmsprop_seed42.yaml | q8tjvkzo |
| anne | fashion_pc_momentum_seed42.yaml | 8zpl7b9v |
---
- TBD: test dropout, fix all optimizers
- TBD: test initialization, fix all optimizers