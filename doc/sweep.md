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
| who | sweep | sweep-id | name | Euler_job_id (Optional)
|---|---|---|
| anne | sine_bp_adagrad_seed42.yaml | mwg5vjm5 | 5312199
| anne | sine_bp_adam_seed42.yaml | 8ocsxge2 | 5312143
| anne | sine_bp_rmsprop_seed42.yaml | twk3ehwy | 5312088
| anne | sine_bp_momentum_seed42.yaml | ba3irv46 | 5312026
| anne | sine_pc_adagrad_adam_seed42.yaml | wc9ctpq8 | 5311940
| anne | sine_pc_adagrad_momentum_seed42.yaml | 8kjc0an3 | 5311988
| anne | sine_pc_adam_adam_seed42.yaml | xzk0ksen | 5311903
| anne | sine_pc_adam_momentum_seed42.yaml | bmk5thrp | 5311823
| anne | sine_pc_rmsprop_adam_seed42.yaml | yumez8sz | 5311716
| anne | sine_pc_rmsprop_momentum_seed42.yaml | o7v5liv6 | 
| anne | sine_pc_momentum_adam_seed42.yaml | xto93w0c | 5311685
| anne | sine_pc_momentum_momentum_seed42.yaml | 5kxn6d81 | 5311648
| andrea | mnist_bp_adagrad_seed42.yaml | 4do451d5 (5311786) |
| andrea | mnist_bp_adam_seed42.yaml | 5yeeq2a4 (5311847) |
| andrea | mnist_bp_rmsprop_seed42.yaml | up3qoeyj (5311883) |
| andrea | mnist_bp_momentum_seed42.yaml | rmpdpbsc (5311939) |
| andrea | mnist_pc_adagrad_adam_seed42.yaml | jvmkpt8s (5312062) |
| andrea | mnist_pc_adam_adam_seed42.yaml | zhqld4ne (5312120) |
| andrea | mnist_pc_rmsprop_adam_seed42.yaml | 11pqrpdd (5312179) |
| andrea | mnist_pc_momentum_adam_seed42.yaml | exahtn7j (5312251) |
| andrea | mnist_pc_adagrad_momentum_seed42.yaml | ev5y7qxc (5312321) |
| andrea | mnist_pc_adam_momentum_seed42.yaml | sc89svlq (5312357) |
| andrea | mnist_pc_rmsprop_momentum_seed42.yaml | gb9m1a0b (5312407) |
| andrea | mnist_pc_momentum_momentum_seed42.yaml | 80zbdwm3 (5312441) |
| anne | fashion_bp_adagrad_seed42.yaml | 0jatqkyi | 5310172
| anne | fashion_bp_adam_seed42.yaml | 1gtby7pv | 5310193
| anne | fashion_bp_rmsprop_seed42.yaml | 8m7yfmsq | 5310306
| anne | fashion_bp_momentum_seed42.yaml | 8jmex28k | 5310475
| anne | fashion_pc_adagrad_adam_seed42.yaml | ujbl0nw9 | 5310598
| anne | fashion_pc_adagrad_momentum_seed42.yaml | nki5ynwb | 5310696
| anne | fashion_pc_adam_adam_seed42.yaml | j6u1zbtz | 5310777
| anne | fashion_pc_adam_momentum_seed42.yaml | slmsr4uw | 5310826
| anne | fashion_pc_rmsprop_adam_seed42.yaml | 2td03ada | 5310896
| anne | fashion_pc_rmsprop_momentum_seed42.yaml | 9jllkbpf | 5311106
| anne | fashion_pc_momentum_adam_seed42.yaml | lct3mndb | 5311137
| anne | fashion_pc_momentum_momentum_seed42.yaml | l6yr9ncr | 5311189

---
- TBD: test dropout, fix all optimizers
- TBD: test initialization, fix all optimizers