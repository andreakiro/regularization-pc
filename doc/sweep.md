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
| who | sweep | sweep-id | name | Euler_job_id (Optional) | Euler-command (Optional sanity check)
|---|---|---|
| anne | sine_bp_adagrad_seed42.yaml | mwg5vjm5 | 5329763 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/mwg5vjm5 --count 20"
| anne | sine_bp_adam_seed42.yaml | 8ocsxge2 | 5329741 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/8ocsxge2 --count 20"
| anne | sine_bp_rmsprop_seed42.yaml | twk3ehwy | 5329714 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/twk3ehwy --count 20"
| anne | sine_bp_momentum_seed42.yaml | ba3irv46 | 5329687 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/ba3irv46 --count 20"
| anne | sine_pc_adagrad_adam_seed42.yaml | wc9ctpq8 | 5329663 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/wc9ctpq8 --count 40"
| anne | sine_pc_adagrad_momentum_seed42.yaml | 8kjc0an3 | 5329622 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/8kjc0an3 --count 40"
| anne | sine_pc_adam_adam_seed42.yaml | xzk0ksen | 5329596 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/xzk0ksen --count 40"
| anne | sine_pc_adam_momentum_seed42.yaml | bmk5thrp | 5329574 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/bmk5thrp --count 40"
| anne | sine_pc_rmsprop_adam_seed42.yaml | yumez8sz | 5329548 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/yumez8sz --count 40"
| anne | sine_pc_rmsprop_momentum_seed42.yaml | o7v5liv6 | 5329512 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/o7v5liv6 --count 40"
| anne | sine_pc_momentum_adam_seed42.yaml | xto93w0c | 5329465 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/xto93w0c --count 40"
| anne | sine_pc_momentum_momentum_seed42.yaml | 5kxn6d81 | 5329440 | sbatch --time=5:00:00 --array=1-20 --wrap="wandb agent the-real-dl/bio-transformers/5kxn6d81 --count 40"
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
| anne | fashion_pc_adagrad_adam_seed42.yaml | 0xreb4ff | 5451741
| anne | fashion_pc_adagrad_momentum_seed42.yaml | p1gq94wf | 5451772
| anne | fashion_pc_adam_adam_seed42.yaml | yr37huql | 5451833
| anne | fashion_pc_adam_momentum_seed42.yaml | 9zcbfhcd | 5451808
| anne | fashion_pc_rmsprop_adam_seed42.yaml | 0mnuocew | 5451863
| anne | fashion_pc_rmsprop_momentum_seed42.yaml | anqqxqtw | 5451889
| anne | fashion_pc_momentum_adam_seed42.yaml | 66zhjwht | 5451947
| anne | fashion_pc_momentum_momentum_seed42.yaml | 55f93huv | 5451923

---
Dropout
| anne | 2_dropout_mnist_bp_seed42.yaml | grnn1330 | 5440452
| anne | 2_dropout_mnist_pc_seed42.yaml | n7mkwdnc | 5440472

| anne | 2_dropout_sine_bp_seed42.yaml | qjtnzrew | 5436988
| anne | 2_dropout_sine_pc_seed42.yaml | czdapdwk | 5440494

dropout with name
| anne | 2_dropout_mnist_bp_seed42.yaml | c1qwx9s1 | 5440645
| anne | 2_dropout_mnist_pc_seed42.yaml | 0fshkrb4 | 5440665
| anne | 2_dropout_sine_bp_seed42.yaml | xn7ki9mc | 5440693
| anne | 2_dropout_sine_pc_seed42.yaml | me29xcjd | 5440715

---
Initialization
| anne | 2_init_mnist_pc_seed42.yaml | w121afgk | 5440516
| anne | 2_init_sine_pc_seed42.yaml | gkhhc6yu | 5440536

init with name
| anne | 2_init_mnist_pc_seed42.yaml | h19w2fjn | 5440608
| anne | 2_init_sine_pc_seed42.yaml | jqdu7i9v | 5440579

---
Dropout
| andrea | 2_dropout_mnist_bp_seed42.yaml | siihf02c | 5515793
| andrea | 2_dropout_mnist_pc_seed42.yaml | u4ir00mo | 5515832

| andrea | 2_dropout_sine_bp_seed42.yaml | bpluj3u2 | 5515703
| andrea | 2_dropout_sine_pc_seed42.yaml | oi23urmo | 5515723

---
Initialization
| andrea | 2_init_sine_pc_seed42.yaml | 92xh7pmj | 5515868
| andrea | 2_init_mnist_pc_seed42.yaml | huvb094j | 5515925

---
3-together
| andrea | 3_together_sine_pc_seed42.yaml | ny0h69tf | 5516202
| andrea | 3_together_mnist_pc_seed42.yaml | 4ppq0ppf | 5516182

---
bugfix-Dropout
| andrea | 2_dropout_sine_bp_seed42.yaml | lp8d7mjq | 5541999
| andrea | 2_dropout_sine_pc_seed42.yaml | xnuu19n5 | 5542017

| andrea | 2_dropout_mnist_bp_seed42.yaml | vyshel3j | 5542034
| andrea | 2_dropout_mnist_pc_seed42.yaml | zpuw1jt2 | 5542049

---
bugfix-Initialization
| andrea | 2_init_sine_pc_seed42.yaml | ng9iw8t0 | 5559237
| andrea | 2_init_mnist_pc_seed42.yaml | 5zmic39y | 5559806

---
bugfix-3-together
| andrea | 3_together_sine_pc_seed42.yaml | 8ji187u4 | 5559356
| andrea | 3_together_mnist_pc_seed42.yaml | y38gvz5w | 5560112


---
LONGER-Dropout
| andrea | 2_dropout_sine_bp_seed42.yaml | me3vrsed | 5564974
| andrea | 2_dropout_sine_pc_seed42.yaml | o5h65l7n | 5565023

| andrea | 2_dropout_mnist_bp_seed42.yaml | sjxh7jtm | 5565071
| andrea | 2_dropout_mnist_pc_seed42.yaml | ax3knz63 | 5565110

---
LONGER-Initialization
| andrea | 2_init_sine_pc_seed42.yaml | l5gvxhno | 5565169
