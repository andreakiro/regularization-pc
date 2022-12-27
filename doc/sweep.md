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
#### Used sweep IDs:
##### Finding best Optimizer for each Dataset:
001: Fashion - BP
002: MNIST - BP
003: Housing - BP
004: Sine - BP
005: Fashion - PC
006: MNIST - PC
007: Housing - PC
008: Sine - PC

##### TBD: fix each dataset with its best optimizer and fix pc-initialization on "forward": find best pc-related parameters inclusively pc-optimizer
Then: test dropout, fix all optimizers
Then: test initialization, fix all optimizers

### Create sweep and run agents (jobs)
```
wandb sweep ./wnb/${SWEEP_FILE}.yaml
sbatch --array=1-${num_agents} --wrap="wandb agent ${SWEEP_ID} --count ${number}"
```

### Run file manually
```
python3 train.py --model ${reg, clf, trf} --training ${bp, pc} --dataset ${sine, housing, mnist, fashion}
```
