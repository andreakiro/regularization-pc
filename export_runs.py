import numpy as np
import pandas as pd 
import warnings
import pickle
import wandb
import time
import tqdm

api = wandb.Api(timeout=60)
entity, project = 'the-real-dl', 'bio-transformers'
runs = api.runs(entity + '/' + project) 

def extract(wandb_runs, maxrun=None):
    
    dfs = []
    configs = []
    summaries = []
    count = 0

    for i, run in enumerate(runs):
        # time.sleep(2) # to avoid 429 Client Error
        print(f'Fetching run #{i}', end='\r')
        if maxrun is not None and count == maxrun: break

        summary = run.summary
        if not 'epoch' in summary.keys(): continue
        if run.state == 'running': continue
        if run.state == 'crashed': continue
        if run.state == 'failed': continue

        run_id = run.id
        sweep_id = run.sweep.id

        config = run.config
        run_df = run.history()

        config.update({'sweep-id': sweep_id, 'run-id': run_id})
        summary.update({'sweep-id': sweep_id, 'run-id': run_id})

        l_epochs = []
        l_train_loss = []
        l_test_loss = []
        l_train_energy = []

        for e in range(summary['epoch']):
            x = run_df[run_df.epoch == e]
            if len(x.index) == 0: continue

            train_losses = x.train_loss.unique()
            test_losses = x.test_loss.unique()

            train_loss_no_nan = train_losses[~np.isnan(train_losses)]
            test_loss_no_nan = test_losses[~np.isnan(test_losses)]

            train_loss = train_loss_no_nan[0] if len(train_loss_no_nan) > 0 else np.nan
            test_loss = test_loss_no_nan[0] if len(test_loss_no_nan) > 0 else np.nan

            l_epochs.append(e)
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)

            if config['training'] == 'pc':
                train_energies = x.train_energy.unique()
                train_energy_no_nan = train_energies[~np.isnan(train_energies)]
                train_energy = train_energy_no_nan[0] if len(train_energy_no_nan) > 0 else np.nan
                l_train_energy.append(train_energy)

            if config['training'] == 'bp':
                run_ids = np.full(len(l_epochs), run_id)
                sweep_ids = np.full(len(l_epochs), sweep_id)
                data = list(zip(sweep_ids, run_ids, l_epochs, l_train_loss, l_test_loss))
                columns = ['sweep-id', 'run-id', 'epoch', 'train_loss', 'test_loss']
            else:
                run_ids = np.full(len(l_epochs), run_id)
                sweep_ids = np.full(len(l_epochs), sweep_id)
                data = list(zip(sweep_ids, run_ids, l_epochs, l_train_loss, l_test_loss, l_train_energy))
                columns = ['sweep-id', 'run-id', 'epoch', 'train_loss', 'test_loss', 'train_energy']

        dfs.append(pd.DataFrame(data, columns=columns))
        configs.append(config)
        summaries.append(summary)
        count += 1


    print(f'Fetched {count} runs in {runs.entity}/{runs.project} (max runs: {"all" if maxrun is None else maxrun})')
    return pd.concat(dfs, axis=0), configs, summaries

df, configs, summaries = extract(runs, maxrun=10)
df.to_csv('wandb-runs-df.csv')