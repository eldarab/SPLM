import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


### THIS IS DEPRECATED BUT STILL HERE FOR DEOMSTRATION REASONS ###
def extract_results(root_dir='figs'):
    successful_experiments = {}
    failed_experiments = {}

    for experiment in os.listdir(root_dir):
        with open(f'{root_dir}/{experiment}/params.yml') as f:
            params = yaml.safe_load(f)
        experiment_files = os.listdir(f'{root_dir}/{experiment}')
        if len(experiment_files) < 2:  # experiment crashed
            failed_experiments[experiment] = (params, 'crash')
        elif len(experiment_files) == 2:
            failed_experiments[experiment] = (params, 'overflow')
        else:
            with open(f'{root_dir}/{experiment}/metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
            successful_experiments[experiment] = (params, metrics)

    return failed_experiments, successful_experiments


def summarize_results(failed, successful):
    df_rows = {}
    for exp_name, (params, metrics) in successful.items():
        df_rows[exp_name] = (
            params['optim']['K'],
            params['optim']['batch_size'],
            params['optim']['beta'],
            params['optim']['epochs'],
            params['optim']['scheduler']['type'],
            params['optim']['scheduler']['step_size'],
            params['optim']['scheduler']['gamma'],
            max(metrics['train_accuracy']),
            max(metrics['eval_accuracy']),
            min(metrics['train_loss']),
            min(metrics['eval_loss']),
            np.std(metrics['train_loss']),
            np.std(metrics['eval_loss']),
            np.mean(metrics['epoch_time']),
            'successful'
        )
    for exp_name, (params, failure_reason) in failed.items():
        df_rows[exp_name] = (
            params['optim']['K'],
            params['optim']['batch_size'],
            params['optim']['beta'],
            params['optim']['epochs'],
            params['optim']['scheduler']['type'],
            params['optim']['scheduler']['step_size'],
            params['optim']['scheduler']['gamma'],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            failure_reason
        )

    df = pd.DataFrame.from_dict(df_rows, orient='index', columns=[
        'K', 'batch_size', 'beta', 'epochs', 'scheduler_type', 'step_size', 'gamma', 'best_train_accuracy', 'best_eval_accuracy',
        'best_train_loss', 'best_eval_loss', 'std_train_loss', 'std_eval_loss', 'mean_epoch_time', 'status'
    ])

    return df


def main(figs_dir, output_dir):
    figs_dir = Path(figs_dir)
    output_dir = Path(output_dir)

    failed, successful = extract_results(figs_dir)
    print(f'Extracted results from: {figs_dir}')

    df = summarize_results(failed, successful)

    csv_path = f'{output_dir}/{figs_dir.name}.csv'
    df.to_csv(csv_path)
    print(f'Saved csv: {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs_dir", type=str, required=True, help="path to figs/ directory with plots.")
    parser.add_argument("--output_dir", type=str, required=True, help="path to output_dir/figs_dir.csv .")
    args = parser.parse_args()
    main(args.figs_dir, args.output_dir)
