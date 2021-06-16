import os
import pickle
import numpy as np
import pandas as pd
import yaml


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
        'K', 'batch_size', 'beta', 'epochs', 'best_train_accuracy', 'best_eval_accuracy',
        'min_train_loss', 'min_eval_loss', 'std_train_loss', 'std_eval_loss', 'mean_epoch_time', 'status'
    ])

    return df


def main():
    failed, successful = extract_results('/home/eldar.a/SPLM2/figs')
    df = summarize_results(failed, successful)
    df.to_csv('/home/eldar.a/SPLM2/results_1.csv')


if __name__ == '__main__':
    main()
