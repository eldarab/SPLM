import pickle
import time
from pathlib import PosixPath

import matplotlib.pyplot as plt

OVERFLOW_VALUE = 10e5


def plot_metrics(metrics, results_dir: PosixPath, delimiter='_'):
    with open(f'{results_dir}/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    metric_types = {delimiter.join(metric_name.split(delimiter)[1:]) for metric_name in metrics
                    if metric_name not in ('epoch_time', 'beta')}

    for metric in metric_types:
        if metric == 'epoch_time':
            continue

        # vs. epochs
        fig, ax = plt.subplots()

        ax.plot(metrics[f'train_{metric}'], label=f'train_{metric}')
        ax.plot(metrics[f'eval_{metric}'], label=f'eval_{metric}')
        ax.set_xlabel('epochs')
        ax.set_ylabel(metric)
        ax.legend()

        ax2 = ax.twinx()
        ax2.semilogy(metrics['beta'], color="black")
        ax2.set_ylabel('beta')

        plt.title(f'{results_dir.name}\n{metric} vs. epochs')
        plt.savefig(f'{results_dir.name}/{metric}_vs_epochs.png')
        plt.clf()

        # vs. time
        fig, ax = plt.subplots()

        times = cumulative_sum(metrics['epoch_time'])

        ax.plot(times, metrics[f'train_{metric}'], label=f'train_{metric}')
        ax.plot(times, metrics[f'eval_{metric}'], label=f'eval_{metric}')
        ax.set_xlabel('time (seconds)')
        ax.set_ylabel(metric)
        ax.legend()

        ax2 = ax.twinx()
        ax2.semilogy(times, metrics['beta'], color="black")
        ax2.set_ylabel('beta')

        plt.title(f'{results_dir.name}\n{metric} vs. time')
        plt.savefig(f'{results_dir.name}/{metric}_vs_time.png')
        plt.clf()


def plot_overflow(results_dir: PosixPath):
    with open(f'{results_dir.name}/overflow.txt', 'w') as f:
        f.write('Overflow happened.')


def cumulative_sum(lst):
    return [sum(lst[0:x:1]) for x in range(0, len(lst) + 1)][1:]


def report_metrics(metrics_values):
    for metric_name, metric_value in metrics_values.items():
        print(f"\t\t{metric_name}={metric_value[-1]:.3f}")
        if metric_value[-1] > OVERFLOW_VALUE:
            raise OverflowError()
    print()


def get_time_str():
    return time.strftime('%Y_%m_%d__%H_%M_%S')
