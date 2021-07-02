import matplotlib.pyplot as plt
import pickle

from utils.utils import cumulative_sum


def plot_metrics(metrics, time_str, title, delimiter='_'):
    with open(f'./figs/{title}__{time_str}/metrics.pkl', 'wb') as f:
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

        plt.title(f'{title}\n{metric} vs. epochs')
        plt.savefig(f'./figs/{title}__{time_str}/{metric}_vs_epochs.png')
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
        ax2.semilogy(metrics['beta'], color="black")
        ax2.set_ylabel('beta')

        plt.title(f'{title}\n{metric} vs. time')
        plt.savefig(f'./figs/{title}__{time_str}/{metric}_vs_time.png')
        plt.clf()


def report_overflow(time_str, title):
    with open(f'./figs/{title}__{time_str}/overflow.txt', 'w') as f:
        f.write('Overflow happened.')
