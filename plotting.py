import matplotlib.pyplot as plt
import pickle

from utils.utils import cumulative_sum


def plot_metrics(metrics, time_str, title, delimiter='_'):
    with open(f'./figs/{title}__{time_str}/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    metric_types = {delimiter.join(metric_name.split(delimiter)[1:]) for metric_name in metrics if metric_name != 'epoch_time'}

    for metric in metric_types:
        if metric == 'epoch_time':
            continue

        # vs. epochs
        plt.title(f'{title}\n{metric} vs. epochs')
        plt.plot(metrics[f'train_{metric}'])
        plt.plot(metrics[f'eval_{metric}'])
        plt.legend([f'train_{metric}', f'eval_{metric}'])
        plt.xlabel('epochs')
        plt.ylabel(metric)
        plt.savefig(f'./figs/{title}__{time_str}/{metric}_vs_epochs.png')
        plt.clf()

        # vs. time
        times = cumulative_sum(metrics['epoch_time'])
        plt.title(f'{title}\n{metric} vs. time')
        plt.plot(times, metrics[f'train_{metric}'])
        plt.plot(times, metrics[f'eval_{metric}'])
        plt.legend([f'train_{metric}', f'eval_{metric}'])
        plt.xlabel('time (seconds)')
        plt.ylabel(metric)
        plt.savefig(f'./figs/{title}__{time_str}/{metric}_vs_time.png')
        plt.clf()


def report_overflow(time_str, title):
    with open(f'./figs/{title}__{time_str}/overflow.txt', 'w') as f:
        f.write('Overflow happened.')
