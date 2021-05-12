import matplotlib.pyplot as plt

from utils.functional import cumulative_sum


def plot_metrics(metrics, title='', delimiter='_'):
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
        plt.show()

        # vs. time
        times = cumulative_sum(metrics['epoch_time'])
        print(times)
        plt.title(f'{title}\n{metric} vs. time')
        plt.plot(metrics[f'train_{metric}'], times)
        plt.plot(metrics[f'eval_{metric}'], times)
        plt.legend([f'train_{metric}', f'eval_{metric}'])
        plt.xlabel('time (seconds)')
        plt.ylabel(metric)
        plt.show()
