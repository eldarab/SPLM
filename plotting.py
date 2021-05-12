import matplotlib.pyplot as plt

from utils.functional import cumulative_sum


def plot_metrics(metrics, title='', delimiter='_'):
    metric_types = {delimiter.join(metric_name.split(delimiter)[1:]) for metric_name in metrics}

    for metric in metric_types:
        if metric == 'epoch_time':
            continue

        # vs. epochs
        plt.title(f'{title}\n{metric} vs. Epochs')
        plt.plot(metrics[f'train_{metric}'])
        plt.plot(metrics[f'eval_{metric}'])
        plt.legend([f'train_{metric}', f'eval_{metric}'])
        plt.xlabel('Epochs')
        plt.show()

        # vs. time
        times = cumulative_sum(metrics['epoch_time'])
        plt.title(f'{title}\n{metric} vs. Time')
        plt.plot(metrics[f'train_{metric}'], times)
        plt.plot(metrics[f'eval_{metric}'], times)
        plt.legend([f'train_{metric}', f'eval_{metric}'])
        plt.xlabel('Time (seconds)')
        plt.show()
