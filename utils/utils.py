def cumulative_sum(lst):
    return [sum(lst[0:x:1]) for x in range(0, len(lst) + 1)][1:]


def report_metrics(epoch, metrics_values):
    print(f"epoch={epoch}")
    for metric_name, metric_value in metrics_values.items():
        print(f"\t\t{metric_name}={metric_value[-1]:.3f}")
    print()