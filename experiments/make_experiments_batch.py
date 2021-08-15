import argparse
import os

import yaml
import numpy as np
import pandas as pd

from utils.logging import get_time_str
from utils.paths import EXPERIMENTS_BATCHES_DIR


def make_config_yml(yml_path, epochs=20, batch_size=1000, optimizer='splm', K=50, beta=20, scheduler=None, step_size=None, gamma=None):
    data = dict(
        general=dict(
            seed=43,
            use_cuda=True
        ),
        data=dict(
            dataset='mnist',
            train_samples=10000,
            eval_samples=2001
        ),
        model=dict(
            input_dim=784,
            hidden_dim=32,
            output_dim=10,
            activation='Softplus',
            num_classes=10,
            loss='hinge',
        ),
        optim=dict(
            epochs=int(round(epochs)),
            batch_size=int(round(batch_size)),
            optimizer=optimizer,
            K=int(round(K)),
            beta=int(round(beta)),
            scheduler=dict(
                type=scheduler,
                step_size=int(round(step_size)),
                gamma=int(round(gamma)),
            ),
        )
    )

    with open(yml_path, 'w') as f:
        yaml.safe_dump(data, f)


def make_splm_experiments_batch(batch_size_interval, K_interval, beta_interval, batch_name):
    data = []

    num_experiments = 0

    for batch_size in np.linspace(batch_size_interval[0], batch_size_interval[1], batch_size_interval[2]):
        for K in np.linspace(K_interval[0], K_interval[1], K_interval[2]):
            for beta in np.logspace(beta_interval[0], beta_interval[1], beta_interval[2]):
                beta = int(round(beta))
                K = int(round(K))
                batch_size = int(round(batch_size))
                yml_path = f'{EXPERIMENTS_BATCHES_DIR}/{batch_name}/batch_{batch_size}__K_{K}__beta_{beta}.yml'

                make_config_yml(yml_path=yml_path, batch_size=batch_size, K=K, beta=beta)

                data.append((batch_size, K, beta))
                num_experiments += 1

    pd.DataFrame(data, columns=['batch_size', 'K', 'beta']).to_csv(f'{EXPERIMENTS_BATCHES_DIR}/{batch_name}/.summary.csv', index=False)

    return num_experiments


def make_splm_experiments_batch_step_beta(beta_interval, step_size_interval, gamma_interval, batch_name):
    data = []

    num_experiments = 0

    for beta in np.linspace(beta_interval[0], beta_interval[1], beta_interval[2]):
        for step_size in np.linspace(step_size_interval[0], step_size_interval[1], step_size_interval[2]):
            for gamma in np.linspace(gamma_interval[0], gamma_interval[1], gamma_interval[2]):
                step_size = int(round(step_size))
                gamma = int(round(gamma))
                beta = int(round(beta))
                yml_path = f'{EXPERIMENTS_BATCHES_DIR}/{batch_name}/scheduler_step__beta_{beta}__step_{step_size}__gamma_{gamma}.yml'

                make_config_yml(yml_path=yml_path, beta=beta, step_size=step_size, gamma=gamma, scheduler='step_beta')

                data.append((beta, step_size, gamma))
                num_experiments += 1

    pd.DataFrame(data, columns=['beta', 'step_size', 'gamma']).to_csv(f'{EXPERIMENTS_BATCHES_DIR}/{batch_name}/.summary.csv', index=False)

    return num_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_batch_name', type=str, default=f'batch_{get_time_str()}',
                        help='Experiments batch folder name to appear under ./experiments/.')
    parser.add_argument('--batch_size', type=tuple, default=(10, 100, 10), help='tuple (start, stop, num) to create a linspace of batch_size.')
    parser.add_argument('--K', type=tuple, default=(50, 100, 1), help='tuple (start, stop, num) to create a linspace of K.')
    parser.add_argument('--beta', type=tuple, default=(20., 90., 8), help='tuple (10^start, 10^stop, num) to create a logspace of beta.')
    parser.add_argument('--step_size', type=tuple, default=(1, 10, 10), help='tuple (start, stop, num) to create a linspace of beta.')
    parser.add_argument('--gamma', type=tuple, default=(1, 10, 10), help='tuple (start, stop, num) to create a linspace of beta.')
    args = parser.parse_args()

    # num_experiments = gen_splm_experiment_set(args.experiments_dir, args.batch_size, args.K, args.beta)
    num_experiments = make_splm_experiments_batch_step_beta(args.beta, args.step_size, args.gamma)

    print(f'Successfully created {num_experiments} experiments under {args.experiments_dir}')


if __name__ == '__main__':
    main()
