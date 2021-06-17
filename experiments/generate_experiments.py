import argparse

import yaml
import numpy as np
import pandas as pd


def gen_splm_yml(yml_path, batch_size, K, beta):
    batch_size = int(round(batch_size))
    K = int(round(K))
    beta = int(round(beta))
    data = dict(
        general=dict(
            seed=42,
            use_cuda=True
        ),
        data=dict(
            dataset='mnist',
            train_samples=1000,
            eval_samples=100
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
            epochs=100 if beta < 500 else 200,
            batch_size=batch_size,
            optimizer='splm',
            K=K,
            beta=beta
        )
    )

    with open(yml_path, 'w') as f:
        yaml.safe_dump(data, f)


def gen_splm_experiment_set(folder_path, batch_size_interval, K_interval, beta_interval, sub_folder='splm_mnist_hinge'):
    data = []

    num_experiments = 0

    for batch_size in np.linspace(batch_size_interval[0], batch_size_interval[1], batch_size_interval[2]):
        batch_size = int(round(batch_size))
        for K in np.linspace(K_interval[0], K_interval[1], K_interval[2]):
            K = int(round(K))
            for beta in np.logspace(beta_interval[0], beta_interval[1], beta_interval[2]):
                beta = int(round(beta))
                gen_splm_yml(f'{folder_path}/{sub_folder}/batch_{batch_size}__K_{K}__beta_{beta}.yml', batch_size, K, beta)
                data.append((batch_size, K, beta))
                num_experiments += 1

    pd.DataFrame(data, columns=['batch_size', 'K', 'beta']).to_csv(f'{folder_path}/{sub_folder}/.summary.csv', index=False)

    return num_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', type=str, default='/home/eldar.a/SPLM2/experiments', help='Experiments folder path.')
    parser.add_argument('--batch_size', type=tuple, default=(10, 100, 10), help='tuple (start, stop, num) to create a linspace of batch_size.')
    parser.add_argument('--K', type=tuple, default=(20, 100, 1), help='tuple (start, stop, num) to create a linspace of K.')
    parser.add_argument('--beta', type=tuple, default=(0.5, 5., 10), help='tuple (10^start, 10^stop, num) to create a logspace of beta.')
    args = parser.parse_args()

    num_experiments = gen_splm_experiment_set(args.experiments_dir, args.batch_size, args.K, args.beta)

    print(f'Successfully created {num_experiments} experiments under {args.experiments_dir}')


if __name__ == '__main__':
    main()
