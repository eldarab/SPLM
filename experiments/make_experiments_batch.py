import argparse
import os

import numpy as np
import yaml

from utils.paths import EXPERIMENTS_BATCHES_DIR
from utils.supported_experiments import MNIST, HINGE_LOSS, SPLM, MLP


def make_config_yml(yml_path, epochs=20, batch_size=1000, optimizer='splm', K=50, beta=20, scheduler=None, step_size=None, gamma=None):
    config_dict = dict(
        data=dict(
            dataset=MNIST,
            train_samples=60000,
            eval_samples=10000
        ),
        general=dict(
            seed=43,
            use_cuda=True
        ),
        model=dict(
            model_name=MLP,
            activation='Softplus',
            loss=HINGE_LOSS,
        ),
        optim=dict(
            epochs=int(epochs),
            batch_size=int(batch_size),
            optimizer=optimizer,
            K=int(K),
            beta=float(beta),
            scheduler=dict(
                use_scheduler=False
            ),
        )
    )

    with open(yml_path, 'w') as f:
        yaml.safe_dump(config_dict, f)


def make_splm_mnist_hinge_mlp(batch_name, batch_sizes, Ks, betas):
    # init batch dir
    batch_dir_path = EXPERIMENTS_BATCHES_DIR / batch_name
    os.makedirs(str(batch_dir_path), exist_ok=True)

    # get linear spaces
    batch_sizes = np.linspace(*batch_sizes).astype(int)
    Ks = np.linspace(*Ks).astype(int)
    betas = np.linspace(*betas).astype(int)

    # create config files
    for bs in batch_sizes:
        for K in Ks:
            for beta in betas:
                config_yml_name = f'batch_{bs}__K_{K}__beta_{beta}.yml'
                config_yml_path = batch_dir_path / config_yml_name
                make_config_yml(yml_path=str(config_yml_path), epochs=20, batch_size=bs, optimizer=SPLM, K=K, beta=beta)

    return batch_dir_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_batch_name', type=str, help='Experiments batch folder name to appear under experiments/batches/')
    parser.add_argument('--batch_sizes', type=tuple, default=(10, 100, 10), help='tuple (start, stop, num) to create a linspace of batch_size.')
    parser.add_argument('--Ks', type=tuple, default=(50, 100, 1), help='tuple (start, stop, num) to create a linspace of K.')
    parser.add_argument('--betas', type=tuple, default=(20., 90., 8), help='tuple (10^start, 10^stop, num) to create a linspace of beta.')
    args = parser.parse_args()

    batch_dir_path = make_splm_mnist_hinge_mlp(args.experiment_batch_name, args.batch_sizes, args.Ks, args.betas)

    print(f'Successfully created {len(os.listdir(str(batch_dir_path)))} experiments under "{batch_dir_path}".')


if __name__ == '__main__':
    main()
