# SPLM: Stochastic proximal linear method for structured non-convex problems

## Paper Authors
**Tamir Hazan, Shoham Sabach, Sergey Voldman**

## Code Author
**[Eldar Abraham](https://github.com/eldarab)**

## Abstract
In this work, motivated by the challenging task of learning a deep neural network, we consider optimization problems that consist of minimizing a finite-sum of non-convex and non-smooth functions, where the non-smoothness appears as the maximum of non-convex functions with Lipschitz continuous gradient. Due to the large size of the sum, in practice, we focus here on stochastic first-order methods and propose the Stochastic Proximal Linear Method (SPLM) that is based on minimizing an appropriate majorizer at each iteration and is guaranteed to almost surely converge to a critical point of the objective function, where we also prove its convergence rate in finding critical points.

## Links

[Paper](https://ssabach.net.technion.ac.il/files/2020/08/HSV2020.pdf)

[Code](https://github.com/eldarab/SPLM)

## Setup

### To set up the project:
```
cd ~
git clone https://github.com/eldarab/SPLM
cd SPLM
conda env create --file env.yml
```

### To run a single template experiment:

Running the experiment
```
conda activate splm
cd ~/SPLM
python main.py experiments/batches/templates/<CONFIG_NAME>.yml
```

Reviewing results is under
```
experiments/results/templates/<CONFIG_NAME>
```

You can create your own `.yml` configuration files and add them to `experiments/batches/templates/`.  

### To run a batch of experiments:

Running the batch (Automatically calls `python main.py <CONFIG_NAME>.yml` for each config file under `BATCH_NAME`).
It is recommended to run a batch inside a `screen` session, because execution may take a while.
```
screen -S splm_batch
conda activate splm
cd ~/SPLM
```
```
sh run_batch.sh experiments/batches/<BATCH_DIR>
```

Reviewing results is under
```
experiments/results/<BATCH_DIR>/<CONFIG_NAME>

```