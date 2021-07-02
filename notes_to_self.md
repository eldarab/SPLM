# Notes to self

## TODO

[comment]: <> (* Extract good results from experiments.)
* Conduct Adam experiments to compare to good results.
* Make sure computed gradients are correct (g_i_y_hat).
* Create a LaTeX doc sewing ML / math / Code.
* Implement more loss functions.
* Write conclusions why project failed.
* Implement more beta schedulers.
* Make use of MNIST model class

[comment]: <> (* "Sniff" what is the dependency parsing task &#40;Tamir/Roi&#41;.)

* Document all code.
* Create a README file.
* Fix all warnings, CTRL+ALT+L, CTRL+ALT+O
* Remove redundant files.
* Create utils __init\__
* Summarize results in a LaTeX table and send to Shoham.
* resolve all device related issues.
* conduct experiments with CIFAR-10 and MNIST architectures.
* add baseline [Ranger](https://lessw.medium.com/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d) (Hadar)
* [code link](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)


[comment]: <> (* Implement CIFAR-10 &#40;from PT website&#41;)

[comment]: <> (* Implement a more advanced architecture for MNIST &#40;from PT website&#41;)

[comment]: <> (* Write Shoham an Email - Jacobian, results.)


## Research questions to check

* Adam vs SPLM w/o scheduler
* Adam vs SPLM w/ scheduler
* SPLM w/ scheduler vs SPLM w/o scheduler --- WOW!
* CIFAR10
* MNIST Different architecture
* Are K and beta dependent? (small beta should come with high K) --- I think not.
* Does batch size matter? -- I think not.
* How many samples to employ?

## Why the project failed

* Can't compute Jacobian using autograd ([link](https://github.com/jshi31/Jacobian_of_MLP)).
* Can't efficiently implement custom loss functions to exploit autograd. C/C++
* Can't parallelize, complexity is linear in \# classes.
* No map_fn like in TF

* check 2 emails to Shoham
* there is a chain of failures:
1. no jacobian
2. manually create one (for loop) is inefficient
3. no map_fn in PT like in TF
4. in addition, needs to customize loss functions (doesn't scale)

## Weird phenomena

* The noisier (small batch_size) the algorithm, the better the accuracy.


## Benchmarks
* ImageNet, CIFAR-10, MNIST
* Think of Datasets from NLP / tabular as well.
* torchvision.classification.models