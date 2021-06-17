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

[comment]: <> (* Implement CIFAR-10 &#40;from PT website&#41;)

[comment]: <> (* Implement a more advanced architecture for MNIST &#40;from PT website&#41;)

[comment]: <> (* Write Shoham an Email - Jacobian, results.)


## Research questions to check

* Adam vs SPLM w/o scheduler
* Adam vs SPLM w/ scheduler
* SPLM w/ scheduler vs SPLM w/o scheduler
* CIFAR10
* MNIST Different architecture
* Are K and beta dependent? (small beta should come with high K)

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

