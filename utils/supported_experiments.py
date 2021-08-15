# models
FF_MNIST_CLASSIFIER = 'mlp'
VGG11_BN = 'vgg11_bn'
RESNET18 = 'resnet18'

SUPPORTED_MODELS = {
    FF_MNIST_CLASSIFIER,
    VGG11_BN,
    RESNET18
}

# losses
MULTI_MARGIN_LOSS = 'MultiMarginLoss'
CE_LOSS = 'CE'
HINGE_LOSS = 'hinge'

SUPPORTED_LOSSES = {
    MULTI_MARGIN_LOSS,
    CE_LOSS,
    HINGE_LOSS
}

# datasets
MNIST = 'mnist'
CIFAR10 = 'cifar10'
SYNTHETIC = 'synthetic'

SUPPORTED_DATASETS = {
    MNIST,
    CIFAR10,
    SYNTHETIC
}

# optimizers
SPLM = 'splm'
ADAM = 'adam'

SUPPORTED_OPTIMIZERS = {
    ADAM,
    SPLM
}

# schedulers
STEP_BETA = 'step_beta'

SUPPORTED_SCHEDULERS = {
    STEP_BETA
}
