import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from train import evaluator
from utils.loss_functions import MulticlassHingeLoss

if __name__ == '__main__':
    torch.manual_seed(42)

    # model parameters
    input_dim = 784
    hidden_dim = 32
    output_dim = num_classes = 10
    # loss_fn = MulticlassHingeLoss(num_classes)
    loss_fn = nn.CrossEntropyLoss()

    # optimization parameters
    epochs = 200
    batch_size = 100
    lr = 0.005
    beta = 1000.0
    use_cuda = False

    # mnist data
    mnist_train = datasets.MNIST("./datasets", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_test = datasets.MNIST("./datasets", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_train.data = mnist_train.data[:1000]
    mnist_test.data = mnist_test.data[:100]
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    # initialize model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TODO return type hinting

    # initialization
    metrics_fns = {'accuracy': accuracy_score}
    metrics_values = {'train_' + metric_name: [] for metric_name in metrics_fns.keys()}
    metrics_values.update({'eval_' + metric_name: [] for metric_name in metrics_fns.keys()})
    metrics_values['train_loss'] = []
    metrics_values['eval_loss'] = []

    classes = [i for i in range(10)]  # TODO
    # TODO model.init_weights_normal()

    for epoch in range(epochs):
        loss = 0
        for i, (x, y) in tqdm(enumerate(train_loader)):
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            # forward pass
            y_hat = model(x.flatten(start_dim=1))

            loss = loss_fn(y_hat, y)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        metrics_values = evaluator(model, train_loader, loss_fn, metrics_fns, metrics_values, use_cuda)
        model.train(False)
        metrics_values = evaluator(model, eval_loader, loss_fn, metrics_fns, metrics_values, use_cuda)
        model.train(True)

    for metric_name, metric_values in metrics_values.items():
        plt.plot(metric_values)
        plt.title(metric_name)
        plt.xlabel('Epochs')
        plt.show()
