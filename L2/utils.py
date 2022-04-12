import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn


def compute_error_rate(model, data_loader, device="cpu"):
    """Evaluate model on all samples from the data loader.
    """
    # Put the model in eval mode, and move to the evaluation device.
    model.eval()
    model.to(device)
    if isinstance(data_loader, InMemDataLoader):
        data_loader.to(device)

    num_errs = 0.0
    num_examples = 0
    # we don't need gradient during eval!
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model.forward(x)
            _, predictions = outputs.data.max(dim=1)
            num_errs += (predictions != y.data).sum().item()
            num_examples += x.size(0)
    return num_errs / num_examples


def plot_history(history):
    """Helper to plot the trainig progress over time."""
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    train_loss = np.array(history["train_losses"])
    plt.semilogy(
        np.arange(train_loss.shape[0]), train_loss, label="batch train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    train_errs = np.array(history["train_errs"])
    plt.plot(np.arange(train_errs.shape[0]),
             train_errs, label="batch train error rate")
    val_errs = np.array(history["val_errs"])
    plt.plot(val_errs[:, 0], val_errs[:, 1],
             label="validation error rate", color="r")
    plt.ylim(0, 0.20)
    plt.legend()


class InMemDataLoader(object):
    """
    A data loader that keeps all data in CPU or GPU memory.
    """

    __initialized = False

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        transform=None,
    ):
        """A torch dataloader that fetches data from memory."""
        batches = []
        for i in tqdm(range(len(dataset))):
            batch = [torch.as_tensor(t) for t in dataset[i]]
            batches.append(batch)
        tensors = [torch.stack(ts) for ts in zip(*batches)]
        dataset = torch.utils.data.TensorDataset(*tensors)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError(
                "sampler option is mutually exclusive with " "shuffle")

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last
            )

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.transform = transform
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ("batch_size", "sampler", "drop_last"):
            raise ValueError(
                "{} attribute should not be set after {} is "
                "initialized".format(attr, self.__class__.__name__)
            )

        super(InMemDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            if self.transform is None:
                yield self.dataset[batch_indices]
            else:
                x, y = self.dataset[batch_indices]
                yield self.transform(x), y

    def __len__(self):
        return len(self.batch_sampler)

    def to(self, device):
        self.dataset.tensors = tuple(t.to(device)
                                     for t in self.dataset.tensors)
        return self


def SGD(
    model,
    data_loaders,
    alpha=1e-4,
    epsilon=0.0,
    decay=0.0,
    norm_constraint=None,
    polyak_averaging=None,
    num_epochs=1,
    max_num_epochs=np.nan,
    patience_expansion=1.5,
    log_every=100,
    device="cpu",
):

    # Put the model in train mode, and move to the evaluation device.
    model.train()
    model.to(device)
    for data_loader in data_loaders.values():
        if isinstance(data_loader, InMemDataLoader):
            data_loader.to(device)

    # Problem 1.3: Initialize momentum variables
    velocities = [torch.zeros_like(p) for p in model.parameters()]

    if polyak_averaging is not None:
        avg_named_parameters = {name: torch.clone(p).detach() for name, p in model.named_parameters()}
        for p_avg in avg_named_parameters.values():
            p_avg.requires_grad = False

    # Problem 1.2: Schedule learning rate
    try:
        _alpha = iter(alpha)
    except TypeError:
        _alpha = itertools.repeat(alpha)

    # Problem 1.1: Schedule learning rate
    try:
        _epsilon = iter(epsilon)
    except TypeError:
        _epsilon = itertools.repeat(epsilon)

    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {"train_losses": [], "train_errs": [], "val_errs": []}
    print("Training the model!")
    print("Interrupt at any time to evaluate the best validation model so far.")
    try:
        tstart = time.time()
        siter = iter_
        while epoch < num_epochs:
            model.train()
            epoch += 1
            if epoch > max_num_epochs:
                break

            # Learning rate and momentum schedule
            alpha = next(_alpha)
            epsilon = next(_epsilon)

            for x, y in data_loaders["train"]:
                x = x.to(device)
                y = y.to(device)
                iter_ += 1
                # This calls the `forward` function: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
                out = model(x)
                loss = model.loss(out, y)
                loss.backward()
                _, predictions = out.max(dim=1)
                batch_err_rate = (predictions != y).sum().item() / out.size(0)

                history["train_losses"].append(loss.item())
                history["train_errs"].append(batch_err_rate)

                # disable gradient computations - we do not want torch to
                # backpropagate through the gradient application!
                with torch.no_grad():
                    for (name, p), v in zip(model.named_parameters(), velocities):
                        if "weight" in name:
                            # Penalty for weight decay (L2 regularization)
                            p -= decay * 2 * p

                        v[...] = epsilon * v - alpha * p.grad
                        p += v

                        if "weight" in name and norm_constraint is not None:
                            norms = torch.linalg.norm(p, dim=list(range(1, p.dim())), keepdim=True)
                            mask = (norms > norm_constraint).squeeze()
                            p[mask] /= norms[mask]


                        # Zero gradients for the next iteration
                        p.grad.zero_()

                if iter_ % log_every == 0:
                    num_iter = iter_ - siter + 1
                    print(
                        "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%, steps/s {3: >5.2f}".format(
                            iter_,
                            loss.item(),
                            batch_err_rate * 100.0,
                            num_iter / (time.time() - tstart),
                        )
                    )
                    tstart = time.time()

            with torch.no_grad():
                if polyak_averaging is not None:
                    state_dict = model.state_dict()
                    avg_state_dict = state_dict.copy()
                    avg_named_parameters = {name: polyak_averaging * p_avg + (1 - polyak_averaging) * state_dict[name] for name, p_avg in avg_named_parameters.items()}
                    avg_state_dict.update(avg_named_parameters)
                    model.load_state_dict(avg_state_dict)

                val_err_rate = compute_error_rate(
                    model, data_loaders["valid"], device)
                history["val_errs"].append((iter_, val_err_rate))

                if val_err_rate < best_val_err:
                    # Adjust num of epochs
                    num_epochs = int(np.maximum(
                        num_epochs, epoch * patience_expansion + 1))
                    best_epoch = epoch
                    best_val_err = val_err_rate
                    best_params = [p.detach().cpu() for p in model.parameters()]

                if polyak_averaging is not None:
                    model.load_state_dict(state_dict)

            clear_output(True)
            m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs".format(
                epoch, val_err_rate * 100.0, num_epochs
            )
            print("{0}\n{1}\n{0}".format("-" * len(m), m))

    except KeyboardInterrupt:
        pass

    if best_params is not None:
        print("\nLoading best params on validation set (epoch %d)\n" %
              (best_epoch))
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_params):
                param[...] = best_param
    plot_history(history)


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.dist = torch.distributions.Bernoulli(self.p)
        self.factor = 1 / (1 - self.p)
        self.zero = torch.Tensor([0])
    
    def forward(self, x):
        if self.training:
            out = self.factor * x
            mask = (self.dist.sample(x.size()) == 1)
            out[mask] = 0
            return out
        return x

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.layers = nn.Sequential(*args, **kwargs)

    def forward(self, X):
        X = X.view(X.size(0), -1)
        return self.layers.forward(X)

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)

class Model2D(Model):
    def forward(self, X):
        return self.layers.forward(X)

class ShapePrinter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = torch.as_tensor(eps)
        self.momentum = torch.as_tensor(momentum)
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.parameter.Parameter(torch.ones((1, self.num_features, 1, 1)))
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.num_features, 1, 1)))

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.ones((1, self.num_features, 1, 1)))
            self.register_buffer("running_var", torch.zeros((1, self.num_features, 1, 1)))
            if self.momentum is None:
                self.register_buffer("n", torch.as_tensor(0))

    def forward(self, x):
        if self.train or not self.track_running_stats:
            batch_mean = x.mean((0, 2, 3), True)
            batch_var = x.var((0, 2, 3), False, True)
            if self.track_running_stats:
                if self.momentum is not None:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                else:
                    self.running_mean = self.running_mean * (self.n / (self.n + 1)) + batch_mean / (self.n + 1)
                    self.running_var = self.running_var * (self.n / (self.n + 1)) + batch_var / (self.n + 1)
                    self.n += 1

        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        y = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        if self.affine:
            return self.weight * y + self.bias
        return y
