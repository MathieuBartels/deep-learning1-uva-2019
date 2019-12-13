"""
Some of this code is inspired using the example of pytorches VAE implementation https://github.com/pytorch/examples/blob/master/vae/main.py
"""

import argparse

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from datasets.bmnist import bmnist
from scipy.stats import norm
import os
device = torch.device("cuda:0")
# device = torch.device("cpu")

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.enhidden = nn.Linear(784, hidden_dim)
        self.enmean = nn.Linear(hidden_dim, z_dim)
        self.enstd = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        hidden = self.enhidden(input).relu()
        mean, std = self.enmean(hidden), self.enstd(hidden)
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.dehidden = nn.Linear(z_dim, hidden_dim)
        self.deimage = nn.Linear(hidden_dim, 784)
    
    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.deimage(self.dehidden(input).relu()).sigmoid()
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        average_negative_elbo = None
        mean, std = self.encoder(input)

        eps = torch.randn_like(mean)
        z = mean + std.exp().sqrt() * eps

        decoded = self.decoder(z)

        BCE = nn.functional.binary_cross_entropy(decoded,input, reduction='sum')
        KLD = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp())
        return (BCE + KLD) / input.shape[0]

    def sample(self, n_samples, z=None):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn([n_samples, self.z_dim]).to(device) if type(z)==type(None) else z
        im_means = self.decoder(z)
        try:
            sampled_ims = torch.bernoulli(im_means)
        except:
            print(im_means.shape)
            print(z)
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    feature_length = 784
    average_epoch_elbo = []
    for sample in data:
        optimizer.zero_grad()
        input = sample.reshape(-1, feature_length).to(device)
        out = model.forward(input)
        if model.training:
            out.backward()
            optimizer.step()
        average_epoch_elbo.append(out.item())

    return np.mean(average_epoch_elbo)


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------
        if not os.path.exists('images'):
            os.makedirs('images')
        im_sample, im_mean = model.sample(25)
        save_image(im_sample.reshape(-1, 1, 28, 28), 'images/sample_' + str(epoch) + '.png', nrow=5, normalize=True)
        save_image(im_mean.reshape(-1, 1, 28, 28), 'images/mean_' + str(epoch) + '.png', nrow=5, normalize=True)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        num = 20
        samples = torch.zeros((400, 1, 28, 28))
        sample = 0
        for x in np.linspace(0.01, 0.99, num):
            for y in np.linspace(0.01, 0.99, num):
                model.sample(1, z=torch.tensor([x, y]).to(device))[1].reshape(1,1,28,28)
                samples[sample, :, :, :] = model.sample(1, z=torch.tensor([norm.ppf(x), norm.ppf(y)]).to(device))[1].reshape(1,1,28,28)
                sample += 1
        save_image(samples, "manifold.png", nrow=20)

    save_elbo_plot(train_curve, val_curve, 'elbo_pdf.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
