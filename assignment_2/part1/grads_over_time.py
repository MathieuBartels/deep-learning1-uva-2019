from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

device = 'cuda:0'

Vanilla = VanillaRNN(100, 10, 256, 10, device).to(device=device)
lstm = LSTM(100, 10, 256, 10, device).to(device=device)

# Initialize the dataset and data loader (note the +1)
dataset = PalindromeDataset(100+1)
data_loader = DataLoader(dataset, 64, num_workers=1)

# Setup the loss and optimizer
criterion = torch.nn.CrossEntropyLoss() # fixme

gradients = []

for model in [Vanilla, lstm]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        # Add more code here ...
        batch_inputs = torch.nn.functional.one_hot(batch_inputs.to(torch.int64), 10).to(torch.double).to(device=device)
        batch_targets = batch_targets.to(device)
        out = model.forward(batch_inputs.to(device))

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10)
    
        ############################################################################

        loss = criterion(out, batch_targets)   
        loss.backward()

        optimizer.step()
        gradlist = [] 
        for h in model.h_list:
            gradlist.append(torch.norm(h.grad))
        gradients.append(gradlist)
        break

plt.plot(gradients[0], label="RNN")
plt.plot(gradients[1], label="LSTM")
plt.title('RNN vs LSTM gradients')
plt.ylabel('Gradient magnitude')
plt.xlabel('timestep')
plt.legend()
plt.show()