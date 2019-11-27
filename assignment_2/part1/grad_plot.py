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

Vanilla = VanillaRNN(30, 10, 256, 10, device).to(device=device)
lstm = LSTM(30, 10, 256, 10, device).to(device=device)

# Initialize the dataset and data loader (note the +1)
dataset = PalindromeDataset(30+1)
data_loader = DataLoader(dataset, 64, num_workers=1)

# Setup the loss and optimizer
criterion = torch.nn.CrossEntropyLoss() # fixme

for step, (batch_inputs, batch_targets) in enumerate(data_loader):

    # Only for time measurement of step through network
    t1 = time.time()
    # Add more code here ...
    batch_inputs = torch.nn.functional.one_hot(batch_inputs.to(torch.int64), 10).to(torch.double).to(device=device)
    batch_targets = batch_targets.to(device)
    out = Vanilla.forward(batch_inputs.to(device))
    out2 = lstm.forward(batch_inputs.to(device))

    ############################################################################
    # QUESTION: what happens here and why?
    ############################################################################
    torch.nn.utils.clip_grad_norm(Vanilla.parameters(), max_norm=10)
    torch.nn.utils.clip_grad_norm(lstm.parameters(), max_norm=10)
 
    ############################################################################

    loss = criterion(out, batch_targets)   
    loss.backward()
    loss2 = criterion(out2, batch_targets)   
    loss2.backward()

    gradlist = []
    gradlistlstm = []
    for h in Vanilla.h_list:
        gradlist.append(torch.norm(h.grad))
    for h in lstm.h_list:
        gradlistlstm.append(torch.norm(h.grad))
    
    plt.plot(gradlist, label="RNN")
    plt.plot(gradlistlstm, label="LSTM")
    plt.title('RNN vs LSTM accuracy')
    plt.ylabel('Gradient magnitude')
    plt.xlabel('timestep')
    plt.legend()
    plt.show()
    break