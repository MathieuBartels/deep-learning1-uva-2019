from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from dataset import PalindromeDataset
from train import train, calc_accuracy
import numpy as np

class Config:
    def __init__(self, batch_size, input_dim, input_length, learning_rate, max_norm, model_type, num_classes, num_hidden, train_steps, device):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_length = input_length
        self.learning_rate = learning_rate
        self.max_norm = max_norm
        self.model_type = model_type
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.train_steps = train_steps
        self.device = device

input_length = 10
device = 'cuda:0'

config = Config(2048, 10, input_length, 0.001, 10.0, 'RNN', 10, 128, 100, device=device)
num_seeds = 10

results = {'RNN': [], 'LSTM': []}

for input_length in range(1,40,1):
    if input_length > 10:
        config.learning_rate = 0.01
    config.input_length = input_length
    dataset = PalindromeDataset(config.input_length+1)
    test_set_batch = 4000
    data_loader = DataLoader(dataset, test_set_batch, num_workers=1)
    for batch_inputs, batch_targets in data_loader:
        batch_inputs = torch.nn.functional.one_hot(batch_inputs.to(torch.int64), 10).to(torch.double).to(device=device)
        batch_targets = batch_targets.to(device)
        for model_type in ['RNN', 'LSTM']:
            config.model_type = model_type
            acc = []
            for i in range(num_seeds):
                # print(model_type, input_length)
                model = train(config)
                model.eval()
                out = model.forward(batch_inputs.to(device))
                acc.append(calc_accuracy(out, batch_targets))
            results[model_type].append({input_length: np.mean(acc)})
        break
print(results)