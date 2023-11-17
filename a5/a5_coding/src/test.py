import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import dataset
import model
import trainer
import utils

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
text = open("birth_dev.tsv", encoding='utf-8').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, 60)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the perceiver models
mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256)
model = model.GPT(mconf).to(device)

# Define a function for your training code
def train_fn(model, loader):
    losses = []
    pbar = tqdm(enumerate(loader), total=len(loader)) if True else enumerate(loader)
    for it, (x, y) in pbar:
        print(x.size(), y.size())
        # forward the model
        #with torch.set_grad_enabled(True):
        logits, loss = model(x, y)

# Use the if __name__ == '__main__': block to protect the entry point
if __name__ == '__main__':
    loader = DataLoader(pretrain_dataset, batch_size=60)

    # Call your training function
    train_fn(model, loader)
