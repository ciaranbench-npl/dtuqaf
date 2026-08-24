# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.
#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
This code is adapted from the work cited below under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) licence. This licence permits sharing and adaptation, provided appropriate credit is given to the original authors and source. The code may include modifications made for the present analysis and should not be treated as an exact reproduction of the original implementation.

@article{bench2025uncertainty,
  title={Uncertainty quantification with approximate variational learning for wearable photoplethysmography prediction tasks},
  author={Bench, Ciaran and Desai, Vivek and Moulaeifard, Mohammad and Strodthoff, Nils and Aston, Philip and Thompson, Andrew},
  journal={Machine Learning: Health},
  volume={1},
  number={1},
  pages={015013},
  year={2025},
  publisher={IOP Publishing},
  doi={10.1088/3049-477X/ae0b74}
}
"""


import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
#import ivon
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np

print('LIBRARIES LOADED')
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#https://ysngshn.github.io/research/why-ivon/#examples-using-ivon
from losses import MCD_loss_fn

from argparser import get_args_AF

from prepare_dataset import create_dataloaders_AF
# main.py
from models import NeuralNetwork_AF
from models import init_weights
from losses import MCD_loss_fn


# In[2]:


#define_hyperparams
args = get_args_AF()
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
weight_decay = args.weight_decay
momentum = args.momentum

KERN = args.KERN
dr = args.dr
model_dir = args.model_dir


# In[4]:


# Define the directory name
directory = model_dir + "/checkpoints_af_mcd_sgd_" + str(learning_rate) + '_' + str(weight_decay)+ '_' + str(batch_size)+ '_' + str(momentum) + '_' + str(KERN) + '_' + str(dr)

# Check if the directory exists
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

hparams = {
    'batch_size': batch_size,
    'learning_rate': learning_rate, 
    'epochs': epochs, 
    'weight_decay': weight_decay,
    'momentum': momentum, 
    'KERN': KERN,
    'dr': dr
}
#save hyperparams...
np.save(directory + '/hparams',hparams)


# In[15]:


torch.manual_seed(0)
device = 'cuda'  # 'cuda'
model = NeuralNetwork_AF().to(device)
loss_fn = MCD_loss_fn
train_dataloader, val_dataloader, test_dataloader, cal_dataloader = create_dataloaders_AF()


# In[ ]:





# In[16]:





# In[17]:


def train_loop(dataloader, model, loss_fn, optimizer,epoch_ind):
    size = len(dataloader.dataset)
    model.train()
    train_losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        logit = model(X)
        loss = loss_fn(logit[:,:2],logit[:,2:], y)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    checkpoint = {
        'epoch': epoch_ind,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f'{directory}/checkpoint_epoch_{epoch_ind}.pth')

    # Validation phase
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            logit = model(X)
            val_loss = loss_fn(logit[:,:2],logit[:,2:], y)
            val_losses.append(val_loss.item())
    
    # Calculate mean losses
    mean_train_loss = np.mean(train_losses)
    mean_val_loss = np.mean(val_losses)
    
    # Log losses to file
    log_file = directory + '/training_log.txt'
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch_ind}: Train Loss = {mean_train_loss:.6f}, Val Loss = {mean_val_loss:.6f}\n")


# In[ ]:


def weight_reset(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        
weight_reset(model)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs*batch_size, eta_min=0)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer,t)


print("Done training with SGD!")


# In[ ]:





# In[ ]:




