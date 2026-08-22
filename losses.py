# Author: Ciaran Bench
# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


#GNLL = nn.MSELoss()
GNLL = nn.GaussianNLLLoss()

def GNLL_loss_fn(outputs,targets):
    pred_val_dbp = torch.unsqueeze(outputs[:,0],dim=-1)

    pred_var_dbp = torch.unsqueeze(outputs[:,2],dim=-1)
    targets_dbp = torch.unsqueeze(targets[:,0],dim=-1)

    pred_val_sbp = torch.unsqueeze(outputs[:,1],dim=-1)
    pred_var_sbp = torch.unsqueeze(outputs[:,3],dim=-1)
    targets_sbp = torch.unsqueeze(targets[:,1],dim=-1)

    
    sum_GNLLs = GNLL(pred_val_sbp,targets_sbp,pred_var_sbp) + GNLL(pred_val_dbp,targets_dbp,pred_var_dbp)
    #avneg_loglik = GNLL(pred_val_sbp,targets_sbp) + GNLL(pred_val_dbp,targets_dbp)
    return sum_GNLLs


def GNLL_loss_fn_exp(outputs,targets):
    pred_val_dbp =torch.unsqueeze(outputs[:,0],dim=-1)

    pred_var_dbp =  torch.unsqueeze(outputs[:,2],dim=-1)
    targets_dbp = torch.unsqueeze(targets[:,0],dim=-1)

    pred_val_sbp = torch.unsqueeze(outputs[:,1],dim=-1)
    pred_var_sbp = torch.unsqueeze(outputs[:,3],dim=-1)
    targets_sbp = torch.unsqueeze(targets[:,1],dim=-1)

    sbp = torch.mean(.5*torch.exp(-pred_var_sbp)*(pred_val_sbp-targets_sbp)**2 + .5*pred_var_sbp)
    dbp = torch.mean(.5*torch.exp(-pred_var_dbp)*(pred_val_dbp-targets_dbp)**2 + .5*pred_var_dbp)
    sum_GNLLs = sbp + dbp
    #avneg_loglik = GNLL(pred_val_sbp,targets_sbp) + GNLL(pred_val_dbp,targets_dbp)
    return sum_GNLLs


import lightning as L
import torch.nn.functional as F
import numpy as np

def MCD_loss_fn(outputs,noises,target):
    T = 100
    Gauss = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))
    target = torch.squeeze(target)
    target = target.long()
    # each output logit distribution will be corrupted by T different noises
    epsilon = Gauss.sample([outputs.shape[0], T]).to(outputs.device)
    
    # repeat the model's output noise param T times
    sigma = noises[:, None, :].repeat(1, T, 1)
    
    # also repeat each output set of logits T times
    f = outputs[:, None, :].repeat(1, T, 1)
    
    # Multiply Gaussian noise with the model's output noise param
    x_t = f + sigma * epsilon
    
    # compute loss
    x_t_sm = F.softmax(x_t, dim=-1)
    x_t_sm_avg = x_t_sm.mean(dim=-2)
    loss = F.nll_loss(torch.log(x_t_sm_avg), target=target, reduction='mean')
    return loss
