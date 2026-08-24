# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.
#!/usr/bin/env python
# coding: utf-8
"""
Disclaimer: This code is adapted from the work cited below under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) licence. This licence permits sharing and adaptation, provided appropriate credit is given to the original authors and source. The code may include modifications made for the present analysis and should not be treated as an exact reproduction of the original implementation.

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

# In[6]:

from __future__ import annotations
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

import numpy as np

print('LIBRARIES LOADED')
import os
import pandas as pd
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#https://ysngshn.github.io/research/why-ivon/#examples-using-ivon
from models import NeuralNetwork_AF
from models import init_weights
from losses import MCD_loss_fn
from utils import flatten_list_of_batches
from argparser import get_args_AF
from prepare_dataset import create_dataloaders_AF,create_dataloaders_AF_noise_aug


# In[ ]:


#define_hyperparams
args = get_args_AF()

checkpoint_path = args.checkpoint_path
calib = args.calib
noise_data = args.addnoise


# In[3]:





# In[39]:


directory = '/'.join(checkpoint_path.split('/')[:-1]) + '/'

hparams = np.load(directory + '/hparams.npy',allow_pickle=True)
hparams = hparams.item()
batch_size = hparams['batch_size']
learning_rate = hparams['learning_rate']
epochs = hparams['epochs']
weight_decay = hparams['weight_decay']
momentum = hparams['momentum']
KERN = hparams['KERN']

dr = hparams['dr']

#test_mc0=2
MCD_SAMPLES = 1


# In[ ]:


torch.manual_seed(0)
device = 'cuda'  # 'cuda'
model = NeuralNetwork_AF(eval=True, dr=dr,KERN=KERN).to(device)
loss_fn = MCD_loss_fn

if noise_data:
        train_dataloader, val_dataloader, test_dataloader,calib_dataloader = create_dataloaders_AF_noise_aug()

else:
    train_dataloader, val_dataloader, test_dataloader,calib_dataloader = create_dataloaders_AF()


# In[66]:


def test_loop(dataloader, model):
    model.eval()
    #for m in model.modules():
    #    if isinstance(m, torch.nn.Dropout):
    #        m.train()
            
    preds = []
    test_entropies = []
    test_H2s = []
    gts = []
    list_of_sampled_probs = []
    with torch.no_grad():
        for X, y in dataloader:

            data=X.to('cuda')
            target = y.to('cuda')
            # prediction = self(data)
            # outputs = prediction[:,:2]
            # noises = prediction[:,2:]
    
            dropout_logits = []
            SM_dropout_logits = []
            noise_dropout_outputs = []
            noise_corrupted_dropout_outputs = []
            entropies_noise_corrupted_probs = []
            entropies_non_noise_corrupted_probs = []
            
            for i in range(MCD_SAMPLES):
                preds_all = model(data)
                print(preds_all)
                print('size of model outputs')
                print(preds_all.size())
                #print(preds_all)
                dropout_logits.append(preds_all[:, :2].cpu())
                SM_logits = F.softmax(preds_all[:, :2], dim=-1)
                SM_dropout_logits.append(SM_logits.cpu())
                noise_dropout_outputs.append(preds_all[:, 2:].cpu())
    
                noise_output = preds_all[:, 2:]
                # print(noise_output.size(0))
                logit_output = preds_all[:, :2]
                print('size of noises')
                print(noise_output.size())
                print('size of logits')
                print(logit_output.size())
                #print(logit_output.size()) #64, 2
                # sampling softmax in the style of Kendall/Gal https://arxiv.org/abs/1703.04977
                T = 1
                Gauss = torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(2), torch.eye(2)
                )
                # each output logit distribution will be corrupted by T different noises, with the variance of each defined by the model's output noise param
                # get the T generic gaussian noises for each set of logits
                epsilon = Gauss.sample([logit_output.shape[0], T]).to(
                    noise_output.device
                )
                print('epsilon shape')
                print(epsilon.size())
                print(epsilon)
                # print('size of epsilon' + str(epsilon.size()))
                # Go from shape: [batch x num_classes] -> [batch x T x num_classes]
                # repeat the model's output noise param T times, so we can use it to define all the variances for the T noise samples applied to each output set of logits
                sigma = (
                    noise_output[:, None, :].repeat(1, T, 1).to(noise_output.device)
                )  
                print('sigma shape')
                print(sigma.size())
                print(sigma)
                # noises
                # print('size of sigma' + str(sigma.size()))
                # also repeat each output set of logits T times.
                f = (
                    logit_output[:, None, :].repeat(1, T, 1).to(noise_output.device)
                )  
                print('f shape')
                print(f.size())
                # logits
                # print('size of f/logits' + str(f.size()))
                # Multiply Gaussian noise with the model's output noise param, and add to the output logits. This gives us \hat{x_t} from Kendall/Gal
                # remember, \hat{x_t} contains T versions of each set of output logits, where each is corruped by its own random noise
                # parameterised by the model's output noise.
                print(sigma * epsilon)
                x_t = f + sigma * epsilon
                print('xt shape')
                print(x_t.size())
                #print(x_t.size()) #64, 100. 2
                # print('size of corrupted logits' + str(x_t.size()))
                # softmax each noise corrupted distribution
                x_t_sm = F.softmax(x_t, dim=-1)
                print('x_t_sm shape')
                print(x_t_sm.size())
                # average over T
                x_t_sm_avg = torch.mean(x_t_sm, dim=-2)
                # print('size of averaged/softmaxxed corrupted logits' + str(x_t_sm_avg.size()))
                print('x_t_sm_avg shape')
                print(x_t_sm_avg.size())
    
                noise_corrupted_dropout_outputs.append(
                    torch.unsqueeze(x_t_sm_avg, dim=-1)
                )
                print('noise_corrupted_dropout_outputs shape')
                print(len(noise_corrupted_dropout_outputs))
                print(noise_corrupted_dropout_outputs[0].size())
    
                # Used to calc H2
                entropy_noise_corrupted_dropout_output = -1.0 * np.sum(
                    x_t_sm_avg.cpu().numpy() * np.log(x_t_sm_avg.cpu().numpy() + 1e-16),
                    axis=-1,
                )
                entropies_noise_corrupted_probs.append(
                    entropy_noise_corrupted_dropout_output
                )
    
                # used to calc H3
                entropy_dropout_output = -1.0 * np.sum(
                    SM_logits.cpu().numpy() * np.log(SM_logits.cpu().numpy() + 1e-16),
                    axis=-1,
                )
                entropies_non_noise_corrupted_probs.append(entropy_dropout_output)
            #print('dropout_logits')
            #print(dropout_logits[0])
            #print(dropout_logits[1])
            H2 = np.mean(entropies_noise_corrupted_probs, axis=0)
            test_H2s.append(H2)
            # self.test_H2[dataloader_idx].append(H2)
            H3 = np.mean(entropies_non_noise_corrupted_probs, axis=0)
            # self.test_H3[dataloader_idx].append(H3)
    
            noise_corrupted_dropout_outputs = torch.cat(
                noise_corrupted_dropout_outputs, dim=-1
            )
            print('noise_corrupted_dropout_outputs catted shape')
            print(noise_corrupted_dropout_outputs.size())
            # take the mean over all the noise corrupted dists
            #print(noise_corrupted_dropout_outputs.size()) #64, 2, 100
            MCD_mean = torch.mean(noise_corrupted_dropout_outputs, dim=-1)
            print('MCD_mean shape')
            print(MCD_mean.size())
            # print('size of MCD_mean' + str(MCD_mean.size()))
    
            # compute the entropy of the distribution.
            entropy_corrupted = -1.0 * np.sum(
                MCD_mean.cpu().numpy() * np.log(MCD_mean.cpu().numpy() + 1e-16), axis=-1
            )
            print('entropy')
            print(np.shape(entropy_corrupted))
            test_entropies.append(entropy_corrupted)
            # self.test_entropy_corrupted[dataloader_idx].append(entropy_corrupted)
    
            # self.test_targs[dataloader_idx].append(test_batch[1].cpu())
    
            # save corrupted preds into test_preds so test eval metrics are computed using them
            preds.append(MCD_mean)
            # self.test_preds_numpy[dataloader_idx].append(MCD_mean.cpu().numpy())
            logit_outputs_mean = np.mean(dropout_logits, axis=0)
            # self.test_logits[dataloader_idx].append(logit_outputs_mean)
    
            # save the mean noise over MCD samples to an array
            noise_dropout_outputs_mean = np.mean(noise_dropout_outputs, axis=0)
            # self.test_noise[dataloader_idx].append(noise_dropout_outputs_mean)
    
            # get the average non-corrupted probabilities over the MCD samples
            SM_dropout_logits_mean = np.mean(SM_dropout_logits, axis=0)
            # save the non-corrupted probs to an array
            # self.test_non_corrupted_preds[dataloader_idx].append(SM_dropout_logits_mean)
    
            # also save the entropy of the non-corrupted distributions...
            entropy = -1.0 * np.sum(
                SM_dropout_logits_mean * np.log(SM_dropout_logits_mean + 1e-16), axis=-1
            )
            # self.test_entropy[dataloader_idx].append(entropy)
            gts.append(target)
            #list_of_sampled_probs.append(SM_dropout_logits)
            print(len(gts))
            print(gts[0].size())
            print(len(preds))
            print(preds[0].size())


        np.save(directory + '/preds_mcd_af',flatten_list_of_batches(preds))
        np.save(directory + '/gts_mcd_af',flatten_list_of_batches(gts))
        np.save(directory + '/entropy_corrupted_mcd_af',flatten_list_of_batches(test_entropies))
        np.save(directory + '/H2_mcd_af',flatten_list_of_batches(test_H2s))
        #with open('sampled_probs_mcd_af.pkl', 'wb') as file:
        #    pickle.dump(list_of_sampled_probs, file)


# In[67]:


import pickle
def flatten_list_of_batches(output_list,concat_axis =0):
    for batch_index in range(len(output_list)):
        if batch_index ==0:
            # get the parameter for all the examples in the batch
            if isinstance(output_list[batch_index], torch.Tensor):
                flattened_array = np.array(output_list[batch_index].detach().cpu())
            else:
                flattened_array = np.array(output_list[batch_index])
        else:
            # add the parameter for all examples in the batch
            if isinstance(output_list[batch_index], torch.Tensor):
                flattened_array = np.concatenate((flattened_array,output_list[batch_index].detach().cpu()),axis=concat_axis)
            else:
                flattened_array = np.concatenate((flattened_array,output_list[batch_index]),axis=concat_axis)
    return flattened_array

def load_checkpoint_and_test(checkpoint_path, test_dataloader,model):
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    #scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cuda'))
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")

    # Set model to evaluation mode
    model.eval()

    # Test the model
    test_loop(test_dataloader, model)


# In[ ]:





# In[ ]:

if calib:
    load_checkpoint_and_test(checkpoint_path,calib_dataloader,model)
else:
    load_checkpoint_and_test(checkpoint_path,test_dataloader,model)


# In[39]:


import numpy as np


# In[40]:


a = np.load(directory + 'preds_mcd_af.npy')


# In[41]:


b = np.load(directory + 'gts_mcd_af.npy')
uncerts =  np.load(directory + 'entropy_corrupted_mcd_af.npy')


# In[42]:


len(a)


# In[43]:


len(uncerts)


# In[44]:


import typing
import warnings

import numpy as np
import sklearn.metrics


def all_binary_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
):
    """Evaluate all binary classification metrics.

    Given a target and a prediction array, this function computes
    all metrics as decided for the QUMPHY common evaluation framework.

    The metrics are returned as a dictionary with the following keys:

    - `auc`: Area under the curve calculated with raw probabilities
    - `f1`: F1-score calculated with a classification threshold of 0.5
    - `mcc_sens`: Matthews correlation coefficient calculated with a threshold achieving a sensitivity of 0.8
    - `mcc_spec`: Matthews correlation coefficient calculated with a threshold achieving a specificity of 0.8
    - `sens`: Sensitivity (with a threshold achieving a sensitivity of 0.8)
    - `spec`: Specificity (with a threshold achieving a specificity of 0.8)


    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions (raw probability of positive
        class).

    Returns
    -------
    Dict[str, float]
        Dictionary with all metrics.
    """
    # Fix the threshold so that recall is 0.8
    # This is hard-coded here for the QUMPHY common evaluation framework.
    recall_value = 0.8
    threshold_sens = recall_score_threshold(
        target, prediction, recall_value=recall_value, pos_label=1
    )
    threshold_spec = recall_score_threshold(
        target, prediction, recall_value=recall_value, pos_label=0
    )

    prediction_05 = prediction > 0.5
    prediction_sens = prediction > threshold_sens
    prediction_spec = prediction > threshold_spec

    metrics_dict = {}
    # metrics_dict["acc_b"] = balanced_accuracy_score(target, prediction)
    # metrics_dict["ppv"] = precision_score(target, prediction)
    metrics_dict["auc"] = auc_score_binary(target, prediction)
    metrics_dict["f1"] = f1_score(target, prediction_05, average="binary")
    metrics_dict["mcc_sens"] = matthews_correlation_coefficient(target, prediction_sens)
    metrics_dict["mcc_spec"] = matthews_correlation_coefficient(target, prediction_spec)
    metrics_dict["sens"] = sensitivity(target, prediction_spec)
    metrics_dict["spec"] = specificity(target, prediction_sens)

    return metrics_dict


def all_regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    baseline_mae: float = None,
) -> dict[str, float]:
    """Evaluate all regression classification metrics.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    baseline_mae : float
        Baseline mean absolute error.

    Returns
    -------
    dict[str, float]
        Dictionary with all metrics.
    """
    metrics_dict = {}
    metrics_dict["mae"] = mean_absolute_error(target, prediction)
    # metrics_dict["rmse"] = root_mean_square_error(target, prediction)
    if baseline_mae is not None:
        metrics_dict["mase"] = mean_absolute_scaled_error(
            baseline_mae, metrics_dict["mae"]
        )
    metrics_dict["ieee_grades"] = ieee_grades(target, prediction)

    return metrics_dict


def auc_score_binary(
    target: np.ndarray, prediction: np.ndarray, axis: int = 0
) -> float | np.ndarray:
    """Compute the area und curve (AUC) score for binary classification.

    Parameters
    ----------
    target : np.ndarray
        Binary ground truth values for different samples.
    prediction : np.ndarray
        Binary model output predictions (raw prob.) associated
        with the positive class.
    axis : int, optional
        Axis to compute AUC over, by default 0.

    Returns
    -------
    :
        Array of AUC values.

    See Also
    --------
    multiclass_auc_score : AUC score for more then two classes.

    Examples
    --------
    >>> target = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    >>> prediction = np.array([.99, .8, .6, .63, .77, .23, .3, .78, .2, 0.01])
    >>> auc_score_binary(target, prediction)
    xx

    >>> target = np.random.randint(0, 2, (100, 2))
    >>> auc_score_binary(target, target, axis=0)
    (1.0, 1.0)

    >>> target = np.random.randint(0, 2, (50, 100, 2, 5))
    >>> auc_score_binary(target, target, axis=1).shape
    (50, 2, 5)
    """
    assert 0 <= axis < target.ndim
    assert np.all([t == p for t, p in zip(target.shape, prediction.shape)])

    if target.ndim == 1:
        return sklearn.metrics.roc_auc_score(target, prediction)

    shape = target.shape
    new_shape = tuple([s for j, s in enumerate(shape) if j != axis])
    target_reshape = np.reshape(np.moveaxis(target, axis, 0), (shape[axis], -1))
    prediction_reshape = np.reshape(np.moveaxis(prediction, axis, 0), (shape[axis], -1))
    auc = np.zeros(target_reshape.shape[-1])
    for j, (t, p) in enumerate(zip(target_reshape.T, prediction_reshape.T)):
        auc[j] = sklearn.metrics.roc_auc_score(t, p)

    return np.reshape(auc, new_shape)


def auc_score_multiclass(
    target: np.ndarray,
    prediction: np.ndarray,
    comparison_type: str = "ovr",
) -> float | np.ndarray:
    """Compute the area und curve (AUC) score.

    Parameters
    ----------
    target : np.ndarray
        Multiclass ground truth values.
    prediction : np.ndarray
        Array of model output probabilities of different classes for different samples.
        If `target` shape is ``(n_samples, ...)`` with ``n_classes`` different class
        values, then `prediction` needs to have shape ``(n_samples, n_classes, ...)``.
        Axis 1 (``n_classes``) needs to sum to one.
    comparison_type : str
        Comparison type for multiclasses, by default \"ovr\".

        ``ovr`` : Stands for one-vs-rest. Computes the AUC for each class against
        the rest of the classes.

        ``ovo`` : Stands for one-vs-one. Computes the average AUC of all possible
        pairwise combinations of classes.

    Returns
    -------
    :
        Array of AUC values.

    See Also
    --------
    auc_score_binary : AUC score for exactly two classes.

    Examples
    --------
    >>> target = np.array([0, 1, 2, 1, 2, 0])
    >>> prediction = np.array([[0.8, 0.1, 0.1],
    >>>                        [0.2, 0.5, 0.3],
    >>>                        [0.8, 0.1, 0.1],
    >>>                        [0.7, 0.2, 0.1],
    >>>                        [0.4, 0.3, 0.3],
    >>>                        [0.5, 0.4, 0.1]])
    >>> auc_score_multiclass(target, prediction, comparison_type="ovo")
    0.6875

    >>> target = np.random.randint(0, 3, (100, 10, 5, 2))
    >>> prediction = np.random.uniform(0, 1, (100, 3, 10, 5, 2))
    >>> prediction /= np.expand_dims(np.sum(prediction, axis=1), 1)
    >>> auc_score_multiclass(target, prediction).shape
    (10, 5, 2)
    """
    assert prediction.ndim == target.ndim + 1
    assert target.shape[0] == prediction.shape[0]
    assert np.all(t == p for t, p in zip(target.shape[1:], prediction.shape[2:]))
    assert (  # second axis is prediction probabilities for classes
        np.unique(target).size == prediction.shape[1]
    )
    assert (  # second axis needs to sum to one
        np.all(prediction >= 0.0)
        and np.all(prediction <= 1.0)
        and np.linalg.norm(np.sum(prediction, axis=1) - 1.0) <= 1e-12
    )
    assert comparison_type in ["ovr", "ovo"]

    if target.ndim == 1:
        return sklearn.metrics.roc_auc_score(
            target, prediction, multi_class=comparison_type
        )

    shape = prediction.shape
    target_reshape = np.reshape(target, (shape[0], -1))
    prediction_reshape = np.reshape(prediction, (shape[0], shape[1], -1))
    auc = np.zeros(target_reshape.shape[-1])
    for j, (t, p) in enumerate(
        zip(target_reshape.T, np.moveaxis(prediction_reshape, -1, 0))
    ):
        auc[j] = sklearn.metrics.roc_auc_score(t, p, multi_class=comparison_type)

    return np.reshape(auc, shape[2:])


def balanced_accuracy_score(
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Compute balanced accuracy score for binary or multi-class classification.

    For the binary case the balanced accuracy score :math:`\\operatorname{Acc}_b` is
    given by the arithmetic mean of sensitivity (Se) and specificity (Sp), i.e.

    .. math::
        \\operatorname{Acc}_b
        = \\frac{1}{2}(\\operatorname{Se} + \\operatorname{Sp})
        = \\frac{1}{2}\\Bigl( \\frac{\\operatorname{TP}}{\\operatorname{TP}+\\operatorname{FN}} + \\frac{\\operatorname{TN}}{\\operatorname{TN}+\\operatorname{FP}} \\Bigr),

    expressed by true positives (TP), true negatives (TN), false positives (FP) and
    false negatives (FN). In general, balanced accuracy is computed by

    .. math::
        \\operatorname{Acc}_b(y_{\\mathrm{true}}, y_{\\mathrm{pred}})
        = \\frac{\\sum_{i=1}^N w_i \\, \\delta(y_{\\mathrm{true},i} = y_{\\mathrm{pred}, i})}{\\sum_{i=1}^N w_i}

    with weights

    .. math::
        w_i = \\frac{1}{\\sum_{j=1}^N \\delta(y_{\\mathrm{true},i} = y_{\\mathrm{true},j})},

    where :math:`\\delta(y_i = y_j)` denotes the Kronecker delta function.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Balanced accuracy score for binary or multi-class classification.

    Examples
    --------
    Computation of balanced accuracy score for binary classification, i.e.,
    a one dimensional array with only 2 classes

    >>> target = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
    >>> prediction = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    >>> balanced_accuracy_score(target, prediction)
    0.6

    Computation of balanced accuracy score for a multi-class scenario, i.e.,
    a 1D array with more then two classes

    >>> target = np.array([1, 2, 2, 2, 1, 2, 1, 0, 1, 1])
    >>> prediction = np.array([1, 1, 2, 0, 0, 1, 1, 0, 0, 2])
    >>> balanced_accuracy_score(target, prediction)
    0.55
    """
    return sklearn.metrics.balanced_accuracy_score(target, prediction)


def f1_score(
    target: np.ndarray,
    prediction: np.ndarray,
    average: str | None = None,
) -> float | np.ndarray:
    """Compute F1-score of binary, multi-class or multi-label classification.

    The :math:`F_1` score is computed using the true positives (TP), false
    positives (FP) and false negatives (FN) via

    .. math::
        F_1 = \\frac{2\\operatorname{TP}}{2\\operatorname{TP} + \\operatorname{FP} + \\operatorname{FN}}.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Predicted values.
    average : str | None, optional
        Averaging of the F1-scores (default None).
        For binary classification, ``average=binary`` is the default case.
        For multi-class and multi-lable classification, ``average=None`` is the
        default case, which results in F1-scores for each individual class.
        For more detail about averaging see the documentation of
        `sklearn.metrics.f1_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_.

    Returns
    -------
    float | np.ndarray
        F1 score(s) for binary, multi-class or multi-lable classification.

    Examples
    --------
    Computation of F1-score for binary classification, i.e.,
    a one dimensional array with only 2 classes

    >>> target = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    >>> prediction = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    >>> f1_score(target, prediction)
    0.5

    Computation of F1-score for a multi-class scenario, i.e., a 1D array with
    more then two classes

    >>> target = np.array([1, 2, 2, 2, 1, 2, 1, 0, 1, 1])
    >>> prediction = np.array([1, 1, 2, 0, 0, 1, 1, 0, 0, 2])
    >>> f1_score(target, prediction)
    array([0.4, 0.44444444, 0.33333333])

    Computation of F1-score for a multi-lable scenario, i.e., a 2D array with
    with columns representing different labels and values 0 or 1 as entries

    >>> target = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    >>> prediction = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
    >>> f1_score(target, prediction)
    array([0.66666667, 0.4, 0.4])

    Computation of F1-score for a multi-class scenario with averaging of F1-scores
    over the different classes

    >>> target = np.array([1, 2, 2, 2, 1, 2, 1, 0, 1, 1])
    >>> prediction = np.array([1, 1, 2, 0, 0, 1, 1, 0, 0, 2])
    >>> f1_score(target, prediction, average='micro')
    0.4
    """
    assert target.shape == prediction.shape
    target = np.squeeze(target)
    prediction = np.squeeze(prediction)

    if np.ndim(target) > 1:
        # multi-lable case
        # NOTE: target and prediction are matrices with only two different entries
        assert np.unique(target).size == np.unique(prediction).size == 2
        avg = average
    elif np.unique(target).size == np.unique(prediction).size == 2:
        # binary case
        avg = "binary"
    else:
        # multi-class case
        avg = average
    return sklearn.metrics.f1_score(target, prediction, average=avg)


def false_discovery_rate(
    target: np.ndarray,
    prediction: np.ndarray,
    average: str | None = None,
) -> float | np.ndarray:
    """Compute false discovery rate (FDR).

    The false discovery rate (FDR) is given by
    :math:`\\operatorname{FDR} = 1 - \\operatorname{PPV}`, where
    :math:`PPV` is the precision (positive predicted value).

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Predicted values.
    average : str | None, optional
        Averaging type (default None).
        For more detail about averaging see the documentation of
        `sklearn.metrics.precision_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_.

    Returns
    -------
    float | np.ndarray
        False discovery rate for binary, multi-class or multi-lable classification.

    See Also
    --------
    precision_score, f1_score
    """
    ppv = precision_score(target, prediction, average)
    return np.ones(ppv.size) - ppv if isinstance(ppv, np.ndarray) else 1 - ppv


def general_threshold(
    target: np.ndarray,
    prediction: np.ndarray,
    metric: typing.Callable[[np.ndarray, np.ndarray], float],
    metric_value: float,
    greater_than: bool = True,
) -> float:
    """
    Find the threshold that sets the given metric closest to `metric_value`.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    metric : Callable[[np.ndarray, np.ndarray], float]
        A metric function that takes target and prediction arrays as input.
    metric_value : float
        The value of the metric to be achieved.
    greater_than : bool
        True if the metric is supposed to be higher than `metric_value`,
        false otherwise.

    Returns
    -------
    float
        The threshold that achieves the metric.
    """
    thresholds = np.unique(prediction)
    if metric(target, prediction > thresholds[-1]) > metric(
        target, prediction > thresholds[-2]
    ):
        if greater_than:
            thresholds = thresholds[::-2]
    else:
        if not (greater_than):
            thresholds = thresholds[::-2]
    for threshold in thresholds:
        prediction_binary = prediction > threshold
        res = metric(target, prediction_binary)
        if greater_than and res > metric_value:
            return threshold
        elif not (greater_than) and res < metric_value:
            return threshold
    warnings.warn("Could not find threshold achieving target metric_value.")
    return -1.0


def ieee_grades(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute the IEEE grades of the predicted values.

    The grades are calculated by comparing the difference between the target
    and the prediction. Returned are the percentage of samples that fall within
    each grade. The grading scores follow the IEEE Std 1708a™-2019 scheme, where
    instead of a mean absolute difference of two measurements with the standard
    device, we use only one measurement.

    The grading for each sample is determined as follows
     - Grade A for error ≤5 mmHg
     - Grade B for error between 5-6 mmHg
     - Grade C for error between 6-7 mmHg
     - Grade D for error ≥7 mmHg

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.

    Returns
    -------
    np.ndarray
    """
    difference = np.abs(prediction - target)
    graded_preds = np.where(
        difference <= 5,
        "A",
        np.where(
            difference <= 6,
            "B",
            np.where(
                difference <= 7,
                "C",
                "D",
            ),
        ),
    )

    grades_dict = {}
    for grade in ["A", "B", "C", "D"]:
        grades_dict[grade] = np.count_nonzero(graded_preds == grade) / len(graded_preds)
    return grades_dict


def l1_norm(array: np.ndarray, axis: int = 0) -> float | np.ndarray:
    """Compute the :math:`L^1`-norm of an array along an axis.

    The :math:`L^2`-norm of an array :math:`x\\in\\mathbb{R}^N` is given by
    :math:`\\Vert x \\Vert_{L^1} = \\frac{1}{N}\\sum_{j=1}^N \\vert x_j \\vert`.

    Parameters
    ----------
    array : np.ndarray
        Data array.
    axis : int, optional
        Axis, by default 0.

    Returns
    -------
    :
        Array of :math:`L^1`-norms over the specified axes.

    See Also
    --------
    mean_absolute_error : Wrapper for ``l1_norm(target - prediction)``.
    l2_norm, root_mean_square_error

    Examples
    --------
    >>> l1_norm(np.array([1, 2, 3, 4]))
    10

    >>> array = np.random.normal(0, 1, (10, 5, 3, 2))
    >>> l1_norm(array, axis=1).shape  # norm over second axis
    (10, 3, 1)
    """
    assert isinstance(array, np.ndarray) and array.ndim > 0
    norm = np.sum(np.abs(array) / array.shape[axis], axis=axis)
    if norm.size == 1:
        return norm.flatten()[0]
    return norm


def l2_norm(array: np.ndarray, axis: int = 0) -> float | np.ndarray:
    """Compute the :math:`L^2`-norm along an axis.

    The :math:`L^2`-norm of an array :math:`x\\in\\mathbb{R}^N` is given by
    :math:`\\Vert x \\Vert_{L^2} = \\sqrt{\\frac{1}{N}\\sum_{j=1}^N x_j^2 }`.

    Parameters
    ----------
    array : np.ndarray
        Data array.
    axis : int, optional
        Axis, by default 0.

    Returns
    -------
    :
        Array of :math:`L^2`-norms over the specified axes.

    See Also
    --------
    root_mean_square_error : Wrapper for ``l2_norm(target - prediction)``.
    l1_norm, mean_absolute_error

    Examples
    --------
    >>> l2_norm(np.array([1, 2, 3, 4]))**2
    30.0

    >>> array = np.random.normal(0, 1, (10, 5, 3, 2))
    >>> l2_norm(array, axis=1).shape  # norm over second axis
    (10, 3, 1)
    """
    assert isinstance(array, np.ndarray) and array.ndim > 0
    norm = np.sqrt(np.sum(array**2, axis=axis) / array.shape[axis])
    if norm.size == 1:
        return norm.flatten()[0]
    return norm


def matthews_correlation_coefficient(
    target: np.ndarray, prediction: np.ndarray
) -> float:
    """Compute Matthews correlation coefficient (Mcc) of binary or multi-class task.

    The Matthews correlation coefficient (Mcc) is computed using the true
    positives (TP), false positives (FP), true negatives (TN) and false negatives (FN)
    via

    .. math::
        \\operatorname{Mcc}
        = \\frac{\\operatorname{TP}\\cdot\\operatorname{TN} - \\operatorname{FP}\\cdot\\operatorname{FN}}{\\sqrt{(\\operatorname{TP}+\\operatorname{FP})(\\operatorname{TP}+\\operatorname{FN})(\\operatorname{TN}+\\operatorname{FP})(\\operatorname{TN}+\\operatorname{FN})}}.

    For the multi-class case, let :math:`C` be the confusion matrix for :math:`K`
    classes and define
    the number of times class :math:`k` truly occurs :math:`t_k = \\sum_{i=1}^K C_{ik}`,
    the number of times class :math:`k` was predicted :math:`p_k = \\sum_{i=1}^K C_{ki}`,
    the total number of samples correctly predicted :math:`c = \\sum_{k=1}^K C_{kk}` and
    the total number of samples :math:`s = \\sum_{i,j=1}^K C_{ij}`.
    Then the multiclass Mcc is defined as

    .. math::
        \\operatorname{Mcc}
        = \\frac{c \\cdot s -\\sum_{k=1}^K p_k \\cdot t_k}{\\sqrt{ (s^2 - \\sum_{k=1}^K p_k^2)(s^2 - \\sum_{k=1}^K t_k^2)}}.

    .. note::
        When there are more than two labels, the value of the MCC will no longer range
        between -1 and +1. Instead the minimum value will be somewhere between -1 and 0
        depending on the number and distribution of ground true labels. The maximum
        value is always +1.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Predicted values.

    Returns
    -------
    float | np.ndarray
        Mcc for binary and multi-class classification.

    Examples
    --------
    Computation of Mcc for binary classification, i.e.,
    a one dimensional array with only 2 classes

    >>> target = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    >>> prediction = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    >>> matthews_correlation_coefficient(target, prediction)
    -0.10206207261596577

    Computation of Mcc for a multi-class scenario, i.e., a 1D array with
    more then two classes

    >>> target = np.array([1, 2, 2, 2, 1, 2, 1, 0, 1, 1])
    >>> prediction = np.array([1, 1, 2, 0, 0, 1, 1, 0, 0, 2])
    >>> matthews_correlation_coefficient(target, prediction)
    0.13130643285972254
    """
    return sklearn.metrics.matthews_corrcoef(target, prediction)


def mean_absolute_error(
    target: np.ndarray, prediction: np.ndarray, axis: int = 0
) -> float | np.ndarray:
    """Compute the mean absolute error (MAE) between target and prediction.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    axis : int, optional
        Axis to sum over, by default 0.

    Returns
    -------
    :
        Array of MAE values (:math:`L^1`-norms) over the specified axes.

    See Also
    --------
    l1_norm : This is a wrapper for ``l1_norm(target - prediction, axis=axis)``.
    l2_norm, root_mean_square_error

    Examples
    --------
    >>> mean_absolute_error(np.array([1, 2, 3]), np.array([1, 2, 3]))
    0.0

    >>> target = np.random.normal(0, 1, (10, 5, 3, 2))
    >>> prediction = np.random.normal(0, 1, (10, 5, 3, 2))
    >>> mean_absolute_error(target, prediction, axis=1).shape  # norm over second axis
    (10, 3, 1)
    """
    return l1_norm(target - prediction, axis=axis)


def mean_absolute_scaled_error(baseline_mae: float, model_mae: float) -> float:
    """Compute mean absolute scaled error (MASE).

    The MASE is a measure of the magnitude of the error relative to a baseline
    error. It is defined as the mean absolute error divided by the baseline
    error.

    Parameters
    ----------
    baseline_mae : float
        Mean absolute scaled error of the baseline.
    model_mae : float
        Mean absolute error.

    Returns
    -------
    float
        Mean absolute scaled error.
    """
    return model_mae / baseline_mae


def precision_score(
    target: np.ndarray,
    prediction: np.ndarray,
    average: str | None = None,
) -> float | np.ndarray:
    """Compute precision (PPV) of binary, multi-class or multi-label classification.

    The precision score (positive predictive value, PPV) is computed using the
    true positives (TP) and false positives (FP) via

    .. math::
        \\operatorname{PPV}
        = \\frac{\\operatorname{TP}}{\\operatorname{TP} + \\operatorname{FP}}.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Predicted values.
    average : str | None, optional
        Averaging type for score (default None).
        For more detail about averaging see the documentation of
        `sklearn.metrics.precision_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_.

    Returns
    -------
    float | np.ndarray
        Precision scores for binary, multi-class or multi-lable classification.

    See Also
    --------
    f1_score, false_discovery_rate

    Examples
    --------
    Computation of precision score for binary classification, i.e.,
    a one dimensional array with only 2 classes

    >>> target = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    >>> prediction = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    >>> precision_score(target, prediction)
    0.375

    Computation of precision score for a multi-class scenario, i.e., a 1D array with
    more then two classes

    >>> target = np.array([1, 2, 2, 2, 1, 2, 1, 0, 1, 1])
    >>> prediction = np.array([1, 1, 2, 0, 0, 1, 1, 0, 0, 2])
    >>> precision_score(target, prediction)
    array([0.25, 0.5, 0.5])

    Computation of precision score for a multi-lable scenario, i.e., a 2D array with
    with columns representing different labels and values 0 or 1 as entries

    >>> target = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    >>> prediction = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
    >>> precision_score(target, prediction)
    array([0.33333333, 0.5, 0.66666667])

    Computation of precision score for a multi-class scenario with averaging of F1-scores
    over the different classes

    >>> target = np.array([1, 2, 2, 2, 1, 2, 1, 0, 1, 1])
    >>> prediction = np.array([1, 1, 2, 0, 0, 1, 1, 0, 0, 2])
    >>> precision_score(target, prediction, average='micro')
    0.4
    """
    assert target.shape == prediction.shape
    target = np.squeeze(target)
    prediction = np.squeeze(prediction)

    if np.ndim(target) > 1:
        # multi-lable case
        # NOTE: target and prediction are matrices with only two different entries
        assert np.unique(target).size == np.unique(prediction).size == 2
        avg = average
    elif np.unique(target).size == np.unique(prediction).size == 2:
        # binary case
        avg = "binary"
    else:
        # multi-class case
        avg = average
    return sklearn.metrics.precision_score(target, prediction, average=avg)


def recall_score_threshold(
    target: np.ndarray,
    prediction: np.ndarray,
    recall_value: float,
    pos_label: 1 | 0 = 1,
    greater_than: bool = True,
    dtype: np.dtype = np.float32,
) -> float:
    """
    Compute the classification threshold so that the recall score is
    closest to the specified value, but greater (or lower).

    Default: The threshold is computed for the sensitivity score.

    The threshold is set as the next floating point number after
    (before) the value of the prediction that needs to be classified
    positive (negative) to achieve the desired recall score.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    recall_value : float
        The desired recall score.
    pos_label : 1 | 0, optional
        1 to compute the threshold for sensitivity, 0 for
        specificity.
    greater_than : bool, optional
        True to let the recall score be higher than recall_value, false
        to let the recall score be lower than recall_value.
    dtype : np.dtype, optional
        The data type of the threshold, by default np.float32

    Returns
    -------
    float
        The classification threshold.
    """

    if np.any(prediction < 0.0) or np.any(prediction > 1.0):
        raise ValueError("All values in prediction must be between 0 and 1.")

    if pos_label:
        label_prediction = np.sort(prediction[target == pos_label])[::-1]
    else:
        label_prediction = np.sort(prediction[target == pos_label])

    if label_prediction.size == 0:
        warnings.warn(
            "Recall is ill-defined and the threshold is set to 0 due to no true samples."
        )
        return 0

    true_label = recall_value * len(label_prediction)
    threshold_index = max(0, np.ceil(true_label - 1).astype(int))

    # set the threshold slightly over/under the prediction value
    # to avoid >= or > issues while computing binary predictions
    # np.mod(x+y,2) is the same as xor(x,y) for two binary values
    threshold = np.nextafter(
        dtype(label_prediction[threshold_index]),
        np.mod(pos_label + greater_than, 2, dtype=dtype),
    )
    return threshold


def root_mean_square_error(
    target: np.ndarray, prediction: np.ndarray, axis: int = 0
) -> float | np.ndarray:
    """Compute the root mean square error (RMSE) between target and prediction.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    axis : int, optional
        Axis to sum over, by default 0.

    Returns
    -------
    :
        Array of RMSE values (:math:`L^2`-norms) over the specified axes.

    See Also
    --------
    l2_norm : This is a wrapper for ``l2_norm(target - prediction, axis=axis)``.
    l1_norm, mean_absolute_error

    Examples
    --------
    >>> root_mean_square_error(np.array([1, 2, 3]), np.array([1, 2, 3]))
    0.0

    >>> target = np.random.normal(0, 1, (10, 5, 3, 2))
    >>> prediction = np.random.normal(0, 1, (10, 5, 3, 2))
    >>> root_mean_square_error(target, prediction, axis=1).shape  # norm over second axis
    (10, 3, 1)
    """
    return l2_norm(target - prediction, axis=axis)


def specificity(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Compute the specificity (or true negative rate). Specificity is
    also known as the recall score of the negative class.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.

    Returns
    -------
    float
        Specificity score.

    See Also
    --------
    sensitivity
    """
    return sklearn.metrics.recall_score(target, prediction, pos_label=0)


def sensitivity(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Compute the sensitivity (or true positive rate). Sensitivity is
    also known as the recall score of the positive class.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.

    Returns
    -------
    float
        Sensitivity score.

    See Also
    --------
    specificity
    """
    return sklearn.metrics.recall_score(target, prediction, pos_label=1)


# In[45]:


all_binary_metrics(b,a[:,1])


# In[46]:


def uncertainty_calibration_error(predictions, target_labels, prediction_entropy_values, M = 10):
    # Define the bin intervals 
    bin_edges = np.linspace(0, 1, M + 1)
    bin_lower_bounds = bin_edges[:-1] 
    bin_upper_bounds = bin_edges[1:]

    # Predicted class label from the predictions array
    predicted_labels = np.argmax(predictions, axis = 1)

    inaccuracies = (predicted_labels != target_labels)
    print(np.shape(inaccuracies))

    UCE = 0
    bin_entropy_values = []
    bin_inaccuracies = []

    for lower, upper in zip(bin_lower_bounds, bin_upper_bounds):
        in_bin = np.logical_and(prediction_entropy_values > lower.item(), prediction_entropy_values <= upper.item())

        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:

            err_in_bin = inaccuracies[in_bin].mean()
            uncert_in_bin = prediction_entropy_values[in_bin].mean()

            UCE += (prob_in_bin) * np.abs(err_in_bin - (uncert_in_bin / 2) ) 

            bin_inaccuracies.append(err_in_bin)
            bin_entropy_values.append(uncert_in_bin)

    return UCE, bin_entropy_values, bin_inaccuracies


# In[47]:


np.shape(a)


# In[48]:


np.shape(b)


# In[49]:


uncerts_bits = uncerts*1.4427


# In[50]:


data = uncertainty_calibration_error(a,b,uncerts_bits)


# In[51]:


import matplotlib.pyplot as plt
plt.plot(data[1],data[2])


# In[52]:


np.max(uncerts*1.4427)


# In[53]:


from scipy.stats import entropy
import numpy as np

b = np.load(directory + 'preds_mcd_af.npy',allow_pickle=True)


# In[54]:



a = np.load(directory + 'entropy_corrupted_mcd_af.npy',allow_pickle=True)


# In[55]:


c = np.load(directory + 'gts_mcd_af.npy',allow_pickle=True)


# In[56]:


# compute differences between predicted and ground truth classes.
# i.e. create an array describing whether each example was predicted correctly or not
differences_array = []
for i in range(len(b)):
    diff = c[i] - np.round(np.argmax(b[i]))
    differences_array.append((np.abs(diff)-1)*-1) #1 is correct, 0 is incorrect


# In[57]:


uncerts = []
for pred in b:
    ent =np.log2(np.exp(1))*(- 1.0 * np.sum(pred * np.log(pred + 1e-16), axis=-1))
    uncerts.append(ent)
    


# In[58]:


import matplotlib.pyplot as plt
def convert_logged_to_np(tag):
    data = event_acc.scalars.Items(tag)
    # Convert the data to a numpy array
    data_np = np.array([(d.step, d.value) for d in data])
    data_np = np.asarray(data_np)
    return data_np

def flatten_list_of_batches(output_list,concat_axis = -1):
    for batch_index in range(len(output_list[-1])):
        if batch_index ==0:
            # get the parameter for all the examples in the batch
            flattened_array = np.array(output_list[-1][batch_index][:])
        else:
            # add the parameter for all examples in the batch to the all_test_entropies array
            flattened_array = np.concatenate((flattened_array,output_list[-1][batch_index][:]),axis=concat_axis)
    return flattened_array

# Calib curve
def plot_calib_curve(pred,differences,UCE=0,dr = 0, title_mod = None,top_left_text=None):

    y_true = np.copy(differences)
    y_pred = np.copy(pred)
    
    # Sort the preds and corresponding difference array
    sorted_indices = np.argsort(y_pred)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]
    
    # Divide the sorted predictions into bins
    n_bins = 10
    bin_edges_values = np.linspace(min(y_pred), max(y_pred), n_bins + 1)
    bin_edges = sorted_y_pred.searchsorted(bin_edges_values)
    
    # Compute the mean prediction and fraction of correct or incorrect predictions in each bin
    prob_true = []
    prob_pred = []
    n_samples = []
    for i in range(n_bins):
        if i == n_bins-1: #gather remaining examples for last bin
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            if UCE == 1:
                prob_true.append(np.mean(1-sorted_y_true[bin_start:])) # average number of incorrect outputs in the bin
            else:
                prob_true.append(np.mean(sorted_y_true[bin_start:])) # average number of correct outputs in the bin
            prob_pred.append(np.mean(sorted_y_pred[bin_start:])) # average value in the bin
            n_samples.append(len(sorted_y_true[bin_start:]))
        else:
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            if UCE == 1:
                prob_true.append(np.mean(1-sorted_y_true[bin_start:bin_end])) # average number of incorrect outputs in the bin
            else:
                prob_true.append(np.mean(sorted_y_true[bin_start:bin_end])) # average number of correct outputs in the bin
            prob_pred.append(np.mean(sorted_y_pred[bin_start:bin_end])) # average value in the bin
            n_samples.append(len(sorted_y_true[bin_start:bin_end]))

    # Number of samples in each bin
    samples_per_bin = np.histogram(y_pred, bins=n_bins)[0]
    if UCE == 1:
        calib_metric = np.sum((n_samples/(np.sum(n_samples)))*np.abs((np.asarray(prob_true) - np.asarray(prob_pred)/2)))
        calib_metric_unweighted = np.sum(np.abs((np.asarray(prob_true) - np.asarray(prob_pred)/2)))
    else:
        calib_metric = np.sum((n_samples/(np.sum(n_samples)))*np.abs((np.asarray(prob_true) - np.asarray(prob_pred))))
        calib_metric_unweighted = np.sum(np.abs((np.asarray(prob_true) - np.asarray(prob_pred))))

    # Plot the calibration curve
    fig = plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o')
    
    if UCE == 1:
        if title_mod is not None:
            plt.title('Uncertainty Calibration Curve' + title_mod, fontsize=20)
        else:
            plt.title('Uncertainty Calibration Curve dropout=' + str(dr) + '%', fontsize=20)
        plt.xlabel('Mean entropy in bin',fontsize=20)
        plt.ylabel('Fraction of incorrect outputs in bin',fontsize=20)
        plt.plot([1, 0], [.5, 0], linestyle='--', color='gray', label='Perfectly calibrated')
    else:
        if title_mod is not None:
            plt.title('Calibration Curve dropout=' + str(dr) + '%' + title_mod, fontsize=20)
        else:
            plt.title('Calibration Curve dropout=' + str(dr) + '%', fontsize=20)
        plt.xlabel('Mean predicted probability in bin',fontsize=20)
        plt.ylabel('Fraction of correct outputs in bin',fontsize=20)
        plt.plot([1, 0.5], [1, 0.5], linestyle='--', color='gray', label='Perfect Calibration')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()


    # Add top left text if provided
    if top_left_text is not None:
        plt.text(-.1,1.15, top_left_text, transform=plt.gca().transAxes, fontsize=30, verticalalignment='top')

    
    plt.legend(fontsize=15)
    plt.show()

    if UCE == 1:
        fig.savefig(('./uce_dr' + str(dr) + str(title_mod[2:]).replace(" ", "")).replace(" ", "") + '.png')
    else:
        fig.savefig('./ece_dr' + str(dr) + str(title_mod[2:]).replace(" ", "") + '.png')
    return [calib_metric, calib_metric_unweighted, fig]

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # create scatter plot with histograms
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    # Example DataFrame (replace with your data)
    df = pd.DataFrame({
        'z': GTs,
    })
    ax.scatter(x, y,marker = 'o',c=df.z, cmap = 'rainbow_r')#,cmap='Set1'
    binwidth = 0.1
    lim = 1
    bins = np.arange(0, lim + binwidth, binwidth)
    x_af = x[GTs ==1]
    x_naf = x[GTs ==0]
    y_af = y[GTs ==1]
    y_naf = y[GTs ==0]
    ax_histx.hist((x_af,x_naf),color=('#8f3fff','#FF5733'), bins=bins)
    ax_histy.hist((y_naf,y_af), color=('#FF5733','#8f3fff'),bins=bins, orientation='horizontal')
    ax.ylim = [0,1]


# In[59]:


# compute differences between predicted and ground truth classes.
# i.e. create an array describing whether each example was predicted correctly or not
differences_array = []
for i in range(len(b)):
    diff = c[i] - np.round(np.argmax(b[i]))
    differences_array.append((np.abs(diff)-1)*-1) #1 is correct, 0 is incorrect


# In[60]:


plot_calib_curve(uncerts,np.asarray(differences_array), 1, 0, '\n all test examples', top_left_text = 'd)')


# In[ ]:




