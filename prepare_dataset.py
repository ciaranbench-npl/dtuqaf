# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.
#load in data

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

from argparser import get_args_BP, get_args_AF
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

def create_dataloaders():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader

def create_dataloaders_aug():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_train[:num_noisy].shape) * noise_std
    signals_train[:num_noisy] = signals_train[:num_noisy] + noise
    #signals_train[int(num_noisy*.99):num_noisy, :, :36] = 0
    
    # Remaining samples
    zero_fraction = zero_frac
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(signals_test[:num_noisy].shape) * noise_std

    #noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy] = signals_test[:num_noisy] + noise
    
    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining


    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader



def create_dataloaders_aug_leda_noise():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    #noise = np.random.randn(*signals_train[:num_noisy].shape) * noise_std

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(signals_train[:num_noisy].shape) * noise_std
    
    signals_train[:num_noisy] = signals_train[:num_noisy] + noise
    signals_train[:num_noisy, :, :36] = 0
    
    # Remaining samples
    zero_fraction = zero_frac
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    #remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(signals_test[:num_noisy].shape) * noise_std

    #noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy] = signals_test[:num_noisy] + noise
    
    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining


    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader

def create_dataloaders_aug_noise_only():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_train[:num_noisy].shape) * noise_std
    signals_train[:num_noisy] = signals_train[:num_noisy] + noise

    
    # Remaining samples
    zero_fraction = zero_frac
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    #remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy] = signals_test[:num_noisy] + noise

    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    #remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining


    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def create_dataloaders_aug_zero_only():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    #noise = np.random.randn(*signals_train[:num_noisy].shape) * noise_std
    #signals_train[:num_noisy] = signals_train[:num_noisy] + noise

    zero_fraction = zero_frac
    # Remaining samples
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(zero_fraction * num_samples)
    #the * unpacks the domensions as arguments
    #noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy, :, :36] = 0

    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining


    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def create_dataloaders_aug_zero_all_test():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_train[:num_noisy].shape) * noise_std
    signals_train[:num_noisy] = signals_train[:num_noisy] + noise

    
    # Remaining samples
    zero_fraction = zero_frac
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(signals_test[:num_noisy].shape) * noise_std
    #noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy] = signals_test[:num_noisy] + noise
    signals_test[:num_noisy, :, :36]=0
    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    #remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining
    

    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def create_dataloaders_AF():
    args = get_args_AF()
    data_dir = args.data_dir
    batch_size = args.batch_size


    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[df['set_revised'] == 0].index
    indices_train_labels = df[df['set_revised'] == 0].label.values
    
    indices_val = df[df['set_revised'] == 1].index
    indices_val_labels = df[df['set_revised'] == 1].label.values
    
    indices_cal = df[df['set_revised'] == 2].index
    indices_cal_labels = df[df['set_revised'] == 2].label.values

    
    indices_test = df[df['set_revised'] == 3].index
    indices_test_labels = df[df['set_revised'] == 3].label.values
    
    print('METADATA sorted')
    
    
    class_sample_count = np.array(
        [len(np.where(indices_train_labels == t)[0]) for t in np.unique(indices_train_labels)])
    weight = 1. / class_sample_count
    
    samples_weight = np.array([weight[t] for t in indices_train_labels])
    
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    class_sample_count = np.array(
        [len(np.where(indices_train_labels == t)[0]) for t in np.unique(indices_train_labels)])
    weight = 1. / class_sample_count
    
    samples_weight = np.array([weight[t] for t in indices_train_labels])
    
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # Extract signals corresponding to indices_train
    signal_min  = np.min(signals)
    signal_max = np.max(signals) - signal_min
    
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    signals_cal = signals[indices_cal]
    print('loading complete')
    
    
    
    def binarise_labels(labels_arr):
        labels = []
        for i in labels_arr:
            #print(i)
            if i ==1:
                labels.append(1)
                #print('yepp')
            else:
                labels.append(0)
        return labels
    
    
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.LongTensor(targets)
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signal_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = binarise_labels(indices_train_labels)
    labels_val = binarise_labels(indices_val_labels) 
    labels_test = binarise_labels(indices_test_labels) 
    labels_cal = binarise_labels(indices_cal_labels) 
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    signals_cal = np.expand_dims(signals_cal,1)

    
    
    # Create the dataset
    custom_dataset = CustomDataset(signals_train,labels_train)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_cal = CustomDataset(signals_cal,labels_cal)

    
    print('datset created')
    
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,sampler = sampler)
    #train_dataloader = DataLoader(custom_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    calib_dataloader = DataLoader(custom_dataset_cal, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, calib_dataloader


def create_dataloaders_aug_noise_all():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_train.shape) * noise_std
    signals_train = signals_train + noise
    #signals_train[int(num_noisy*.99):num_noisy, :, :36] = 0
    
    # Remaining samples
    zero_fraction = zero_frac
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy] = signals_test[:num_noisy] + noise
    
    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining


    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader

def create_dataloaders_aug_non_LEDA():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    #noise = np.random.randn(*signals_train[:num_noisy].shape) * noise_std
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(signals_train[:num_noisy].shape) * noise_std
    signals_train[:num_noisy] = signals_train[:num_noisy] + noise
    #signals_train[int(num_noisy*.99):num_noisy, :, :36] = 0
    
    # Remaining samples
    zero_fraction = zero_frac
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    #remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy] = signals_test[:num_noisy] + noise
    
    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining


    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader

def create_dataloaders_aug_non_spurious():
    args = get_args_BP()
    data_dir = args.data_dir
    batch_size = args.batch_size
    zero_frac  = args.zero
    noise_frac  = args.noise
    # Load the CSV file
    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[(df['set_calib'] == 0) & (df['source'] == 1)].index
    indices_train_labels = df[(df['set_calib'] == 0) & (df['source'] == 1)].label.values
    
    indices_val = df[(df['set_calib'] == 1) & (df['source'] == 1)].index
    indices_val_labels = df[(df['set_calib'] == 1) & (df['source'] == 1)].label.values
    
    indices_test = df[(df['set_calib'] == 3) & (df['source'] == 1)].index
    indices_test_labels = df[(df['set_calib'] == 3) & (df['source'] == 1)].label.values
    
    
    print('INDICES PARSED')
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # compute min and max for normalisation
    signal_min  = np.min(signals)
    signals_max = np.max(signals) - signal_min
    # parse the train val and test sets
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    print('loading complete')
    
    
    #create a dataset that normalises each example when fed to the model
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.tensor(targets, dtype=torch.float32).squeeze()
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signals_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = indices_train_labels
    labels_val = indices_val_labels
    labels_test = indices_test_labels
    
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    
    
    
    def process_targets(labels):
        processed_targets = []
        
        for target in labels:
            # Split the string by comma and convert each element to float
            processed_target = [float(x) for x in target.split(',')]
            processed_targets.append(processed_target)
        
        # Convert to NumPy array
        processed_targets_np = np.array(processed_targets, dtype=np.float32)
        return processed_targets_np
    
    labels_train = process_targets(labels_train)
    labels_val = process_targets(labels_val)
    labels_test = process_targets(labels_test)
    
    
    
    

    # ✅ Add noise to the first X% of training samples
    noise_fraction = noise_frac   # e.g. first 10%
    noise_std = args.noiselevel
    
    num_samples = signals_train.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    #noise = np.random.randn(*signals_train[:num_noisy].shape) * noise_std
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(signals_train[:num_noisy].shape) * noise_std
    
    signals_train[:num_noisy] = signals_train[:num_noisy] + noise
    signals_train[int(num_noisy*.5):num_noisy, :, :36] = 0
    
    # Remaining samples
    zero_fraction = zero_frac
    remaining = signals_train[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:int(num_remaining*.5), :, :36] = 0
    
    # Put back into original array
    signals_train[num_noisy:] = remaining


    num_samples = signals_test.shape[0]
    num_noisy = int(noise_fraction * num_samples)
    #the * unpacks the domensions as arguments
    noise = np.random.randn(*signals_test[:num_noisy].shape) * noise_std
    signals_test[:num_noisy] = signals_test[:num_noisy] + noise
    
    
    # Remaining samples
    remaining = signals_test[num_noisy:]
    
    # Take first X% of remaining
    num_remaining = remaining.shape[0]
    num_zero = int(zero_fraction * num_remaining)
    
    # Zero out first 8 elements of those samples
    remaining[:num_zero, :, :36] = 0
    
    # Put back into original array
    signals_test[num_noisy:] = remaining


    # Create the datasets
    custom_dataset = CustomDataset(signals_train,labels_train)
    
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    
    print('datsets created')
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader



def create_dataloaders_AF_noise_aug():
    args = get_args_AF()
    data_dir = args.data_dir
    batch_size = args.batch_size
    noise_level = args.noise


    df = pd.read_csv(data_dir + '/metadata.csv')
    print('METADATA LOADED')
    
    indices_train = df[df['set_revised'] == 0].index
    indices_train_labels = df[df['set_revised'] == 0].label.values
    
    indices_val = df[df['set_revised'] == 1].index
    indices_val_labels = df[df['set_revised'] == 1].label.values
    
    indices_cal = df[df['set_revised'] == 2].index
    indices_cal_labels = df[df['set_revised'] == 2].label.values

    
    indices_test = df[df['set_revised'] == 3].index
    indices_test_labels = df[df['set_revised'] == 3].label.values
    
    print('METADATA sorted')
    
    
    class_sample_count = np.array(
        [len(np.where(indices_train_labels == t)[0]) for t in np.unique(indices_train_labels)])
    weight = 1. / class_sample_count
    
    samples_weight = np.array([weight[t] for t in indices_train_labels])
    
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    class_sample_count = np.array(
        [len(np.where(indices_train_labels == t)[0]) for t in np.unique(indices_train_labels)])
    weight = 1. / class_sample_count
    
    samples_weight = np.array([weight[t] for t in indices_train_labels])
    
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    
    # Load the signal file.npy
    print('begin loading signals')
    signals = np.load(data_dir + '/signals.npy')
    # Extract signals corresponding to indices_train
    signal_min  = np.min(signals)
    signal_max = np.max(signals) - signal_min
    
    signals_train = signals[indices_train]
    signals_val = signals[indices_val]
    signals_test = signals[indices_test]
    signals_cal = signals[indices_cal]
    print('loading complete')
    
    
    
    def binarise_labels(labels_arr):
        labels = []
        for i in labels_arr:
            #print(i)
            if i ==1:
                labels.append(1)
                #print('yepp')
            else:
                labels.append(0)
        return labels
    
    
    class CustomDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = torch.FloatTensor(data)
            self.targets = torch.LongTensor(targets)
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]
            x = x-signal_min
            x = x/signal_max
    
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    
    labels_train = binarise_labels(indices_train_labels)
    labels_val = binarise_labels(indices_val_labels) 
    labels_test = binarise_labels(indices_test_labels) 
    labels_cal = binarise_labels(indices_cal_labels) 
    
    signals_train = np.expand_dims(signals_train,1)
    signals_val = np.expand_dims(signals_val,1)
    signals_test = np.expand_dims(signals_test,1)
    signals_cal = np.expand_dims(signals_cal,1)

    np.random.seed(1234)  # repeatable

    noise = np.random.normal(loc=0.0, scale=noise_level, size=signals_test.shape)
    signals_test = signals_test +noise

    
    
    # Create the dataset
    custom_dataset = CustomDataset(signals_train,labels_train)
    custom_dataset_val = CustomDataset(signals_val,labels_val)
    custom_dataset_test = CustomDataset(signals_test,labels_test)
    custom_dataset_cal = CustomDataset(signals_cal,labels_cal)

    
    print('datset created')
    
    
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size,sampler = sampler)
    #train_dataloader = DataLoader(custom_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(custom_dataset_val, batch_size=batch_size)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size)
    calib_dataloader = DataLoader(custom_dataset_cal, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, calib_dataloader
