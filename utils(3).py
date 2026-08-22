# Author: Ciaran Bench
# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.
import torch
import numpy as np

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
