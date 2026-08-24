# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.

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
