# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.

"""
This code is adapted from source code released under the Apache License 2.0 and associated with the publication cited below. The Apache License 2.0 permits use, modification, and distribution subject to its terms, including preservation of the applicable copyright, licence, and attribution notices. The adapted code may contain modifications made for the present analysis and should not be treated as an exact reproduction of the original implementation.
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
import argparse

def get_args_BP():
    parser = argparse.ArgumentParser(description='PyTorch BP Model Training')

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train (default: 200)')
    parser.add_argument('--weight-decay', type=float, default=1e-8, help='Weight decay (default: 1e-8)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
    parser.add_argument('--print-mod', type=int, default=450, help='Print modulo (default: 450)')
    #parser.add_argument('--KERN', type=int, default=8, help='Kernel size for AF model (default: 8)')
    parser.add_argument('--h0', type=float, default=0.001, help='Initialisation value (default: 0.001)')
    parser.add_argument('--train-samples', type=int, default=60, help='Number of training samples for ivon (default: 60)')
    parser.add_argument('--dr', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--model_dir', type = str, default = '<path>', help='Dir to store the checkpoints/loss')
    parser.add_argument('--data_dir', type = str, default = '<path>', help='Dir containing training data')
    parser.add_argument('--checkpoint-path', type = str, default = '', help='path to model checkpoint')
    parser.add_argument('--zero', type = float, default = 1, help='zero fraction')
    parser.add_argument('--noise', type = float, default = .5, help='noise fraction')
    parser.add_argument('--zeroonly', action='store_true', help='zero only data')
    parser.add_argument('--aug', action='store_true', help='aug data')
    parser.add_argument('--testspecial',action='store_true')
    parser.add_argument('--noiseonly',action='store_true')
    parser.add_argument('--noiselevel', type = float, default = .1,)
    parser.add_argument('--noaug',action='store_true')
    parser.add_argument('--noiseall',action='store_true')
    parser.add_argument('--nonleda',action='store_true')
    parser.add_argument('--nonspurious',action='store_true')
    parser.add_argument('--noiseleda',action='store_true')
    parser.add_argument('--seed', type = int, default = 20, help='model seed')

    args = parser.parse_args()
    return args


def get_args_AF():
    parser = argparse.ArgumentParser(description='PyTorch AF Model Training')

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train (default: 500)')
    parser.add_argument('--weight-decay', type=float, default=1e-10, help='Weight decay (default: 1e-10)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
    parser.add_argument('--KERN', type=int, default=8, help='Kernel size for AF model (default: 8)')
    parser.add_argument('--h0', type=float, default=0.001, help='Initial hidden state value (default: 0.001)')
    parser.add_argument('--train-samples', type=int, default=60, help='Number of training samples (default: 60)')
    parser.add_argument('--dr', type=float, default=0.05, help='Dropout rate (default: 0.05)')
    parser.add_argument('--model_dir', type = str, default = '<path>', help='Dir to store the checkpoints/loss')
    parser.add_argument('--data_dir', type = str, default = '<path>', help='Dir containing training data')
    parser.add_argument('--checkpoint-path', type = str, default = '<path>', help='path to model checkpoint')
    parser.add_argument('--calib',action='store_true')
    parser.add_argument('--noise',type=float, default=0.0)
    parser.add_argument('--addnoise',action='store_true')


    args = parser.parse_args()
    return args
