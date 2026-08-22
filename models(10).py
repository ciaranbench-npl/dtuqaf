# Author: Ciaran Bench
# Disclaimer: This code is provided solely for the purpose of reproducing results as described in the associated work. 
# The authors and contributors are not responsible for any consequences arising from the use of this code or its outputs 
# beyond this intended purpose. Use at your own risk.
import torch.nn as nn
from argparser import get_args_BP, get_args_AF
import torch.nn.init as init
import torch.nn.functional as F
import torch


class GNLL_activation(nn.Module):
    def forward(self, x):
        # Apply softmax to the first two elements
        #softmax_part = F.softmax(x[:, :2], dim=-1)
        softplus_part = F.softplus(x[:, :2])
        # Apply softplus to the last two elements
        exp_part = torch.log(x[:, 2:])
        # Concatenate the results
        return torch.cat((softplus_part, exp_part), dim=1)

class NeuralNetwork(nn.Module):
    def __init__(self, eval=False, dr = 0, KERN = 8):
        super().__init__()
        if eval == True:
            dr = dr
            KERN = KERN
        else:
            args = get_args_BP()
            KERN = args.KERN
            dr = args.dr
            
        
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.Conv1d(128, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(128),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 64, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(dr),
            nn.Conv1d(64, 64, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 32, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(32),
            nn.Dropout(dr),
            nn.Conv1d(32, 32, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(32),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 8, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(8),
            nn.Dropout(dr),
            nn.Conv1d(8, 8, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(8),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 1, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(29, 4),
            nn.Softplus()
        )

    def forward(self, x):
        preds = self.conv_relu_stack(x)
        return preds

class NeuralNetwork_exp(nn.Module):
    def __init__(self, eval=False, dr = 0, KERN = 8):
        super().__init__()
        if eval == True:
            dr = dr
            KERN = KERN
        else:
            args = get_args_BP()
            KERN = args.KERN
            dr = args.dr
            
        
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.Conv1d(128, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(128),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 64, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(dr),
            nn.Conv1d(64, 64, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 32, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(32),
            nn.Dropout(dr),
            nn.Conv1d(32, 32, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(32),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 8, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(8),
            nn.Dropout(dr),
            nn.Conv1d(8, 8, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(8),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 1, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(29, 4),
            GNLL_activation()
        )

    def forward(self, x):
        preds = self.conv_relu_stack(x)
        return preds
        

class NeuralNetwork_less_dr(nn.Module):
    def __init__(self, eval=False, dr = 0, KERN = 8):
        super().__init__()
        if eval == True:
            dr = dr
            KERN = KERN
        else:
            args = get_args_BP()
            KERN = args.KERN
            dr = args.dr
            
        
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.Dropout(dr),
            nn.Conv1d(128, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(128),
            #nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 64, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(64),
            #nn.Dropout(dr),
            nn.Conv1d(64, 64, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(64),
            #nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 32, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(32),
            #nn.Dropout(dr),
            nn.Conv1d(32, 32, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(32),
            #nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 8, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(8),
            #nn.Dropout(dr),
            nn.Conv1d(8, 8, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(8),
            #nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 1, kernel_size=KERN),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1),
            #nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(29, 4),
            nn.Dropout(dr),
            nn.Softplus()
        )

    def forward(self, x):
        preds = self.conv_relu_stack(x)
        return preds


            

def init_weights(m, init_type='xavier_normal', init_gain=1):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if init_type == 'xavier_uniform':
            init.xavier_uniform_(m.weight, gain=init_gain)
        elif init_type == 'xavier_normal':
            init.xavier_normal_(m.weight, gain=init_gain)
        elif init_type == 'kaiming_normal':
            init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif init_type == 'kaiming_uniform':
            init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def init_weights_no_bias(m, init_type='xavier_normal', init_gain=1):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if init_type == 'xavier_uniform':
            init.xavier_uniform_(m.weight, gain=init_gain)
        elif init_type == 'xavier_normal':
            init.xavier_normal_(m.weight, gain=init_gain)
        elif init_type == 'kaiming_normal':
            init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif init_type == 'kaiming_uniform':
            init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)



class MCDActivation(nn.Module):
    def forward(self, x):
        # Apply softmax to the first two elements
        #softmax_part = F.softmax(x[:, :2], dim=-1)
        softmax_part = x[:, :2]
        # Apply softplus to the last two elements
        softplus_part = F.softplus(x[:, 2:])
        # Concatenate the results
        return torch.cat((softmax_part, softplus_part), dim=1)




class NeuralNetwork_AF(nn.Module):
    def __init__(self, eval=False, dr = 0, KERN = 8):
        super().__init__()
        if eval == True:
            dr = dr
            KERN = KERN
        else:
            args = get_args_AF()
            KERN = args.KERN
            dr = args.dr
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 64, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 32, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 16, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 1, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(18, 4),
            #MCDActivation(),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        logits = self.conv_relu_stack(x)
        return logits

class NeuralNetwork_AF_more(nn.Module):
    def __init__(self, eval=False, dr = 0, KERN = 8):
        super().__init__()
        if eval == True:
            dr = dr
            KERN = KERN
        else:
            args = get_args_AF()
            KERN = args.KERN
            dr = args.dr
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 256, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(256, 384, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(384, 128, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 1, kernel_size=KERN),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(18, 4),
            #MCDActivation(),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        logits = self.conv_relu_stack(x)
        return logits


import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.2):
        """
        1D Residual Block for ResNet architecture with dropout
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for matching dimensions. Defaults to None.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResidualBlock1D, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, #3
                               stride=stride, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate) #Dropout1d
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, #3
                               stride=1, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout_rate)#Dropout1d
        
        # Downsampling layer to match dimensions if needed
        self.downsample = downsample
        
    def forward(self, x):
        """
        Forward pass for the residual block
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after residual block processing
        """
        identity = x
        
        # First conv, batch norm, dropout, and activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)
        
        # Second conv, batch norm, and dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Apply downsampling if necessary
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1, dr=0, eval=False):
        super().__init__()
        if eval == True:
            dropout_rate = dr
        else:
            args = get_args_BP()
            dropout_rate = args.dr
    #def __init__(self, block, layers, num_classes=10, in_channels=1, dropout_rate=0.2):
        """
        1D ResNet model with dropout
        
        Args:
            block (nn.Module): ResNet block type (usually ResidualBlock1D)
            layers (list): Number of blocks in each layer
            num_classes (int, optional): Number of output classes. Defaults to 10.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResNet1D, self).__init__()
        
        # Initial layers
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) #Dropout1d
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with dropout
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 1, layers[3], stride=2, dropout_rate=dropout_rate)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(100)#1
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(100, num_classes)
        self.softplus = nn.Softplus()
        
    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.2):
        """
        Create a layer with multiple ResNet blocks
        
        Args:
            block (nn.Module): ResNet block type
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in the layer
            stride (int, optional): Stride for the first block. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        
        Returns:
            nn.Sequential: Layer of ResNet blocks
        """
        downsample = None
        
        # If stride is not 1 or input/output channels differ, we need downsampling
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, #1
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the entire ResNet model
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        # Initial convolution layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout before fully connected layer
        x = self.fc_dropout(x)
        x = self.fc(x)
        x = self.softplus(x)
        
        return x

def resnet18_1d(num_classes=4, in_channels=1, dr=0, eval=False):
    """
    Create a ResNet-18 model for 1D data with dropout
    
    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
    
    Returns:
        ResNet1D: ResNet-18 model
    """
    return ResNet1D(ResidualBlock1D, [2, 2, 2, 2], num_classes, in_channels, dr, eval)

# Example usage
# model = resnet18_1d(num_classes=10, in_channels=1, dropout_rate=0.3)
# x = torch.randn(32, 1, 100)  # batch_size, channels, sequence_length
# output = model(x)


import torch
import torch.nn as nn

class ResidualBlockBatchNorm1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.2):
        """
        1D Residual Block for ResNet architecture with batch normalization
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for matching dimensions. Defaults to None.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResidualBlockBatchNorm1D, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout1d(p=dropout_rate)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout1d(p=dropout_rate)
        
        # Downsampling layer to match dimensions if needed
        self.downsample = downsample
        
    def forward(self, x):
        """
        Forward pass for the residual block
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after residual block processing
        """
        identity = x
        
        # First conv, batch norm, dropout, and activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)
        
        # Second conv, batch norm, and dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Apply downsampling if necessary
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out

class ResNetBatchNorm1D(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1, dropout_rate=0.2):
        """
        1D ResNet model with batch normalization
        
        Args:
            block (nn.Module): ResNet block type (usually ResidualBlockBatchNorm1D)
            layers (list): Number of blocks in each layer
            num_classes (int, optional): Number of output classes. Defaults to 10.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResNetBatchNorm1D, self).__init__()
        
        # Initial layers
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with dropout
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.2):
        """
        Create a layer with multiple ResNet blocks
        
        Args:
            block (nn.Module): ResNet block type
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in the layer
            stride (int, optional): Stride for the first block. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        
        Returns:
            nn.Sequential: Layer of ResNet blocks
        """
        downsample = None
        
        # If stride is not 1 or input/output channels differ, we need downsampling
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the entire ResNet model
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        # Initial convolution layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout before fully connected layer
        x = self.fc_dropout(x)
        x = self.fc(x)
        
        return x

class ResidualBlockNoBatchNorm1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0):
        """
        1D Residual Block for ResNet architecture without batch normalization
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for matching dimensions. Defaults to None.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResidualBlockNoBatchNorm1D, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9,#3 
                               stride=stride, padding=4, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout1d(p=dropout_rate)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9,#3 
                               stride=1, padding=4, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout1d(p=dropout_rate)
        
        # Downsampling layer to match dimensions if needed
        self.downsample = downsample
        
    def forward(self, x):
        """
        Forward pass for the residual block
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after residual block processing
        """
        identity = x
        
        # First conv, activation, and dropout
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv, activation, and dropout
        out = self.conv2(out)
        
        # Apply downsampling if necessary
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return out

class ResNetNoBatchNorm1D(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1, dr=0, eval=False):
        super().__init__()
        if eval == True:
            dropout_rate = dr
        else:
            args = get_args_BP()
            dropout_rate = args.dr
   # def __init__(self, block, layers, num_classes=10, in_channels=1, dropout_rate=0.2):
        """
        1D ResNet model without batch normalization
        
        Args:
            block (nn.Module): ResNet block type (usually ResidualBlockNoBatchNorm1D)
            layers (list): Number of blocks in each layer
            num_classes (int, optional): Number of output classes. Defaults to 10.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResNetNoBatchNorm1D, self).__init__()
        
        # Initial layers
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, 
                               stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 1, layers[3], stride=2, dropout_rate=dropout_rate)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(100)#1
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(100, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.0):
        """
        Create a layer with multiple ResNet blocks
        
        Args:
            block (nn.Module): ResNet block type
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in the layer
            stride (int, optional): Stride for the first block. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        
        Returns:
            nn.Sequential: Layer of ResNet blocks
        """
        downsample = None
        
        # If stride is not 1 or input/output channels differ, we need downsampling
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                                   stride=stride, bias=True)
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the entire ResNet model
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        # Initial convolution layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        
        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout before fully connected layer
        x = self.fc_dropout(x)
        x = self.fc(x)
        x = self.softplus(x)
        
        return x

def resnet18_batchnorm_1d(num_classes=10, in_channels=1, dropout_rate=0.0):
    """
    Create a ResNet-18 model for 1D data with batch normalization
    
    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
    
    Returns:
        ResNetBatchNorm1D: ResNet-18 model with batch normalization
    """
    return ResNetBatchNorm1D(ResidualBlockBatchNorm1D, [2, 2, 2, 2], num_classes, in_channels, dropout_rate)

def resnet18_no_batchnorm_1d(num_classes=4, in_channels=1, dr=0, eval=False):
    """
    Create a ResNet-18 model for 1D data without batch normalization
    
    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
    
    Returns:
        ResNetNoBatchNorm1D: ResNet-18 model without batch normalization
    """
    return ResNetNoBatchNorm1D(ResidualBlockNoBatchNorm1D, [2, 2, 2, 2], num_classes, in_channels, dr, eval)

# Example usage
# batchnorm_model = resnet18_batchnorm_1d(num_classes=10, in_



'''
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.2):
        """
        1D Residual Block for ResNet architecture with dropout
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for matching dimensions. Defaults to None.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResidualBlock1D, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout1d(p=dropout_rate)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout1d(p=dropout_rate)
        
        # Downsampling layer to match dimensions if needed
        self.downsample = downsample
        
    def forward(self, x):
        """
        Forward pass for the residual block
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after residual block processing
        """
        identity = x
        
        # First conv, batch norm, dropout, and activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)
        
        # Second conv, batch norm, and dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Apply downsampling if necessary
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out
'''
class ResNet1D_af(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1, dropout_rate=0.2):
        """
        1D ResNet model with dropout
        
        Args:
            block (nn.Module): ResNet block type (usually ResidualBlock1D)
            layers (list): Number of blocks in each layer
            num_classes (int, optional): Number of output classes. Defaults to 10.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResNet1D_af, self).__init__()
        
        # Initial layers
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with dropout
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        self.softplus = nn.Softplus()
        
    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.2):
        """
        Create a layer with multiple ResNet blocks
        
        Args:
            block (nn.Module): ResNet block type
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in the layer
            stride (int, optional): Stride for the first block. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        
        Returns:
            nn.Sequential: Layer of ResNet blocks
        """
        downsample = None
        
        # If stride is not 1 or input/output channels differ, we need downsampling
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the entire ResNet model
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        # Initial convolution layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout before fully connected layer
        x = self.fc_dropout(x)
        x = self.fc(x)
        
        return x

def resnet18_1d_af(num_classes=4, in_channels=1, dropout_rate=0.1):
    """
    Create a ResNet-18 model for 1D data with dropout
    
    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
    
    Returns:
        ResNet1D: ResNet-18 model
    """
    return ResNet1D_af(ResidualBlock1D, [2, 2, 2, 2], num_classes, in_channels, dropout_rate)




class ResNetNoBatchNorm1D_af(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1, dropout_rate=0.2):
        """
        1D ResNet model without batch normalization
        
        Args:
            block (nn.Module): ResNet block type (usually ResidualBlockNoBatchNorm1D)
            layers (list): Number of blocks in each layer
            num_classes (int, optional): Number of output classes. Defaults to 10.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(ResNetNoBatchNorm1D_af, self).__init__()
        
        # Initial layers
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, 
                               stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.2):
        """
        Create a layer with multiple ResNet blocks
        
        Args:
            block (nn.Module): ResNet block type
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in the layer
            stride (int, optional): Stride for the first block. Defaults to 1.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        
        Returns:
            nn.Sequential: Layer of ResNet blocks
        """
        downsample = None
        
        # If stride is not 1 or input/output channels differ, we need downsampling
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                                   stride=stride, bias=True)
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the entire ResNet model
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        # Initial convolution layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        
        # Pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout before fully connected layer
        x = self.fc_dropout(x)
        x = self.fc(x)
        
        return x


def resnet18_no_batchnorm_1d_af(num_classes=4, in_channels=1, dropout_rate=0):
    """
    Create a ResNet-18 model for 1D data without batch normalization
    
    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
    
    Returns:
        ResNetNoBatchNorm1D: ResNet-18 model without batch normalization
    """
    return ResNetNoBatchNorm1D_af(ResidualBlockNoBatchNorm1D, [2, 2, 2, 2], num_classes, in_channels, dropout_rate)

# Example usage
# batchnorm_model = resnet18_batchnorm_1d(num_classes=10, in_