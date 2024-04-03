"""
Author: Louis Winder

A full WaveNet model, as described in "WaveNet: A Generative Model for Raw Audio" by van den Oord et al.
With influence from:
 https://github.com/vincentherrmann/pytorch-wavenet
 and
 https://github.com/evinpinar/wavenet_pytorch/
"""

import torch.nn as nn
import torch.nn.functional as func

class WaveNet(nn.Module):
    def __init__(self,
                 layers = 13, # Number of layers in a stack
                 stacks = 5, # Number of stacks
                 dilation_channels = 64, # Number of dilation input channels
                 residual_channels = 64, # Number of residual input channels
                 skip_channels = 1024, # Number of skip input channels
                 classes = 256, # Number of classes to classify (256 for mu-encoded input)
                 kernel_size = 2): # Base size of the convolution kernel (without dilation)
        super(WaveNet, self).__init__()

        self.layers = layers
        self.stacks = stacks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size

        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.skips = nn.ModuleList()

        # DROPOUT (not often used)
        dropoutProb = 0.0  # Chance for neurons to be dropped
        self.dropout = nn.Dropout(dropoutProb)

        receptive_field = 0

        # Create initial causal convolution (dilation = 1)
        self.causal = nn.Conv1d(in_channels=classes,
                                out_channels=residual_channels,
                                kernel_size=1,
                                dilation=1,
                                padding=0,
                                bias=True)

        ############# RESIDUAL BLOCK #############

        for stack in range(stacks):
            dilation = 1
            rf = kernel_size - 1
            for layer in range(layers):
                padding = (kernel_size - 1) * dilation # Determine appropriate input padding
                # "Filter" dilated convolution
                self.filters.append(nn.Conv1d(in_channels=residual_channels,
                                              out_channels=dilation_channels,
                                              kernel_size=kernel_size,
                                              dilation=dilation,
                                              padding=padding,
                                              bias=True))

                # "Gate" dilated convolution
                self.gates.append(nn.Conv1d(in_channels=residual_channels,
                                            out_channels=dilation_channels,
                                            kernel_size=kernel_size,
                                            dilation=dilation,
                                            padding=padding,
                                            bias=True))

                # Residual convolution
                self.residuals.append(nn.Conv1d(in_channels=dilation_channels,
                                                out_channels=residual_channels,
                                                kernel_size=1,
                                                bias=True))

                # Skip convolution
                self.skips.append(nn.Conv1d(in_channels=dilation_channels,
                                            out_channels=skip_channels,
                                            kernel_size=1,
                                            bias=True))

                receptive_field += rf
                dilation *= 2
                rf *= 2

        ############# OUTPUT BLOCK #############

        # First 1x1 output convolution
        self.end1 = nn.Conv1d(in_channels=skip_channels,
                              out_channels=skip_channels,
                              kernel_size=1,
                              bias=True)

        # Second 1x1 output convolution
        self.end2 = nn.Conv1d(in_channels=skip_channels,
                              out_channels=classes, # Output classes to classify via Softmax
                              kernel_size=1,
                              bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        residual_input = self.causal(input)
        residual_input = self.dropout(residual_input)
        skip_sum = 0 # Skip connection accumulator

        ############# RESIDUAL BLOCK #############

        for i in range(self.layers * self.stacks):
            residual = residual_input # Store residual connection

            # "Filter" dilated convolution
            filter = self.filters[i](residual_input)
            filter = func.tanh(filter)

            # "Gate" dilated convolution
            gate = self.gates[i](residual_input)
            gate = func.sigmoid(gate)

            # Reshape to correct size
            filter = filter[:, :, :residual.size(-1)]
            gate = gate[:, :, :residual.size(-1)]

            dilated = filter * gate
            dilated = self.dropout(dilated)

            # Accumulate skip connections
            s = self.skips[i](dilated)
            skip_sum = s + skip_sum

            # Add residual to dilated output as input to next layer
            r = self.residuals[i](dilated)
            r = self.dropout(r)
            residual_input = r + residual

        ############## OUTPUT BLOCK #############

        skip_sum = func.relu(skip_sum)
        skip_sum = self.dropout(skip_sum)
        output = self.end1(skip_sum)

        output = func.relu(output)
        output = self.dropout(output)
        output = self.end2(output).transpose(1, 2)

        return output # No softmax needed as done internally by CrossEntropyLoss expecting logits
