########################################################
###
###     From PDEArena: https://github.com/mliuschi/pdearena/
###         - Modifications are only in terms of file structure and
###         to process the data appropriately
###
########################################################


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from functools import partialmethod
import sys


# From https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
def partialclass(name, cls, *args, **kwds):
    new_cls = type(name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwds)})

    # The following is copied nearly ad verbatim from `namedtuple's` source.

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).

    try:
        new_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass

    return new_cls


ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

# Complex multiplication 2d
def batchmul2d(input, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", input, weights)

class SpectralConv2d(nn.Module):
    """2D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes to keep in the first spatial direction
        modes2 (int): Number of Fourier modes to keep in the second spatial direction
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )

    def forward(self, x, x_dim=None, y_dim=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], torch.view_as_complex(self.weights2)
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


#######################################################################
#######################################################################
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.bn1 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups, num_channels=planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.activation(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = out + self.shortcut(x)
        # out = self.activation(out)
        return out


class DilatedBasicBlock(nn.Module):
    """Basic block for Dilated ResNet

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x


class FourierBasicBlock(nn.Module):
    """Basic block for Fourier Neural Operators

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        modes1 (int, optional): number of modes for the first spatial dimension. Defaults to 16.
        modes2 (int, optional): number of modes for the second spatial dimension. Defaults to 16.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to False.

    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        modes1: int = 16,
        modes2: int = 16,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        assert not norm
        self.fourier1 = SpectralConv2d(in_planes, planes, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv2d(planes, planes, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)

        # So far shortcut connections are not helping
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1)
        #     )

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        out = self.activation(x1 + x2)

        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2
        # out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    """Class to support ResNet like feedforward architectures

    Args:
        n_input_scalar_components (int): Number of input scalar components in the model
        n_input_vector_components (int): Number of input vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        block (Callable): BasicBlock or DilatedBasicBlock or FourierBasicBlock
        num_blocks (List[int]): Number of blocks in each layer
        time_history (int): Number of time steps to use in the input
        time_future (int): Number of time steps to predict in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
    """

    padding = 9

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = True,
        diffmode: bool = False,
        usegrid: bool = False,
    ):
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        if self.usegrid:
            insize += 2
        self.conv_in1 = nn.Conv2d(
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.in_planes,
            time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2),
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def _make_layer(
        self,
        block: Callable,
        planes: int,
        num_blocks: int,
        stride: int,
        activation: str,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        # prev = x.float()
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
            # x = x + prev[:, -1:, ...].detach()
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )


#######################################################################
#######################################################################