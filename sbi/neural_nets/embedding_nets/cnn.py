# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from sbi.neural_nets.embedding_nets.fully_connected import FCEmbedding

from sbi.neural_nets.embedding_nets import spherical as sp
from healpy import nside2npix

def calculate_filter_output_size(input_size, padding, dilation, kernel, stride) -> int:
    """Returns output size of a filter given filter arguments.

    Uses formulas from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.
    """

    return int(
        (int(input_size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1)
        / int(stride)
        + 1
    )


def get_new_cnn_output_size(
    input_shape: Tuple,
    conv_layer: Union[nn.Conv1d, nn.Conv2d],
    pool: Union[nn.MaxPool1d, nn.MaxPool2d],
) -> Union[Tuple[int], Tuple[int, int]]:
    """Returns new output size after applying a given convolution and pooling.

    Args:
        input_shape: tup.
        conv_layer: applied convolutional layers
        pool: applied pooling layer

    Returns:
        new output dimension of the cnn layer.

    """
    assert isinstance(input_shape, Tuple), "input shape must be Tuple."
    assert 0 < len(input_shape) < 3, "input shape must be 1 or 2d."
    assert isinstance(conv_layer.padding, Tuple), "conv layer attributes must be Tuple."
    assert isinstance(pool.padding, int), "pool layer attributes must be integers."

    out_after_conv = [
        calculate_filter_output_size(
            input_shape[i],
            conv_layer.padding[i],
            conv_layer.dilation[i],
            conv_layer.kernel_size[i],
            conv_layer.stride[i],
        )
        for i in range(len(input_shape))
    ]
    out_after_pool = [
        calculate_filter_output_size(
            out_after_conv[i],
            pool.padding,
            pool.dilation,
            pool.kernel_size,
            pool.stride,
        )
        for i in range(len(input_shape))
    ]
    return tuple(out_after_pool)  # pyright: ignore[reportReturnType]

class hpCNNEmbedding(nn.Module):
    '''
    CNN emedding network for use on the healpix sphere.
    '''
    def __init__(self,
        nside: int,
        in_channels: int = 1,
        out_channels_per_layer: Optional[List[int]] = None,
        nest: bool = True,
        n_blocks: None | int = None,
        num_fc_units: int = 48,
        num_fc_layers: int = 2,
        output_dim: int = 20
    ) -> None:
        '''
        :param nside: Nside of input healpy map.
        :param in_channels: Number of input channels.
        :param out_channels_per_layer: Number of output channels for each layer.
            If None, defaults to [8, 16, 32, 64, 128, 256].
        :param n_blocks: Number of network building blocks to use, as defined
            in Krachmalnicoff & Tomasi (2019). A minimum of 1 block needs to be
            specified, up to a maximum of log2(nside), representing a maximally
            deep network (default).
        :param nest: Whether or not the healpy map uses nested ordering.
            This always needs to be true it seems (required for pooling).
        '''
        super().__init__()

        if n_blocks == None:
            self.n_blocks = torch.log2(torch.as_tensor(nside))

        if out_channels_per_layer is None:
            out_channels_per_layer = [8, 16, 32, 64, 128, 256]
        
        # Ensure we don't have more layers than blocks
        max_blocks = int(self.n_blocks)
        out_channels_per_layer = out_channels_per_layer[:max_blocks]

        npix = nside2npix(nside)
        self.input_shape = (in_channels, npix) # input map is always 1d?
        cur_nside = nside
        cnn_layers = []
        current_in_channels = in_channels
        current_out_channels = current_in_channels # for no blocks/linter

        for i in range(self.n_blocks.to(dtype=torch.int)):
            # Get output channels for this layer
            # (or use last value if list is shorter)
            if i < len(out_channels_per_layer):
                current_out_channels = out_channels_per_layer[i]
            else:
                current_out_channels = out_channels_per_layer[-1]
                
            conv_layer = sp.sphericalConv(
                NSIDE=cur_nside,
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                nest=nest
            )
            pool = sp.sphericalDown(cur_nside)
            cnn_layers += [conv_layer, nn.ReLU(inplace=True), pool]
            cur_nside //= 2
            cnn_output_size = int(nside2npix(cur_nside))
            
            # Update input channels for next layer
            current_in_channels = current_out_channels
        
        self.cnn_subnet = nn.Sequential(*cnn_layers)

        # Construct linear post processing net
        self.linear_subnet = FCEmbedding(
            input_dim=current_out_channels * cnn_output_size,
            output_dim=output_dim,
            num_layers=num_fc_layers,
            num_hiddens=num_fc_units,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        # print(x.shape)
        # print(x.view(batch_size, *self.input_shape).shape)
        # reshape to account for single channel data
        x = self.cnn_subnet(x.view(batch_size, *self.input_shape))

        # flatten for linear layers
        x = x.view(batch_size, -1)
        x = self.linear_subnet(x)
        
        return x

class CNNEmbedding(nn.Module):
    """Convolutional embedding network (1D or 2D convolutions)."""

    def __init__(
        self,
        input_shape: Tuple,
        in_channels: int = 1,
        out_channels_per_layer: Optional[List] = None,
        num_conv_layers: int = 2,
        num_linear_layers: int = 2,
        num_linear_units: int = 50,
        output_dim: int = 20,
        kernel_size: int = 5,
        pool_kernel_size: int = 2,
    ):
        """Convolutional embedding network.

        First two layers are convolutional, followed by fully connected layers.

        Automatically infers whether to apply 1D or 2D convolution depending on
        input_shape.
        Allows usage of multiple (color) channels by passing in_channels > 1.

        Args:
            input_shape: Dimensionality of input, e.g., (28,) for 1D, (28, 28) for 2D.
            in_channels: Number of image channels, default 1.
            out_channels_per_layer: Number of out convolutional out_channels for each
                layer. Must match the number of layers passed below.
            num_cnn_layers: Number of convolutional layers.
            num_linear_layers: Number fully connected layer.
            num_linear_units: Number of hidden units in fully-connected layers.
            output_dim: Number of output units of the final layer.
            kernel_size: Kernel size for both convolutional layers.
            pool_size: pool size for MaxPool1d operation after the convolutional
                layers.
        """
        super(CNNEmbedding, self).__init__()

        assert isinstance(input_shape, Tuple), (
            "input_shape must be a Tuple of size 1 or 2, e.g., (width, [height])."
        )
        assert (
            0 < len(input_shape) < 3
        ), """input_shape must be a Tuple of size 1 or 2, e.g.,
            (width, [height]). Number of input channels are passed separately"""

        use_2d_cnn = len(input_shape) == 2
        conv_module = nn.Conv2d if use_2d_cnn else nn.Conv1d
        pool_module = nn.MaxPool2d if use_2d_cnn else nn.MaxPool1d

        if out_channels_per_layer is None:
            out_channels_per_layer = [6, 12]
        assert len(out_channels_per_layer) == num_conv_layers, (
            "out_channels needs as many entries as num_cnn_layers."
        )

        # define input shape with channel
        self.input_shape = (in_channels, *input_shape)

        # Construct CNN feature extractor.
        cnn_layers = []
        cnn_output_size = input_shape
        stride = 1
        padding = 1

        for ii in range(num_conv_layers):
            # Defining another 2D convolution layer
            conv_layer = conv_module(
                in_channels=in_channels if ii == 0 else out_channels_per_layer[ii - 1],
                out_channels=out_channels_per_layer[ii],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            pool = pool_module(kernel_size=pool_kernel_size)
            cnn_layers += [conv_layer, nn.ReLU(inplace=True), pool]
            # Calculate change of output size of each CNN layer
            cnn_output_size = get_new_cnn_output_size(cnn_output_size, conv_layer, pool)

            assert all(
                cnn_output_size
            ), f"""CNN output size is zero at layer {ii + 1}. Either reduce
                 num_cnn_layers to {ii} or adjust the kernel_size
                 and pool_kernel_size accordingly."""

        self.cnn_subnet = nn.Sequential(*cnn_layers)

        # Construct linear post processing net.
        self.linear_subnet = FCEmbedding(
            input_dim=out_channels_per_layer[-1]
            * torch.prod(torch.tensor(cnn_output_size)),
            output_dim=output_dim,
            num_layers=num_linear_layers,
            num_hiddens=num_linear_units,
        )

    # Defining the forward pass
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        # reshape to account for single channel data.
        x = self.cnn_subnet(x.view(batch_size, *self.input_shape))
        # flatten for linear layers.
        x = x.view(batch_size, -1)
        x = self.linear_subnet(x)
        return x
