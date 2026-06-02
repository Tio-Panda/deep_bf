from dataclasses import astuple

import torch.nn as nn
from ..constants.model import Conv2DInitOptions, ActivationType
from ..config_registery import Conv2dInitConfig, ActivationConfig

def set_conv2d_layer(ch_in, ch_out, kernel, padding, bias, conv2d_init_config: Conv2dInitConfig):
    m = nn.Conv2d(
        in_channels=ch_in,
        out_channels=ch_out,
        kernel_size=kernel,
        padding=padding,
        bias=bias,
    )

    _, weight_init, bias_init = astuple(conv2d_init_config)
    
    if weight_init == Conv2DInitOptions.INIT_W_XAVIER_UNIFORM:
        nn.init.xavier_uniform_(m.weight)

    if m.bias is not None:
        if bias_init == Conv2DInitOptions.INIT_B_ZEROS:
            nn.init.zeros_(m.bias)

    return m

def set_activation_layer(activation_config: ActivationConfig):
    _, activation_type, params = astuple(activation_config)

    if activation_type == ActivationType.LEAKYRELU:
        a = nn.LeakyReLU(**params)
    else:
        a = nn.LeakyReLU(negative_slope=0.01)

    return a

