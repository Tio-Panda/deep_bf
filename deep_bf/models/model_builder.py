from typing import List
from dataclasses import astuple

import torch.nn as nn

from .sorter_layer import ClassicSorter

from .model_modules import BasicConv2dModule
from ..webdataset.gsi.gsi_for_training import GlobalSamplesIdxForTraining
from .bf_layers.das import DAS
from .model_bases import Binn, BinnOG, Sandwich

from ..beamformers.resampler.resampler_builder import get_resampler_for_training

from ..config_registery import (
    ModelPacking,
    ArchitectureCnnBfConfig,
    BeamformerPacking,
    Conv2dInitConfig,
    ActivationConfig,
)

INIT_W_XAVIER_UNIFORM = "XavierUniform"
INIT_B_ZEROS = "Zeros"


def set_conv2d_layer(
    ch_in, ch_out, kernel, padding, bias, conv2d_init_config: Conv2dInitConfig
):
    m = nn.Conv2d(
        in_channels=ch_in,
        out_channels=ch_out,
        kernel_size=kernel,
        padding=padding,
        bias=bias,
    )

    _, weight_init, bias_init = astuple(conv2d_init_config)

    if INIT_W_XAVIER_UNIFORM == weight_init:
        nn.init.xavier_uniform_(m.weight)

    if m.bias is not None:
        if INIT_B_ZEROS == bias_init:
            nn.init.zeros_(m.bias)

    return m


LEAKYRELU = "LeakyReLU"


def set_activation_layer(activation_config: ActivationConfig):
    _, activation_type, params = astuple(activation_config)

    if activation_type == LEAKYRELU:
        a = nn.LeakyReLU(**params)
    else:
        a = nn.LeakyReLU(negative_slope=0.01)

    return a


DAS_TYPE = "DAS"


def set_beamformer_layer(
    beamformer_packing: BeamformerPacking, gsi: GlobalSamplesIdxForTraining, batch_size
):
    nz, nx = gsi.nz, gsi.nx
    bf_config = beamformer_packing.beamformer_config
    resampler_config = beamformer_packing.resampler_config

    resampler = get_resampler_for_training(gsi, resampler_config, batch_size)

    if bf_config.type == DAS_TYPE:
        bf_layer = DAS(nz, nx, resampler)
    else:
        bf_layer = DAS(nz, nx, resampler)

    return bf_layer


CONV2D_TYPE = "BasicConv2d"
BF_TYPE = "BF"

BINN_FAMILY = "BINN"
BINN_OG_FAMILY = "BINN_OG"
SANDWICH_FAMILY = "SANDWICH"


def model_builder(
    mode, config: ModelPacking, gsi: GlobalSamplesIdxForTraining, batch_size
):
    asC: List[ArchitectureCnnBfConfig] = config.architecture_configs
    bP: BeamformerPacking = config.beamformer
    cC: Conv2dInitConfig = config.conv2d_init_config
    aC: ActivationConfig = config.activation_config
    layers = nn.ModuleList()
    is_iq = mode == "IQ"
    pre_bf = True
    if is_iq:
        layers.append(ClassicSorter())  # [B,C,nc,ns,2] -> [B,2C,nc,ns]
    for layer_config in asC:
        _, family, _, layer_type, ch_in, ch_out, kernel, padding, bias = astuple(
            layer_config
        )
        if layer_type == CONV2D_TYPE:
            if is_iq and pre_bf:
                ch_in_eff = ch_in * 2
                ch_out_eff = ch_out * 2
            else:
                ch_in_eff = ch_in
                ch_out_eff = ch_out
            conv2d_layer = set_conv2d_layer(
                ch_in_eff, ch_out_eff, kernel, padding, bias, cC
            )
            activation_layer = set_activation_layer(aC)
            layers.append(BasicConv2dModule(conv2d_layer, activation_layer))
        else:
            layers.append(set_beamformer_layer(bP, gsi, batch_size))
            pre_bf = False
    if family == BINN_FAMILY:
        model = Binn(layers)
    elif family == BINN_OG_FAMILY:
        model = BinnOG(layers)
    elif family == SANDWICH_FAMILY:
        model = Sandwich(layers)
    else:
        model = Binn(layers)
    return model


def model_toy_builder(
    mode, config: ModelPacking, gsi: GlobalSamplesIdxForTraining, batch_size
):
    bP: BeamformerPacking = config.beamformer
    layers = nn.ModuleList()
    is_iq = mode == "IQ"
    if is_iq:
        layers.append(ClassicSorter())  # [B,C,nc,ns,2] -> [B,2C,nc,ns]
    layers.append(set_beamformer_layer(bP, gsi, batch_size))

    model = Binn(layers)
    return model


# def model_builder(mode, config: ModelPacking, gsi: GlobalSamplesIdxForTraining, batch_size):
#     asC: List[ArchitectureCnnBfConfig] = config.architecture_configs
#     bP: BeamformerPacking = config.beamformer
#     cC: Conv2dInitConfig = config.conv2d_init_config
#     aC: ActivationConfig = config.activation_config
#
#     layers = nn.ModuleList()
#
#     if mode == "IQ":
#         layers.append(ClassicSorter())
#
#     for layer_config in asC:
#         _, family, _, layer_type, ch_in, ch_out, kernel, padding, bias = astuple(layer_config)
#
#         if layer_type == CONV2D_TYPE:
#             conv2d_layer = set_conv2d_layer(ch_in, ch_out, kernel, padding, bias, cC)
#             activation_layer = set_activation_layer(aC)
#             m = BasicConv2dModule(conv2d_layer, activation_layer)
#             layers.append(m)
#
#         # if layer_type == BF_TYPE:
#         else:
#             bf_layer = set_beamformer_layer(bP, gsi, batch_size)
#             layers.append(bf_layer)
#
#     if family == BINN_FAMILY:
#         model = Binn(layers)
#
#     elif family == BINN_OG_FAMILY:
#         model = BinnOG(layers)
#
#     elif family == SANDWICH_FAMILY:
#         model = Sandwich(layers)
#
#     else:
#         model = Binn(layers)
#
#     return model
