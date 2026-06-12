from typing import List
from dataclasses import astuple

import torch.nn as nn

from .model_modules import BasicConv2dModule
from .model_bases import Binn, BinnOG, Sandwich

from .bf.bf_layer import get_beamformer_layer
from .conv2d_builder import set_conv2d_layer, set_activation_layer

from ..config_registery import (
    ModelPack,
    ArchitectureCnnBfConfig,
    BeamformerSetup,
    Conv2dInitConfig,
    ActivationConfig,
)

from ..constants.model import LayerOption, ModelType

def model_builder(config: ModelPack, batch_size=1, location="local"):
    asC: List[ArchitectureCnnBfConfig] = config.architecture_configs
    bP: BeamformerSetup = config.beamformer_setup
    cC: Conv2dInitConfig = config.conv2d_init_config
    aC: ActivationConfig = config.activation_config
    layers = nn.ModuleList()


    for layer_config in asC:
        _, family, _, layer_type, ch_in, ch_out, kernel, padding, bias = astuple(layer_config)
        
        if layer_type == LayerOption.CONV2D_TYPE:
            conv2d_layer = set_conv2d_layer(ch_in, ch_out, kernel, padding, bias, cC)
            activation_layer = set_activation_layer(aC)
            layers.append(BasicConv2dModule(conv2d_layer, activation_layer))
        else:
            layers.append(get_beamformer_layer(bP, batch_size, location))

        if family == ModelType.BINN_FAMILY:
            model = Binn(layers)
        elif family == ModelType.BINN_OG_FAMILY:
            model = BinnOG(layers)
        elif family == ModelType.SANDWICH_FAMILY:
            model = Sandwich(layers)
        else:
            model = Binn(layers)

    return model

def model_toy_builder(config: ModelPack, batch_size=1):
    bP: BeamformerSetup = config.beamformer_setup
    layers = nn.ModuleList()
    layers.append(get_beamformer_layer(bP, batch_size))

    model = Binn(layers)
    return model

# #layer option
# CONV2D_TYPE = "BasicConv2d"
# BF_TYPE = "BF"
#
# # model option
# BINN_FAMILY = "BINN"
# BINN_OG_FAMILY = "BINN_OG"
# SANDWICH_FAMILY = "SANDWICH"
#
# from .sorter_layer import ClassicSorter
# def model_builder1(mode, config: ModelPack, batch_size):
#     asC: List[ArchitectureCnnBfConfig] = config.architecture_configs
#     bP: BeamformerSetup = config.beamformer_setup
#     cC: Conv2dInitConfig = config.conv2d_init_config
#     aC: ActivationConfig = config.activation_config
#     layers = nn.ModuleList()
#     is_iq = mode == "IQ"
#     pre_bf = True
#     if is_iq:
#         layers.append(ClassicSorter())  # [B,C,nc,ns,2] -> [B,2C,nc,ns]
#     for layer_config in asC:
#         _, family, _, layer_type, ch_in, ch_out, kernel, padding, bias = astuple(
#             layer_config
#         )
#         if layer_type == CONV2D_TYPE:
#             if is_iq and pre_bf:
#                 ch_in_eff = ch_in * 2
#                 ch_out_eff = ch_out * 2
#             else:
#                 ch_in_eff = ch_in
#                 ch_out_eff = ch_out
#             conv2d_layer = set_conv2d_layer(
#                 ch_in_eff, ch_out_eff, kernel, padding, bias, cC
#             )
#             activation_layer = set_activation_layer(aC)
#             layers.append(BasicConv2dModule(conv2d_layer, activation_layer))
#         else:
#             layers.append(set_beamformer_layer(bP, gsi, batch_size))
#             pre_bf = False
#     if family == BINN_FAMILY:
#         model = Binn(layers)
#     elif family == BINN_OG_FAMILY:
#         model = BinnOG(layers)
#     elif family == SANDWICH_FAMILY:
#         model = Sandwich(layers)
#     else:
#         model = Binn(layers)
#     return model


# def model_toy_builder(
#     mode, config: ModelPack, gsi: GlobalSamplesIdxForTraining, batch_size
# ):
#     bP: BeamformerSetup = config.beamformer_setup
#     layers = nn.ModuleList()
#     is_iq = mode == "IQ"
#     if is_iq:
#         layers.append(ClassicSorter())  # [B,C,nc,ns,2] -> [B,2C,nc,ns]
#     layers.append(set_beamformer_layer(bP, gsi, batch_size))
#
#     model = Binn(layers)
#     return model
