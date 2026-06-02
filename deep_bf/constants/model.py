from enum import StrEnum

class Conv2DInitOptions(StrEnum):
    INIT_W_XAVIER_UNIFORM = "XavierUniform"
    INIT_B_ZEROS = "Zeros"

class ActivationType(StrEnum):
    LEAKYRELU = "LeakyReLU"

# ?
# DAS_TYPE = "DAS"

class LayerOption(StrEnum):
    CONV2D_TYPE = "BasicConv2d"
    BF_TYPE = "BF"

class ModelType(StrEnum):
    BINN_FAMILY = "BINN"
    BINN_OG_FAMILY = "BINN_OG"
    SANDWICH_FAMILY = "SANDWICH"
