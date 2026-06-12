from __future__ import annotations

from .resampler import GridSample, LinearInterpolation, IntegerDelayFractionalFIR
from .resampler_module import ResamplerByIdsAndAngles, ResamplerSimple
from .resamplers.gridSampler import GridSample3D, GridSample4D
from ...config_registery import ResamplerConfig

from ...constants.bf import PWDataType, ResamplerType

RF_TYPE = "RF"
IQ_TYPE = "IQ"
IQCOMPLEX_TYPE = "IQComplex"

GRID_SAMPLE = "GridSample"
LINEAR_INTERPOLATION = "LinearInterpolation"
INTEGER_FRACTIONAL_FIR = "IntegerDelayFractionalFIR"

def resampler_builder(config: ResamplerConfig, data_type:str):
    type = config.type
    params = config.params

    # TODO: Para calcular las distancias con IQ Demod hay que hacerlo con fc y no con fs, hay que cambiar eso.
    if data_type != PWDataType.IQ_SPLIT:
        if type == ResamplerType.GRID_SAMPLE:
            resampler = GridSample3D(**params)
        else:
            resampler = GridSample3D(**params)
    else:
        if type == ResamplerType.GRID_SAMPLE:
            resampler = GridSample4D(**params)
        else:
            resampler = GridSample4D(**params)

    return resampler

def resampler_builder2(config: ResamplerConfig):
    type = config.type
    params = config.params

    if type == GRID_SAMPLE:
        resampler = GridSample(**params)
    elif type == LINEAR_INTERPOLATION:
        resampler = LinearInterpolation(**params)
    elif type == INTEGER_FRACTIONAL_FIR:
        resampler = IntegerDelayFractionalFIR(
            n_taps=9, n_phases=64, window="hamming", z_chunk=128, eps=1e-12
        )
    else:
        resampler = GridSample(**params)

    return resampler
