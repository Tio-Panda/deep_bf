import torch
from .apod import dynamic_receive_aperture
from ...config_registery import ApodConfig

DYNAMIC_RECEIVE_APERTURE = "DynamicReceiveAperture"
NONE = "None"
def apod_builder(Z, X, probe_geometry, config: ApodConfig):
    type = config.type
    params = config.params

    nc = int(probe_geometry.shape[0])

    if type == DYNAMIC_RECEIVE_APERTURE:
        apod = dynamic_receive_aperture(Z, X, probe_geometry, **params)
    elif type == NONE:
        apod = torch.ones((nc, Z.shape[0], Z.shape[1]), dtype=Z.dtype, device=Z.device)
    else:
        apod = dynamic_receive_aperture(Z, X, probe_geometry, **params)

    return apod
