from __future__ import annotations

from .das import DAS3D, DAS4D
from .fdmas7 import FDMAS3D
from .mv4 import MV3D, MV4D
from .cf2 import CF3D, CF4D
from .imap2 import IMAP3D, IMAP4D
from .sr import SR3D, SR4D
from .sr2 import SR2_3D, SR2_4D
from .cb import CB3D

from ...constants.bf import BeamformerType, PWDataType
from ...config_registery import BeamformerConfig


def bf_builder(config: BeamformerConfig, pw, nz=None, nx=None):
    bf_type = config.type
    bf_params = dict(config.params)
    data_type = pw.type

    if data_type != PWDataType.IQ_SPLIT:
        if bf_type == BeamformerType.DAS:
            bf = DAS3D(**bf_params)
        elif bf_type == BeamformerType.FDMAS:
            bf_params["fs"] = pw.fs
            bf_params["f0"] = pw.fc
            bf = FDMAS3D(**bf_params)
        elif bf_type == BeamformerType.MV:
            bf = MV3D(**bf_params)
        elif bf_type == BeamformerType.CF:
            bf = CF3D(**bf_params)
        elif bf_type == BeamformerType.IMAP:
            bf = IMAP3D(**bf_params)
        elif bf_type == BeamformerType.SR1:
            bf = SR3D(**bf_params)
        elif bf_type == BeamformerType.SR2:
            bf = SR2_3D(**bf_params)
        elif bf_type == BeamformerType.CB:
            if data_type != PWDataType.RF:
                raise NotImplementedError(
                    "Compressing Beamforming is implemented only for RF raw 3D data"
                )
            if nz is None or nx is None:
                raise ValueError("Compressing Beamforming requires nz and nx")
            bf = CB3D(pw=pw, nz=nz, nx=nx, **bf_params)
        else:
            bf = DAS3D(**bf_params)
    else:
        if bf_type == BeamformerType.DAS:
            bf = DAS4D(**bf_params)
        elif bf_type == BeamformerType.FDMAS:
            raise NotImplementedError("F-DMAS is implemented only for RF 3D data")
        elif bf_type == BeamformerType.MV:
            bf = MV4D(**bf_params)
        elif bf_type == BeamformerType.CF:
            bf = CF4D(**bf_params)
        elif bf_type == BeamformerType.IMAP:
            bf = IMAP4D(**bf_params)
        elif bf_type == BeamformerType.SR1:
            bf = SR4D(**bf_params)
        elif bf_type == BeamformerType.SR2:
            bf = SR2_4D(**bf_params)
        elif bf_type == BeamformerType.CB:
            raise NotImplementedError(
                "Compressing Beamforming is implemented only for RF raw 3D data"
            )
        else:
            bf = DAS4D(**bf_params)

    return bf
