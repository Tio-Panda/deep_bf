import matplotlib.pyplot as plt

from ..reconstruction.reconstruction import Reconstruction
from ..reconstruction.model_reconstruction import ModelReconstruction
from ...config_registery import BeamformerConfig

def get_bf_title(reconstruction: Reconstruction, custom=""):
    bfC: BeamformerConfig = reconstruction.beamformer_config
    bf_type = bfC.type

    return f"{bf_type}"


def get_bf_title_custom(reconstruction: Reconstruction, custom):
    bfC: BeamformerConfig = reconstruction.beamformer_config
    bf_type = bfC.type

    return f"{bf_type}-{custom}"

def get_compounding_title(reconstruction: Reconstruction, custom=""):
    cC = reconstruction.compounding_config
    cmp_type = cC.type

    return f"{cmp_type}"

def get_exp_desc_title(reconstruction: ModelReconstruction, custom=""):
    title  = reconstruction.experiment.description
    return f"{title}"


def get_custom_title(title, custom=""):
    return f"{custom}"

# TODO: Agregar como parametro una funcion que con una reconstruccion me genere un subtitulo para el grafico.
def simple_bmode(ax, reconstruction:Reconstruction|ModelReconstruction, fn_title, vmin=-60, vmax=0, eps=1e-10, custom=""):
    bmode = reconstruction.get_bmode(vmin, vmax, eps)

    title = fn_title(reconstruction, custom)

    zlims = reconstruction.zlims
    xlims = reconstruction.xlims

    extent = (xlims[0], xlims[-1], zlims[-1], zlims[0])

    ax.imshow(bmode, cmap="grey", vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_axis_off()

    return ax
