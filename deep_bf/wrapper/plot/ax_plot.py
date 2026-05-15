import matplotlib.pyplot as plt

from ..reconstruction.reconstruction import Reconstruction
from ...config_registery import BeamformerConfig

def get_bf_title(reconstruction: Reconstruction):
    bfC: BeamformerConfig = reconstruction.beamformer_config
    bf_type = bfC.type

    return f"{bf_type}"


def get_compounding_title(reconstruction: Reconstruction):
    cC = reconstruction.compounding_config
    cmp_type = cC.type

    return f"{cmp_type}"

# TODO: Agregar como parametro una funcion que con una reconstruccion me genere un subtitulo para el grafico.
def simple_bmode(ax, reconstruction:Reconstruction, fn_title, vmin=-60, vmax=0, eps=1e-10):
    bmode = reconstruction.get_bmode(vmin, vmax, eps)

    # TODO: setear el titulo fuera de esta funcion
    title = fn_title(reconstruction)

    zlims = reconstruction.zlims
    xlims = reconstruction.xlims

    extent = (xlims[0], xlims[-1], zlims[-1], zlims[0])

    ax.imshow(bmode, cmap="grey", vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_axis_off()

    return ax
