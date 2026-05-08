import matplotlib.pyplot as plt

from ..reconstruction.reconstruction import Reconstruction

def simple_bmode(ax, reconstruction:Reconstruction, title, vmin=60, vmax=0, eps=1e-10):
    bmode = reconstruction.get_bmode(vmin, vmax, eps)
    nz, nx = bmode.shape

    # TODO: setear el titulo fuera de esta funcion
    title = f"{nz} x {nx}"

    zlims = reconstruction.zlims
    xlims = reconstruction.xlims

    extent = (xlims[0], xlims[-1], zlims[-1], zlims[0])

    ax.imshow(bmode, cmap="grey", vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_axis_off()

    return ax, nz, nx
