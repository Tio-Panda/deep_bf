import matplotlib.pyplot as plt

def show_simple_plot(bmode, pw, vmin=-60, vmax=0):

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])  # left, bottom, width, height
    ax.set_axis_off()
    fig.add_axes(ax)

    zlims = pw.zlims
    xlims = pw.xlims

    extent = (xlims[0], xlims[-1], zlims[-1], zlims[0])
    ax.imshow(bmode, cmap="gray", vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
    plt.show()
