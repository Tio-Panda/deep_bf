import matplotlib.pyplot as plt


def get_axes_grid(rows, cols, figsize=(18,6), squeeze=False, **subplot_kwargs):
    """
    Crea una grilla regular de filas x columnas y retorna axes flatten.

    Returns
    -------
    fig
    axes
    axes_flat : list[matplotlib.axes.Axes]
    """
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("'rows' debe ser un entero positivo.")
    if not isinstance(cols, int) or cols <= 0:
        raise ValueError("'cols' debe ser un entero positivo.")

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=figsize,
        squeeze=squeeze,
        **subplot_kwargs,
    )

    if hasattr(axes, "ravel"):
        axes_flat = list(axes.ravel())
    else:
        axes_flat = [axes]

    return fig, axes_flat


def get_axes_flex(layout, figsize=None, hspace=0.2, wspace=0.2, **figure_kwargs):
    """
    Crea una grilla flexible por filas y retorna axes flatten.

    layout es una lista/tupla con la cantidad de columnas por fila.
    Ejemplo: [2, 4, 4]

    Returns
    -------
    fig
    axes_by_row : list[list[matplotlib.axes.Axes]]
    axes_flat : list[matplotlib.axes.Axes]
    """
    if not isinstance(layout, (list, tuple)) or len(layout) == 0:
        raise ValueError("'layout' debe ser una lista/tupla no vacia de enteros positivos.")

    for idx, ncols in enumerate(layout):
        if not isinstance(ncols, int) or ncols <= 0:
            raise ValueError(
                f"layout[{idx}] debe ser un entero positivo, se recibio: {ncols!r}"
            )

    fig = plt.figure(figsize=figsize, **figure_kwargs)
    root = fig.add_gridspec(nrows=len(layout), ncols=1, hspace=hspace)

    axes_by_row = []
    axes_flat = []

    for row_idx, ncols in enumerate(layout):
        sub = root[row_idx].subgridspec(1, ncols, wspace=wspace)
        row_axes = []

        for col_idx in range(ncols):
            ax = fig.add_subplot(sub[0, col_idx])
            row_axes.append(ax)
            axes_flat.append(ax)

        axes_by_row.append(row_axes)

    return fig, axes_flat
