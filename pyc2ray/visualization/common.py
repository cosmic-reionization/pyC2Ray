import matplotlib.pyplot as plt


def xfrac_plot(
    data,
    ax,
    xmin=None,
    xmax=1.0,
    cmap="jet",
    interp=None,
    fs=10,
    boxsize=None,
    time=None,
    label=r"$x_{HI}$",
):
    if boxsize is None:
        unitstr = "[Grid Coordinates]"
        ext = None
    else:
        unitstr = "[kpc]"
        ext = (0, boxsize, 0, boxsize)
    im = ax.imshow(
        data,
        origin="lower",
        norm="log",
        cmap=cmap,
        interpolation=interp,
        vmin=xmin,
        vmax=xmax,
        extent=ext,
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(label=label, size=1.5 * fs)
    ax.set_xlabel("$x$ " + unitstr, fontsize=fs)
    ax.set_ylabel("$y$ " + unitstr, fontsize=fs)
    if time is not None:
        pass
        # ttl = ax.text(
        #     0.01,
        #     0.99,
        #     "$t={:.2f}$ Myr".format(time),
        #     horizontalalignment="left",
        #     verticalalignment="top",
        #     color="white",
        #     transform=ax.transAxes,
        # )

    return im, cb


def resid_plot(
    data, ax, clim=None, cmap="bwr", interp=None, fs=10, boxsize=None, time=None
):
    if boxsize is None:
        unitstr = "[Grid Coordinates]"
        ext = None
    else:
        unitstr = "[kpc]"
        ext = (0, boxsize, 0, boxsize)
    if clim is None:
        vlim = max(abs(data.min()), data.max())
    else:
        vlim = clim
    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        extent=ext,
        interpolation="none",
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(
        label=r"$\langle x_{HI}\rangle/\langle x_{HI}^{ref}\rangle - 1$", size=1.5 * fs
    )
    ax.set_xlabel("$x$ " + unitstr, fontsize=fs)
    ax.set_ylabel("$y$ " + unitstr, fontsize=fs)
    if time is not None:
        pass
        # ttl = ax.text(
        #     0.01,
        #     0.99,
        #     "$t={:.2f}$ Myr".format(time),
        #     horizontalalignment="left",
        #     verticalalignment="top",
        #     color="white",
        #     transform=ax.transAxes,
        # )

    return im, cb
